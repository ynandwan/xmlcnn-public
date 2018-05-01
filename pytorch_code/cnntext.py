import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils as tutils

cuda = False

def select_model(args, num_labels, vocab_size, embedding_init):
    if args.has_titles:
        print('Using titles')
        model = Xmlcnn_titles(num_labels, vocab_size, embedding_size=embedding_init.shape[1], num_features=args.num_features,hidden_size=args.hidden_size, kernels=args.kernels, channels=args.channels, lstm = args.lstm, attention = args.attn, lstm_hidden_size = args.lstm_hidden_size, embedding_init=embedding_init, title_hidden_size = args.title_hidden_size)
        
    else:
        model = Xmlcnn_lstm_attn(num_labels, vocab_size, embedding_size=embedding_init.shape[1],num_features=args.num_features,hidden_size=args.hidden_size, kernels=args.kernels, channels=args.channels, lstm = args.lstm, attention = args.attn, lstm_hidden_size = args.lstm_hidden_size, embedding_init=embedding_init)

    if cuda:
        model = model.cuda()
    return model

class Xmlcnn(nn.Module):
    def __init__(self, num_labels, vocab_size, embedding_size=300, num_features=32, hidden_size = 512, kernels = [3,5,9], channels = 128, embedding_init = None):
        super(Xmlcnn, self).__init__()
        self.embedding_size = embedding_size
        self.num_labels = num_labels
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.kernels = kernels
        self.channels = channels
       
        self.word_embeddings = nn.Embedding(vocab_size,embedding_size)
        if embedding_init is not None:
            self.word_embeddings.weight.data.copy_(torch.FloatTensor(embedding_init))

        #  
        self.embedding_dropout = nn.Dropout(p = 0.25)
        ml = []
        for kernel_size in kernels:
            conv1d = nn.Conv1d(self.embedding_size, self.channels, kernel_size,stride = 1)
            apool1d  = nn.AdaptiveMaxPool1d(self.num_features)
            ml.append(nn.Sequential(conv1d, nn.ReLU(),  apool1d))
        #
        self.convList = nn.ModuleList(ml)
    
        self.fc1 = nn.Sequential(
                #nn.Dropout(0.5),
                 nn.Linear(self.channels*self.num_features*len(self.kernels), self.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.5)
                )

        self.classifier = nn.Linear(self.hidden_size, self.num_labels)


    
    def forward(self, x):
        #Pdb().set_trace()
        emb = self.word_embeddings(x)
        emb = self.embedding_dropout(emb).transpose(1,2)
        if emb.size()[-1] < max(self.kernels):
            pad_size = max(self.kernels) - emb.size()[-1] + 1
            emb = F.pad(emb,(0,pad_size), "constant", 0)
        features = [conv(emb) for conv in self.convList]
        features = torch.cat(features,dim = 2)
        features = features.view(len(features),-1)
        features = self.fc1(features)
        y = self.classifier(features)
        return y




class Xmlcnn_lstm(nn.Module):
    def __init__(self, num_labels, vocab_size, embedding_size=300, num_features=32, hidden_size = 512, kernels = [3,5,9], channels = 128, lstm_hidden_size= 128,  embedding_init = None):
        super(Xmlcnn_lstm, self).__init__()
        self.embedding_size = embedding_size
        self.num_labels = num_labels
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.kernels = kernels
        self.channels = channels
        self.lstm_hidden_size = lstm_hidden_size 
        self.word_embeddings = nn.Embedding(vocab_size,embedding_size)
        if embedding_init is not None:
            self.word_embeddings.weight.data.copy_(torch.FloatTensor(embedding_init))

        #  
        self.embedding_dropout = nn.Dropout(p = 0.25)
        ml = []
        for kernel_size in kernels:
            conv1d = nn.Conv1d(self.embedding_size, self.channels, kernel_size,stride = 1)
            apool1d  = nn.AdaptiveMaxPool1d(self.num_features)
            ml.append(nn.Sequential(conv1d, nn.ReLU(),  apool1d))
        #
        self.convList = nn.ModuleList(ml)
    
        self.rnn = nn.LSTM(self.channels, self.lstm_hidden_size, bidirectional = True, batch_first= True, dropout = 0.5)


        self.fc1 = nn.Sequential(
                #nn.Dropout(0.5),
                 nn.Linear(2*self.lstm_hidden_size*len(self.kernels), self.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.5)
                )

        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, x):

        emb = self.word_embeddings(x)
        emb = self.embedding_dropout(emb).transpose(1,2)
        if emb.size()[-1] < max(self.kernels):
            pad_size = max(self.kernels) - emb.size()[-1] + 1
            emb = F.pad(emb,(0,pad_size), "constant", 0)
        features = [conv(emb) for conv in self.convList]
        features = [self.rnn(f.transpose(1,2),None)[1][0] for f in features] 
        features = [torch.cat([f[0],f[1]], dim = 1) for f in features]
        features = torch.cat(features,dim = 1)
        #features = features.view(len(features),-1)
        features = self.fc1(features)
        y = self.classifier(features)
        return y




class Xmlcnn_titles(nn.Module):
    def __init__(self, num_labels, vocab_size, embedding_size=300, num_features=32, hidden_size = 512, kernels = [3,5,9], channels = 128, lstm_hidden_size= 128,  lstm = False, attention=False, embedding_init = None, title_hidden_size= 256):
        super(Xmlcnn_titles, self).__init__()
        self.title_hidden_size = title_hidden_size

        self.cnn_features_extractor = Xmlcnn_feature_extractor(num_labels, vocab_size, embedding_size, num_features, hidden_size, kernels, channels , lstm_hidden_size,  lstm , attention, embedding_init)

        self.title_rnn = nn.LSTM(embedding_size, self.title_hidden_size, bidirectional = True, batch_first= True, dropout = 0.5)
       
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.fc1 = nn.Sequential(
                 nn.Linear(self.cnn_features_extractor.features_dim+self.title_hidden_size*2, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.5)
                )

        self.classifier = nn.Linear(self.hidden_size, self.num_labels)


    def forward(self,x,titles, title_len):
        cnn_features = self.cnn_features_extractor(x)
        emb = self.cnn_features_extractor.word_embeddings(titles)
        emb = self.cnn_features_extractor.embedding_dropout(emb)
        emb_ =  tutils.rnn.pack_padded_sequence(emb, title_len.data.tolist(), batch_first = True)

        _,th = rnn_features = self.title_rnn(emb_,None)
        features = torch.cat([cnn_features, th[0][0], th[0][1]], dim = 1)
        features = self.fc1(features)
        features = self.classifier(features)
        return (features)



class Xmlcnn_feature_extractor(nn.Module):
    def __init__(self, num_labels, vocab_size, embedding_size=300, num_features=32, hidden_size = 512, kernels = [3,5,9], channels = 128, lstm_hidden_size= 128,  lstm = False, attention=False, embedding_init = None):
        super(Xmlcnn_feature_extractor, self).__init__()
        self.embedding_size = embedding_size
        self.num_labels = num_labels
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.kernels = kernels
        self.channels = channels
        self.lstm_hidden_size = lstm_hidden_size 
        self.is_lstm = lstm
        self.attention = attention

        self.word_embeddings = nn.Embedding(vocab_size,embedding_size,padding_idx = 1)
        if embedding_init is not None:
            self.word_embeddings.weight.data.copy_(torch.FloatTensor(embedding_init))

        #  
        self.embedding_dropout = nn.Dropout(p = 0.25)
        ml = []
        for kernel_size in kernels:
            conv1d = nn.Conv1d(self.embedding_size, self.channels, kernel_size,stride = 1)
            apool1d  = nn.AdaptiveMaxPool1d(self.num_features)
            ml.append(nn.Sequential(conv1d, nn.ReLU(),  apool1d))
        #
        self.convList = nn.ModuleList(ml)
  
        fc1_input = self.channels*self.num_features*len(self.kernels)

        if self.is_lstm:
            self.rnn = nn.LSTM(self.channels, self.lstm_hidden_size, bidirectional = True, batch_first= True, dropout = 0.5)
    
            fc1_input = 2*self.lstm_hidden_size*len(self.kernels)
            if self.attention:
                self.h2e  = nn.Linear(self.lstm_hidden_size*2, self.lstm_hidden_size*2)
                self.attn = nn.Linear(self.lstm_hidden_size*2,1,bias = False)

        #   
        self.features_dim = fc1_input


    def get_features(self,x):
        emb = self.word_embeddings(x)
        emb = self.embedding_dropout(emb).transpose(1,2)
        if emb.size()[-1] < max(self.kernels):
            pad_size = max(self.kernels) - emb.size()[-1] + 1
            emb = F.pad(emb,(0,pad_size), "constant", 0)
        features = [conv(emb) for conv in self.convList]
        if self.is_lstm:
            if self.attention:
                features = [self.rnn(f.transpose(1,2),None)[0] for f in features] 
                emb_features = [F.tanh(self.h2e(f)) for f in features]
                attn_wts = [self.attn(f) for f in emb_features]
                attn_wts = [F.softmax(f.squeeze(-1),dim = 1) for f in attn_wts]
                features = [torch.bmm(a.unsqueeze(1), f).squeeze(1) for (a,f) in zip(attn_wts,features)]

            else:
                features = [self.rnn(f.transpose(1,2),None)[1][0] for f in features]
                features = [torch.cat([f[0],f[1]], dim = 1) for f in features]
            #
            features = torch.cat(features,dim = 1)
        else:
            features = torch.cat(features,dim = 2)
            features = features.view(len(features),-1)
        return features

    def forward(self, x):
        return (self.get_features(x))


class Xmlcnn_lstm_attn(nn.Module):
    def __init__(self, num_labels, vocab_size, embedding_size=300, num_features=32, hidden_size = 512, kernels = [3,5,9], channels = 128, lstm_hidden_size= 128,  lstm = False, attention=False, embedding_init = None):
        super(Xmlcnn_lstm_attn, self).__init__()

        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.feature_extractor = Xmlcnn_feature_extractor(num_labels, vocab_size, embedding_size, num_features, hidden_size, kernels, channels , lstm_hidden_size,  lstm , attention, embedding_init)

        self.fc1 = nn.Sequential(
                 nn.Linear(self.feature_extractor.features_dim, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.5)
                )

        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.fc1(features)
        y = self.classifier(features)
        return y

