
import train_xml 
import cnntext
import utils
import xml_dataset
import argparse
import torch
import pickle
import html
import data_helpers as tdh
import re
import numpy as np
from torch.autograd import Variable

from nltk.tokenize.moses import MosesTokenizer
moses_tokenizer = MosesTokenizer()

tkn = moses_tokenizer.tokenize

from nltk import TweetTokenizer

tweet_tokenizer = TweetTokenizer(reduce_len=True)



multi_occur_regex = re.compile(r'(\W)(?=\1)')

cuda = False

class XMLCNNDemo:
    def __init__(self, args):
        global cuda
        cuda = torch.cuda.is_available()
        self.cuda = cuda
        if cuda:
            train_xml.cuda = cuda
            cnntext.cuda = cuda
            xml_dataset.cuda = cuda	

        utils.log('start loading vocab')
        self.vocabs = pickle.load(open(args.vocab_path, 'rb'))
        utils.log('done loading vocab')

        _ = xml_dataset.xml_dataset(max_length= args.max_length, vocab_size = args.vocab_size, min_count = args.min_count, word_counts = self.vocabs['word_counts'], vocabulary_inv= self.vocabs['vocabulary_inv'], only_initialize_vocab = True)
        self.vocabulary = {x: i for i, x in enumerate(xml_dataset.xml_dataset.vocabulary_inv)} 
        self.vocab_size = xml_dataset.xml_dataset.vocab_size
        embedding_init = self.vocabs['embedding_init']
        embedding_init = embedding_init[:self.vocab_size]
        self.num_labels = len(self.vocabs['labels_inv'])
        self.model = cnntext.select_model(args, self.num_labels, self.vocab_size, embedding_init = embedding_init)
        self.my_loss_fn = train_xml.get_loss_fn(args)
        self.model.eval()
        cp = torch.load(args.checkpoint)
        self.model.load_state_dict(cp['model'])
        self.which_data = args.which_data



    def preprocess(self,line):
        if line is None:
            return [[0]]

        if self.which_data == 'amazon':
            
            line = html.unescape(line)
            line = ' '.join(tkn(multi_occur_regex.sub('',line))).lower()
            sentences = [(tdh.clean_str(line).split(" "))]
            return (tdh.build_input_data(sentences,self.vocabulary))

        elif self.which_data == 'eurlex':
            pass
        elif self.which_data == 'tweets':
            line = ' '.join(tweet_tokenizer.tokenize(line))
            sentences = [(tdh.clean_str(line).split(" "))]
            return (tdh.build_input_data(sentences,self.vocabulary))


        return [[0]]

    def predict_labels(self,s,t=None,gold=None, topk = 5):
        self.model.eval()
        ns = self.preprocess(s)
        nt = self.preprocess(t)
        if gold is None:
            ngold = [np.ones(self.num_labels)]
        else:
            ngold = [np.zeros(self.num_labels)]
            ngold[0][np.array(gold)]  = 1
        #

        var = [torch.LongTensor(ns), torch.LongTensor(nt), torch.LongTensor([len(nt[0])]), torch.FloatTensor(ngold), torch.LongTensor([0])]

        for index in range(len(var)-1):
            var[index] = Variable(var[index], volatile = True)
            if self.cuda:
                var[index] = var[index].cuda()

        y,ypred,loss = self.my_loss_fn(var, self.model)
        patk, gatk,p1,topk_pred,act,ec = train_xml.calculate_accuracies(ypred.data.cpu(), y.data.cpu(), [topk])
        topk_pred1 = topk_pred.numpy().astype(int)
        predicted_labels = utils.get_inv_list(topk_pred1[0], self.vocabs['labels_inv'])
        return predicted_labels

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
   
    
    parser.add_argument('--which_data',
                        help='eurlex or amazon or twitter?', type=str,
                        default='amazon')

    parser.add_argument('--vocab_path',
                        help='vocab in pickle', type=str,
                        #default='/home/cse/phd/csz178057/phd/nlp/project/xmlcnn/data/eurlex/vocab2.pkl')
                        default='/home/cse/phd/csz178057/phd/nlp/project/xmlcnn/data/amazon13k/vocab2.pkl')
    
    parser.add_argument('--max_length',
                        help='max_sentence length', type=int,
                        default=500)

    parser.add_argument('--vocab_size',
                        help='max_sentence length', type=int,
                        default=150000)

    parser.add_argument('--min_count',
                        help='max_sentence length', type=int,
                        default=2)


    #model creation params
    parser.add_argument('--has_titles', help=' has titles in addn to description',
                        action='store_true')

    parser.add_argument('--title_hidden_size', help=' has titles in addn to description',type=int,
            default = 256)
    
    
    parser.add_argument('--kernels',
                        help='number of filter sizes (could be a list of integer)', type=int,
                        default=[2, 4, 8], nargs='+')
    parser.add_argument('--channels',
                        help='number of filters (i.e. kernels) in CNN model', type=int,
                        default=128)
    parser.add_argument('--num_features',
                        help='number of pooling units in 1D pooling layer', type=int,
                        default=8)
    parser.add_argument('--hidden_size',
                        help='number of hidden units', type=int,
                        default=512)
    parser.add_argument('--lstm', help=' should apply lstm?',
                        action='store_true')
    parser.add_argument('--attn', help=' should apply attn',
                        action='store_true')
    parser.add_argument('--lstm_hidden_size',
                        help='number of hidden units in lstm', type=int,
                        default=256)
    
    parser.add_argument('--checkpoint',
                        help='continue from this checkpoint', type=str,
                        default='')

    args = parser.parse_args()
    

    #args.has_titles = True
    #args.checkpoint = '/home/cse/phd/csz178057/phd/nlp/project/xmlcnn/output/amazon13k/title1/eurlex_lr_1e-05_nf_8_ch_128_decay_0.0_hd_512_kernels_2-4-8_ml_500_vmc_2_vs_150000_l_False_a_False_best_checkpoint.pth0'

    #d = demo.XMLCNNDemo(args)
    #d.predict_labels('this is a great product')

