
import pickle,os,sys
import data_helpers as tdh
import numpy as np

data_path = '../data/eurlex/eurlex_raw_text.p'
num_features = 300
model_name = os.path.join('../data/word2vec_models/', 'glove.6B.%dd.txt' % (num_features))

train_output_file = '../data/eurlex/train2.pkl'
test_output_file = '../data/eurlex/test2.pkl'
vocab_output_file = '../data/eurlex/vocab2.pkl'
word_count_txt_file = '../data/eurlex/eurlex_vocab_freq.txt'

with open(os.path.join(data_path), 'rb') as fin:
    [train, test, vocab, catgy] = pickle.load(fin)


num_labels = len(catgy)
tdh.create_test_train_vocabs(train, test, num_labels, train_output_file, test_output_file, vocab_output_file, word_count_txt_file, model_name, num_features)

##### parse 1st 10 and last 10 sentences and treat them as titles

train = pickle.load(open(train_output_file,'rb'))
test = pickle.load(open(test_output_file,'rb'))
N = 10
def get_titles(N,x):
    z = []
    for xi in x:
        if len(xi) < 2*N:
            z.append(xi)
        else:
            z.append(xi[:N] + xi[-N:])
    return z


train['titles'] = get_titles(N,train['x'])
test['titles'] = get_titles(N,test['x'])
pickle.dump(train, open(train_output_file,'wb'))
pickle.dump(test, open(test_output_file,'wb'))
raw = pickle.load(open(data_path,'rb'))

labels_inv = [None]*len(raw[-1])
for k in raw[-1]:
    labels_inv[raw[-1][k]] = k

vocabs  = pickle.load(open(vocab_output_file,'rb'))
vocabs['labels_inv'] = labels_inv
pickle.dump(vocabs, open(vocab_output_file,'wb'))





