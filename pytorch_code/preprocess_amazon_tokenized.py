import pickle,os,sys
import data_helpers as tdh
import numpy as np
import utils

test_label_file = '../data/amazon13k/test_labels.txt'
test_features_file = '../data/amazon13k/test_features_tokenized.txt'
train_label_file = '../data/amazon13k/train_labels.txt'
train_features_file = '../data/amazon13k/train_features_tokenized.txt'

train_output_file = '../data/amazon13k/train2.pkl'
test_output_file = '../data/amazon13k/test2.pkl'
vocab_output_file = '../data/amazon13k/vocab2.pkl'
word_count_txt_file = '../data/amazon13k/amazon_vocab_freq.txt'
labels_path ='../data/amazon13k/Y.txt'

num_features = 300
model_name = os.path.join('../data/word2vec_models/', 'glove.6B.%dd.txt' % (num_features))

def create_list_of_dict(label_file, features_file):
    labels = open(label_file).readlines()
    label_header = labels[0]
    num_labels = int(label_header.split()[-1])
    labels = labels[1:]
    labels = np.array(labels)
    ffh= open(features_file,'r')
    lines = ffh.readlines()
    rv = []
    for i in range(len(labels)):
        rv.append({'text': lines[i][:-1], 'catgy': [int(x) for x in labels[i].split(',')]})
    # 
    return (rv, num_labels)

test_data, num_labels = create_list_of_dict(test_label_file, test_features_file)
train_data, num_labels = create_list_of_dict(train_label_file, train_features_file)

tdh.create_test_train_vocabs(train_data, test_data, num_labels, train_output_file, test_output_file, vocab_output_file, word_count_txt_file, model_name, num_features)

#### add titles to existing pickle files
test_title_file = '../data/amazon13k/test_titles.txt'
train_title_file = '../data/amazon13k/train_titles.txt'

vocabs = pickle.load(open(vocab_output_file,'rb'))
train_data = pickle.load(open(train_output_file,'rb'))
test_data = pickle.load(open(test_output_file,'rb'))

train_titles = [x[:-1] for x in open(train_title_file,'r').readlines()]
train_titles_sents = [tdh.clean_str(x).split() for x in train_titles]
test_titles = [x[:-1] for x in open(test_title_file,'r').readlines()]
test_titles_sents = [tdh.clean_str(x).split() for x in test_titles]

X_trn_titles = tdh.build_input_data(train_titles_sents, vocabs['vocabulary'])
X_tst_titles = tdh.build_input_data(test_titles_sents,vocabs['vocabulary'])

train_data['titles'] = X_trn_titles
test_data['titles'] = X_tst_titles
pickle.dump(train_data, open(train_output_file,'wb'))
pickle.dump(test_data, open(test_output_file,'wb'))

#### add labels inv to existing vocabs
vocabs = pickle.load(open(vocab_output_file,'rb'))
labels_inv = [x[:-1] for x in open(labels_path,'r',errors='ignore').readlines()]
vocabs['labels_inv'] = labels_inv
pickle.dump(vocabs, open(vocab_output_file, 'wb'))







