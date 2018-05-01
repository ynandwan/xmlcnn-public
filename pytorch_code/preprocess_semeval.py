import pickle,os,sys
import data_helpers as tdh
import numpy as np
import utils
import pandas as pd

train_input_file = '../data/semeval/train.json'
test_input_file = '../data/semeval/test.json'
val_input_file = '../data/semeval/val.json'

label_mapping_file = '../data/semeval/us_mapping.txt'


train_output_file = '../data/semeval/train2.pkl'
test_output_file = '../data/semeval/test2.pkl'
val_output_file = '../data/semeval/val2.pkl'
vocab_output_file = '../data/semeval/vocab2.pkl'
word_count_txt_file = '../data/semeval/semeval_vocab_freq.txt'


#create_test_train_split

labels_map = pd.read_csv(label_mapping_file, sep='\t',header=None)
labels_map.columns = ['idx','emoji','emoji_desc','_']

def read_tweets_json(semeval_file):
    semeval = pd.read_json(semeval_file,orient= 'records',lines = True)
    semeval['labels'] = semeval.label.apply(lambda x: [x])
    nwords = semeval['tokens'].apply(len)
    semeval['nwords'] = nwords
    semeval = semeval[semeval.nwords > 0]
    return semeval
    

train_semeval = read_tweets_json(train_input_file)
val_semeval = read_tweets_json(val_input_file)
test_semeval = read_tweets_json(test_input_file)


label_set = set()
for l in train_semeval['labels']:
    label_set = label_set.union(set(l))
#
num_labels = max(label_set) + 1


train_tweet_list = [{'text': ' '.join(row[4]), 'catgy': row[5]} for row in train_semeval.itertuples()]
test_tweet_list = [{'text': ' '.join(row[4]), 'catgy': row[5]} for row in test_semeval.itertuples()]
val_tweet_list = [{'text': ' '.join(row[4]), 'catgy': row[5]} for row in val_semeval.itertuples()]



trn_sents, Y_trn = tdh.load_data_and_labels(train_tweet_list,num_labels)
tst_sents, Y_tst = tdh.load_data_and_labels(test_tweet_list,num_labels)
val_sents, Y_val = tdh.load_data_and_labels(val_tweet_list,num_labels)

embedding_file = '/home/cse/phd/csz178057/scratch/squad/data/glove.6B.300d.txt'
vocabs = tdh.get_vocabs_embeddings(trn_sents, embedding_file,num_features= 300)
labels_inv = list(labels_map.emoji_desc)
vocabs['labels_inv'] = labels_inv


print('\n'.join(['{},{}'.format(x[0],x[1]) for x in vocabs['word_counts'].most_common(None)]),file = open(word_count_txt_file, 'w'))

X_trn = tdh.build_input_data(trn_sents, vocabs['vocabulary'])
X_tst = tdh.build_input_data(tst_sents, vocabs['vocabulary'])
X_val = tdh.build_input_data(val_sents, vocabs['vocabulary'])

pickle.dump({'x': X_trn, 'y': Y_trn}, open(train_output_file,'wb'))
pickle.dump({'x': X_tst, 'y': Y_tst}, open(test_output_file,'wb'))
pickle.dump({'x': X_val, 'y': Y_val}, open(val_output_file,'wb'))
pickle.dump(vocabs, open(vocab_output_file,'wb'))


