
import numpy as np
import utils
import pandas as pd

all_tweets_file = '../data/tweets/multilabel.json'
label_mapping_file = '../data/tweets/multilabel.tsv'


train_output_file = '../data/tweets/train2.pkl'
test_output_file = '../data/tweets/test2.pkl'
val_output_file = '../data/tweets/val2.pkl'
vocab_output_file = '../data/tweets/vocab2.pkl'
word_count_txt_file = '../data/tweets/tweets_vocab_freq.txt'


#create_test_train_split

tweets = pd.read_json(all_tweets_file,orient= 'records',lines = True)
labels_map = pd.read_csv(label_mapping_file, sep='\t',header=None)
labels_map.columns = ['idx','emoji','emoji_desc']

#el tweets['raw']
#el tweets['text']
np.random.seed(173)
samples = np.random.sample(len(tweets))
split = np.zeros(len(tweets))
split[samples < 0.2] = 1
split[(0.2 <= samples) & (samples < 0.4)] = 2

tweets['split']  = split
tweets.split.value_counts()/ len(tweets)

nlabels = tweets['labels'].apply(len)
tweets['nlabels'] = nlabels
nlabels.value_counts()



label_set = set()
for l in tweets['labels']:
    label_set = label_set.union(set(l))


num_labels = max(label_set) + 1

#weets.columns = ['catgy','text','split']
nwords = tweets['tokens'].apply(len)
tweets['nwords'] = nwords

tweets = tweets[tweets.nwords > 0]

train_tweets = tweets[tweets.split == 0]
val_tweets = tweets[tweets.split == 1]
test_tweets = tweets[tweets.split == 2]

train_tweet_list = [{'text': ' '.join(row[4]), 'catgy': row[1]} for row in train_tweets.itertuples()]
test_tweet_list = [{'text': ' '.join(row[4]), 'catgy': row[1]} for row in test_tweets.itertuples()]
val_tweet_list = [{'text': ' '.join(row[4]), 'catgy': row[1]} for row in val_tweets.itertuples()]



trn_sents, Y_trn = tdh.load_data_and_labels(train_tweet_list,num_labels)
tst_sents, Y_tst = tdh.load_data_and_labels(test_tweet_list,num_labels)
val_sents, Y_val = tdh.load_data_and_labels(val_tweet_list,num_labels)

embedding_file = os.path.join('../data/word2vec_models/', 'glove.6B.%dd.txt' % (num_features))

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

trt = train_tweets.to_json(orient='records',lines=True)
print(trt,file = open('../data/tweets/train_tweets.json','w'))
print(test_tweets.to_json(orient ='records',lines=True),file = open('../data/tweets/test_tweets.json','w'))
print(val_tweets.to_json(orient ='records',lines=True),file = open('../data/tweets/val_tweets.json','w'))



