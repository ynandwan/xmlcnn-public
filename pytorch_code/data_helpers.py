#Adapted from source - http://manikvarma.org/downloads/XC/XMLRepository.html - From XML-CNN authors

import numpy as np
import os
import re
import itertools
import scipy.sparse as sp
#import cPickle as pickle
import pickle
from collections import Counter
from nltk.corpus import stopwords

cachedStopWords = stopwords.words("english")

def get_vocabs_embeddings(token_list,embedding_file,num_features):
    vocabulary, vocabulary_inv, word_counts  = build_vocab(token_list)
    embedding_model = {}
    for line in open(embedding_file , 'r'):
        tmp = line.strip().split()
        word, vec = tmp[0], list(map(float, tmp[1:]))
        assert(len(vec) == num_features)
        if word not in embedding_model:
            embedding_model[word] = vec 
    #   
    embedding_weights = [embedding_model[w] if w in embedding_model
                            else np.random.uniform(-0.25, 0.25, num_features)
                        for w in vocabulary_inv]
    #   
    embedding_weights = np.array(embedding_weights).astype('float64')
    return {'word_counts':  word_counts, 'vocabulary_inv': vocabulary_inv, 'embedding_init': embedding_weights, 'vocabulary': vocabulary}


def clean_str(string):
    # remove stopwords
    # string = ' '.join([word for word in string.split() if word not in cachedStopWords])
    string = re.sub(r"(\W)(?=\1)","", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\'", " \' ", string)
    string = re.sub(r"\`", " \` ", string)
    return string.strip().lower()


def pad_sentences(sentences, padding_word="<PAD/>", max_length=500):
    sequence_length = min(max(len(x) for x in sentences), max_length)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence) < max_length:
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:max_length]
        padded_sentences.append(new_sentence)
    return padded_sentences


def load_data_and_labels(data, num_labels = 0):
    x_text = [clean_str(doc['text']) for doc in data]
    x_text = [s.split() for s in x_text]
    labels = [doc['catgy'] for doc in data]
    row_idx, col_idx, val_idx = [], [], []
    for i in range(len(labels)):
        l_list = list(set(labels[i])) # remove duplicate cateories to avoid double count
        for y in l_list:
            row_idx.append(i)
            col_idx.append(y)
            val_idx.append(1)
    m = max(row_idx) + 1
    n = max(col_idx) + 1
    n = max(n,num_labels)
    Y = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n))
    return [x_text, Y]


def build_vocab(sentences, vocab_size=None, min_count = 1):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_size)]
    vocabulary_inv = ['<UNK/>','<PAD/>'] + [x for x in vocabulary_inv if word_counts[x] >= min_count]
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    # append <UNK/> symbol to the vocabulary
    #vocabulary['<UNK/>'] = len(vocabulary)
    #vocabulary_inv.append('<UNK/>')
    return [vocabulary, vocabulary_inv,word_counts]


def build_input_data(sentences, vocabulary):
    x = np.array([[vocabulary[word] if word in vocabulary else vocabulary['<UNK/>'] for word in sentence] for sentence in sentences])
    #x = np.array([[vocabulary[word] if word in vocabulary else len(vocabulary) for word in sentence] for sentence in sentences])
    return x



def create_test_train_vocabs(train_data, test_data, num_labels, train_output_file, test_output_file, vocab_output_file, word_count_txt_file, glove_file, num_features = 300):
    trn_sents, Y_trn = load_data_and_labels(train_data, num_labels)
    tst_sents, Y_tst = load_data_and_labels(test_data, num_labels)

    vocabulary, vocabulary_inv, word_counts = build_vocab(trn_sents)
    X_trn = build_input_data(trn_sents, vocabulary)
    X_tst = build_input_data(tst_sents, vocabulary)
    
    model_name = glove_file
    assert(os.path.exists(model_name))
    print('Loading existing Word2Vec model (Glove.6B.%dd)' % (num_features))

    # dictionary, where key is word, value is word vectors
    embedding_model = {}
    for line in open(model_name, 'r'):
        tmp = line.strip().split()
        word, vec = tmp[0], list(map(float, tmp[1:]))
        assert(len(vec) == num_features)
        if word not in embedding_model:
            embedding_model[word] = vec

    embedding_weights = [embedding_model[w] if w in embedding_model
                            else np.random.uniform(-0.25, 0.25, num_features)
                        for w in vocabulary_inv]


    embedding_weights = np.array(embedding_weights).astype('float64')

    pickle.dump({'x': X_trn, 'y': Y_trn}, open(train_output_file,'wb'))
    pickle.dump({'x': X_tst, 'y': Y_tst}, open(test_output_file,'wb'))
    pickle.dump({'word_counts':  word_counts, 'vocabulary_inv': vocabulary_inv, 'embedding_init': embedding_weights, 'vocabulary': vocabulary}, open(vocab_output_file,'wb'))

    print('\n'.join(['{},{}'.format(x[0],x[1]) for x in word_counts.most_common(None)]),file = open(word_count_txt_file, 'w'))

