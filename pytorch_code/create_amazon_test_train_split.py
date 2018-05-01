import os
import numpy as np


titles_file = '../data/amazon13k/X_tokenized.txt'

label_file = '../data/amazon13k/labels.txt'
features_file= '../data/amazon13k/text_feat_tokenized.txt'

split_file = '../data/amazon13k/split.0.txt'

test_label_file = '../data/amazon13k/test_labels.txt'
test_features_file = '../data/amazon13k/test_features_tokenized.txt'

train_label_file = '../data/amazon13k/train_labels.txt'
train_features_file = '../data/amazon13k/train_features_tokenized.txt'

test_titles_file= '../data/amazon13k/test_titles.txt'
train_titles_file = '../data/amazon13k/train_titles.txt'


split = [int(x[:-1]) for x in open(split_file).readlines()]
split = np.array(split)
# 1 - test, 0 - train

titles = open(titles_file,errors='ignore').readlines()
titles = np.array(titles)

with open(test_titles_file,'w') as ttf:
    print(''.join(titles[split==1]), file = ttf)


with open(train_titles_file,'w') as ttf:
    print(''.join(titles[split==0]), file = ttf)



labels = open(label_file).readlines()
label_header = labels[0]
num_labels = int(label_header.split()[-1])

labels = labels[1:]
labels = np.array(labels)

test_labels = labels[split == 1]
train_labels = labels[split == 0]

with open(test_label_file,'w') as tlf:
    print('{} {}'.format(len(test_labels), num_labels), file = tlf)
    print(''.join(test_labels), file = tlf)


with open(train_label_file,'w') as tlf:
    print('{} {}'.format(len(train_labels), num_labels), file = tlf)
    print(''.join(train_labels), file = tlf)


ffh= open(features_file,'r')
lines = ffh.readlines()

test_feat_fh = open(test_features_file,'w')
train_feat_fh = open(train_features_file,'w')

for i,line in enumerate(lines):
    if split[i] == 1:
        #test
        print(line[:-1],file = test_feat_fh)
    else:
        #train
        print(line[:-1], file = train_feat_fh)

test_feat_fh.close()
train_feat_fh.close()


