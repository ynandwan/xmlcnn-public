import pickle
import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', help='exp name',
                    type=str, default='eurlex')

parser.add_argument('--train_path',
                    help='numericalized data in pickle', type=str,
                    default='/home/cse/phd/csz178057/phd/nlp/project/xmlcnn/data/eurlex/train2.pkl')
                    #default='/home/cse/phd/csz178057/phd/nlp/project/xmlcnn/data/amazon13k/train2.pkl')
parser.add_argument('--val_path',
                    help='val numericalized data in pickle', type=str,
                    default='/home/cse/phd/csz178057/phd/nlp/project/xmlcnn/data/eurlex/test2.pkl')
                    #default='/home/cse/phd/csz178057/phd/nlp/project/xmlcnn/data/amazon13k/test2.pkl')
parser.add_argument('--vocab_path',
                    help='vocab in pickle', type=str,
                    default='/home/cse/phd/csz178057/phd/nlp/project/xmlcnn/data/eurlex/vocab2.pkl')
                    #default='/home/cse/phd/csz178057/phd/nlp/project/xmlcnn/data/amazon13k/vocab2.pkl')

args = parser.parse_args()
train = pickle.load(open(args.train_path,'rb'))
val= pickle.load(open(args.val_path,'rb'))
vocab = pickle.load(open(args.vocab_path,'rb'))

N,L = train['y'].shape
M = len(val['x'])
D = len(vocab['word_counts'])

Lbar = train['y'].sum()*1.0/N
Ltild = train['y'].sum()*1.0/L
Wbar = sum(map(len, train['x']))/ N
Wcap=  sum(map(len, val['x']))/ M

print(N,M,D,L,Lbar,Ltild,Wbar,Wcap)


