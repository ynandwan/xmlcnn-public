
from sklearn.metrics import precision_recall_fscore_support as prfs
from torch.autograd import Variable
from IPython.core.debugger import Pdb
import torch.nn.functional as F
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import scipy as sp
import re
import shutil
import collections

import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle
from datetime import datetime as dt
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
import time
from sklearn.ensemble import RandomForestClassifier

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from nltk import pos_tag
import copy

from torchtext import data

from functools import reduce

import csv
import html

import torch.utils as tutils

CONSOLE_FILE = 'IPYTHON_CONSOLE'




def log(s, file=None):
    print('{},{}'.format(dt.now().strftime('%Y%m%d%H%M%S'), s), file=file)
    print('{},{}'.format(dt.now().strftime('%Y%m%d%H%M%S'), s), file=None)
    print('{},{}'.format(dt.now().strftime('%Y%m%d%H%M%S'), s),
          file=open(CONSOLE_FILE, 'a'))
    if file is not None:
        file.flush()

def process(line):
    line = html.unescape(line)
    line = MULTI_OCCUR_REGEX.sub('', line)
    return line


def save_checkpoint(state, epoch, isBest, checkpoint_file, best_file):
    torch.save(state, checkpoint_file)
    if isBest:
        print("isBest True. Epoch: {0}, bestError: {1}".format(
            state['epoch'], state['best_score']))
        best_file = best_file + str(0)
        shutil.copyfile(checkpoint_file,
                        best_file)



def get_inv(x,inv,sep=' '):
    return sep.join(map(lambda temp: inv[temp], x.astype(int)))

def get_inv_list(x,inv):
    return list(map(lambda temp: inv[temp], x.astype(int)))



