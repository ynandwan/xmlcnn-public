import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.utils as tutils
import utils
from IPython.core.debugger import Pdb 
import pickle
import copy
import data_samplers 
import itertools

MAX_SENTENCE_LENGTH = 999999


def sort_sentence_collate(x):
    #return (x)
    transposed = list(zip(*x))
    transposed[1] = [temp if len(temp) > 0 else [0] for temp in transposed[1]]
    summary_len  = torch.LongTensor([len(temp) for temp in transposed[1]])
    _,y = summary_len.sort(descending = True)
    max_summary_len = max(summary_len)
    
    titles =torch.LongTensor(np.array(list(itertools.zip_longest(*transposed[1], fillvalue=1))).T)
    transposed[1]= titles
    x = [tutils.data.dataloader.default_collate(samples) for samples in transposed]
    return (x[0][y], x[1][y], summary_len[y], x[2][y], x[3][y])


def get_data_loaders(args, vocabs):
    train_ds = xml_dataset(max_length = args.max_length, vocab_size = args.vocab_size, min_count = args.min_count, word_counts = vocabs['word_counts'], root_dir = '.', train_val_test = 'train', pkl_data_path = args.train_path, has_titles = args.has_titles)
    val_ds = xml_dataset(
        root_dir = '.', train_val_test = 'val', pkl_data_path = args.val_path, max_length=args.max_length, has_titles = args.has_titles)

    if args.has_titles:
        collate_fn = sort_sentence_collate
    else:
        collate_fn = tutils.data.dataloader.default_collate

    train_loader = DataLoader(train_ds, batch_sampler=data_samplers.BatchSampler(list(
        map(lambda x: min(args.max_length, len(x[0])), train_ds)), args.batch_size, shuffle=True), num_workers=4, collate_fn = collate_fn)

    val_loader = DataLoader(val_ds, batch_sampler=data_samplers.BatchSampler(list(
        map(lambda x: min(args.max_length, len(x[0])), val_ds)), args.batch_size, shuffle=False), num_workers=4,collate_fn = collate_fn)

    return (train_loader,val_loader)



class xml_dataset(torch.utils.data.Dataset):
    vocabulary_inv = None
    vocab_size = None

    
    def __init__(self, max_length= None, vocab_size = 500000, min_count = 1, word_counts = None, vocabulary_inv = None, root_dir='.', train_val_test= 'train', pkl_data_path = None, has_titles = False, only_initialize_vocab = False):

        if max_length is None:
            self.max_length = MAX_SENTENCE_LENGTH
        else:
            self.max_length = max_length
        
        if only_initialize_vocab or xml_dataset.vocabulary_inv is None:    
            assert(vocabulary_inv is not None)
            vocab_size = min(vocab_size, len(vocabulary_inv)) + 2
            xml_dataset.vocabulary_inv = vocabulary_inv[:vocab_size]
            xml_dataset.vocabulary_inv = vocabulary_inv[:2] + [x for x in xml_dataset.vocabulary_inv[2:] if word_counts[x] >= min_count]
            xml_dataset.vocab_size = len(xml_dataset.vocabulary_inv)
            #
            
       
        if not only_initialize_vocab:
            self.root_dir=  root_dir
            self.data = pickle.load(open(os.path.join(root_dir,pkl_data_path),'rb'))
            self.has_titles = has_titles
            #
            #
            self.data['raw_x'] = copy.deepcopy(self.data['x'])
            #
            for i in range(len(self.data['x'])):
                self.data['x'][i] = np.array(self.data['x'][i])
                self.data['x'][i][self.data['x'][i] >= xml_dataset.vocab_size] = 0
                n = min(len(self.data['x'][i]),self.max_length)
                self.data['x'][i] = self.data['x'][i][:n]
    
            #
            if self.has_titles:
                for i in range(len(self.data['x'])):
                    self.data['titles'][i] = np.array(self.data['titles'][i])
                    self.data['titles'][i][self.data['titles'][i] >= xml_dataset.vocab_size] = 0

    def __len__(self):
        return len(self.data['x'])



    def __getitem__(self,idx):
        if self.has_titles:
            return (self.data['x'][idx], self.data['titles'][idx], np.ravel(self.data['y'][idx].todense()), idx)
        else:
            return (self.data['x'][idx], np.ravel(self.data['y'][idx].todense()), idx)



