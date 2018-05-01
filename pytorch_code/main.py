from datetime import datetime as dt
import torch.optim as optim
import torch.nn as nn
import os
import time
import argparse
import pickle
import torch
from IPython.core.debugger import Pdb
from torch.utils.data import Dataset, DataLoader
import utils
import xml_dataset
import cnntext
import numpy as np
import data_samplers
import train_xml
import error_analysis
cuda = False


def main(args):
    global cuda
    cuda = torch.cuda.is_available()
    if cuda:
        train_xml.cuda = cuda
        cnntext.cuda = cuda
        #criterion.cuda()

    utils.log('start loading vocab')
    vocabs = pickle.load(open(args.vocab_path, 'rb'))
    utils.log('done loading vocab')


    _ = xml_dataset.xml_dataset(max_length= args.max_length, vocab_size = args.vocab_size, min_count = args.min_count, word_counts = vocabs['word_counts'], vocabulary_inv= vocabs['vocabulary_inv'], only_initialize_vocab = True)

    vocab_size = xml_dataset.xml_dataset.vocab_size
    embedding_init = vocabs['embedding_init']
    embedding_init = embedding_init[:vocab_size]
    num_labels = len(vocabs['labels_inv']) 
    model = cnntext.select_model(args, num_labels, vocab_size, embedding_init)
    my_loss_fn = train_xml.get_loss_fn(args)


    (train_loader, val_loader) = xml_dataset.get_data_loaders(args,vocabs)
    utils.log('done loading vocab')



    #optimizer  = optim.SGD(model.parameters(), momentum = 0.9, lr = lr, weight_decay = 0.0005)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.decay)


    exp_name = '{}_lr_{}_nf_{}_ch_{}_decay_{}_hd_{}_kernels_{}_ml_{}_vmc_{}_vs_{}_l_{}_a_{}'.format ( args.exp_name, args.lr, args.num_features, args.channels, args.decay, args.hidden_size, '-'.join([str(x) for x in args.kernels]), args.max_length, args.min_count, args.vocab_size, args.lstm, args.attn)


    log_file = '{}.csv'.format(exp_name)
    checkpoint_file = os.path.join(args.output_path, '{}_checkpoint.pth'.format(exp_name))

    best_checkpoint_file = os.path.join(args.output_path, '{}_best_checkpoint.pth'.format(exp_name))

    utils.log('save checkpoints at {} and best checkpoint at : {}'.format(checkpoint_file, best_checkpoint_file))

    if not os.path.exists(args.output_path):
        try:
            os.makedirs(args.output_path)
        except:
            pass

    #

    if args.checkpoint != '':
        utils.log('start from checkpoint: {}'.format(args.checkpoint))
        tfh = open(os.path.join(args.output_path, log_file), 'a')
        load_checkpoint_file = args.checkpoint
        cp = torch.load(os.path.join(args.output_path, args.checkpoint))
        model.load_state_dict(cp['model'])
        optimizer.load_state_dict(cp['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
            param_group['weight_decay'] = args.decay

    else:
        tfh = open(os.path.join(args.output_path, log_file), 'w')


    start_epoch = 0
    num_epochs = args.num_epochs
    # Pdb().set_trace()
    utils.log('start train/validate cycle')
    best_score = 0
    for epoch in range(start_epoch, num_epochs):
        train_xml.compute_xml_title(epoch, model, train_loader,
                              optimizer, 'train', tfh, args.backprop_batch_size, [args.lr, exp_name],loss_fn = my_loss_fn)
        rec,i,topk_pred,actual_labels,ec  = train_xml.compute_xml_title(epoch, model, val_loader,
                                    None, 'eval', tfh, args.backprop_batch_size, [args.lr, exp_name], loss_fn = my_loss_fn)

        is_best = False
        utils.log('best score: {}, this score: {}'.format(best_score, rec[i]))
        if rec[i] > best_score:
            best_score = rec[i]
            is_best = True
        #
        utils.save_checkpoint( {
        'epoch': epoch,
        'best_score': best_score,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'is_best': is_best
        } , epoch, is_best, checkpoint_file, best_checkpoint_file)
    #
    tfh.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='exp name',
                        type=str, default='eurlex')
    
    parser.add_argument('--lstm', help=' should apply lstm?',
                        action='store_true')
    
    parser.add_argument('--attn', help=' should apply attn',
                        action='store_true')

    parser.add_argument('--train_path',
                        help='numericalized data in pickle', type=str,
                        default='../data/eurlex/train2.pkl')
    
    parser.add_argument('--val_path',
                        help='val numericalized data in pickle', type=str,
                        default='../data/eurlex/test2.pkl')
    parser.add_argument('--vocab_path',
                        help='vocab in pickle', type=str,
                        default='../data/eurlex/vocab2.pkl')
    parser.add_argument('--output_path',
                        help='output path', type=str,
                        default='../output/eurlex/best_models')
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
    parser.add_argument('--lstm_hidden_size',
                        help='number of hidden units in lstm', type=int,
                        default=256)
    parser.add_argument('--batch_size',
                        help='number of batch size', type=int,
                        default=256)
    parser.add_argument('--num_epochs',
                        help='number of epcohs for training', type=int,
                        default=200)
    parser.add_argument('--lr',
                        help='learning rate', type=float,
                        default=0.001)
    parser.add_argument('--decay',
                        help='decay in adam', type=float,
                        default=0.0)
    parser.add_argument('--max_length',
                        help='max_sentence length', type=int,
                        default=500)

    parser.add_argument('--vocab_size',
                        help='max_sentence length', type=int,
                        default=75000)

    parser.add_argument('--min_count',
                        help='max_sentence length', type=int,
                        default=2)

    parser.add_argument('--backprop_batch_size',
                        help='batch size for backprop', type = int, default=256)

    parser.add_argument('--has_titles', help=' has titles in addn to description',
                        action='store_true')

    parser.add_argument('--title_hidden_size', help=' has titles in addn to description',type=int,
            default = 128)
    
    parser.add_argument('--checkpoint',
                        help='continue from this checkpoint', type=str,
                        default='')
    
    args = parser.parse_args()

    #Pdb().set_trace()
    main(args)

