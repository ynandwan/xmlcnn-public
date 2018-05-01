from datetime import datetime as dt
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import os
import sys
import time
import argparse
import pickle
import yaml
import torch
import shutil
import utils
import cnntext
import data_samplers
import xml_dataset
from IPython.core.debugger import Pdb
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support as prfs
import sklearn.metrics as metrics
import numpy as np

cuda = False

def calculate_accuracies(ypred, y, topk_list):
    p1ind = ypred.numpy().argmax(axis=1)
    p1ind = np.ravel(p1ind)
    p1 = y.numpy()[np.arange(y.numpy().shape[0]), p1ind].sum()
    topk = max(topk_list)
    a, b = torch.topk(ypred, topk, dim=1)
    p = torch.stack([y[np.arange(len(y)), b[:, i]]
                     for i in range(topk)], dim=1)
    
    patk = torch.stack([p[:, :tk].mean(dim=1) for tk in topk_list], dim=1)
    wt = torch.Tensor([1 / np.log2(x + 2) for x in range(topk)])
    wt = wt.repeat(len(y), 1)
    pw = p*wt
    lo_mask = (y.sum(dim=1).unsqueeze(-1) >
               torch.arange(0, topk).unsqueeze(0)).float()

    wt_mask = wt * lo_mask
    #pw = p * wt_mask
    gatk = torch.stack(
        [pw[:, :tk].sum(dim=1) / wt_mask[:, :tk].sum(dim=1) for tk in topk_list], dim=1)

    act = [np.flatnonzero(x) for x in y.numpy()]
    return (patk.sum(dim=0), gatk.sum(dim=0), p1,b,act, p)


def get_loss_fn(args):
    criterion = nn.BCEWithLogitsLoss()
    if cuda:
        criterion.cuda()
    if args.has_titles:
        return (lambda var,model: title_loss_fn(criterion,var,model))
    else:
        return (lambda var, model: standard_loss_fn(criterion, var, model))


def standard_loss_fn(criterion, var,model):
    ypred = model(var[0])
    loss = criterion(ypred,var[-2].float())
    return (var[-2].float(), ypred, loss)

def title_loss_fn(criterion,var,model):
    ypred = model(var[0], var[1], var[2])
    loss = criterion(ypred,var[-2].float())
    return (var[-2].float(), ypred, loss)


def compute_xml_title(epoch, model, loader, optimizer=None, mode='eval', fh=None, backprop_batch_size=None, tolog=[], loss_fn=standard_loss_fn):
    global cuda
    #Pdb().set_trace() 
    if backprop_batch_size is None:
        backprop_batch_size = loader.batch_sampler.batch_size
    t1 = time.time()
    if mode == 'train':
        model.train()
    else:
        model.eval()

    topk_list = [1, 3, 5]
    last_print = 0
    count = 0
    cum_loss = 0
    cum_correct = 0
    cum_patk = np.zeros(len(topk_list))
    cum_gatk = np.zeros(len(topk_list))
    max_k = max(topk_list)
    topk_pred = np.zeros((len(loader.dataset),max_k))
    actual_labels = np.array([None]*len(loader.dataset))
    ec = np.zeros((len(loader.dataset),max_k))
    if mode == 'train':
        this_backprop_count = 0
        optimizer.zero_grad()
        backprop_loss = 0

    for var in loader:
        var = list(var)
        #var = [var[0], var[2], var[1], var[3]]
        # break
        idx = var[-1]
        count += len(idx)
        # print(len(idx))
        #
        volatile = True
        if mode == 'train':
            this_backprop_count += len(idx)
            volatile = False

        
        for index in range(len(var)-1):
            var[index] = Variable(var[index], volatile = volatile)
            if cuda:
                var[index] = var[index].cuda()

        y, ypred, loss = loss_fn(var, model)
        if mode == 'train':
            backprop_loss += loss
            if this_backprop_count >= backprop_batch_size:
                #utils.log("backproping now: {0}".format(this_backprop_count))
                backprop_loss.backward()
                # loss.backward()
                optimizer.step()
                this_backprop_count = 0
                backprop_loss = 0
                optimizer.zero_grad()
        #
        patk, gatk,p1,b,act,tec = calculate_accuracies(ypred.data.cpu(), y.data.cpu(), topk_list)
        ec[idx] = tec
        actual_labels[idx] = act
        topk_pred[idx] = b
        cum_patk += patk.numpy()
        cum_gatk += gatk.numpy()
        cum_correct += p1
        #cum_correct  += calculate_accuracy(ypred.data.cpu().numpy(), y.data.cpu().numpy())
        #ypred_cum[idx] = ypred.data.cpu().numpy()
        #idx_mask[idx] = 1
        cum_loss += loss.data[0] * len(idx)
        if (count - last_print) >= 200000:
            last_print = count
            #p1 = 1.0*cum_correct/count
            rec = [epoch, mode, 1.0 * cum_loss / count,
                   count, time.time() - t1] + list(1.0*cum_patk / count) + list(1.0*cum_gatk / count) + tolog + [1*cum_correct/count]
            utils.log(','.join([str(round(x, 5)) if isinstance(
                x, float) else str(x) for x in rec]))

    #
    #p1 = patk[0]/len(loader.dataset)
    #compute macro as well from 
    pred_f = topk_pred[:,0]
    actual_f = [x[0] for x in actual_labels]
    macro_f1 = metrics.f1_score(actual_f, pred_f, average = 'macro')
    print('Macro_f1: {}'.format(macro_f1))
    #Pdb().set_trace()


    rec = [epoch, mode, 1.0 * cum_loss /
           len(loader.dataset), len(loader.dataset), time.time() - t1] + list(cum_patk / len(loader.dataset)) + list(cum_gatk / len(loader.dataset)) + tolog + [1.0*cum_correct/len(loader.dataset)]

    utils.log(','.join([str(round(x, 5)) if isinstance(
        x, float) else str(x) for x in rec]), file=fh)
    return (rec,-1,topk_pred, actual_labels, ec)

