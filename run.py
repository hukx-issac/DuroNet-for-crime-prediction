#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:50:12 2019

@author: issac
"""

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from drnet import Model
from transformer.Optim import ScheduledOptim
import torch.optim as optim
import time
from preprocess import init_seed, obtain_time, caltrend, calMAE, norm
import argparse

def train(model, iters, criterion, src, trg, srcEval, trgEval, save_model=None, sava_log=None):
    model.train()
    optimizer = ScheduledOptim(
        optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-06),
        2.0, 16, 75)
    loss_min = np.inf
    if sava_log:
        log_file = sava_log + os.sep + 'train.log'
        with open(log_file, 'w') as log:
            log.write('epoch\t loss\n')
    for iter in range(iters):
        model.train()
        outputs, = model(src)
        optimizer.zero_grad()
        loss = criterion(outputs, trg)
        loss.backward()
        optimizer.step_and_update_lr()

        model.eval()
        with torch.no_grad():
            y_test_pre, = model(srcEval)
            loss_test = criterion(y_test_pre, trgEval).item()

        if loss_test < loss_min:
            loss_min = loss_test
            checkpoint = {'epoch': iter, 'status': model.state_dict(), 'loss': loss_min}
            if save_model:
                model_name = save_model + os.sep + 'best.chkpt'
                torch.save(checkpoint, model_name)

        if log_file and (iter % 100 == 0):
            with open(log_file, 'a') as log:
                log.write('{epoch}, {loss: 8.5f}\n'.format(epoch=iter, loss=loss))

def obtain_parser(parser):
    parser.add_argument('--device',   type=int, default=0, help='cuda number (default: 0)')
    parser.add_argument('--date',     type=str, default='2017-02-01', help='first predict date (default: "2015-02-01")')
    parser.add_argument('--tgtWin',   type=int, default=1, help='target data window (default: 1)')
    parser.add_argument('--srcWin',   type=int, default=15, help='source data window (default: 15)')
    parser.add_argument('--d_inner',  type=int, default=8, help='d_inner (default: 8)')
    parser.add_argument('--n_layers', type=int, default=6, help='n_layers (default: 2)')
    parser.add_argument('--n_head',   type=int, default=8, help='n_head (default: 4)')
    parser.add_argument('--d_k', type=int, default=16, help='d_k (default: 16)')
    parser.add_argument('--d_v', type=int, default=16, help='d_v (default: 16)')
    parser.add_argument('--kernel', type=str, default='linear', help='kernel (default: "linear")')
    parser.add_argument('--kernel_size_tcn', type=int, default=3, help='kernel_size_tcn (default: 3)')
    parser.add_argument('--kernel_size_scn', type=int, default=2, help='kernel_size_scn (default: 2)')
    parser.add_argument('--city',    type=str, default=2, help='city (default: "NewYork")')



def main():
    init_seed(init_seed=1)
    parser = argparse.ArgumentParser()
    obtain_parser(parser)
    args = parser.parse_args()

    device = torch.device("cuda:%s" % args.device if torch.cuda.is_available() else "cpu")
    time_pre = pd.Timestamp(args.date)
    date = '%s%02d' % (time_pre.year, time_pre.month) +'N'
    saveDirectory = os.getcwd() + os.sep + 'log' + os.sep + date
    total_file = os.getcwd() + os.sep + 'log' + os.sep + 'drnet.log'
    data_directory = os.path.abspath(os.path.dirname(os.getcwd())) + os.sep + 'data' + os.sep + args.city + os.sep + date

    sourceDataTrain = np.load(data_directory + os.sep + 'sourceDataTrainN.npy')
    targetDataTrain = np.load(data_directory + os.sep + 'targetDataTrainN.npy')
    sourceDataTest = np.load(data_directory + os.sep + 'sourceDataTestP.npy')
    targetDataTest = np.load(data_directory + os.sep + 'targetDataTestP.npy')
    sourceDataTrain, sourceDataTest = norm(sourceDataTrain, sourceDataTest)


    sourceDataTrain = torch.Tensor(sourceDataTrain).to(device)
    targetDataTrain = torch.Tensor(targetDataTrain).to(device)
    sourceDataTest = torch.Tensor(sourceDataTest).to(device)
    targetDataTest = torch.Tensor(targetDataTest).to(device)
    batch = sourceDataTrain.shape[0]
    sourceDataEval = sourceDataTest #sourceDataTrain[batch - 15:, :, :]
    targetDataEval = targetDataTest #targetDataTrain[batch - 15:, :, :]

    model = Model(d_word_vec=77, n_layers=args.n_layers, n_head=args.n_head, d_k=args.d_k, d_v=args.d_v,
                  d_model=77, d_inner=args.d_inner, dropout=0.1, n_position=200,
                  seq_len=args.srcWin,con_size=1,days=args.tgtWin, kernel = args.kernel, kernel_size_tcn=args.kernel_size_tcn, kernel_size_scn = args.kernel_size_scn).to(device)
    criterion = nn.MSELoss()

    start = time.time()
    train(model,1000,criterion,sourceDataTrain,targetDataTrain,sourceDataEval,targetDataEval,saveDirectory,saveDirectory)
    end = time.time()
    h, m, s = obtain_time(start, end)

    checkpoint = torch.load(saveDirectory + os.sep + 'best.chkpt', map_location=device)
    model.load_state_dict(checkpoint['status'])
    loss_train = checkpoint['loss']
    epoch = checkpoint['epoch']

    # predict and print log
    model.eval()
    with torch.no_grad():
        y_test_pre, = model(sourceDataTest)
        loss_test = criterion(y_test_pre,targetDataTest).item()
        MAE = calMAE(y_test_pre, targetDataTest)
        precision_trend = caltrend(y_test_pre, targetDataTest)

    result_log = '\n \
    %s is predicted.\n \
    Model description: %s. \n \
    Time expense of train: %02d minutes %02d seconds\n \
    Train dataset loss(%s): %s \n \
    test  dataset loss: %s \n' \
                 % (date, model.description(), m, s, epoch, np.sqrt(loss_train), np.sqrt(loss_test))

    np.save(saveDirectory + os.sep + 'result', y_test_pre.detach().cpu().numpy())
    res_file = saveDirectory + os.sep + 'result.log'
    with open(res_file, 'w') as log:
        log.write(result_log)
    with open(total_file, 'a') as total_log:
        total_log.write('date:{date}, loss:{loss: 8.5f}, MAE:{MAE: 8.4f}, precision_trend:{precision_trend: 8.4f}, time_cost:{time_cost}, '
                        'd_k:{d_k}, d_v:{d_v}, d_inner:{d_inner}, n_head:{n_head}, n_layers:{n_layers}, tcn:{tcn} ,scn:{scn}\n'.format(
            date=date, loss=np.sqrt(loss_test), MAE=MAE, precision_trend=precision_trend, time_cost=end-start,
        d_k=args.d_k, d_v=args.d_v, d_inner=args.d_inner, n_head=args.n_head, n_layers=args.n_layers, tcn=args.kernel_size_tcn,scn=args.kernel_size_scn))

if __name__ == '__main__':
    main()
#    pass
