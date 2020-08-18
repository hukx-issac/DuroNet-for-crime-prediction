#!/usr/bin/env python
# coding:utf-8
"""
Name : tcn.py
Author  : issac
Time    : 2020/3/25 22:57
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import os
import numpy as np

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out+res) #res


class TemporalConvNet(nn.Module):
    '''
    num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
    num_inputs: int， 输入通道数
    '''
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=int((kernel_size-1)/2) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        '''
        param x: size of (Batch, input_channel, seq_len)
        return: size of (Batch, output_channel, seq_len)
        '''
        return self.network(x)

class SpatialTemporalConvNet(nn.Module):
    def __init__(self, seq_len, d_model, num_channels, kernel_size_tcn=3, kernel_size_scn = 2, dropout=0.2):
        super(SpatialTemporalConvNet, self).__init__()
        self.seq_len = seq_len
        self.tcn = TemporalConvNet(d_model, num_channels, kernel_size=kernel_size_tcn, dropout=dropout)

        self.conv1 = weight_norm(nn.Conv1d(seq_len, 1, kernel_size_scn))
        self.relu1 = nn.ReLU()
        self.fc = nn.Linear(d_model-kernel_size_scn+1,num_channels[-1])
        self.relu2 = nn.ReLU()

        self.scn = nn.Sequential(self.conv1, self.relu1, self.fc, self.relu2)

        self.align_weight = nn.Linear(num_channels[-1]*2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h_tcn = self.tcn(x)
        h_scn = self.scn(x.transpose(1, 2))

        h_tcn = h_tcn.transpose(1, 2)
        h_scn = h_scn.repeat(1, self.seq_len, 1)
        h_st = torch.cat([h_tcn,h_scn],dim=2)
        g = self.sigmoid(self.align_weight(h_st))

        if True:
            for i in range(15):
                att = g[i,:,0]
                log_file = os.getcwd() + os.sep + 'data' + os.sep + '%s.csv'%(16+i)
                att = att.detach().cpu().numpy()
                att = map(lambda x:str(x),att)
                att = ', '.join(att)
                with open(log_file, 'a') as log:
                    log.write('{str}\n'.format(str=att))


        output = (1-g)*h_tcn + g*h_scn
        output = output.transpose(1, 2)
        return output
