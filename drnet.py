#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:46:13 2020

@author: issac
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformer.Models import Encoder, Transformer

'''
d_word_vec:嵌入字典大小，即输入向量维度
n_layers:Encoder层数
n_head：多头数量    
d_model：输入向量维度
d_inner：前馈隐变量维度
n_position:　生成位置编码的长度，default:200
   
'''


class Model(nn.Module):
    ''' Transformer Encoder + Conv1d + MLP '''

    def __init__(self, d_word_vec=77, n_layers=3, n_head=1, d_k=16, d_v=16,
            d_model=77, d_inner=16, dropout=0.1, n_position=200, seq_len=15, con_size=3, days=1, kernel = 'linear',kernel_size_tcn=3, kernel_size_scn = 2):

        super().__init__()
        self.encoder = Encoder(d_word_vec, n_layers, n_head, d_k, d_v,d_model, d_inner, dropout, n_position, kernel = kernel, kernel_size_tcn=kernel_size_tcn, kernel_size_scn = kernel_size_scn)
        self.con1 = nn.Conv1d(d_model, days, con_size)
        self.ff1 = nn.Linear(seq_len-con_size+1, d_word_vec)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        

    def forward(self, src_seq, return_attns=False):
        enc_output, enc_slf_attn_list = self.encoder(src_seq,True)
        enc_output = enc_output.transpose(1, 2)
        out = self.con1(enc_output)
        out = self.ff1(out)
        if return_attns:
            return out, enc_slf_attn_list
        return out,
    
    def description(self):
        return 'drnet'
