#!/usr/bin/env python
# coding:utf-8
"""
Name : executor.py
Author  : issac
Time    : 2020/3/22 16:59
"""

import os
from preprocess import data_cache_creater

def date_list_creater():
    dateList = ['2017-%02d-01'%(mon) for mon in range(2,13,2)]
    saveDirectory = os.getcwd() + os.sep + 'data'
    data_cache_creater(1,15,dateList,saveDirectory)

def run_drnet():
    device = 0
    date = ['2017-%02d-01'%(mon) for mon in [3]  ]  #['20170201', '20170301', '20170601', '20170701', '20171001', '20171101']  #['20170301','20170701','20171101'] 
    d_k = 16  #[8,16,24,32,40,48,56,64]
    d_v = 16  #[8,16,24,32,40,48,56,64] 
    n_layers = 2# [1,2,3,4,5,6,7,8]
    n_head = 4
    kernel = 'stcn'
    kernel_size_tcn = 3
    kernel_size_scn = 2
    city='Chicago'
    
    for para in date:
        os.system('python ./run.py --device %s --date %s --d_k %s --d_v %s --n_layers %s --n_head %s --kernel %s --kernel_size_tcn %s --kernel_size_scn %s --city %s'\
                  %(device, para, d_k, d_v, n_layers, n_head , kernel, kernel_size_tcn, kernel_size_scn, city))

if __name__=='__main__':
    run_drnet()
