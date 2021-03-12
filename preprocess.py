#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:46:13 2020

@author: issac
"""

import pandas as pd
import numpy as np
import os
import torch
import heapq

def dataPreprocess(path,year):
    originData = pd.read_csv(path)
    originData = originData.rename(index=str, columns={'Community Area': 'communityArea'})
    data = originData[originData['communityArea'].notnull()] # 去除'Community Area'缺失值后的数据
    data['occurDate'] = data['Date'].apply(lambda x: pd.Timestamp(x).date())# 以天作为最小时间步
    data = data.groupby(['communityArea','occurDate']).agg({'Case Number':'count'})    # goupby
    ts = pd.date_range('%s-01-01'%year, '%s-12-31'%year, freq='D')
    data = data.unstack(level=0).reindex(ts).fillna(0)
    data.index = map(lambda e: e.strftime('%Y-%m-%d'), data.index)
    data.index.name = 'occurDate'
    data = data.stack('communityArea')
    return data # index contains 'Community Area' and 'occurDate'

def dataCreater(data, startTime, endTime, srcWin, tgtWin): # including 
    '''
    Dividing data into training data including features and targets.
    
    Args:
        data: original data. 
        srcWin: source data window. 
        tgtWin: target data window. 
        startTime: '2015-01-01'. 
        endTime: '2015-12-31'. 
        
    Returns:
        sourceData: [batch,seq_len,input_size].\
        targetData: [[batch,seq_len,output_size].
    '''
    window = srcWin + tgtWin -1
    totalDay = pd.Timestamp(endTime) - pd.Timestamp(startTime)
    totalDay = totalDay.days - window + 1
    sourceData = []
    targetData = []
    lastStartDate = ''
    lastEndDate =  ''
    for step in range(totalDay):
        startTmp = pd.Timestamp(startTime) + pd.Timedelta(days = step)
        endTmp = startTmp + pd.Timedelta(days = window)
        startTmp = startTmp.strftime('%Y-%m-%d')
        endTmp = endTmp.strftime('%Y-%m-%d')
        windowData = data.loc[startTmp:endTmp]
        windowData = windowData.unstack(level=0)
        windowData = windowData.loc[1:]
        lastStartDate = startTmp
        lastEndDate = endTmp
        srcData = windowData.iloc[:,range(srcWin)]
        tgtData = windowData.iloc[:,range(srcWin,srcWin+tgtWin)]
        srcData = srcData.values.T
        tgtData = tgtData.values.T
        sourceData.append(srcData)
        targetData.append(tgtData)
    sourceData = np.array(sourceData) 
    targetData = np.array(targetData)
    return sourceData, targetData, lastStartDate, lastEndDate

def data_cache_creater(tgtWin,srcWin,monthList,saveDirectory,noise=False):
    yearList = [2016, 2017]
    dataList = []
    for year in yearList:
        path = os.path.abspath(os.path.dirname(os.getcwd())) + \
               os.sep + 'data' + os.sep + 'Chicago' + os.sep + 'crime' + os.sep + 'Crimes_%s.csv' % year
        dataList.append(dataPreprocess(path, year))
    data = pd.concat(dataList)
    if noise:
        record = add_noise(data)
        print(record)
    # obtain train and test data
    for firstPredictDate in monthList:
        firstDate = pd.Timestamp(firstPredictDate)
        endTime = firstDate - pd.Timedelta(days=1)
        endTime = endTime.strftime('%Y-%m-%d')
        startTime = pd.Timestamp('%s-%s-%s'%(firstDate.year-1,firstDate.month,firstDate.day))
        startTime = startTime.strftime('%Y-%m-%d')
        sourceDataTrain, targetDataTrain, lastStartDate, lastEndDate = \
            dataCreater(data, startTime=startTime, endTime=endTime, srcWin=srcWin, tgtWin=tgtWin)
        nextStartDate = pd.Timestamp(lastStartDate) + pd.Timedelta(days=tgtWin)
        nextEndDate = pd.Timestamp('%s-%s-%s' % (
        firstDate.year + int(firstDate.month / 12), (firstDate.month + 1) % 12, firstDate.day)) - pd.Timedelta(days=1)
        nextStartDate = nextStartDate.strftime('%Y-%m-%d')
        nextEndDate = nextEndDate.strftime('%Y-%m-%d')
        sourceDataTest, targetDataTest, _, _ = \
            dataCreater(data, startTime=nextStartDate, endTime=nextEndDate, srcWin=srcWin, tgtWin=tgtWin)
        if saveDirectory:
            savePath = saveDirectory + os.sep + '%s%02d_noise' % (firstDate.year, firstDate.month)
            os.mkdir(savePath) if not os.path.exists(savePath) else print('%s exist'%firstPredictDate)
            np.save(savePath + os.sep + 'sourceDataTrain', sourceDataTrain)
            np.save(savePath + os.sep + 'targetDataTrain', targetDataTrain)
            np.save(savePath + os.sep + 'sourceDataTest', sourceDataTest)
            np.save(savePath + os.sep + 'targetDataTest', targetDataTest)


def norm(sourceDataTrain, sourceDataTest):
    mean = np.mean(sourceDataTrain,axis=0)
    std = np.std(sourceDataTrain,axis=0)
    sourceDataTrain = (sourceDataTrain - mean)/std
    sourceDataTest = (sourceDataTest - mean) / std
    return sourceDataTrain, sourceDataTest

def init_seed(init_seed = 1):
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    np.random.seed(init_seed) # 用于numpy的随机数

def obtain_time(start, end):
    delta = end - start
    m, s = divmod(delta, 60)
    h, m = divmod(m, 60)
    return h,m,s

def calHitRate(y_pre, y_true):
    batch, step, region = y_true.shape
    hit5 = 0
    hit10 = 0
    for b in range(batch):
        rank_true = y_true[b, 0, :].cpu().numpy()
        rank_pre = y_pre[b, 0, :].cpu().numpy()
        rank_true20 = np.array(heapq.nlargest(20, range(len(rank_true)), rank_true.take))
        rank_pre20 = np.array(heapq.nlargest(20, range(len(rank_pre)), rank_pre.take))
        # rank_true10 = np.array(heapq.nlargest(10, range(len(rank_true)), rank_true.take))
        # rank_pre10 = np.array(heapq.nlargest(10, range(len(rank_pre)), rank_pre.take))
        hit5 += len(list(set(rank_true20).intersection(set(rank_pre20)))) #(rank_true5==rank_pre5).sum()
        # hit10 += len(list(set(rank_true10).intersection(set(rank_pre10)))) #(rank_true10 == rank_pre10).sum()
    hitRate20 = hit5/(batch*20)
    # hitRate10 = hit10/(batch*10)
    return hitRate20

def caltrend(y_pre,y_true):
    batch, step, region = y_true.shape
    trend_sum = 0
    for r in range(region):
        yy_true = y_true[:, 0, r].cpu().numpy()
        yy_pre = y_pre[:, 0, r].cpu().numpy()
        trend_true = (yy_true[1:batch] - yy_true[0:batch-1])>0
        trend_pre = (yy_pre[1:batch] - yy_pre[0:batch - 1])>0
        trend_sum += np.sum(trend_true==trend_pre)
    precison = trend_sum/(region*(batch-1))
    return precison

def calMAE(y_pre,y_true):
    yy_true = y_true.cpu().numpy()
    yy_pre = y_pre.cpu().numpy()
    res = np.abs(yy_true - yy_pre)
    res = np.mean(res)
    return res

def trainAndTestDataCreater(tgtWin=1,srcWin=15,FirstPredictDate='2015-11-01', savePath=None):
    '''
    tgtWin: 预测天数
    srcWin: 历史天数
    FirstPredictDate:　第一个预测日期
    '''
    # obtian origin data
    yearList = [2013,2014,2015]
    dataList = []
    for year in yearList:
        path = os.path.abspath(os.path.dirname(os.getcwd())) + \
        os.sep + 'data' + os.sep + 'Chicago' + os.sep + 'crime' + os.sep + 'Crimes_%s.csv'%year
        dataList.append(dataPreprocess(path, year))
    data = pd.concat(dataList)
    # obtain train and test data
    endTime = pd.Timestamp(FirstPredictDate) - pd.Timedelta(days = 1)
    endTime = endTime.strftime('%Y-%m-%d')
    sourceDataTrain, targetDataTrain, lastStartDate, lastEndDate = \
    dataCreater(data, startTime='%s-01-01'%yearList[0], endTime=endTime, srcWin=srcWin, tgtWin=tgtWin)
    nextStartDate = pd.Timestamp(lastStartDate)+pd.Timedelta(days = tgtWin)
    nextEndDate = pd.Timestamp(lastEndDate)+pd.Timedelta(days = tgtWin)
    nextStartDate = nextStartDate.strftime('%Y-%m-%d')
    nextEndDate = nextEndDate.strftime('%Y-%m-%d')
    sourceDataTest, targetDataTest, _, _= \
    dataCreater(data, startTime=nextStartDate, endTime=nextEndDate, srcWin=srcWin, tgtWin=tgtWin)
    if savePath:
        os.mkdir(savePath) if not os.path.exists(savePath) else print('%s exist' % FirstPredictDate)
        np.save(savePath+os.sep+'sourceDataTrain', sourceDataTrain)
        np.save(savePath+os.sep+'targetDataTrain', targetDataTrain)
        np.save(savePath+os.sep+'sourceDataTest', sourceDataTest)
        np.save(savePath+os.sep+'targetDataTest', targetDataTest)
    return sourceDataTrain, targetDataTrain, sourceDataTest, targetDataTest


if __name__=='__main__':
    pass