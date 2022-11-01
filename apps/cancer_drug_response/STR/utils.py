#coding=utf-8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import pandas as pd

def pcc_cal(path):
    df_gene = pd.read_csv(path,header=None)
    df_gene.columns=['cancer','ccle','pub_chemid','pred','gt']
    drugs = list(set(df_gene['pub_chemid']))
    pccs_drug = drugpcc_cal(df_gene,drugs)
    pccs_ccle = cclepcc_cal(df_gene)
    return np.nanmean(pccs_drug), np.nanmean(pccs_ccle)

def drugpcc_cal(df,drugs):
    max_min = []
    pccs = []
    for drug in drugs:
        df_temp = df[df['pub_chemid']==drug]
        pred = df_temp['pred'].values
        gt = df_temp['gt'].values
        if len(gt) <= 2:
            continue
        pcc = pearsonr(pred,gt)[0]
        sp = spearmanr(pred,gt)[0]
        max_ = np.floor(max(max(gt),max(pred)))+1
        min_ = np.floor(min(min(gt),min(pred)))
        max_min.append(max_ - min_)
        pccs.append(pcc)
    pccs_drug = np.array(pccs)
    return pccs_drug
def cclepcc_cal(df):
    ccles = list(set(df['ccle']))
    pccs = []
    for ccle in ccles:
        df_temp = df[df['ccle']==ccle]
        if len(df_temp)<=2:
            continue
        pred = df_temp['pred'].values
        gt = df_temp['gt'].values
        pcc = pearsonr(pred,gt)[0]
        sp = spearmanr(pred,gt)
        max_ = np.floor(max(max(gt),max(pred)))+1
        min_ = np.floor(min(min(gt),min(pred)))
        pccs.append(pcc)
    pccs_ccle = np.array(pccs)
    return pccs_ccle