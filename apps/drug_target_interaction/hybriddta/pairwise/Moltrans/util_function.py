#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Util Functions."""

import pandas as pd
import numpy as np
import os
import json
import random
import pickle
from itertools import repeat
from collections import OrderedDict
from preprocess import drug_encoder, target_encoder


def convert_y_unit(y, from_, to_):
    """
    Convert Kd/pKd
    """
    array_flag = False
    if isinstance(y, (int, float)):
        y = np.array([y])
        array_flag = True
    y = y.astype(float)    
    # basis as nM
    if from_ == 'nM':
        y = y
    elif from_ == 'p':
        y = 10 ** (-y) / 1e-9
        
    if to_ == 'p':
        zero_idxs = np.where(y == 0.)[0]
        y[zero_idxs] = 1e-10
        y = -np.log10(y * 1e-9)
    elif to_ == 'nM':
        y = y
        
    if array_flag:
        return y[0]
    return y


def length_func(list_or_tensor):
    """
    Get length of list or tensor
    """
    if type(list_or_tensor) == list:
        return len(list_or_tensor)
    return list_or_tensor.shape[0]


def load_davis_dataset():
    """
    Load benchmark DAVIS for regression
    """
    trainn_fold = json.load(
        open(os.path.join('dataset', 'regression', 'benchmark', 'DAVIStest', 'folds', 'train_fold_setting1.txt')))
    train_fold = []
    for e in zip(*trainn_fold):
        for ee in e:
            train_fold.append(ee)
    #train_fold = [ee for e in trainn_fold for ee in e]
    test_fold = json.load(
        open(os.path.join('dataset', 'regression', 'benchmark', 'DAVIStest', 'folds', 'test_fold_setting1.txt')))
    ligands = json.load(
        open(os.path.join('dataset', 'regression', 'benchmark', 'DAVIStest', 'ligands_can.txt')),
        object_pairs_hook=OrderedDict)
    proteins = json.load(
        open(os.path.join('dataset', 'regression', 'benchmark', 'DAVIStest', 'proteins.txt')),
        object_pairs_hook=OrderedDict)
    
    affinity = pickle.load(open(os.path.join('dataset', 'regression', 'benchmark', 'DAVIStest', 'Y'), 
                                'rb'), encoding='latin1')
    smiles_lst, protein_lst = [], []

    for k in ligands.keys():
        smiles = ligands[k]
        smiles_lst.append(smiles)
    for k in proteins.keys():
        protein_lst.append(proteins[k])

    affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)
    
    os.makedirs(os.path.join('dataset', 'regression', 'benchmark', 'DAVIStest', 'processed'), exist_ok=True)
    train_test_dataset = []
    for split in ['train', 'test']:
        split_dir = os.path.join('dataset', 'regression', 'benchmark', 'DAVIStest', 'processed', split)
        os.makedirs(split_dir, exist_ok=True)
        fold = train_fold if split == 'train' else test_fold
        rows, cols = np.where(np.isnan(affinity) == False)
        rows, cols = rows[fold], cols[fold]
        
        data_lst = [[] for _ in range(1)]
        for idx in range(len(rows)):
            data = {}
            data['smiles'] = smiles_lst[rows[idx]]
            data['protein'] = protein_lst[cols[idx]]
            af = affinity[rows[idx], cols[idx]]
            data['aff'] = af

            data_lst[idx % 1].append(data)
        random.shuffle(data_lst)
        train_test_dataset.append(data_lst[0])
    return train_test_dataset


def load_kiba_dataset():
    """
    Load benchmark Kiba for regression
    """
    trainn_fold = json.load(
        open(os.path.join('dataset', 'regression', 'benchmark', 'KIBAtest', 'folds', 'train_fold_setting1.txt')))
    train_fold = []
    for e in zip(*trainn_fold):
        for ee in e:
            train_fold.extend(ee)
    #train_fold = [ee for e in trainn_fold for ee in e]
    test_fold = json.load(
        open(os.path.join('dataset', 'regression', 'benchmark', 'KIBAtest', 'folds', 'test_fold_setting1.txt')))
    ligands = json.load(
        open(os.path.join('dataset', 'regression', 'benchmark', 'KIBAtest', 'ligands_can.txt')),
        object_pairs_hook=OrderedDict)
    proteins = json.load(
        open(os.path.join('dataset', 'regression', 'benchmark', 'KIBAtest', 'proteins.txt')),
        object_pairs_hook=OrderedDict)
    
    affinity = pickle.load(open(os.path.join('dataset', 'regression', 'benchmark', 'KIBAtest', 'Y'), 
                                'rb'), encoding='latin1')
    smiles_lst, protein_lst = [], []

    for k in ligands.keys():
        smiles = ligands[k]
        smiles_lst.append(smiles)
    for k in proteins.keys():
        protein_lst.append(proteins[k])

    affinity = np.asarray(affinity)
    
    os.makedirs(os.path.join('dataset', 'regression', 'benchmark', 'KIBAtest', 'processed'), exist_ok=True)
    train_test_dataset = []
    for split in ['train', 'test']:
        split_dir = os.path.join('dataset', 'regression', 'benchmark', 'KIBAtest', 'processed', split)
        os.makedirs(split_dir, exist_ok=True)
        fold = train_fold if split == 'train' else test_fold
        rows, cols = np.where(np.isnan(affinity) == False)
        rows, cols = rows[fold], cols[fold]
        
        data_lst = [[] for _ in range(1)]
        for idx in range(len(rows)):
            data = {}
            data['smiles'] = smiles_lst[rows[idx]]
            data['protein'] = protein_lst[cols[idx]]
            af = affinity[rows[idx], cols[idx]]
            data['aff'] = af

            data_lst[idx % 1].append(data)
        random.shuffle(data_lst)
        train_test_dataset.append(data_lst[0])
    return train_test_dataset


def load_DAVIS(convert_to_log=True):
    """
    Load raw DAVIS
    """
    affinity = pd.read_csv('./dataset/regression/DAVIS/affinity.txt', header=None, sep=' ')
    with open('./dataset/regression/DAVIS/target_seq.txt') as f1:
        target = json.load(f1)
    with open('./dataset/regression/DAVIS/SMILES.txt') as f2:
        drug = json.load(f2)
        
    target = list(target.values())
    drug = list(drug.values())
    
    SMILES = []
    Target_seq = []
    y = []
    
    for i in range(len(drug)):
        for j in range(len(target)):
            SMILES.append(drug[i])
            Target_seq.append(target[j])
            y.append(affinity.values[i, j])
            
    if convert_to_log:
        y = convert_y_unit(np.array(y), 'nM', 'p')
    else:
        y = y
    return np.array(SMILES), np.array(Target_seq), np.array(y)


def load_KIBA():
    """
    Load raw KIBA
    """
    affinity = pd.read_csv('./dataset/regression/KIBA/affinity.txt', header=None, sep='\t')
    affinity = affinity.fillna(-1)
    with open('./dataset/regression/KIBA/target_seq.txt') as f:
        target = json.load(f)
    with open('./dataset/regression/KIBA/SMILES.txt') as f:
        drug = json.load(f)
        
    target = list(target.values())
    drug = list(drug.values())
    
    SMILES = []
    Target_seq = []
    y = []
    
    for i in range(len(drug)):
        for j in range(len(target)):
            if affinity.values[i, j] != -1:
                SMILES.append(drug[i])
                Target_seq.append(target[j])
                y.append(affinity.values[i, j])

    y = y
    return np.array(SMILES), np.array(Target_seq), np.array(y)


def load_ChEMBL_pkd():
    """
    Load raw ChEMBL with pKd
    """
    affinity = pd.read_csv('./dataset/regression/ChEMBL/Chem_Affinity.txt', header=None)
    affinity = affinity.fillna(-1)
    target = pd.read_csv('./dataset/regression/ChEMBL/ChEMBL_Target_Sequence.txt', header=None)
    drug = pd.read_csv('./dataset/regression/ChEMBL/Chem_SMILES_only.txt', header=None)
    
    SMILES=[]
    Target=[]
    y=[]
    drugcnt=[]
    
    for i in range(len(target)):
        Target.append(target[0][i])
        y.append(affinity[0][i])
        SMILES.append(drug[0][i])

    aff=[]
    total=[]
    for i in range(len(target)):
        aff.insert(i, y[i].split(" "))
    for i in aff:
        total += i
    for i in range(len(SMILES)):
        drugcnt.insert(i, len(SMILES[i].split()))
    
    smile = []
    for segments in SMILES:
        for x in segments.split():
            smile.extend(x)
    #smile = [x for segments in SMILES for x in segments.split()]
    smiles_res=[]
    y_tmp=[]
    target_res=[]
    tmp=[]
    
    for i in range(len(drugcnt)):
        tmp.extend(repeat(Target[i], drugcnt[i]))
    for i in range(len(total)):
        if total[i] != '-1':
            y_tmp.append(total[i])
            smiles_res.append(smile[i])
            target_res.append(tmp[i])

    y_res = [float(i) for i in y_tmp]
    return np.array(smiles_res), np.array(target_res), np.array(y_res)


def load_ChEMBL_kd():
    """
    Load raw ChEMBL with Kd
    """
    affinity = pd.read_csv('./dataset/regression/ChEMBL/Chem_Kd_nM.txt', header=None)
    target = pd.read_csv('./dataset/regression/ChEMBL/ChEMBL_Target_Sequence.txt', header=None)
    drug = pd.read_csv('./dataset/regression/ChEMBL/Chem_SMILES_only.txt', header=None)
    
    SMILES=[]
    Target=[]
    y=[]
    drugcnt=[]
    
    for i in range(len(target)):
        Target.append(target[0][i])
        y.append(affinity[0][i])
        SMILES.append(drug[0][i])

    aff=[]
    total=[]
    for i in range(len(target)):
        aff.insert(i, y[i].split(" "))
    for i in aff:
        total += i
    for i in range(len(SMILES)):
        drugcnt.insert(i, len(SMILES[i].split()))

    smile = []
    for segments in SMILES:
        for x in segments.split():
            smile.extend(x)
    #smile = [x for segments in SMILES for x in segments.split()]
    smiles_res=[]
    y_tmp=[]
    target_res=[]
    tmp=[]

    for i in range(len(drugcnt)):
        tmp.extend(repeat(Target[i], drugcnt[i]))
    for i in range(len(total)):
        if total[i] != '-1':
            y_tmp.append(total[i])
            smiles_res.append(smile[i])
            target_res.append(tmp[i])

    y_res = [float(i) for i in y_tmp]
    y_res = convert_y_unit(np.array(y_res), 'nM', 'p')
    return np.array(smiles_res), np.array(target_res), np.array(y_res)


def load_BindingDB_kd():
    """
    Load raw BindingDB with Kd
    """
    affinity = pd.read_csv('./dataset/regression/BindingDB/BindingDB_Kd.txt', header=None)
    target = pd.read_csv('./dataset/regression/BindingDB/BindingDB_Target_Sequence_new.txt', header=None)
    drug = pd.read_csv('./dataset/regression/BindingDB/BindingDB_SMILES_new.txt', header=None)
    
    SMILES=[]
    Target=[]
    y=[]
    drugcnt=[]
    
    for i in range(len(target)):
        Target.append(target[0][i])
        y.append(affinity[0][i])
        SMILES.append(drug[0][i])

    aff=[]
    total=[]
    for i in range(len(target)):
        aff.insert(i, y[i].split(" "))
    for i in aff:
        total += i
    for i in range(len(SMILES)):
        drugcnt.insert(i, len(SMILES[i].split()))

    smile = []
    for segments in SMILES:
        for x in segments.split():
            smile.extend(x)
    #smile = [x for segments in SMILES for x in segments.split()]
    smiles_res=[]
    y_tmp=[]
    target_res=[]
    tmp=[]

    for i in range(len(drugcnt)):
        tmp.extend(repeat(Target[i], drugcnt[i]))
    for i in range(len(total)):
        if total[i] != '-1':
            y_tmp.append(total[i])
            smiles_res.append(smile[i])
            target_res.append(tmp[i])

    y_res = [float(i) for i in y_tmp]
    y_res = convert_y_unit(np.array(y_res), 'nM', 'p')
    return np.array(smiles_res), np.array(target_res), np.array(y_res)


def data_process(X_drug, X_target, y, frac, drug_encoding='Transformer', target_encoding='Transformer', 
                 split_method='protein_split', random_seed=1, sample_frac=1, mode='DTI'):
    """
    Raw data preprocessing
    """
    if isinstance(X_target, str):
        X_target = [X_target]
    if len(X_target) == 1:
        X_target = np.tile(X_target, (length_func(X_drug), ))

    df_data = pd.DataFrame(zip(X_drug, X_target, y))
    df_data.rename(columns={0:'SMILES', 1: 'Target Sequence', 2: 'Label'}, inplace=True)
    
    if sample_frac != 1:
        df_data = df_data.sample(frac = sample_frac).reset_index(drop = True)
    
    # Drop overall duplicates
    df_data = df_data.drop_duplicates()
    # Only keep unique protein+target pairs by keeping the max label value
    d_t = pd.DataFrame(df_data.groupby(['Target Sequence', 'SMILES']).apply(lambda x: max(x.Label)).index.tolist())
    label = pd.DataFrame(df_data.groupby(['Target Sequence', 'SMILES']).apply(lambda x: max(x.Label)).tolist())
    df_data = pd.concat([d_t, label], 1)
    df_data.columns = ['Target Sequence', 'SMILES', 'Label']

    # Apply BPE for drug and target
    df_data['drug_encoding'] = df_data['SMILES'].apply(drug_encoder)
    df_data['target_encoding'] = df_data['Target Sequence'].apply(target_encoder)
    
    # DTI split
    if split_method == 'random_split': 
        train, val, test = random_split_dataset(df_data, random_seed, frac)
    elif split_method == 'drug_split':
        train, val, test = drug_split_dataset(df_data, random_seed, frac)
    elif split_method == 'protein_split':
        train, val, test = protein_split_dataset(df_data, random_seed, frac)
    elif split_method == 'no_split':
        return df_data.reset_index(drop=True)
    else:
        raise AttributeError("Please select one of the three split method: random, cold_drug, cold_target!")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def data_process_whole(X_drug, X_target, y, drug_encoding='Transformer', 
                       target_encoding='Transformer', mode='DTI'):
    """
    Raw test data preprocessing
    """
    if isinstance(X_target, str):
        X_target = [X_target]
    if len(X_target) == 1:
        X_target = np.tile(X_target, (length_func(X_drug), ))
        
    df_data = pd.DataFrame(zip(X_drug, X_target, y))
    df_data.rename(columns={0:'SMILES', 1: 'Target Sequence', 2: 'Label'}, inplace=True)

    # Drop overall duplicates
    df_data = df_data.drop_duplicates()
    # Only keep unique protein+target pairs by keeping the max label value
    d_t = pd.DataFrame(df_data.groupby(['Target Sequence', 'SMILES']).apply(lambda x: max(x.Label)).index.tolist())
    label = pd.DataFrame(df_data.groupby(['Target Sequence', 'SMILES']).apply(lambda x: max(x.Label)).tolist())
    df_data = pd.concat([d_t, label], 1)
    df_data.columns = ['Target Sequence', 'SMILES', 'Label']
    
    # Apply BPE for drug and target
    df_data['drug_encoding'] = df_data['SMILES'].apply(drug_encoder)
    df_data['target_encoding'] = df_data['Target Sequence'].apply(target_encoder)

    return df_data.reset_index(drop=True)


def random_split_dataset(df, fold_seed, frac):
    """
    Random split
    """
    _, val_frac, test_frac = frac
    test = df.sample(frac=test_frac, replace=False, random_state=fold_seed)
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(frac=val_frac / (1 - test_frac), replace=False, random_state=1)
    train = train_val[~train_val.index.isin(val.index)]

    return train, val, test


def drug_split_dataset(df, fold_seed, frac):
    """
    Split by drug
    """
    _, val_frac, test_frac = frac
    drug_drop = df['SMILES'].drop_duplicates().sample(frac=test_frac, replace=False, random_state=fold_seed).values
    test = df[df['SMILES'].isin(drug_drop)]
    train_val = df[~df['SMILES'].isin(drug_drop)]
    
    drug_drop_val = train_val['SMILES'].drop_duplicates().sample(frac=val_frac / (1 - test_frac), 
                        replace=False, random_state=fold_seed).values
    val = train_val[train_val['SMILES'].isin(drug_drop_val)]
    train = train_val[~train_val['SMILES'].isin(drug_drop_val)]
    
    return train, val, test


def protein_split_dataset(df, fold_seed, frac):
    """
    Split by protein
    """
    _, val_frac, test_frac = frac
    gene_drop = df['Target Sequence'].drop_duplicates().sample(frac=test_frac, replace=False, 
                                                               random_state=fold_seed).values
    test = df[df['Target Sequence'].isin(gene_drop)]
    train_val = df[~df['Target Sequence'].isin(gene_drop)]
    
    gene_drop_val = train_val['Target Sequence'].drop_duplicates().sample(frac=val_frac / (1 - test_frac), 
                                                                replace=False, random_state=fold_seed).values
    val = train_val[train_val['Target Sequence'].isin(gene_drop_val)]
    train = train_val[~train_val['Target Sequence'].isin(gene_drop_val)]
    
    return train, val, test