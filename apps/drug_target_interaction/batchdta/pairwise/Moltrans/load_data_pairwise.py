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

"""Scipts for loading pairwise data."""

import pandas as pd
import numpy as np
import os
import json
from preprocess import drug_encoder, target_encoder
import pdb


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
        y = 10**(-y) / 1e-9
        
    if to_ == 'p':
        zero_idxs = np.where(y == 0.)[0]
        y[zero_idxs] = 1e-10
        y = -np.log10(y*1e-9)
    elif to_ == 'nM':
        y = y
        
    if array_flag:
        return y[0]
    return y

def load_DAVIS(binary = False, convert_to_log = True, threshold = 30):
    """
    Load raw DAVIS
    """
    affinity = pd.read_csv('./dataset/DAVIS/affinity.txt', header=None, sep=' ')
    with open('./dataset/DAVIS/target_seq.txt') as f1:
        target = json.load(f1)
    with open('./dataset/DAVIS/SMILES.txt') as f2:
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
            
    if binary:
        print("Default binary threshold for Kd is 30, you can adjust it via threshold parameter")
        y = [1 if i else 0 for i in np.array(y) < threshold]
    else:
        if convert_to_log:
            y = convert_y_unit(np.array(y), 'nM', 'p')
        else:
            y = y
    return np.array(SMILES), np.array(Target_seq), np.array(y)


def load_KIBA(binary = False, threshold = 9):
    """
    Load raw KIBA
    """
    affinity = pd.read_csv('./dataset/KIBA/affinity.txt', header=None, sep='\t')
    affinity = affinity.fillna(-1)
    
    with open('./dataset/KIBA/target_seq.txt') as f:
        target = json.load(f)
        
    with open('./dataset/KIBA/SMILES.txt') as f:
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
   
    if binary:
        print("Note that KIBA is not suitable for binary classification as it is a modified score. Default binary threshold for pKd is 9, you can adjust it via threshold parameter")
        y = [1 if i else 0 for i in np.array(y) < threshold]
    else:
        y = y
    return np.array(SMILES), np.array(Target_seq), np.array(y)





def data_process(X_drug=None, X_target=None, y=None, drug_encoding='Transformer', target_encoding='Transformer', split_method='protein_split', frac=[0.8, 0.1, 0.1], random_seed=1, sample_frac=1, mode='DTI'):
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
    
    ### drop duplicates
    # drop overall duplicates
    df_data = df_data.drop_duplicates()
    # only keep unique 'protein+target' pairs by keeping the max label value
    d_t = pd.DataFrame(df_data.groupby(['Target Sequence','SMILES']).apply(lambda x: max(x.Label)).index.tolist())
    label = pd.DataFrame(df_data.groupby(['Target Sequence','SMILES']).apply(lambda x: max(x.Label)).tolist())
    df_data = pd.concat([d_t,label],1)
    df_data.columns = ['Target Sequence','SMILES','Label']
    ### drop duplicates

    # apply BPE encoder for d and t
    df_data['drug_encoding'] = df_data['SMILES'].apply(drug_encoder)
    df_data['target_encoding'] = df_data['Target Sequence'].apply(target_encoder)
    
    # dti split
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



def protein_split_dataset(df, fold_seed, frac):
    """
    Split by protein
    """
    train_frac, val_frac, test_frac = frac
    gene_drop = df['Target Sequence'].drop_duplicates().sample(frac=test_frac, replace=False, random_state=fold_seed).values
    test = df[df['Target Sequence'].isin(gene_drop)]
    train_val = df[~df['Target Sequence'].isin(gene_drop)]
    
    gene_drop_val = train_val['Target Sequence'].drop_duplicates().sample(frac=val_frac/(1-test_frac), replace=False, random_state=fold_seed).values
    val = train_val[train_val['Target Sequence'].isin(gene_drop_val)]
    train = train_val[~train_val['Target Sequence'].isin(gene_drop_val)]
    
    return train, val, test



def results_prepare_pairwise(data,groupID='Target ID',label='Label',BPE='BPE_dt',BPE_mask='BPE_dt_mask'):

    results = []
    for i in range(data.shape[0]):

        res = []
        # res.append(data['Target Sequence'][i])
        res.append(data[groupID][i])
        res.append(data[label][i])
        res.extend(data[BPE][i])
        res.extend(data[BPE_mask][i])

        results.append(res)
    results=np.array(results)
    return results

def results_prepare_pairwise_mixed(data,groupID='Target ID',label=['KI','KD','IC50','EC50'],BPE='BPE_dt'):

    results = []
    for i in range(data.shape[0]):

        res = []
        # res.append(data['Target Sequence'][i])
        res.append(data[groupID][i])
        for i_label in label:
            res.append(data[i_label][i])
        res.extend(data[BPE][i])

        results.append(res)
    results=np.array(results)
    return results


def process_data_pairwise(dataset):
    if dataset == 'KIBA':
        d, t, y = load_KIBA()
    if dataset == 'DAVIS':
        d, t, y = load_DAVIS(binary = False, convert_to_log = True, threshold = 30)

    train, val, test = data_process(d, t, y)
    
    ### generate the results arrary for pairwise
    train['BPE_dt'] = train['drug_encoding'].apply(lambda x: x[0].tolist()) + train['target_encoding'].apply(lambda x: x[0].tolist())
    val['BPE_dt'] = val['drug_encoding'].apply(lambda x: x[0].tolist()) + val['target_encoding'].apply(lambda x: x[0].tolist())
    test['BPE_dt'] = test['drug_encoding'].apply(lambda x: x[0].tolist()) + test['target_encoding'].apply(lambda x: x[0].tolist())

    ### rename the protein sequence
    unique_protein = np.unique(train['Target Sequence'].append(val['Target Sequence']).append(test['Target Sequence']))
    map_protein = ['P' + str(i) for i in range(len(unique_protein))]

    train['Target Sequence'] = train['Target Sequence'].replace(unique_protein,map_protein)
    val['Target Sequence'] = val['Target Sequence'].replace(unique_protein,map_protein)
    test['Target Sequence'] = test['Target Sequence'].replace(unique_protein,map_protein)
    ### rename the protein sequence
    
    r_train = results_prepare_pairwise(train)
    r_test = results_prepare_pairwise(test)
    r_val = results_prepare_pairwise(val)
    
    return r_train, r_val, r_test

def load_customised_Davis(r_train):   
    # apply BPE encoder for d and t
    r_train['drug_encoding'] = r_train['SMILES'].apply(drug_encoder)
    r_train['target_encoding'] = r_train['Target Sequence'].apply(target_encoder)
    r_train['BPE_dt'] = r_train['drug_encoding'].apply(lambda x: x[0].tolist()) + r_train['target_encoding'].apply(lambda x: x[0].tolist())
    r_train['BPE_dt_mask'] = r_train['drug_encoding'].apply(lambda x: x[1].tolist()) + r_train['target_encoding'].apply(lambda x: x[1].tolist())

    
    r_train = results_prepare_pairwise(r_train)

    return r_train
    
def load_customised_KIBA():
    path = './apps/pairwise/Data_for_ALL/KIBA/'
    # path = './Data_for_ALL/DAVIS/'
    train_path = path + 'KIBA_unseenP_seenD_train.csv'
    val_path = path + 'KIBA_unseenP_seenD_val.csv'
    test_path = path + 'KIBA_unseenP_seenD_test.csv'
    r_train = pd.read_csv(train_path)
    r_val = pd.read_csv(val_path)
    r_test = pd.read_csv(test_path)

    
    # apply BPE encoder for d and t
    r_train['drug_encoding'] = r_train['SMILES'].apply(drug_encoder)
    r_train['target_encoding'] = r_train['Target Sequence'].apply(target_encoder)
    r_train['BPE_dt'] = r_train['drug_encoding'].apply(lambda x: x[0].tolist()) + r_train['target_encoding'].apply(lambda x: x[0].tolist())

    r_val['drug_encoding'] = r_val['SMILES'].apply(drug_encoder)
    r_val['target_encoding'] = r_val['Target Sequence'].apply(target_encoder)
    r_val['BPE_dt'] = r_val['drug_encoding'].apply(lambda x: x[0].tolist()) + r_val['target_encoding'].apply(lambda x: x[0].tolist())
  
    r_test['drug_encoding'] = r_test['SMILES'].apply(drug_encoder)
    r_test['target_encoding'] = r_test['Target Sequence'].apply(target_encoder)
    r_test['BPE_dt'] = r_test['drug_encoding'].apply(lambda x: x[0].tolist()) + r_test['target_encoding'].apply(lambda x: x[0].tolist())
    
    r_train = results_prepare_pairwise(r_train)
    r_val = results_prepare_pairwise(r_val)
    r_test = results_prepare_pairwise(r_test)


    return r_train, r_val, r_test

def load_customised_BindingDB(r_train,groupID='groupID'):    
    #
    
    # apply BPE encoder for d and t
    r_train['drug_encoding'] = r_train['SMILES'].astype('string').apply(drug_encoder)
    r_train['target_encoding'] = r_train['Target'].astype('string').apply(target_encoder)
    r_train['BPE_dt'] = r_train['drug_encoding'].apply(lambda x: x[0].tolist()) + r_train['target_encoding'].apply(lambda x: x[0].tolist())
    r_train['BPE_dt_mask'] = r_train['drug_encoding'].apply(lambda x: x[1].tolist()) + r_train['target_encoding'].apply(lambda x: x[1].tolist())

    label =   r_train.columns[4]  

    r_train = results_prepare_pairwise(r_train,groupID=groupID,label=label,BPE='BPE_dt')

    return r_train

def load_customised_BindingDB_mixed():
    path = './apps/pairwise/Data_for_ALL/BindingDB/'
    train_path = path + 'BindingDB_values_mixed_train.csv'
    test_path = path + 'BindingDB_values_mixed_test.csv'
    r_train = pd.read_csv(train_path)
    r_test = pd.read_csv(test_path)
    # apply BPE encoder for d and t
    r_train['drug_encoding'] = r_train['SMILES'].astype('string').apply(drug_encoder)
    r_train['target_encoding'] = r_train['Target'].astype('string').apply(target_encoder)
    r_train['BPE_dt'] = r_train['drug_encoding'].apply(lambda x: x[0].tolist()) + r_train['target_encoding'].apply(lambda x: x[0].tolist())

    r_test['drug_encoding'] = r_test['SMILES'].apply(drug_encoder)
    r_test['target_encoding'] = r_test['Target'].apply(target_encoder)
    r_test['BPE_dt'] = r_test['drug_encoding'].apply(lambda x: x[0].tolist()) + r_test['target_encoding'].apply(lambda x: x[0].tolist())

    r_train = results_prepare_pairwise_mixed(r_train,groupID='groupID',label=['Kd','KI','IC50','EC50'],BPE='BPE_dt')
    r_test = results_prepare_pairwise_mixed(r_test,groupID='groupID',label=['Kd','KI','IC50','EC50'],BPE='BPE_dt')

    return r_train, r_test

def load_customised_BindingDB_mixed_KD_KI():
    path = './apps/pairwise/Data_for_ALL/BindingDB/'
    train_path = path + 'BindingDB_values_mixed_train.csv'
    test_path = path + 'BindingDB_values_mixed_test.csv'
    r_train = pd.read_csv(train_path)
    # only keep Kd and KI
    r_train = r_train.drop(columns=['IC50','EC50']).dropna(subset=['Kd','KI'],how='all').reset_index(drop=True)
    r_train = r_train[0:1000]
    r_test = pd.read_csv(test_path)
    
    # apply BPE encoder for d and t
    r_train['drug_encoding'] = r_train['SMILES'].astype('string').apply(drug_encoder)
    r_train['target_encoding'] = r_train['Target'].astype('string').apply(target_encoder)
    r_train['BPE_dt'] = r_train['drug_encoding'].apply(lambda x: x[0].tolist()) + r_train['target_encoding'].apply(lambda x: x[0].tolist())

    r_test['drug_encoding'] = r_test['SMILES'].apply(drug_encoder)
    r_test['target_encoding'] = r_test['Target'].apply(target_encoder)
    r_test['BPE_dt'] = r_test['drug_encoding'].apply(lambda x: x[0].tolist()) + r_test['target_encoding'].apply(lambda x: x[0].tolist())

    r_train = results_prepare_pairwise_mixed(r_train,groupID='groupID',label=['Kd','KI'],BPE='BPE_dt')
    r_test = results_prepare_pairwise_mixed(r_test,groupID='groupID',label=['Kd','KI','IC50','EC50'],BPE='BPE_dt')

    return r_train, r_test

if __name__ == "__main__":
    d, t, y = load_DAVIS(binary = False, convert_to_log = True, threshold = 30)
    train, val, test = data_process(d, t, y)
    
    ### generate the results arrary for pairwise
    train['BPE_dt'] = train['drug_encoding'].apply(lambda x: x[0].tolist()) + train['target_encoding'].apply(lambda x: x[0].tolist())
    val['BPE_dt'] = val['drug_encoding'].apply(lambda x: x[0].tolist()) + val['target_encoding'].apply(lambda x: x[0].tolist())
    test['BPE_dt'] = test['drug_encoding'].apply(lambda x: x[0].tolist()) + test['target_encoding'].apply(lambda x: x[0].tolist())

    ### rename the protein sequence
    unique_protein = np.unique(train['Target Sequence'].append(val['Target Sequence']).append(test['Target Sequence']))
    map_protein = ['P' + str(i) for i in range(len(unique_protein))]

    train['Target Sequence'] = train['Target Sequence'].replace(unique_protein,map_protein)
    val['Target Sequence'] = val['Target Sequence'].replace(unique_protein,map_protein)
    test['Target Sequence'] = test['Target Sequence'].replace(unique_protein,map_protein)
    ### rename the protein sequence
    
    r_train = results_prepare_pairwise(train)
    r_test = results_prepare_pairwise(test)
    r_val = results_prepare_pairwise(val)
    
    
    
    