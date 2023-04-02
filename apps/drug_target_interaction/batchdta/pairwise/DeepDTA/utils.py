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

"""Utils scripts for DeepDTA."""

import numpy as np
import pandas as pd
from paddle.io import Dataset
import paddle.distributed as dist

import paddle
import paddle.nn as nn
import random
import time
import os
from lifelines.utils import concordance_index
from random import randint, sample

def group_by(data, qid_index):
    """
    group documents by query-id
    :param data: input_data which contains multiple query and corresponding documents
    :param qid_index: the column num where qid locates in input data
    :return: a dict group by qid
    """
    qid_doc_map = {}
    idx = 0
    #print(data)
    N_data = data.shape[0]
    for i in range(N_data):
        record = data.iloc[i,:]
        #print(type(record[qid_index]))
        qid_doc_map.setdefault(record[qid_index], [])
        qid_doc_map[record[qid_index]].append(i)
    return qid_doc_map



def sample_index(pairs,sampling_method = None):
    '''
    pairs: the score pairs for train or test
    
    return:
    index of x1 and x2
    '''
    x1_index = []
    x2_index = []

    for i_data in pairs:
        if sampling_method == '500 times':
            sampled_data = pd.DataFrame(i_data).sample(n=500,replace=True)
        if sampling_method == None:
            sampled_data = pd.DataFrame(i_data)
        
        x1_index.append(sampled_data.iloc[:,0].values)
        x2_index.append(sampled_data.iloc[:,1].values)
        
    return x1_index, x2_index

def get_pairs(scores,K,eps=0.2,seed=0):
    """
    compute the ordered pairs whose firth doc has a higher value than second one.
    :param scores: given score list of documents for a particular query
    :param K: times of sampling
    :return: ordered pairs.  List of tuple, like [(1,2), (2,3), (1,3)]
    """
    pairs = []
    random.seed(seed)
    for i in range(len(scores)):
        #for j in range(len(scores)):
        # sampling K times
        if K < 1:
            K_ = 1
        else:
            K_ = K
        
        for _ in range(K_):
            idx = random.randint(0, len(scores) - 1)
            score_diff = float(scores[i]) - float(scores[idx])
            if abs(score_diff) >  eps:
                pairs.append((i, idx, score_diff, len(scores)))

    if K < 1:
        N_pairs = len(pairs)
        pairs = sample(pairs, int(N_pairs*K))

    return pairs


def split_pairs(order_pairs, true_scores):
    """
    split the pairs into two list, named relevant_doc and irrelevant_doc.
    relevant_doc[i] is prior to irrelevant_doc[i]

    :param order_pairs: ordered pairs of all queries
    :param ture_scores: scores of docs for each query
    :return: relevant_doc and irrelevant_doc
    """
    relevant_doc = []
    irrelevant_doc = []
    score_diff = []
    N_smiles = []
    doc_idx_base = 0
    query_num = len(order_pairs)
    for i in range(query_num):
        pair_num = len(order_pairs[i])
        docs_num = len(true_scores[i])
        for j in range(pair_num):
            d1, d2, score, N = order_pairs[i][j]
            d1 += doc_idx_base
            d2 += doc_idx_base
            relevant_doc.append(d1)
            irrelevant_doc.append(d2)
            score_diff.append(score)
            N_smiles.append(N)
        doc_idx_base += docs_num
    return relevant_doc, irrelevant_doc, score_diff, N_smiles


def filter_pairs(data,order_paris,threshold):
    # filterred the pairs which have score diff less than 0.2
    order_paris_filtered = []
    for i_pairs in order_paris:
        pairs1_score = data[pd.DataFrame(i_pairs).iloc[:,0].values][:,1].astype('float32')
        pairs2_score = data[pd.DataFrame(i_pairs).iloc[:,1].values][:,1].astype('float32')

        # filtered |score|<threshold
        score = pairs1_score-pairs2_score
        temp_mask = abs(score) > threshold # 0.2 threshold
        i_pairs_filtered = np.array(i_pairs)[temp_mask].tolist()
        if len(i_pairs_filtered)>0:
            order_paris_filtered.append(i_pairs_filtered)
    return order_paris_filtered

def sample_pairs(true_scores,K,eps,seed):
    # get all the pairs after filtering based on scores
    order_paris = []
    for scores in true_scores:
        order_paris.append(get_pairs(scores,K=K,eps=eps,seed=seed))
    x1_index, x2_index, train_scores, N_smiles = split_pairs(order_paris ,true_scores)
    print('Number of training dataset is {}'.format(len(x1_index)))
    # change labels to binary
    Y = np.array(train_scores).astype('float32')

    Y[Y<0] = 0
    Y[Y>0] = 1

    return x1_index, x2_index, train_scores, Y


drug_dic = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

def encodeDrug(drug_seq, drug_dic):
    max_drug = 100

    e_drug = [drug_dic[aa] for aa in drug_seq]
    ld = len(e_drug)
    if ld < max_drug:
        d_seq = np.pad(e_drug,(0,max_drug-ld),'constant',constant_values=0)
    else:
        d_seq = e_drug[:max_drug]
    return d_seq



pro_rdic = ['A','I','L','V','F','W','Y','N','C','Q','M','S','T','D','E','R','H','K','G','P','O','U','X','B','Z']
pro_dic = {w: i+1 for i,w in enumerate(pro_rdic)}


def encodePro(protein_seq, pro_dic):
    max_pro = 1000

    e_pro = [pro_dic[aa] for aa in protein_seq]
    lp = len(e_pro)
    if lp < max_pro:
        p_seq = np.pad(e_pro,(0,max_pro-lp),'constant',constant_values=0)
    else:
        p_seq = e_pro[:max_pro]
    return p_seq

class Data_Encoder_flow(Dataset):
    def __init__(self, X1_index, X2_index,Y,data):
        super(Data_Encoder_flow, self).__init__()
        self.X1_index = X1_index
        self.X2_index = X2_index
        self.Y = Y
        self.data = data

    def __len__(self):
        return len(self.X1_index)
    
    def __getitem__(self, idx):

        return_x1_index = self.X1_index[idx]
        return_x2_index = self.X2_index[idx]
        return_x1 = self.data.iloc[return_x1_index]
        return_x2 = self.data.iloc[return_x2_index]

        return_d1 = return_x1['SMILES']
        return_t1 = return_x1['Target']
        return_d2 = return_x2['SMILES']
        return_t2 = return_x2['Target']

        #Encode Label
        return_d1 = encodeDrug(return_d1,drug_dic)
        return_d2 = encodeDrug(return_d2,drug_dic)
        return_t1 = encodePro(return_t1,pro_dic)
        return_t2 = encodePro(return_t2,pro_dic)

        return_y = self.Y[idx]

        return_d1 = np.asarray(return_d1)
        return_t1 = np.asarray(return_t1)
        return_d2 = np.asarray(return_d2)
        return_t2 = np.asarray(return_t2)
        return_y = np.asarray(return_y)

        return return_d1, return_t1, return_d2, return_t2, return_y


class Data_test(Dataset):
    def __init__(self, test_index, processed_data):
        super(Data_test, self).__init__()
        self.test_index = test_index
        self.processed_data = processed_data
        self.max_len = max([len(i) for i in self.test_index])

    def __len__(self):
        return len(self.test_index)
    
    def __getitem__(self, idx):

        return_test_index = self.test_index[idx]
        return_data = self.processed_data.iloc[return_test_index,:]
        
        return_len = len(return_test_index)
        
        # get scores
        return_y= return_data.iloc[:,-1].values.astype('float32')
        return_y = paddle.to_tensor(return_y)
        # get featueres
        return_d = return_data['SMILES'].values
        return_t = return_data['Target'].values

        #Encode Label
        return_d = [encodeDrug(data_d,drug_dic) for data_d in return_d]
        return_t = [encodePro(data_t,pro_dic) for data_t in return_t]
        return_d = paddle.to_tensor(return_d)
        return_t = paddle.to_tensor(return_t)

        # pad the dataset
        if self.max_len != return_data.shape[0]:
            padded_d = paddle.zeros(shape=[self.max_len-return_d.shape[0],return_d.shape[1]]).astype('int')
            padded_t = paddle.zeros(shape=[self.max_len-return_t.shape[0],return_t.shape[1]]).astype('int')
            padded_y = paddle.zeros(shape=[self.max_len-return_y.shape[0]]).astype('float32')

            return_d = paddle.concat([return_d,padded_d],0)
            return_t = paddle.concat([return_t,padded_t],0)
            return_y = paddle.concat([return_y,padded_y],0)
                
        return return_d, return_t, return_y, return_len


class Data_Encoder(Dataset):
    def __init__(self, X1, X2,Y):
        super(Data_Encoder, self).__init__()
        self.X1 = X1
        self.X2 = X2
        self.Y = Y

    def __len__(self):
        return len(self.X1)
    
    def __getitem__(self, idx):
        return_x1 = self.X1[idx]
        return_x2 = self.X2[idx]
        return_y = self.Y[idx]

        return return_x1, return_x2, return_y


def model_eval(model,val_dataloader):
    model.eval()
    ## validation
    CI_list = []
    weighted_CI_list = []
    weights_len = []

    for _, data in enumerate(val_dataloader()):
        batch_smiles = data[0]
        batch_protein = data[1]
        batch_y = data[2]
        batch_len = data[3]
        
        for i_target_score in range(batch_smiles.shape[0]):
            
            i_target_len = int(batch_len[i_target_score])
            smiles = batch_smiles[i_target_score][0:i_target_len]
            target = batch_protein[i_target_score][0:i_target_len]
            y_label = batch_y[i_target_score][0:i_target_len].numpy()

            test_DS = Data_Encoder(smiles, target, y_label)
            test_loader = paddle.io.DataLoader(test_DS, batch_size=16, shuffle=False)

            i_target_pred_scores = []
            for data_test in test_loader:
                test_smiles = data_test[0]
                test_target = data_test[1]
                test_label = data_test[2]

                pred_scores = model.forward_single(test_smiles, test_target)
                pred_scores = pred_scores.squeeze(1).numpy().tolist()
                i_target_pred_scores.extend(pred_scores)


            i_target_pred_scores = np.array(i_target_pred_scores)
            i_target_y_label = y_label

            # compute CI
            try:
                CI = concordance_index(i_target_y_label,i_target_pred_scores)
                CI_list.append(CI)
                weighted_CI_list.append(i_target_len*CI)
                weights_len.append(i_target_len)
            except:
                pass

    average_CI = np.mean(CI_list)
    weighted_CI = np.sum(weighted_CI_list)/np.sum(weights_len)

    print("Average CI is {}".format(average_CI))
    print("weighted CI is {}".format(weighted_CI))
    return average_CI, weighted_CI