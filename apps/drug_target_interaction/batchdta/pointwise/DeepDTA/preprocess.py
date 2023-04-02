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

"""Preprocessing scripts for DeepDTA."""

import os
import numpy as np
import pandas as pd
import codecs
import paddle
from paddle import io

# Drug dictionary
drug_dic = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
			"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
			"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
			"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
			"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
			"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
			"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
            "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

# Protein dictionary
pro_temp = ['A','I','L','V','F','W','Y','N','C','Q','M','S','T','D','E','R','H','K','G','P','O','U','X','B','Z']
pro_dic = {w: i+1 for i,w in enumerate(pro_temp)}


def encode_drug(drug_seq, drug_dic):
    """Drug encoding.

    Args:
        drug_seq: Input of DTI drugs.
        drug_dic: Drug dictionary.

    Returns:
        v_d: Output of processed drugs.
    """
    max_drug = 100
    e_drug = [drug_dic[aa] for aa in drug_seq]
    ld = len(e_drug)
    if ld < max_drug:
        v_d = np.pad(e_drug,(0,max_drug-ld),'constant',constant_values=0)
    else:
        v_d = e_drug[:max_drug]
    return v_d


def encode_protein(protein_seq, pro_dic):
    """Target encoding.

    Args:
        protein_seq: Input of DTI targets.
        pro_dic: Target dictionary.

    Returns:
        v_t: Output of processed targets.
    """
    max_pro = 1000
    e_pro = [pro_dic[aa] for aa in protein_seq]
    lp = len(e_pro)
    if lp < max_pro:
        v_t = np.pad(e_pro,(0,max_pro-lp),'constant',constant_values=0)
    else:
        v_t = e_pro[:max_pro]
    return v_t


def mse(y_true, y_pred):
    """Compute the MSE.
    
    Args:
        y_true (ndarray): 1-dim ndarray representing the Kd from the ground truth.
        y_pred (ndarray): 1-dim ndarray representing the predicted Kd from the model.

    Returns:
        mse (float): the mse result.
    """
    mse = np.mean((y_true - y_pred)**2)
    return mse


def concordance_index(y_true, y_pred):
    """Compute the concordance index (CI).

    Args:
        y_true (ndarray): 1-dim ndarray representing the Kd from the ground truth.
        y_pred (ndarray): 1-dim ndarray representing the predicted Kd from the model.

    Returns:
        ci (float): the concordance index.
    """
    ind = np.argsort(y_true)
    y_true = y_true[ind]
    y_pred = y_pred[ind]
    i = len(y_true)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y_true[i] > y_true[j]:
                z = z+1
                u = y_pred[i] - y_pred[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci


class BindingDB_Encoder(paddle.io.Dataset):
    """Data Encoder for BindingDB.
    
    Args:
        data_id: Index of data.
        all_data: All data inputs.
        
    Returns:
        d_out: Output of drug.
        t_out: Output of target.
        labels: DTI labels.
        groups: DTI groups.
    """
    def __init__(self, data_id, all_data):
        """Initialization."""
        self.all_data = all_data
        self.data_id = data_id

    def __len__(self):
        """Get size of input data."""
        return len(self.data_id)

    def __getitem__(self, index):
        """Get items from raw data."""
        index = self.data_id[index]
        t_input, d_input, labels = self.all_data.iloc[index].iloc[0], self.all_data.iloc[index].iloc[1], self.all_data.iloc[index].iloc[4]
        groups = self.all_data.iloc[index].iloc[2]
        
        # Input encoding
        d_out = encode_drug(d_input, drug_dic)
        t_out = encode_protein(t_input, pro_dic)

        return np.asarray(d_out), np.asarray(t_out), np.asarray([labels]), np.asarray(groups)


class Basic_Encoder(paddle.io.Dataset):
    """Data Encoder for Davis/KIBA.
    
    Args:
        data_id: Index of data.
        all_data: All data inputs.
        
    Returns:
        d_out: Output of drug.
        t_out: Output of target.
        labels: DTI labels.
    """
    def __init__(self, data_id, all_data):
        """Initialization."""
        self.all_data = all_data
        self.data_id = data_id

    def __len__(self):
        """Get size of input data."""
        return len(self.data_id)

    def __getitem__(self, index):
        """Get items from raw data."""
        index = self.data_id[index]
        t_input, d_input, labels = self.all_data.iloc[index].iloc[1], self.all_data.iloc[index].iloc[2], self.all_data.iloc[index].iloc[3]
        
        # Input encoding
        d_out = encode_drug(d_input, drug_dic)
        t_out = encode_protein(t_input, pro_dic)

        return np.asarray(d_out), np.asarray(t_out), np.asarray([labels])