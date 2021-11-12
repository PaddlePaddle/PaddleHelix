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

"""Preprocessing scripts for MolTrans."""

import os
import numpy as np
import pandas as pd
import codecs
import paddle
from paddle import io
from sklearn.preprocessing import OneHotEncoder
from subword_nmt.apply_bpe import BPE

# Global settings
D_MAX = 50
T_MAX = 545

drug_vocab_path = './vocabulary/drug_bpe_chembl_freq_100.txt'
drug_codes_bpe = codecs.open(drug_vocab_path)
drug_bpe = BPE(drug_codes_bpe, merges=-1, separator='')
drug_temp = pd.read_csv('./vocabulary/subword_list_chembl_freq_100.csv')
drug_index2word = drug_temp['index'].values
drug_idx = dict(zip(drug_index2word, range(0, len(drug_index2word))))

target_vocab_path = './vocabulary/target_bpe_uniprot_freq_500.txt'
target_codes_bpe = codecs.open(target_vocab_path)
target_bpe = BPE(target_codes_bpe, merges=-1, separator='')
target_temp = pd.read_csv('./vocabulary/subword_list_uniprot_freq_500.csv')
target_index2word = target_temp['index'].values
target_idx = dict(zip(target_index2word, range(0, len(target_index2word))))


def drug_encoder(input_smiles):
    """Drug encoder to parse input SMILES strings.
    
    Args:
        input_smiles: Input of DTI drugs.
    
    Returns:
        v_d: Output of processed drugs.
        v_mask_d: Output of processed drug maskings.
    """
    temp_d = drug_bpe.process_line(input_smiles).split()
    try:
        idx_d = np.asarray([drug_idx[i] for i in temp_d])
    except:
        idx_d = np.array([0])

    flag = len(idx_d)
    if flag < D_MAX:
        v_d = np.pad(idx_d, (0, D_MAX - flag), 'constant', constant_values=0)
        temp_mask_d = [1] * flag + [0] * (D_MAX - flag)
    else:
        v_d = idx_d[:D_MAX]
        temp_mask_d = [1] * D_MAX
    v_mask_d = np.asarray(temp_mask_d)
    
    return v_d, v_mask_d


def target_encoder(input_seq):
    """Target encoder to parse input FASTA sequence.
    
    Args:
        input_seq: Input of DTI targets.
        
    Returns:
        v_t: Output of processed targets.
        v_mask_t: Output of processed target maskings.
    """
    temp_t = target_bpe.process_line(input_seq).split()
    try:
        idx_t = np.asarray([target_idx[i] for i in temp_t])
    except:
        idx_t = np.array([0])

    flag = len(idx_t)
    if flag < T_MAX:
        v_t = np.pad(idx_t, (0, T_MAX - flag), 'constant', constant_values=0)
        temp_mask_t = [1] * flag + [0] * (T_MAX - flag)
    else:
        v_t = idx_t[:T_MAX]
        temp_mask_t = [1] * T_MAX
    v_mask_t = np.asarray(temp_mask_t)

    return v_t, v_mask_t


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
        mask_d_out: Output of drug maskings.
        t_out: Output of target.
        mask_t_out: Output of target maskings.
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
        d_out, mask_d_out = drug_encoder(d_input)
        t_out, mask_t_out = target_encoder(t_input)
        
        return d_out, mask_d_out, t_out, mask_t_out, labels, np.asarray(groups)


class Basic_Encoder(paddle.io.Dataset):
    """Data Encoder for Davis/KIBA.
    
    Args:
        data_id: Index of data.
        all_data: All data inputs.
        
    Returns:
        d_out: Output of drug.
        mask_d_out: Output of drug maskings.
        t_out: Output of target.
        mask_t_out: Output of target maskings.
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
        d_out, mask_d_out = drug_encoder(d_input)
        t_out, mask_t_out = target_encoder(t_input)
        
        return d_out, mask_d_out, t_out, mask_t_out, labels