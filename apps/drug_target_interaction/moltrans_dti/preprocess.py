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
"""
Data preprocessing
"""

from helper import utils
import paddle
from paddle import io
import os
import numpy as np
import pandas as pd
import codecs
from subword_nmt.apply_bpe import BPE

# Set global variable, drug max position, target max position
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
    """
    Drug Encoder

    Args:
        input_smiles: input drug sequence.

    Returns:
        v_d: padded drug sequence.
        temp_mask_d: masked drug sequence.
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
    
    return v_d, np.asarray(temp_mask_d)


def target_encoder(input_seq):
    """
    Target Encoder

    Args:
        input_seq: input target sequence.

    Returns:
        v_t: padded target sequence.
        temp_mask_t: masked target sequence.
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

    return v_t, np.asarray(temp_mask_t)


def concordance_index1(y, f):
    """
    Compute the concordance index (CI)

    Args:
        y (ndarray): 1-dim ndarray representing the Kd from the ground truth.
        f (ndarray): 1-dim ndarray representing the predicted Kd from the model.

    Returns:
        ci (float): the concordance index.
    """
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


class DataEncoder(io.Dataset):
    """
    Data Encoder
    """
    def __init__(self, ids, label, dti_data):
        """
        Initialization
        """
        super(DataEncoder, self).__init__()
        self.ids = ids
        self.label = label
        self.data = dti_data

    def __len__(self):
        """
        Get size
        """
        return len(self.ids)
    
    def __getitem__(self, idx):
        """
        Get embeddings of drug and target, label
        """
        idx = self.ids[idx]
        d_input = self.data.iloc[idx]['SMILES']
        t_input = self.data.iloc[idx]['Target Sequence']
        res = []

        d_out, mask_d_out = drug_encoder(d_input)
        res.append(d_out)
        res.append(mask_d_out)
        t_out, mask_t_out = target_encoder(t_input)
        res.append(t_out)
        res.append(mask_t_out)

        labels = self.label[idx]
        res.append(labels)
        return res


class DataEncoderTest(io.Dataset):
    """
    Data Encoder for Test
    """
    def __init__(self, ids, dti_data):
        """
        Initialization
        """
        super(DataEncoderTest, self).__init__()
        self.ids = ids
        self.data = dti_data

    def __len__(self):
        """
        Get size
        """
        return len(self.ids)
    
    def __getitem__(self, idx):
        """
        Get embeddings of drug and target
        """
        idx = self.ids[idx]
        d_input = self.data.iloc[idx]['SMILES']
        t_input = self.data.iloc[idx]['Target Sequence']
        res = []

        d_out, mask_d_out = drug_encoder(d_input)
        res.append(d_out)
        res.append(mask_d_out)
        t_out, mask_t_out = target_encoder(t_input)
        res.append(t_out)
        res.append(mask_t_out)
        return res