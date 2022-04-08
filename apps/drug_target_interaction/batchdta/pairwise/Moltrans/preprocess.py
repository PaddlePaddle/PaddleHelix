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
import json
import codecs

from sklearn.preprocessing import OneHotEncoder
from subword_nmt.apply_bpe import BPE

d_max_seq = 50
t_max_seq = 545

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
    temp_d = drug_bpe.process_line(input_smiles).split()
    try:
        idx_d = np.asarray([drug_idx[i] for i in temp_d])
    except:
        idx_d = np.array([0])

    flag = len(idx_d)
    if flag < d_max_seq:
        v_d = np.pad(idx_d, (0, d_max_seq - flag), 'constant', constant_values=0)
        temp_mask_d = [1] * flag + [0] * (d_max_seq - flag)
    else:
        v_d = idx_d[:d_max_seq]
        temp_mask_d = [1] * d_max_seq
    
    return v_d, np.asarray(temp_mask_d)

def target_encoder(input_seq):
    temp_t = target_bpe.process_line(input_seq).split()
    try:
        idx_t = np.asarray([target_idx[i] for i in temp_t])
    except:
        idx_t = np.array([0])

    flag = len(idx_t)
    if flag < t_max_seq:
        v_t = np.pad(idx_t, (0, t_max_seq - flag), 'constant', constant_values=0)
        temp_mask_t = [1] * flag + [0] * (t_max_seq - flag)
    else:
        v_t = idx_t[:t_max_seq]
        temp_mask_t = [1] * t_max_seq

    return v_t, np.asarray(temp_mask_t)