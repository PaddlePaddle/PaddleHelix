#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
DataLoader generator for the sequence-based pretrain models for protein.
"""

import os
from os.path import exists, basename
import json
import math
import random
import linecache

import math
import numpy as np

import paddle
from paddle.io import IterableDataset, DataLoader, get_worker_info
from paddle.distributed import fleet

from .protein_tools import ProteinTokenizer


def sequence_pad(items, max_len, pad_id):
    new_items = []
    for _, item in enumerate(items):
        seq_len = len(item)
        if seq_len <= max_len:
            pad_len = max_len - seq_len
            item = np.pad(item, (0, pad_len), 'constant', constant_values=pad_id)
        else:
            crop_start = np.random.randint(seq_len - max_len)
            crop_end = crop_start + max_len
            item = item[crop_start:crop_end]
        new_items.append(item)
    return np.array(new_items)


def transform_text_to_bert_feature(text, for_eval=False):
    tokenizer = ProteinTokenizer()

    seq = np.array(tokenizer.gen_token_ids(text))  # will pad start and end token
    position = np.arange(1, len(seq) + 1)
    data = {
        'sequence': seq,
        'position': position,
    }
    return data


def collate_bert_features(data_list):
    pad_id = ProteinTokenizer.padding_token_id

    max_len = max([len(data['sequence']) for data in data_list])
    sequence = sequence_pad([data['sequence'] for data in data_list], max_len, pad_id)
    position = sequence_pad([data['position'] for data in data_list], max_len, pad_id)

    batch = {
        'sequence': np.array(sequence, 'int64'),
        'position': np.array(position, 'int64'),
    }
    return batch

