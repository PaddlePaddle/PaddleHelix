#!/usr/bin/python
#-*-coding:utf-8-*-
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
tbd
"""

import os
from os.path import join, exists
import pandas as pd
import numpy as np

from pahelix.datasets.inmemory_dataset import InMemoryDataset


__all__ = [
    'get_default_ogbg_molpcba_task_names',
    'load_ogbg_molpcba_dataset',
    'split_ogbg_molpcba_by_ogbg_scaffold',
]


def get_default_ogbg_molpcba_task_names(data_path):
    """Get that default ogbg_molpcba task names and return class label"""
    input_df = pd.read_csv(os.path.join(data_path, "mapping", "mol.csv.gz"), sep=',')
    outcomes = input_df.set_index("smiles").drop(["mol_id"], axis=1)
    return list(outcomes.columns)


def load_ogbg_molpcba_dataset(data_path, task_names=None):
    """tbd"""
    if task_names is None:
        task_names = get_default_ogbg_molpcba_task_names(data_path)
    
    input_df = pd.read_csv(os.path.join(data_path, "mapping", "mol.csv.gz"), sep=',')
    smiles_list = input_df['smiles']
    labels = input_df[task_names]
    labels = labels.replace(0, -1)  # convert 0 to -1
    labels = labels.fillna(0)   # convert nan to 0

    data_list = []
    for i in range(len(smiles_list)):
        data = {}
        data['smiles'] = smiles_list[i]
        data['label'] = labels.values[i]
        data_list.append(data)
    dataset = InMemoryDataset(data_list)
    return dataset


def get_scaffold_split_idx(data_path):
    """tbd"""
    index_path = join(data_path, "split", "scaffold")
    train_df = pd.read_csv(join(index_path, 'train.csv.gz'), compression='gzip', names=['train_idx'])
    train_idx = train_df['train_idx'].tolist()
    valid_df = pd.read_csv(join(index_path, 'valid.csv.gz'), compression='gzip', names=['valid_idx'])
    valid_idx = valid_df['valid_idx'].tolist()
    test_df = pd.read_csv(join(index_path, 'test.csv.gz'), compression='gzip', names=['test_idx'])
    test_idx = test_df['test_idx'].tolist()
    return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}


def split_ogbg_molpcba_by_ogbg_scaffold(data_path, dataset):
    """tbd"""
    split_idx = get_scaffold_split_idx(data_path)
    train_dataset  = dataset[split_idx['train']]
    valid_dataset = dataset[split_idx['valid']]
    test_dataset = dataset[split_idx['test']]
    return train_dataset, valid_dataset, test_dataset