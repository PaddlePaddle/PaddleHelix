#!/usr/bin/python
#-*-coding:utf-8-*-
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
Processing of qm7 dataset.
"""

import os
from os.path import join, exists
import pandas as pd
import numpy as np

from pahelix.datasets.inmemory_dataset import InMemoryDataset


def get_default_qm7_task_names():
    """Get that default qm7 task names and return measured expt"""
    return ['u0_atom']


def load_qm7_dataset(data_path, task_names=None):
    """
    min/max/mean: -2192.0/-404.88/-1544.8360893118595 
    """
    if task_names is None:
        task_names = get_default_qm7_task_names()

    csv_file = join(data_path, 'raw/qm7.csv')
    input_df = pd.read_csv(csv_file, sep=',')
    smiles_list = input_df['smiles']
    labels = input_df[task_names]

    data_list = []
    for i in range(len(labels)):
        data = {
            'smiles': smiles_list[i],
            'label': labels.values[i],
        }
        data_list.append(data)
    dataset = InMemoryDataset(data_list)
    return dataset


def get_qm7_stat(data_path, task_names):
    """Return mean and std of labels"""
    csv_file = join(data_path, 'raw/qm7.csv')
    input_df = pd.read_csv(csv_file, sep=',')
    labels = input_df[task_names].values
    return {
        'mean': np.mean(labels, 0),
        'std': np.std(labels, 0),
        'N': len(labels),
    }
