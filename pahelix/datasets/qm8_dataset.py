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
Processing of qm8 dataset.
"""

import os
from os.path import join, exists
import pandas as pd
import numpy as np

from pahelix.datasets.inmemory_dataset import InMemoryDataset


def get_default_qm8_task_names():
    """Get that default freesolv task names and return measured expt"""
    return ['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2', 
            'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0', 
            'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM']


def load_qm8_dataset(data_path, task_names=None):
    """
    tbd 
    """
    if task_names is None:
        task_names = get_default_qm8_task_names()

    csv_file = join(data_path, 'raw/qm8.csv')
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


def get_qm8_stat(data_path, task_names):
    """Return mean and std of labels"""
    csv_file = join(data_path, 'raw/qm8.csv')
    input_df = pd.read_csv(csv_file, sep=',')
    labels = input_df[task_names].values
    return {
        'mean': np.mean(labels, 0),
        'std': np.std(labels, 0),
        'N': len(labels),
    }