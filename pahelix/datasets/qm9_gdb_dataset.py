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
Processing of qm9_gdb dataset.
"""

import os
from os.path import join, exists
import pandas as pd
import numpy as np
from rdkit import Chem

from pahelix.datasets.inmemory_dataset import InMemoryDataset


def get_default_qm9_gdb_task_names():
    """Get that default freesolv task names and return measured expt"""
    return ['homo', 'lumo', 'gap']


def load_qm9_gdb_dataset(data_path, task_names=None):
    """
    tbd
    """
    if task_names is None:
        task_names = get_default_qm9_gdb_task_names()

    sdf_file = join(data_path, 'gdb9.sdf')
    supplier = Chem.SDMolSupplier(sdf_file)
    label_file = join(data_path, 'gdb9.sdf.csv')
    label_df = pd.read_csv(label_file, sep=',')
    labels = label_df[task_names]

    data_list = []
    for i in range(len(labels)):
        data = {}
        mol = supplier[i]
        if mol is None:
            continue
        data['mol'] = mol
        data['label'] = labels.values[i]
        data_list.append(data)
    dataset = InMemoryDataset(data_list)
    return dataset


def get_qm9_gdb_stat(data_path, task_names):
    """Return mean and std of labels"""
    label_file = join(data_path, 'gdb9.sdf.csv')
    label_df = pd.read_csv(label_file, sep=',')
    labels = label_df[task_names].values
    return {
        'mean': np.mean(labels, 0),
        'std': np.std(labels, 0),
        'N': len(labels),
    }