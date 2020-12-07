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
Tox21 dataset
"""

import os
from os.path import join, exists
import pandas as pd
import numpy as np

from pahelix.datasets.inmemory_dataset import InMemoryDataset


__all__ = ['get_default_muv_task_names', 'load_muv_dataset']


def get_default_muv_task_names():
    """tbd"""
    return ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
           'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
           'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']


def load_muv_dataset(data_path, task_names=None, featurizer=None):
    """tbd"""
    if task_names is None:
        task_names = get_default_muv_task_names()

    file = os.listdir(data_path)[0]
    input_df = pd.read_csv(join(data_path, file), sep=',')
    smiles_list = input_df['smiles']
    labels = input_df[task_names]
    labels = labels.replace(0, -1)  # convert 0 to -1
    labels = labels.fillna(0)   # convert nan to 0

    data_list = []
    for i in range(len(smiles_list)):
        raw_data = {}
        raw_data['smiles'] = smiles_list[i]        
        raw_data['label'] = labels.values[i]

        if not featurizer is None:
            data = featurizer.gen_features(raw_data)
        else:
            data = raw_data

        if not data is None:
            data_list.append(data)

    dataset = InMemoryDataset(data_list)
    return dataset
