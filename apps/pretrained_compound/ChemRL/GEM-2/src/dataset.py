#!/usr/bin/python                                                                                                
#-*-coding:utf-8-*- 
#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
dataset
"""

import os
from os.path import join, exists
import numpy as np
import pandas as pd
from rdkit import Chem

import paddle
import torch
from pahelix.datasets.inmemory_dataset import InMemoryDataset

from .utils import tree_map


class PCQMv2Dataset(paddle.io.Dataset):
    def __init__(self, dataset_config):
        self.data_dir = dataset_config.data_dir
        self.load_sdf = dataset_config.load_sdf
        self.task_names = dataset_config.task_names

        self.raw_dir = join(self.data_dir, 'raw')
        self.sdf_file = join(self.data_dir, 'pcqm4m-v2-train.sdf')
        self.split_dict_file = os.path.join(self.data_dir, 'split_dict.pt')
        
    def load_dataset_dict(self):
        csv_file = os.listdir(self.raw_dir)[0]
        input_df = pd.read_csv(join(self.raw_dir, csv_file), sep=',')
        smiles_list = input_df['smiles']
        labels = input_df[self.task_names]
        if self.load_sdf:
            suppl = Chem.SDMolSupplier(self.sdf_file)

        data_list = []
        for i in range(len(smiles_list)):
            data = {}
            data['smiles'] = smiles_list[i]        
            data['label'] = labels.values[i]
            if self.load_sdf and i < len(suppl):
                data['mol'] = suppl[i]
            data_list.append(data)
        dataset = InMemoryDataset(data_list)

        split_dict = torch.load(self.split_dict_file)
        dataset_dict = tree_map(lambda x: dataset[list(x)], split_dict)
        return dataset_dict

    def get_task_names(self):
        return self.task_names

    def get_label_stat(self):
        """Return mean and std of labels"""
        csv_file = join(self.raw_dir, 'data.csv.gz')
        input_df = pd.read_csv(csv_file, sep=',')
        labels = input_df[self.task_names].dropna().values
        return {
            'mean': np.mean(labels, 0),
            'std': np.std(labels, 0),
            'N': len(labels),
        }

