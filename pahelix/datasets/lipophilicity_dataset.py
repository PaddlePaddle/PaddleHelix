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
Processing of lipohilicity dataset.

Lipophilicity is a dataset curated from ChEMBL database containing experimental results on octanol/water distribution coefficient (logD at pH=7.4).As the Lipophilicity plays an important role in membrane permeability and solubility. Related work deserves more attention.

You can download the dataset from
http://moleculenet.ai/datasets-1 and load it into pahelix reader creators.

"""

import os
from os.path import join, exists
import pandas as pd
import numpy as np

from pahelix.datasets.inmemory_dataset import InMemoryDataset


def get_default_lipophilicity_task_names():
    """Get that default lipophilicity task names and return measured expt"""
    return ['exp']


def load_lipophilicity_dataset(data_path, task_names=None):
    """Load lipophilicity dataset,process the input information.
    
    Description:

        The data file contains a csv table, in which columns below are used:
            
            smiles: SMILES representation of the molecular structure
            
            exp: Measured octanol/water distribution coefficient (logD) of the compound, used as label
    
    Args:
        data_path(str): the path to the cached npz path.
        task_names(list): a list of header names to specify the columns to fetch from 
            the csv file.
    
    Returns:
        an InMemoryDataset instance.
    
    Example:
        .. code-block:: python

            dataset = load_lipophilicity_dataset('./lipophilicity')
            print(len(dataset))

    References:
    
    [1]Hersey, A. ChEMBL Deposited Data Set - AZ dataset; 2015. https://doi.org/10.6019/chembl3301361

    """
    if task_names is None:
        task_names = get_default_lipophilicity_task_names()

    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
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


def get_lipophilicity_stat(data_path, task_names):
    """Return mean and std of labels"""
    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    labels = input_df[task_names].values
    return {
        'mean': np.mean(labels, 0),
        'std': np.std(labels, 0),
        'N': len(labels),
    }
