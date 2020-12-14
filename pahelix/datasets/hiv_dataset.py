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
Processing of hiv dataset.

The HIV dataset was introduced by the Drug Therapeutics Program (DTP) AIDS Antiviral Screen, which tested the ability to inhibit HIV replication for over 40,000 compounds. Screening results were evaluated and placed into three categories: confirmed inactive (CI),confirmed active (CA) and confirmed moderately active (CM). We further combine the latter two labels, making it a classification task between inactive (CI) and active (CA and CM).

You can download the dataset from
http://moleculenet.ai/datasets-1 and load it into pahelix reader creators

"""

import os
from os.path import join, exists
import pandas as pd
import numpy as np

from pahelix.datasets.inmemory_dataset import InMemoryDataset


__all__ = ['get_default_hiv_task_names', 'load_hiv_dataset']


def get_default_hiv_task_names():
    """Get that default hiv task names and return class label"""
    return ['HIV_active']


def load_hiv_dataset(data_path, task_names=None, featurizer=None):
    """Load hiv dataset,process the input information and the featurizer.
    
   The data file contains a csv table, in which columns below are used:

    :smiles:  SMILES representation of the molecular structure
    :activity: Three-class labels for screening results: CI/CM/CA
    :HIV_active: Binary labels for screening results: 1 (CA/CM) and 0 (CI)
    :Valid ratio:1.0
    :Task evaluated:1/1

    Args:
        data_path(str): the path to the cached npz path.
        task_names:get the default lipophilicity task names.
        featurizer: the featurizer to use for processing the data.    
   
    Returns:
        dataset(InMemoryDataset): the data_list(list of dict of numpy ndarray).


    References:
    [1] AIDS Antiviral Screen Data. https://wiki.nci.nih.gov/display/NCIDTPdata/AIDS+Antiviral+Screen+Data

    """
    if task_names is None:
        task_names = get_default_hiv_task_names()

    csv_file = os.listdir(data_path)[0]
    input_df = pd.read_csv(join(data_path, csv_file), sep=',')
    smiles_list = input_df['smiles']
    labels = input_df[task_names]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans

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
