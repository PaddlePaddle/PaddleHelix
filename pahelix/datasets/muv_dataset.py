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
Processing of muv dataset.

The Maximum Unbiased Validation (MUV) group is a benchmark dataset selected from PubChem BioAssay by applying a refined nearest neighbor analysis. The MUV dataset contains 17 challenging tasks for around 90,000 compounds and is specifically designed for validation of virtual screening techniques.


You can download the dataset from
http://moleculenet.ai/datasets-1 and load it into pahelix reader creators.

"""

import os
from os.path import join, exists
import pandas as pd
import numpy as np

from pahelix.datasets.inmemory_dataset import InMemoryDataset


__all__ = ['get_default_muv_task_names', 'load_muv_dataset']


def get_default_muv_task_names():
    """Get that default hiv task names and return the measured results for bioassays"""

    return ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
           'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
           'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']


def load_muv_dataset(data_path, task_names=None, featurizer=None):
    """Load muv dataset,process the input information and the featurizer.

    Description：
        The data file contains a csv table, in which columns below are used:
            smiles:  SMILES representation of the molecular structure.
            mol_id:  PubChem CID of the compound.
            MUV-XXX: Measured results (Active/Inactive) for bioassays.

    Args:
        data_path(str): the path to the cached npz path.
        task_names(list): a list of header names to specify the columns to fetch from 
            the csv file.
        featurizer(pahelix.featurizers.Featurizer): the featurizer to use for 
            processing the data. If not none, The ``Featurizer.gen_features`` will be 
            applied to the raw data.
    
    Returns:
        an InMemoryDataset instance.
    
    Example:
        .. code-block:: python

            dataset = load_muv_dataset('./muv/raw')
            print(len(dataset))

    References:
    [1]Rohrer, Sebastian G., and Knut Baumann. “Maximum unbiased validation (MUV) data sets for virtual screening based on PubChem bioactivity data.” Journal of chemical information and modeling 49.2 (2009): 169-184.

    """
    if task_names is None:
        task_names = get_default_muv_task_names()

    csv_file = os.listdir(data_path)[0]
    input_df = pd.read_csv(join(data_path, csv_file), sep=',')
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
