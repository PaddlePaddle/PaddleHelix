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
Processing of bace dataset.

It contains quantitative IC50 and qualitative (binary label) binding results for 
a set of inhibitors of human beta-secretase 1 (BACE=1).
The data are experimental values collected from the scientific literature which 
contains 152 compounds and their 2D structures and properties。


You can download the dataset from
http://moleculenet.ai/datasets-1 and load it into pahelix reader creators
"""

import os
from os.path import join, exists
import pandas as pd
import numpy as np

from pahelix.datasets.inmemory_dataset import InMemoryDataset


__all__ = ['get_default_bace_task_names', 'load_bace_dataset']


def get_default_bace_task_names():
    """Get that default bace task names."""
    return ['Class']


def load_bace_dataset(data_path, task_names=None, featurizer=None):
    """Load bace dataset ,process the classification labels and the input information.

    Description:
        The data file contains a csv table, in which columns below are used:
            mol: The smile representation of the molecular structure;
            pIC50: The negative log of the IC50 binding affinity;
            class: The binary labels for inhibitor.
   
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

            dataset = load_bace_dataset('./bace/raw')
            print(len(dataset))

    References:
    [1]Subramanian, Govindan, et al. “Computational modeling of β-secretase 1 (BACE-1) inhibitors using ligand based approaches.” Journal of chemical information and modeling 56.10 (2016): 1936-1949.
    """

    if task_names is None:
        task_names = get_default_bace_task_names()

    csv_file = os.listdir(data_path)[0]
    input_df = pd.read_csv(join(data_path, csv_file), sep=',')
    smiles_list = input_df['mol']
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




