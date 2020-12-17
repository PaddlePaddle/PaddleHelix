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
Processing of esol dataset.

ESOL (delaney) is a standard regression data set,which is also called delaney dataset. In the dataset, you can find  the structure and water solubility data of 1128 compounds.  It's a good choice to validate machine learning models and to estimate solubility directly based on molecular structure which was encoded in SMILES string.

You can download the dataset from
http://moleculenet.ai/datasets-1 and load it into pahelix reader creators.

"""

import os
from os.path import join, exists
import pandas as pd
import numpy as np

from pahelix.datasets.inmemory_dataset import InMemoryDataset


__all__ = ['get_default_esol_task_names', 'load_esol_dataset']


def get_default_esol_task_names():
    """get that default esol task names and return measured values"""
    return ['measured log solubility in mols per litre']


def load_esol_dataset(data_path, task_names=None, featurizer=None):
    """load esol dataset ,process the classification labels and the input information.

    Description：
        The data file contains a csv table, in which columns below are used:
            smiles:SMILES representation of the molecular structure
            Compound ID:Name of the compound
            measured log solubility in mols per litre - Log-scale water solubility of the compound, used as label
   
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

            dataset = load_esol_dataset('./esol/raw')
            print(len(dataset))
    
    References:
    [1] Delaney, John S. "ESOL: estimating aqueous solubility directly from molecular structure." Journal of chemical information and computer sciences 44.3 (2004): 1000-1005.

    """
    if task_names is None:
        task_names = get_default_esol_task_names()

    # NB: some examples have multiple species
    csv_file = os.listdir(data_path)[0]
    input_df = pd.read_csv(join(data_path, csv_file), sep=',')
    smiles_list = input_df['smiles']
    labels = input_df[task_names]

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
