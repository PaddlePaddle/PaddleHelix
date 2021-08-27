#!/usr/bin/env python3
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
Processing of ddi dataset.
The DDI dataset includes 23,052 Drug-Drug Synergy pairs from 39 celllines. 
You can download the dataset from
http://www.bioinf.jku.at/software/DeepSynergy/labels.csv and load it into pahelix reader creators
"""
import os
from os.path import join, exists
import pandas as pd
import numpy as np
from pahelix.datasets.inmemory_dataset import InMemoryDataset
__all__ = ['get_default_ddi_task_names', 'load_ddi_dataset']
def get_default_ddi_task_names():
    """Get that default ddi task names and return class label"""
    return ['drug_a_name', 'drug_b_name', 'cell_line', 'synergy']

    
def load_ddi_dataset(data_path, task_names=None, cellline=None):
    """Load ddi dataset,process the input information.

    Description:

        The data file contains a csv table, in which columns below are used:
            
            drug_a_name: drug name
            
            drug_b_name: drug name
            
            cell_line: cell line which the drug pairs were tested on

            synergy: continuous values represent the synergy effect, we use 30 as threshold to binarize the data into binary labels.
            1 as positive and 0 as negative

    Args:
        data_path(str): the path to the cached npz path.
        task_names(list): a list of header names to specify the columns to fetch from 
            the csv file.
        cellline: the exact cellline model you want to test on.
    
    Returns:
        an InMemoryDataset instance.
    
    Example:
        .. code-block:: python

            dataset = load_hddi_dataset('./ddi/raw')
            print(len(dataset))
    
    References:
    
    [1] Drug-Drug Dynergy Data. https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btx806/4747884
    
    """
    if task_names is None:
        task_names = get_default_ddi_task_names()
    if cellline is None:
        cellline = 'A2058'
    csv_file = os.listdir(data_path)[0]
    input_df = pd.read_csv(join(data_path, csv_file), sep=',', index_col=0)
    input_df = input_df[input_df['cell_line']==cellline]
    input_df['label'] = [1 if x > 30 else -1 for x in input_df['synergy']]
    input_df.index = range(input_df.shape[0])
    #sample_list = input_df['synergy']
    labels = input_df['label']
    # convert 0 to -1
    #labels = labels.replace(0, -1)
    # there are no nans
    data_list = []
    for i in range(input_df.shape[0]):
        raw_data = {}
        raw_data['pair'] = input_df.loc[i, 'drug_a_name'], input_df.loc[i, 'drug_b_name']
        raw_data['label'] = labels.values[i]
        
        data = raw_data
        if not data is None:
            data_list.append(data)
    dataset = InMemoryDataset(data_list)
    return dataset