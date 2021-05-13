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
Processing of PPI dataset.
The DDI dataset were extracted from DrugCombDB. 
You can download the dataset from
http://drugcombdb.denglab.org/download/protein_protein_links.rar and load it into pahelix reader creators
"""
import os
from os.path import join, exists
import pandas as pd
import numpy as np
from pahelix.datasets.inmemory_dataset import InMemoryDataset
__all__ = ['get_default_ppi_task_names', 'load_ppi_dataset']
def get_default_ppi_task_names():
    """Get that default ppi task names"""
    return ['protein1', 'protein2'] 


def load_ppi_dataset(data_path, task_names=None, featurizer=None):
    """Load ppi dataset,process the input information and the featurizer.
    Description:
        
        The data file contains a txt file, in which columns below are used:
            
            protein1: protein1 name;
            
            protein2: protein2 name.
        
    Args:
        data_path(str): the path to the cached npz path.
        task_names(list): a list of header names to specify the columns to fetch from 
            the txt file.
    
    Returns:
        an InMemoryDataset instance.
    
    Example:
        .. code-block:: python
            dataset = load_ppi_dataset('./ppi/raw')
            print(len(dataset))
    """
    if task_names is None:
        task_names = get_default_ppi_task_names()
    
    txt_file = os.listdir(data_path)[0]
    input_df = pd.read_csv(join(data_path, txt_file), sep=' ')
    
    # there are no nans
    data_list = []
    for i in range(input_df.shape[0]):
        raw_data = {}
        raw_data['pair'] = input_df.loc[i, 'protein1'], input_df.loc[i, 'protein2']
        
        data = raw_data
        if not data is None:
            data_list.append(data)
    dataset = InMemoryDataset(data_list)
    return dataset