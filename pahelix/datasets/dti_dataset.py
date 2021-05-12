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
Processing of DTi dataset.
The DTI dataset were extracted from the DrugCombDB. 
You can download the dataset from
http://drugcombdb.denglab.org/download/drug_protein_links.rar and load it into pahelix reader creators
"""
import os
from os.path import join, exists
import pandas as pd
import numpy as np
from pahelix.datasets.inmemory_dataset import InMemoryDataset
__all__ = ['get_default_dti_task_names', 'load_dti_dataset']
def get_default_dti_task_names():
    """Get that default dti task names"""
    return ['chemical', 'protein']


def load_dti_dataset(data_path, task_names=None, featurizer=None):
    """Load dti dataset,process the input information and the featurizer.
    Description:
        
        The data file contains a tsv table, in which columns below are used:
            
            chemical: drug name;
            
            protein: targeted protein name.
            
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
            dataset = load_hddi_dataset('./dti/raw')
            print(len(dataset))
    """
    if task_names is None:
        task_names = get_default_dti_task_names()
    
    tsv_file = os.listdir(data_path)[0]
    input_df = pd.read_csv(join(data_path, tsv_file), sep='\t')
    
    # there are no nans
    data_list = []
    for i in range(input_df.shape[0]):
        raw_data = {}
        raw_data['pair'] = input_df.loc[i, 'chemical'], input_df.loc[i, 'protein']
        
        if not featurizer is None:
            data = featurizer.gen_features(raw_data)
        else:
            data = raw_data
        if not data is None:
            data_list.append(data)
    dataset = InMemoryDataset(data_list)
    return dataset