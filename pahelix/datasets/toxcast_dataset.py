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
Processing of toxcast dataset.

ToxCast is an extended data collection from the same initiative as Tox21, providing toxicology data for a large library of compounds based on in vitro high-throughput screening. The processed collection includes qualitative results of over 600 experiments on 8k compounds.

You can download the dataset from
http://moleculenet.ai/datasets-1 and load it into pahelix reader creators.

"""

import os
from os.path import join, exists
import pandas as pd
import numpy as np

from pahelix.datasets.inmemory_dataset import InMemoryDataset


__all__ = ['get_default_toxcast_task_names', 'load_toxcast_dataset']


def get_default_toxcast_task_names(data_path):
    """Get that default toxcast task names and return the list of the input information"""
    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    return list(input_df.columns)[1:]


def load_toxcast_dataset(data_path, task_names=None):
    """Load toxcast dataset,process the input information.

    Description:
        
        The data file contains a csv table, in which columns below are used:
            
            smiles:  SMILES representation of the molecular structure.
            
            ACEA_T47D_80hr_Negative: “Tanguay_ZF_120hpf_YSE_up” - Bioassays results
            
            SR-XXX: Stress response bioassays results

    Args:
        data_path(str): the path to the cached npz path.
        task_names(list): a list of header names to specify the columns to fetch from 
            the csv file.
    
    Returns:
        an InMemoryDataset instance.
    
    Example:
        .. code-block:: python

            dataset = load_toxcast_dataset('./toxcast')
            print(len(dataset))


    References:
    
    [1]Richard, Ann M., et al. “ToxCast chemical landscape: paving the road to 21st century toxicology.” Chemical research in toxicology 29.8 (2016): 1225-1251.
    
    [2]please refer to the section “high-throughput assay information” at https://www.epa.gov/chemical-research/toxicity-forecaster-toxcasttm-data for details.

    """
    if task_names is None:
        task_names = get_default_toxcast_task_names(data_path)

    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    smiles_list = input_df['smiles']
    from rdkit.Chem import AllChem
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # Some smiles could not be successfully converted
    # to rdkit mol object so them to None
    preprocessed_rdkit_mol_objs_list = [m if not m is None else None 
            for m in rdkit_mol_objs_list]
    smiles_list = [AllChem.MolToSmiles(m) if not m is None else None 
            for m in preprocessed_rdkit_mol_objs_list]
    labels = input_df[task_names]
    labels = labels.replace(0, -1)  # convert 0 to -1
    labels = labels.fillna(0)   # convert nan to 0

    data_list = []
    for i in range(len(smiles_list)):
        if smiles_list[i] is None:
            continue
        data = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data_list.append(data)
    dataset = InMemoryDataset(data_list)
    return dataset


