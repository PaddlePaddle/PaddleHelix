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
Processing of Blood-Brain Barrier Penetration dataset

The Blood-brain barrier penetration (BBBP) dataset is extracted from a study on the modeling and 
prediction of the barrier permeability. As a membrane separating circulating blood and brain extracellular fluid, the blood-brain barrier blocks most drugs, hormones and neurotransmitters. Thus penetration of the barrier forms a long-standing issue in development of drugs targeting central nervous system.
This dataset includes binary labels for over 2000 compounds on their permeability properties.

You can download the dataset from
http://moleculenet.ai/datasets-1 and load it into pahelix reader creators
"""

import os
from os.path import join, exists
import pandas as pd
import numpy as np

from pahelix.datasets.inmemory_dataset import InMemoryDataset


__all__ = ['get_default_bbbp_task_names', 'load_bbbp_dataset']


def get_default_bbbp_task_names():
    """Get that default bbbp task names and return the binary labels"""
    return ['p_np']


def load_bbbp_dataset(data_path, task_names=None):
    """Load bbbp dataset ,process the classification labels and the input information.

    Description:

        The data file contains a csv table, in which columns below are used:
            
            Num:number
            
            name:Name of the compound
            
            smiles:SMILES representation of the molecular structure
            
            p_np:Binary labels for penetration/non-penetration

    Args:
        data_path(str): the path to the cached npz path.
        task_names(list): a list of header names to specify the columns to fetch from 
            the csv file.
    
    Returns:
        an InMemoryDataset instance.
    
    Example:
        .. code-block:: python

            dataset = load_bbbp_dataset('./bbbp')
            print(len(dataset))
        
    References:
    
    [1] Martins, Ines Filipa, et al. “A Bayesian approach to in silico blood-brain barrier penetration modeling.” Journal of chemical information and modeling 52.6 (2012): 1686-1697.
    
    """

    if task_names is None:
        task_names = get_default_bbbp_task_names()

    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    smiles_list = input_df['smiles']
    from rdkit.Chem import AllChem
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    preprocessed_rdkit_mol_objs_list = [m if not m is None else None for m in
                                                          rdkit_mol_objs_list]
    smiles_list = [AllChem.MolToSmiles(m) if not m is None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    labels = input_df[task_names]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans

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
