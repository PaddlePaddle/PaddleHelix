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
Processing of clintox dataset

The ClinTox dataset compares drugs approved by the FDA and drugs that have failed clinical trials for toxicity reasons. The dataset includes two classification tasks for 1491 drug compounds with known chemical structures: (1) clinical trial toxicity (or absence of toxicity) and (2) FDA approval status. List of FDA-approved drugs are compiled from the SWEETLEAD database, and list of drugs that failed clinical trials for toxicity reasons are compiled from the Aggregate Analysis of ClinicalTrials.gov(AACT) database.

You can download the dataset from
http://moleculenet.ai/datasets-1 and load it into pahelix reader creators

"""

import os
from os.path import join, exists
import pandas as pd
import numpy as np

from pahelix.datasets.inmemory_dataset import InMemoryDataset


__all__ = ['get_default_clintox_task_names', 'load_clintox_dataset']


def get_default_clintox_task_names():
    """Get that default clintox task names and return class"""
    return ['FDA_APPROVED', 'CT_TOX']


def load_clintox_dataset(data_path, task_names=None):
    """Load Clintox dataset ,process the classification labels and the input information.

    Description:

        The data file contains a csv table, in which columns below are used:
            
            smiles: SMILES representation of the molecular structure
            
            FDA_APPROVED: FDA approval status
            
            CT_TOX: Clinical trial results

    Args:
        data_path(str): the path to the cached npz path.
        task_names(list): a list of header names to specify the columns to fetch from 
            the csv file.
    
    Returns:
        an InMemoryDataset instance.
    
    Example:
        .. code-block:: python

            dataset = load_clintox_dataset('./clintox')
            print(len(dataset))
    
    References:
    
    [1] Gayvert, Kaitlyn M., Neel S. Madhukar, and Olivier Elemento. “A data-driven approach to predicting successes and failures of clinical trials.” Cell chemical biology 23.10 (2016): 1294-1301.
    
    [2] Artemov, Artem V., et al. “Integrated deep learned transcriptomic and structure-based predictor of clinical trials outcomes.” bioRxiv (2016): 095653.
    
    [3] Novick, Paul A., et al. “SWEETLEAD: an in silico database of approved drugs, regulated chemicals, and herbal isolates for computer-aided drug discovery.” PloS one 8.11 (2013): e79568.
    
    [4] Aggregate Analysis of ClincalTrials.gov (AACT) Database. https://www.ctti-clinicaltrials.org/aact-database
    
    """
    if task_names is None:
        task_names = get_default_clintox_task_names()

    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    smiles_list = input_df['smiles']
    from rdkit.Chem import AllChem
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    preprocessed_rdkit_mol_objs_list = [m if not m is None else None 
            for m in rdkit_mol_objs_list]
    smiles_list = [AllChem.MolToSmiles(m) if not m is None else None 
            for m in preprocessed_rdkit_mol_objs_list]
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

