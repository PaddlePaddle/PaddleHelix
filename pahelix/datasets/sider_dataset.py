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
Processing of sider dataset.

The Side Effect Resource (SIDER) is a database of marketed drugs and adverse drug reactions (ADR). The version of the SIDER dataset in DeepChem has grouped drug side effects into 27 system organ classes following MedDRA classifications measured for 1427 approved drugs.

You can download the dataset from
http://moleculenet.ai/datasets-1 and load it into pahelix reader creators.

"""

import os
from os.path import join, exists
import pandas as pd
import numpy as np

from pahelix.datasets.inmemory_dataset import InMemoryDataset


__all__ = ['get_default_sider_task_names', 'load_sider_dataset']


def get_default_sider_task_names():
    """Get that default sider task names and return the side results for the drug"""

    return ['Hepatobiliary disorders',
           'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
           'Investigations', 'Musculoskeletal and connective tissue disorders',
           'Gastrointestinal disorders', 'Social circumstances',
           'Immune system disorders', 'Reproductive system and breast disorders',
           'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
           'General disorders and administration site conditions',
           'Endocrine disorders', 'Surgical and medical procedures',
           'Vascular disorders', 'Blood and lymphatic system disorders',
           'Skin and subcutaneous tissue disorders',
           'Congenital, familial and genetic disorders',
           'Infections and infestations',
           'Respiratory, thoracic and mediastinal disorders',
           'Psychiatric disorders', 'Renal and urinary disorders',
           'Pregnancy, puerperium and perinatal conditions',
           'Ear and labyrinth disorders', 'Cardiac disorders',
           'Nervous system disorders',
           'Injury, poisoning and procedural complications']


def load_sider_dataset(data_path, task_names=None, featurizer=None):
    """Load sider dataset,process the input information and the featurizer.

    Description：
        The data file contains a csv table, in which columns below are used:
            smiles:  SMILES representation of the molecular structure.
            Hepatobiliary disorders ~ Injury, poisoning and procedural complications:Recorded side effects for the drug
    
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

            dataset = load_sider_dataset('./sider/raw')
            print(len(dataset))

    References:
    [1]Kuhn, Michael, et al. “The SIDER database of drugs and side effects.” Nucleic acids research 44.D1 (2015): D1075-D1079.
    [2]Altae-Tran, Han, et al. “Low data drug discovery with one-shot learning.” ACS central science 3.4 (2017): 283-293.
    [3]Medical Dictionary for Regulatory Activities. http://www.meddra.org/
    [4]Please refer to http://sideeffects.embl.de/se/?page=98 for details on ADRs.

    """
    if task_names is None:
        task_names = get_default_sider_task_names()

    csv_file = os.listdir(data_path)[0]
    input_df = pd.read_csv(join(data_path, csv_file), sep=',')
    smiles_list = input_df['smiles']
    labels = input_df[task_names]
    labels = labels.replace(0, -1)  # convert 0 to -1

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
