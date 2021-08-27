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
Processing of davis dataset
"""
import os
import sys
import json
import random
import pickle
import argparse
import numpy as np
from rdkit import Chem
from collections import OrderedDict

import json
from pahelix.datasets.internal_inmemory_dataset import InMemoryDataset
from pahelix.utils.compound_tools import mol_to_md_graph_data
from pahelix.utils.compound_tools import smiles_to_graph_data
from pahelix.utils.protein_tools import ProteinTokenizer
from pahelix.utils.data_utils import save_data_list_to_npz

__all__ = ['load_pdbbind_dataset']


def load_pdbbind_dataset(data_path, featurizer):
    """tbd"""
    tokenizer = ProteinTokenizer()
    file = os.path.join(data_path, 'raw.txt')
    data_list = []
    with open(file, 'r') as f:
        for line in f:
            protein, smiles, affinity = line.strip().split(',')
            smiles = smiles.split()[0]
            affinity = float(affinity)

            data = {}
            mol_graph = featurizer.gen_features({'smiles': smiles})
            if mol_graph is None:
                continue
            data.update(mol_graph)
            data['protein_token_ids'] = np.array(tokenizer.gen_token_ids(protein))
            data['affinity'] = np.array([affinity])
            data_list.append(data)
        dataset = InMemoryDataset(data_list=data_list)
    return dataset