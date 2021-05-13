#!/usr/bin/python3                                                                                                
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
make dataset sequential
"""
from __future__ import print_function
from past.builtins import range

import os
import sys
import numpy as np
import math
import random
from tqdm import tqdm

sys.path.append('../mol_common')
from mol_tree import AnnotatedTree2MolTree
from cmd_args import cmd_args

sys.path.append('../mol_decoder')
from attribute_tree_decoder import batch_make_att_masks

sys.path.append('../cfg_parser')
import cfg_parser as parser

import h5py
import pdb

def parse_smiles_with_cfg(smiles_file, grammar_file):
    """
    tbd
    """
    grammar = parser.Grammar(cmd_args.grammar_file)

    cfg_tree_list = []

    with open(smiles_file, 'r') as f:
        for row in tqdm(f):
            smiles = row.strip()
            ts = parser.parse(smiles, grammar)
            assert isinstance(ts, list) and len(ts) == 1
            n = AnnotatedTree2MolTree(ts[0])
            cfg_tree_list.append(n)

    return cfg_tree_list

if __name__ == '__main__':

    cfg_tree_list = parse_smiles_with_cfg(cmd_args.smiles_file, cmd_args.grammar_file)

    all_true_binary, all_rule_masks = batch_make_att_masks(cfg_tree_list)
    
    print(all_true_binary.shape, all_rule_masks.shape)

    f_smiles = '.'.join(cmd_args.smiles_file.split('/')[-1].split('.')[0:-1])

    out_file = '%s/%s.h5' % (cmd_args.save_dir, f_smiles)    
    h5f = h5py.File(out_file, 'w')

    h5f.create_dataset('x', data=all_true_binary)
    h5f.create_dataset('masks', data=all_rule_masks)
    h5f.close()
