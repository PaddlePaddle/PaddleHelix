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
dump cfg trees
"""

from __future__ import print_function
from past.builtins import range

import os
import sys
import numpy as np
import math
import random



from joblib import Parallel, delayed

sys.path.append('../mol_common')
from cmd_args import cmd_args
from mol_tree import AnnotatedTree2MolTree

sys.path.append('../mol_decoder')
from attribute_tree_decoder import create_tree_decoder,batch_make_att_masks

sys.path.append('../cfg_parser')
import cfg_parser as parser

import pickle 
from tqdm import tqdm

def parse_single(smiles, grammar):
    """
    tbd
    """
    ts = parser.parse(smiles, grammar)
    assert isinstance(ts, list) and len(ts) == 1
    n = AnnotatedTree2MolTree(ts[0])
    return n


def parse_many(chunk, grammar):
    """
    tbd
    """
    return [parse_single(smiles, grammar) for smiles in chunk]


def parse(chunk, grammar):
    """
    tbd
    """
    size = 100
    result_list = Parallel(n_jobs=-1)(delayed(parse_many)(chunk[i: i + size], grammar) \
                                            for i in range(0, len(chunk), size))

    return_value = []
    for _0 in result_list:
        for  _1 in _0:
            return_value.append(_1)

    return return_value


if __name__ == '__main__':
    smiles_file = cmd_args.smiles_file 
    fname = '.'.join(smiles_file.split('.')[0:-1]) + '.cfg_dump'
    fout = open(fname, 'wb')
    grammar = parser.Grammar(cmd_args.grammar_file)

    # load smiles strings as a list
    with open(smiles_file, 'r') as f:
        smiles = f.readlines()

    for i in range(len(smiles)):
        smiles[i] = smiles[i].strip()

    # cfg_tree_list = parse(smiles, grammar)
    # cp.dump(cfg_tree_list, fout, cp.HIGHEST_PROTOCOL)
    
    for i in tqdm(range(len(smiles))):
        ts = parser.parse(smiles[i], grammar)
        assert isinstance(ts, list) and len(ts) == 1
        n = AnnotatedTree2MolTree(ts[0])
        pickle.dump(n, fout, pickle.HIGHEST_PROTOCOL)

    fout.close()
