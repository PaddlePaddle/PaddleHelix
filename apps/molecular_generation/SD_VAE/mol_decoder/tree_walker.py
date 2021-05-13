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
tree walker
"""



from __future__ import print_function

import os
import sys
import csv
import numpy as np
import math
import random
from collections import defaultdict

sys.path.append('../mol_common')
from mol_util import atom_valence, bond_types, bond_valence, \
                        prod, TOTAL_NUM_RULES, rule_ranges, DECISION_DIM
from mol_tree import Node, get_smiles_from_tree, AnnotatedTree2MolTree
from cmd_args import cmd_args

sys.path.append('../cfg_parser')
import cfg_parser


class DecodingLimitExceeded(Exception):
    """
    tbd
    """
    def __init__(self):
        pass
    def __str__(self):
        return 'DecodingLimitExceeded'


class TreeWalker(object):
    """
    tbd
    """
    def __init__(self):
        pass

    def reset(self):
        """
        tbd
        """
        raise NotImplementedError

    def sample_index_with_mask(self, node, idxes):
        """
        tbd
        """
        raise NotImplementedError


class OnehotBuilder(TreeWalker):
    """
    tbd
    """
    def __init__(self):
        super(OnehotBuilder, self).__init__()
        self.reset()
        
    def reset(self):
        """
        tbd
        """
        self.num_steps = 0
        self.global_rule_used = []
        self.mask_list = []

    def sample_index_with_mask(self, node, idxes):
        """
        tbd
        """
        assert node.rule_used is not None
        g_range = rule_ranges[node.symbol]
        global_idx = g_range[0] + node.rule_used
        self.global_rule_used.append(global_idx)
        self.mask_list.append(np.array(idxes))

        self.num_steps += 1

        result = None
        for i in range(len(idxes)):
            if idxes[i] == global_idx:
                result = i
        if result is None:
            print(rule_ranges[node.symbol], node.symbol, idxes, global_idx)
        assert result is not None
        return result

    def sample_att(self, node, candidates):
        """
        tbd
        """
        assert hasattr(node, 'bond_idx')
        assert node.bond_idx in candidates

        global_idx = TOTAL_NUM_RULES + node.bond_idx
        self.global_rule_used.append(global_idx)
        self.mask_list.append(np.array(candidates) + TOTAL_NUM_RULES)

        self.num_steps += 1
        
        return node.bond_idx


class ConditionalDecoder(TreeWalker):
    """
    tbd
    """ 
    def __init__(self, raw_logits, use_random):
        super(ConditionalDecoder, self).__init__()
        self.raw_logits = raw_logits
        self.use_random = use_random
        assert len(raw_logits.shape) == 2 and raw_logits.shape[1] == DECISION_DIM

        self.reset()

    def reset(self):
        """
        tbd
        """
        self.num_steps = 0

    def _get_idx(self, cur_logits):
        """
        tbd
        """
        if self.use_random:
            cur_prob = np.exp(cur_logits)
            cur_prob = cur_prob / np.sum(cur_prob)

            result = np.random.choice(len(cur_prob), 1, p=cur_prob)[0]
            result = int(result)  # enusre it's converted to int
        else:
            result = np.argmax(cur_logits)

        self.num_steps += 1
        return result

    def sample_index_with_mask(self, node, idxes):
        """
        tbd
        """
        if self.num_steps >= self.raw_logits.shape[0]:
            raise DecodingLimitExceeded()
        cur_logits = self.raw_logits[self.num_steps][idxes]
        
        return self._get_idx(cur_logits)

    def sample_att(self, node, candidates):    
        """
        tbd
        """    
        if self.num_steps >= self.raw_logits.shape[0]:
            raise DecodingLimitExceeded()
        cur_logits = self.raw_logits[self.num_steps][np.array(candidates) + TOTAL_NUM_RULES]

        idx = self._get_idx(cur_logits)
        
        return candidates[idx]


if __name__ == '__main__':
    pass

    
    
