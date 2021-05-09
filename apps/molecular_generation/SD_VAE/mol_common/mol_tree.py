#!/usr/bin/python3                                                                                                
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
mol tree
"""
from mol_util import prod, rule_ranges, TOTAL_NUM_RULES, MAX_NESTED_BONDS, DECISION_DIM
import numpy as np

import sys
import os
sys.path.append( '../cfg_parser')
import cfg_parser
from cmd_args import cmd_args

class Node(object)
    """
    tbd
    """
    def __init__(self,s, father = None):
        self.symbol = s
        self.children = []
        self.rule_used = None    
        self.init_atts()
        self.father = father
        if self.father is None:
            assert self.symbol == 'smiles'

    def init_atts(self):
        """
        tbd
        """
        self.left_remain = None
        self.right_remain = None
        self.single_atom = None
        self.task = False
        self.pre_node = None
        self.atom_pos = None
        self.is_aromatic = None

    def is_created(self):
        """
        tbd
        """
        if self.rule_used is None:
            return False
        return len(prod[self.symbol][self.rule_used]) == len(self.children)

    def add_child(self, child, pos = None):
        """
        tbd
        """
        if self.is_created():
            return
        if pos is None:
            self.children.append(child)
        else:
            self.children.insert(pos, child)

    def get_pre(self):
        """
        tbd
        """
        if self.pre_node is None:
            if self.father is None:
                self.pre_node = -1
            else:
                self.pre_node = self.father.get_pre()
        assert self.pre_node is not None
        return self.pre_node


def dfs(node, result):
    """
    tbd
    """
    if len(node.children):
        for c in node.children:
            dfs(c, result)
    else:
        assert node.symbol[0] == node.symbol[-1] == '\''
        result.append(node.symbol[1:-1])


def get_smiles_from_tree(root):
    """
    tbd
    """
    result = []
    dfs(root, result)
    st = ''.join(result)
    return st


def _AnnotatedTree2MolTree(annotated_root, bond_set, father):
    """
    tbd
    """
    n = Node(str(annotated_root.symbol), father=father)
    n.rule_used = annotated_root.rule_selection_id    
    for c in annotated_root.children:
        new_c = _AnnotatedTree2MolTree(c, bond_set, n)
        n.children.append(new_c)
    if n.symbol == 'ringbond':
        assert len(n.children)
        d = n.children[-1]
        assert d.symbol == 'DIGIT'
        st = d.children[0].symbol
        assert len(st) == 3
        idx = int(st[1 : -1]) - 1
        if idx in bond_set:
            n.bond_idx = idx
            bond_set.remove(idx)
        else:
            bond_set.add(idx)
            n.bond_idx = MAX_NESTED_BONDS
    if isinstance(annotated_root.symbol, cfg_parser.Nonterminal): # it is a non-terminal        
        assert len(n.children)
        assert n.is_created()
    else:
        assert isinstance(annotated_root.symbol, str)
        assert len(n.symbol) < 3 or (n.symbol[0] != '\'' and n.symbol[-1] != '\'')        
        n.symbol = '\'' + n.symbol + '\''        
    return n


def AnnotatedTree2MolTree(annotated_root):
    """
    tbd
    """
    bond_set = set()
    ans = _AnnotatedTree2MolTree(annotated_root, bond_set, father=None)
    assert len(bond_set) == 0
    return ans


def dfs_indices(node, result):
    """
    tbd
    """
    if len(node.children):
        assert node.rule_selection_id >= 0
        g_range = rule_ranges[str(node.symbol)]
        idx = g_range[0] + node.rule_selection_id
        assert idx >= 0 and idx < TOTAL_NUM_RULES

        result.append(idx)
        for c in node.children:
            dfs_indices(c, result)


def AnnotatedTree2RuleIndices(annotated_root):
    """
    tbd
    """
    result = []
    dfs_indices(annotated_root, result)
    return np.array(result)


def AnnotatedTree2Onehot(annotated_root, max_len):
    """
    tbd
    """
    cur_indices = AnnotatedTree2RuleIndices(annotated_root)
    assert len(cur_indices) <= max_len

    x_cpu = np.zeros(( DECISION_DIM, max_len), dtype=np.float32)
    x_cpu[cur_indices, np.arange(len(cur_indices))] = 1.0
    x_cpu[-1, np.arange(len(cur_indices), max_len)] = 1.0 # padding

    return x_cpu

if __name__ == '__main__':

    smiles = 'OSC'
    grammar = cfg_parser.Grammar(cmd_args.grammar_file)


    ts = cfg_parser.parse(smiles, grammar)
    assert isinstance(ts, list) and len(ts) == 1

    print(AnnotatedTree2RuleIndices(ts[0]))
