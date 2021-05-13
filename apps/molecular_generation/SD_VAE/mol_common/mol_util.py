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
mol utils
"""

from collections import defaultdict
from cmd_args import cmd_args

info_folder = cmd_args.info_fold

prod = defaultdict(list)

_total_num_rules = 0
rule_ranges = {}
terminal_idxes = {}

avail_atoms = {}
aliphatic_types = []
aromatic_types = []
bond_types = []
with open(cmd_args.grammar_file, 'r') as f:
    for row in f:
        s = row.split('->')[0].strip()
        rules = row.split('->')[1].strip().split('|')
        rules = [w.strip() for w in rules]
        for rule in rules:
            rr = rule.split()
            prod[s].append(rr)
            for x in rr:
                if x[0] == '\'' and not x in terminal_idxes:
                    idx = len(terminal_idxes)
                    terminal_idxes[x] = idx
        rule_ranges[s] = (_total_num_rules, _total_num_rules + len(rules))
        _total_num_rules += len(rules)

        if s == 'aliphatic_organic':
            for x in prod[s]:
                assert len(x) == 1
                aliphatic_types.append(x[0])
        if s == 'aromatic_organic':
            for x in prod[s]:
                assert len(x) == 1
                aromatic_types.append(x[0])
        if s == 'bond':
            for x in prod[s]:
                assert len(x) == 1
                bond_types.append(x[0])

def load_valence(fname, info_dict):
    """
    tbd
    """
    with open(fname, 'r') as f:
        for row in f:
            row = row.split()
            info_dict[row[0]] = int(row[1])

avail_atoms['aliphatic_organic'] = aliphatic_types
avail_atoms['aromatic_organic'] = aromatic_types
TOTAL_NUM_RULES = _total_num_rules
atom_valence = {}
bond_valence = {}
load_valence(info_folder + '/atom.valence', atom_valence)
load_valence(info_folder + '/bond.valence', bond_valence)
bond_valence[None] = 1
MAX_NESTED_BONDS = 8

DECISION_DIM = MAX_NESTED_BONDS + TOTAL_NUM_RULES + 2

if __name__ == '__main__':
    print(terminal_idxes)