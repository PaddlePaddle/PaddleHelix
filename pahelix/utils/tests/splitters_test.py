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
import sys
import numpy as np
import unittest

from pahelix.utils.splitters import \
        RandomSplitter, IndexSplitter, ScaffoldSplitter, RandomScaffoldSplitter
from pahelix.datasets.inmemory_dataset import InMemoryDataset
from pahelix.featurizers.featurizer import Featurizer


class RandomSplitterTest(unittest.TestCase):  
    def test_split(self):
        raw_data_list = [
            {'smiles': 'CCOc1ccc2nc(S(N)(=O)=O)sc2c1'},
            {'smiles': 'CCOc1ccc2nc(S(N)(=O)=O)sc2c1'},
            {'smiles': 'CCOc1ccc2nc(S(N)(=O)=O)sc2c1'},
            {'smiles': 'CCOc1ccc2nc(S(N)(=O)=O)sc2c1'},
            {'smiles': 'CC(C)CCCCCCCOP(OCCCCCCCC(C)C)Oc1ccccc1'},
            {'smiles': 'CC(C)CCCCCCCOP(OCCCCCCCC(C)C)Oc1ccccc1'},
            {'smiles': 'CC(C)CCCCCCCOP(OCCCCCCCC(C)C)Oc1ccccc1'},
            {'smiles': 'CC(C)CCCCCCCOP(OCCCCCCCC(C)C)Oc1ccccc1'},
            {'smiles': 'CC(C)CCCCCCCOP(OCCCCCCCC(C)C)Oc1ccccc1'},
            {'smiles': 'CCCCCCCCCCOCC(O)CN'},
            {'smiles': 'CCCCCCCCCCOCC(O)CN'},
            {'smiles': 'CCCCCCCCCCOCC(O)CN'},
            {'smiles': 'CCCCCCCCCCOCC(O)CN'},
        ]
        dataset = InMemoryDataset(raw_data_list)
        splitter = RandomSplitter()
        train_dataset, valid_dataset, test_dataset = splitter.split(
                dataset, frac_train=0.34, frac_valid=0.33, frac_test=0.33)
        n = len(train_dataset) + len(valid_dataset) + len(test_dataset)
        self.assertEqual(n, len(dataset))


class IndexSplitterTest(unittest.TestCase):  
    def test_split(self):
        raw_data_list = [
            {'smiles': 'CCOc1ccc2nc(S(N)(=O)=O)sc2c1'},
            {'smiles': 'CCOc1ccc2nc(S(N)(=O)=O)sc2c1'},
            {'smiles': 'CCOc1ccc2nc(S(N)(=O)=O)sc2c1'},
            {'smiles': 'CCOc1ccc2nc(S(N)(=O)=O)sc2c1'},
            {'smiles': 'CC(C)CCCCCCCOP(OCCCCCCCC(C)C)Oc1ccccc1'},
            {'smiles': 'CC(C)CCCCCCCOP(OCCCCCCCC(C)C)Oc1ccccc1'},
            {'smiles': 'CC(C)CCCCCCCOP(OCCCCCCCC(C)C)Oc1ccccc1'},
            {'smiles': 'CC(C)CCCCCCCOP(OCCCCCCCC(C)C)Oc1ccccc1'},
            {'smiles': 'CC(C)CCCCCCCOP(OCCCCCCCC(C)C)Oc1ccccc1'},
            {'smiles': 'CCCCCCCCCCOCC(O)CN'},
            {'smiles': 'CCCCCCCCCCOCC(O)CN'},
            {'smiles': 'CCCCCCCCCCOCC(O)CN'},
            {'smiles': 'CCCCCCCCCCOCC(O)CN'},
        ]
        dataset = InMemoryDataset(raw_data_list)
        splitter = IndexSplitter()
        train_dataset, valid_dataset, test_dataset = splitter.split(
                dataset, frac_train=0.34, frac_valid=0.33, frac_test=0.33)
        n = len(train_dataset) + len(valid_dataset) + len(test_dataset)
        self.assertEqual(n, len(dataset))


class ScaffoldSplitterTest(unittest.TestCase):  
    def test_split(self):
        raw_data_list = [
            {'smiles': 'CCOc1ccc2nc(S(N)(=O)=O)sc2c1'},
            {'smiles': 'CCOc1ccc2nc(S(N)(=O)=O)sc2c1'},
            {'smiles': 'CCOc1ccc2nc(S(N)(=O)=O)sc2c1'},
            {'smiles': 'CCOc1ccc2nc(S(N)(=O)=O)sc2c1'},
            {'smiles': 'CC(C)CCCCCCCOP(OCCCCCCCC(C)C)Oc1ccccc1'},
            {'smiles': 'CC(C)CCCCCCCOP(OCCCCCCCC(C)C)Oc1ccccc1'},
            {'smiles': 'CC(C)CCCCCCCOP(OCCCCCCCC(C)C)Oc1ccccc1'},
            {'smiles': 'CC(C)CCCCCCCOP(OCCCCCCCC(C)C)Oc1ccccc1'},
            {'smiles': 'CC(C)CCCCCCCOP(OCCCCCCCC(C)C)Oc1ccccc1'},
            {'smiles': 'CCCCCCCCCCOCC(O)CN'},
            {'smiles': 'CCCCCCCCCCOCC(O)CN'},
            {'smiles': 'CCCCCCCCCCOCC(O)CN'},
            {'smiles': 'CCCCCCCCCCOCC(O)CN'},
        ]
        dataset = InMemoryDataset(raw_data_list)
        splitter = ScaffoldSplitter()
        train_dataset, valid_dataset, test_dataset = splitter.split(
                dataset, frac_train=0.34, frac_valid=0.33, frac_test=0.33)
        n = len(train_dataset) + len(valid_dataset) + len(test_dataset)
        self.assertEqual(n, len(dataset))


class RandomScaffoldSplitterTest(unittest.TestCase):  
    def test_split(self):
        raw_data_list = [
            {'smiles': 'CCOc1ccc2nc(S(N)(=O)=O)sc2c1'},
            {'smiles': 'CCOc1ccc2nc(S(N)(=O)=O)sc2c1'},
            {'smiles': 'CCOc1ccc2nc(S(N)(=O)=O)sc2c1'},
            {'smiles': 'CCOc1ccc2nc(S(N)(=O)=O)sc2c1'},
            {'smiles': 'CC(C)CCCCCCCOP(OCCCCCCCC(C)C)Oc1ccccc1'},
            {'smiles': 'CC(C)CCCCCCCOP(OCCCCCCCC(C)C)Oc1ccccc1'},
            {'smiles': 'CC(C)CCCCCCCOP(OCCCCCCCC(C)C)Oc1ccccc1'},
            {'smiles': 'CC(C)CCCCCCCOP(OCCCCCCCC(C)C)Oc1ccccc1'},
            {'smiles': 'CC(C)CCCCCCCOP(OCCCCCCCC(C)C)Oc1ccccc1'},
            {'smiles': 'CCCCCCCCCCOCC(O)CN'},
            {'smiles': 'CCCCCCCCCCOCC(O)CN'},
            {'smiles': 'CCCCCCCCCCOCC(O)CN'},
            {'smiles': 'CCCCCCCCCCOCC(O)CN'},
        ]
        dataset = InMemoryDataset(raw_data_list)
        splitter = RandomScaffoldSplitter()
        train_dataset, valid_dataset, test_dataset = splitter.split(
                dataset, frac_train=0.34, frac_valid=0.33, frac_test=0.33)
        n = len(train_dataset) + len(valid_dataset) + len(test_dataset)
        self.assertEqual(n, len(dataset))


if __name__ == '__main__':
    unittest.main()