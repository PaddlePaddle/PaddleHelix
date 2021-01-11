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
sys.path.append("../../../")
import unittest
from rdkit.Chem import AllChem
from pahelix.featurizers.pretrain_gnn_featurizer import PreGNNAttrMaskFeaturizer,PreGNNSupervisedFeaturizer,PreGNNContextPredFeaturizer
from pahelix.featurizers.featurizer import Featurizer
from pahelix.utils.compound_tools import mol_to_graph_data
from pahelix.featurizers import pretrain_gnn_featurizer


class PreGNNAttrMaskFeaturizer(unittest.TestCase):
    def setUp(self):
        self.model = PreGNNAttrMaskFeaturizer()
    
    def test_gen_features(self):
        raw_data_list = [
            {'smiles': 'CCOc1ccc2nc(S(N)(=O)=O)sc2c1'},
            {'smiles': 'CCCCCCCCCCOCC(O)CN'},
        ]
        smiles = raw_data_list[0]['smiles']
        mol = AllChem.MolFromSmiles(smiles)
        data1 = mol_to_graph_data(mol)
        self.assertTrue(data1)
    
class PreGNNSupervisedFeaturizer(unittest.TestCase):
    
    def test_gen_features(self):
        raw_data_list = [
            {'smiles': 'CCOc1ccc2nc(S(N)(=O)=O)sc2c1','label': '0'},
            {'smiles': 'CCCCCCCCCCOCC(O)CN','label': '1'},
        ]
        smiles = raw_data_list[0]['smiles']
        mol = AllChem.MolFromSmiles(smiles)
        data2 = mol_to_graph_data(mol)
        self.assertTrue(data2)
    
class PreGNNContextPredFeaturizer(unittest.TestCase):
    
    def test_gen_features(self):
        raw_data_list = [
            {'smiles': 'CCOc1ccc2nc(S(N)(=O)=O)sc2c1'},
            {'smiles': 'CCCCCCCCCCOCC(O)CN'},
        ]
        smiles = raw_data_list[0]['smiles']
        mol = AllChem.MolFromSmiles(smiles)
        data3 = mol_to_graph_data(mol)
        self.assertTrue(data3)
   

if __name__ == '__main__':
    unittest.main()