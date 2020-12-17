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
import unittest
from rdkit import Chem
from rdkit.Chem import AllChem

from pahelix.utils.compound_tools import smiles_to_graph_data
from pahelix.utils.compound_tools import mol_to_graph_data
from pahelix.utils.compound_tools import get_gasteiger_partial_charges
from pahelix.utils.compound_tools import create_standardized_mol_id
from pahelix.utils.compound_tools import split_rdkit_mol_obj
from pahelix.utils.compound_tools import CompoundConstants


class CompoundToolsTest(unittest.TestCase):
    def test_mol_to_graph_data(self, add_self_loop=True):
        smiles ='CCOc1ccc2nc(S(N)(=O)=O)sc2c1'
        mol = AllChem.MolFromSmiles(smiles)
        data = mol_to_graph_data(mol)
        self.assertTrue(data)
  
    def test_smiles_to_graph_data(self, add_self_loop=True):
        smiles ='CCOc1ccc2nc(S(N)(=O)=O)sc2c1'
        data = smiles_to_graph_data(smiles)
        self.assertTrue(data)
       
    def test_get_gasteiger_partial_charges(self, n_iter=12):
        smiles ='CCOc1ccc2nc(S(N)(=O)=O)sc2c1'
        mol = AllChem.MolFromSmiles(smiles)
        charges = get_gasteiger_partial_charges(mol)
        self.assertEqual(len(charges), 16)

    def test_create_standardized_mol_id(self):
        smiles ='CCOc1ccc2nc(S(N)(=O)=O)sc2c1'
        id1 = create_standardized_mol_id(smiles)
        id2 = create_standardized_mol_id(smiles)
        self.assertEqual(id1, id2)


if __name__ == '__main__':
    unittest.main()