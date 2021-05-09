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
"""grah message passing network"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import rdkit.Chem as Chem
import numpy as np
from src.nnutils import index_select_ND
from src.chemutils import get_mol
from src.utils import onek_encoding_unk

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn',
             'H', 'Cu', 'Mn', 'unknown']

ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6


def atom_features(atom):
    """return atom one-hot embedding"""
    return np.array(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)
                    + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
                    + onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
                    + onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3])
                    + [atom.GetIsAromatic()])



def bond_features(bond):
    """return bond one-hot embedding"""
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [int(bt == Chem.rdchem.BondType.SINGLE), int(bt == Chem.rdchem.BondType.DOUBLE), 
             int(bt == Chem.rdchem.BondType.TRIPLE), int(bt == Chem.rdchem.BondType.AROMATIC), 
             int(bond.IsInRing())]
    fstereo = onek_encoding_unk(stereo, [0, 1, 2, 3, 4, 5])
    return np.array(fbond + fstereo)



class MPN(nn.Layer):
    """Graph message passing layer"""
    def __init__(self, hidden_size, depth):
        super(MPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias_attr=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias_attr=False)
        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def forward(self, fatoms, fbonds, agraph, bgraph, scope):
        """Forward"""
        fatoms = paddle.to_tensor(fatoms)
        fbonds = paddle.to_tensor(fbonds) 
        agraph = paddle.to_tensor(agraph) 
        bgraph = paddle.to_tensor(bgraph)

        binput = self.W_i(fbonds)
        message = F.relu(binput)

        for i in range(self.depth - 1):
            nei_message = index_select_ND(message, 0, bgraph)
            nei_message = paddle.sum(nei_message, axis=1)
            nei_message = self.W_h(nei_message)
            message = F.relu(binput + nei_message)

        nei_message = index_select_ND(message, 0, agraph)
        nei_message = paddle.sum(nei_message, axis=1)
        ainput = paddle.concat([fatoms, nei_message], axis=1)
        atom_hiddens = F.relu(self.W_o(ainput))

        max_len = max([x for _, x in scope])
        batch_vecs = []
        for st, le in scope:
            cur_vecs = paddle.mean(atom_hiddens[st: st + le], axis=0)
            batch_vecs.append(cur_vecs)

        mol_vecs = paddle.stack(batch_vecs, axis=0)
        return mol_vecs

    @staticmethod
    def tensorize(mol_batch):
        """transform mol object into graph feature"""
        padding = np.zeros([ATOM_FDIM + BOND_FDIM]).astype('int64')

        fatoms, fbonds = [], [padding]  
        in_bonds, all_bonds = [], [(-1, -1)]  
        scope = []
        total_atoms = 0

        for smiles in mol_batch:
            mol = get_mol(smiles)
            n_atoms = mol.GetNumAtoms()
            for atom in mol.GetAtoms():
                fatoms.append(atom_features(atom))
                in_bonds.append([])

            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                x = a1.GetIdx() + total_atoms
                y = a2.GetIdx() + total_atoms

                b = len(all_bonds)
                all_bonds.append((x, y))
                fbonds.append(np.concatenate([fatoms[x], bond_features(bond)], 0))
                in_bonds[y].append(b)

                b = len(all_bonds)
                all_bonds.append((y, x))
                fbonds.append(np.concatenate([fatoms[y], bond_features(bond)], 0))
                in_bonds[x].append(b)

            scope.append((total_atoms, n_atoms))
            total_atoms += n_atoms

        total_bonds = len(all_bonds)

        fatoms = np.stack(fatoms, 0).astype('float32')
        fbonds = np.stack(fbonds, 0).astype('float32')
        agraph = np.zeros([total_atoms, MAX_NB]).astype('int64')
        bgraph = np.zeros([total_bonds, MAX_NB]).astype('int64')
        for a in range(total_atoms):
            for i, b in enumerate(in_bonds[a]):
                agraph[a, i] = b

        for b1 in range(1, total_bonds):
            x, y = all_bonds[b1]
            for i, b2 in enumerate(in_bonds[x]):
                if all_bonds[b2][0] != y:
                    bgraph[b1, i] = b2

        return {'fatoms': fatoms, 
                'fbonds': fbonds, 
                'agraph': agraph, 
                'bgraph': bgraph, 
                'scope': scope}

