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
"""subgraph message passing network"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import rdkit.Chem as Chem
import numpy as np
from src.nnutils import index_select_ND
from src.utils import onek_encoding_unk



ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn',
             'H', 'Cu', 'Mn', 'unknown']

ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 1
BOND_FDIM = 5
MAX_NB = 15


def atom_features(atom):
    """return atom one-hot embedding"""
    return np.array(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)
                    + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
                    + onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
                    + [atom.GetIsAromatic()])


def bond_features(bond):
    """return bond one-hot embedding"""
    bt = bond.GetBondType()
    return np.array(
        [int(bt == Chem.rdchem.BondType.SINGLE),
         int(bt == Chem.rdchem.BondType.DOUBLE),
         int(bt == Chem.rdchem.BondType.TRIPLE),
         int(bt == Chem.rdchem.BondType.AROMATIC),
         int(bond.IsInRing())])



class JTMPN(nn.Layer):
    """subgraph message passing layer"""
    def __init__(self, hidden_size, depth):
        super(JTMPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias_attr=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias_attr=False)
        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def forward(self, fatoms, fbonds, agraph, bgraph, scope, tree_message):
        """Forward"""
        fatoms = paddle.to_tensor(fatoms)
        fbonds = paddle.to_tensor(fbonds)
        agraph = paddle.to_tensor(agraph)
        bgraph = paddle.to_tensor(bgraph)

        binput = self.W_i(fbonds)
        graph_message = F.relu(binput)

        for i in range(self.depth - 1):
            message = paddle.concat([tree_message, graph_message], axis=0)
            nei_message = index_select_ND(message, 0, bgraph)
            nei_message = paddle.sum(nei_message, axis=1)
            nei_message = self.W_h(nei_message)
            graph_message = F.relu(binput + nei_message)

        message = paddle.concat([tree_message, graph_message], axis=0)
        nei_message = index_select_ND(message, 0, agraph)
        nei_message = paddle.sum(nei_message, axis=1)
        ainput = paddle.concat([fatoms, nei_message], axis=1)
        atom_hiddens = F.relu(self.W_o(ainput))

        mol_vecs = []
        for st, le in scope:
            mol_vec = paddle.sum(paddle.slice(atom_hiddens, [0], [st], [st + le]), axis=0) / le
            mol_vecs.append(mol_vec)

        mol_vecs = paddle.stack(mol_vecs, axis=0)
        return mol_vecs

    @staticmethod
    def tensorize(cand_batch, mess_dict):
        """Return tree feature"""
        fatoms, fbonds = [], []
        in_bonds, all_bonds = [], []
        total_atoms = 0
        total_mess = len(mess_dict) + 1
        scope = []
        for smiles, all_nodes, ctr_node in cand_batch:
            mol = Chem.MolFromSmiles(smiles)
            Chem.Kekulize(mol)
            n_atoms = mol.GetNumAtoms()
            ctr_bid = ctr_node.idx

            for atom in mol.GetAtoms():
                fatoms.append(atom_features(atom))
                in_bonds.append([])

            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                x = a1.GetIdx() + total_atoms
                y = a2.GetIdx() + total_atoms
                x_nid, y_nid = a1.GetAtomMapNum(), a2.GetAtomMapNum()
                x_bid = all_nodes[x_nid - 1].idx if x_nid > 0 else -1
                y_bid = all_nodes[y_nid - 1].idx if y_nid > 0 else -1

                bfeature = bond_features(bond)

                b = total_mess + len(all_bonds)
                all_bonds.append((x, y))
                fbonds.append(np.concatenate([fatoms[x], bfeature], 0))
                in_bonds[y].append(b)

                b = total_mess + len(all_bonds)
                all_bonds.append((y, x))
                fbonds.append(np.concatenate([fatoms[y], bfeature], 0))
                in_bonds[x].append(b)

                if x_bid >= 0 and y_bid >= 0 and x_bid != y_bid:
                    if (x_bid, y_bid) in mess_dict:
                        mess_idx = mess_dict[(x_bid, y_bid)]
                        in_bonds[y].append(mess_idx)
                    if (y_bid, x_bid) in mess_dict:
                        mess_idx = mess_dict[(y_bid, x_bid)]
                        in_bonds[x].append(mess_idx)

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

        for b1 in range(total_bonds):
            x, y = all_bonds[b1]
            for i, b2 in enumerate(in_bonds[x]):
                if b2 < total_mess or all_bonds[b2 - total_mess][0] != y:
                    bgraph[b1, i] = b2

        return {'fatoms': fatoms,
                'fbonds':fbonds,
                'agraph': agraph,
                'bgraph': bgraph,
                'scope': scope}

