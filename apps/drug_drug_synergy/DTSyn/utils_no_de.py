#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
utility functions for data loading and featurizing
"""
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from pgl.utils.data import Dataset, Dataloader
import numpy as np
import pgl


def atom_features(atom):
    """extract atomic features"""
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    """set max atom number equals to 100"""
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    #features = np.empty([c_size, 78])
    mask = [0] * 100
    features = np.zeros([100, 78])
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum == 0:
            return None
        
        feature = atom_features(atom)
        features[i, :] = feature / sum(feature)
        mask[i] = 1
        
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))
    #g = nx.Graph(edges).to_directed()
    
    
    g = pgl.Graph(num_nodes=100,
              edges=edges,
              node_feat={'node_feat':features})
    
    return g, mask

"""def gem_graph(smile):

    gem_feat = np.load('atom_feat_GEM.npy', allow_pickle=True).item()
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    mask = [0] * 100
    features = np.zeros([100, 1024])
    feat = gem_feat[smile]
    features[:feat.shape[0], :] = feat
    mask[:feat.shape[0]] = (1,) * feat.shape[0]

    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))

    g = pgl.Graph(num_nodes=100,
              edges=edges,
              node_feat={'node_feat':features})

    return g, mask"""


class DDsData(Dataset):
    """data"""
    def __init__(self, d1, d2, cg, label):
        super(Dataset, self).__init__()
        self.d1 = d1
        self.d2 = d2
        self.cg = cg
        
        self.label = label
    def __getitem__(self, index):
        data1 = self.d1[index]
        data2 = self.d2[index]
        data3 = self.cg[index]
        
        label = self.label[index]
        
        return data1, data2, data3, label
    def __len__(self):
        return len(self.label)

class DBData(Dataset):
    """data"""
    def __init__(self, d1, d2, cg):
        super(Dataset, self).__init__()
        self.d1 = d1
        self.d2 = d2
        self.cg = cg
        
    def __getitem__(self, index):
        data1 = self.d1[index]
        data2 = self.d2[index]
        data3 = self.cg[index]
        
        return data1, data2, data3
    def __len__(self):
        return len(self.cg)

def collate(batch):
    """data process during data loading"""
    d1_list , d2_list = [], []
    mask1, mask2 = [], []
    #dea, deb = [], []
    cells = []
    lbs = []
    for cd1, cd2, cell, label in batch:
        sm1, sm2 = cd1, cd2 #pub_dict[cd1], pub_dict[cd2]
        dg1, n_dg1 = smile_to_graph(sm1)
        dg2, n_dg2 = smile_to_graph(sm2)
        #dg1, n_dg1 = gem_graph(sm1)
        #dg2, n_dg2 = gem_graph(sm2)
        mask1.append(n_dg1)
        mask2.append(n_dg2)
        
        d1_list.append(dg1)
        d2_list.append(dg2)
        cells.append(cell)
        lbs.append(label)
        
    join_graph1 = pgl.Graph.batch(d1_list)
    join_mask1 = np.array(mask1)
    join_mask2 = np.array(mask2)
    join_graph2 = pgl.Graph.batch(d2_list)
    
    join_cells = np.array(cells)
    labels = np.array(lbs)
        
    return join_graph1, join_graph2, join_mask1, join_mask2, join_cells, labels 

def join_cell(ddi, cell):
    """join all cell feature vectors in a batch"""
    cgs = []
    for c in ddi['cell']:
        cgs.append(cell.loc[c, :].values)
    
    return cgs
