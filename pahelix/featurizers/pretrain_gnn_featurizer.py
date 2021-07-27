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
| Featurizers for pretrain-gnn.

| Adapted from https://github.com/snap-stanford/pretrain-gnns/tree/master/chem/utils.py
"""

import numpy as np
import networkx as nx
import pgl
from rdkit.Chem import AllChem

from pahelix.utils.compound_tools import mol_to_graph_data


class AttrmaskTransformFn(object):
    """Gen features for attribute mask model of pretrain gnns"""
    def __init__(self):
        pass

    def __call__(self, raw_data):
        """
        Gen features according to raw data and return a single graph data.

        Args:
            raw_data: It contains smiles and label,we convert smiles to mol
            by rdkit,then convert mol to graph data.
        
        Returns:
            data: It contains reshape label and smiles.
        """
        smiles = raw_data['smiles']
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            return None
        data = mol_to_graph_data(mol)
        return data


class AttrmaskCollateFn(object):
    """CollateFn for attribute mask model of pretrain gnns"""
    def __init__(self, atom_names, bond_names, mask_ratio=0.15):
        self.atom_names = atom_names
        self.bond_names = bond_names
        self.mask_ratio = mask_ratio
    
    def __call__(self, data_list):
        """
        Collate features about a sublist of graph data and return join_graph, 
        masked_node_indice and masked_node_labels.

        Args:
            data_list : the graph data in AttrmaskCollateFn.for data in data_list,
            create node features and edge features according to pgl graph,and then 
            use graph wrapper to feed join graph, then the label can be arrayed to batch label.

        Returns:
            The batch data contains finetune label and valid,which are 
            collected from batch_label and batch_valid.
        """
        g_list = []
        for data in data_list:
            g = pgl.Graph(num_nodes=len(data[self.atom_names[0]]),
                    edges=data['edges'],
                    node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_names})
            g_list.append(g)

        join_graph = pgl.Graph.batch(g_list)
        for name in join_graph.node_feat:
            join_graph.node_feat[name] = join_graph.node_feat[name].reshape([-1])
        for name in join_graph.edge_feat:
            join_graph.edge_feat[name] = join_graph.edge_feat[name].reshape([-1])

        ### mask atom
        N = join_graph.num_nodes
        masked_size = int(N * self.mask_ratio)
        masked_node_indice = np.random.choice(range(N), size=masked_size, replace=False)
        masked_node_labels = join_graph.node_feat['atomic_num'][masked_node_indice]
        for name in join_graph.node_feat:
            join_graph.node_feat[name][masked_node_indice] = 0       # 0: OOV

        return join_graph, masked_node_indice, masked_node_labels


class SupervisedTransformFn(object):
    """Gen features for supervised model of pretrain gnns"""
    def __init__(self):
        pass

    def __call__(self, raw_data):
        """
        Gen features according to raw data and return a single graph data.

        Args:
            raw_data: It contains smiles and label,we convert smiles 
            to mol by rdkit,then convert mol to graph data.

        Returns:
            data: It contains reshape label and smiles.
        """
        smiles, label = raw_data['smiles'], raw_data['label']
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            return None
        data = mol_to_graph_data(mol)
        data['label'] = label.reshape([-1])
        return data


class SupervisedCollateFn(object):
    """CollateFn for supervised model of pretrain gnns"""
    def __init__(self, atom_names, bond_names):
        self.atom_names = atom_names
        self.bond_names = bond_names
    
    def __call__(self, data_list):
        """
        Collate features about a sublist of graph data and return a big batch feed dictionary.

        Args:
            data_list : the graph data in SupervisedTransformFn for data in data_list,
            create node features and edge features according to pgl graph,and then 
            use graph wrapper to feed join graph, then the label can be arrayed to batch label.

        Returns:
            The batch data contains finetune label and valid,which are 
            collected from batch_label and batch_valid.      
        """
        g_list = []
        label_list = []
        for data in data_list:
            g = pgl.Graph(num_nodes=len(data[self.atom_names[0]]),
                    edges=data['edges'],
                    node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_names})
            g_list.append(g)
            label_list.append(data['label'])

        join_graph = pgl.Graph.batch(g_list)
        for name in join_graph.node_feat:
            join_graph.node_feat[name] = join_graph.node_feat[name].reshape([-1])
        for name in join_graph.edge_feat:
            join_graph.edge_feat[name] = join_graph.edge_feat[name].reshape([-1])

        labels = np.array(label_list)
        labels = (labels + 1.0) / 2
        valids = labels != 0.5
        return join_graph, labels, valids

