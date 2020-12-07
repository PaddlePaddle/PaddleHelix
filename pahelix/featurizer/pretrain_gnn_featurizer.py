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
Compound datasets from pretrain-gnn.
"""

import numpy as np
import networkx as nx
import pgl
from rdkit.Chem import AllChem

from pahelix.featurizer.featurizer import Featurizer
from pahelix.utils.compound_tools import mol_to_graph_data

__all__ = [
    'PreGNNAttrMaskFeaturizer',
    'PreGNNSupervisedFeaturizer',
    'PreGNNContextPredFeaturizer',
]


class PreGNNAttrMaskFeaturizer(Featurizer):
    """docstring for PreGNNAttrMaskFeaturizer"""
    def __init__(self, graph_wrapper, atom_type_num=None, mask_ratio=None):
        super(PreGNNAttrMaskFeaturizer, self).__init__()
        self.graph_wrapper = graph_wrapper
        self.atom_type_num = atom_type_num
        self.mask_ratio = mask_ratio
    
    def gen_features(self, raw_data):
        """tbd"""
        smiles = raw_data['smiles']
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            return None
        data = mol_to_graph_data(mol)
        data['smiles'] = smiles
        return data

    def collate_fn(self, batch_data_list):
        """tbd"""
        g_list = []
        for data in batch_data_list:
            g = pgl.graph.Graph(num_nodes = len(data['atom_type']),
                    edges = data['edges'],
                    node_feat = {
                        'atom_type': data['atom_type'].reshape([-1, 1]),
                        'chirality_tag': data['chirality_tag'].reshape([-1, 1]),
                    },
                    edge_feat ={
                        'bond_type': data['bond_type'].reshape([-1, 1]),
                        'bond_direction': data['bond_direction'].reshape([-1, 1]),
                    })
            g_list.append(g)
        join_graph = pgl.graph.MultiGraph(g_list)

        ### mask atom
        num_node = len(join_graph.node_feat['atom_type'])
        masked_size = int(num_node * self.mask_ratio)
        masked_node_indice = np.random.choice(range(num_node), size=masked_size, replace=False)
        masked_node_labels = join_graph.node_feat['atom_type'][masked_node_indice]

        join_graph.node_feat['atom_type'][masked_node_indice] = self.atom_type_num
        join_graph.node_feat['chirality_tag'][masked_node_indice] = 0

        feed_dict = self.graph_wrapper.to_feed(join_graph)
        feed_dict['masked_node_indice'] = np.reshape(masked_node_indice, [-1, 1]).astype('int64')
        feed_dict['masked_node_label'] = np.reshape(masked_node_labels, [-1, 1]).astype('int64')
        return feed_dict


class PreGNNSupervisedFeaturizer(Featurizer):
    """docstring for PreGNNSupervisedFeaturizer"""
    def __init__(self, graph_wrapper):
        super(PreGNNSupervisedFeaturizer, self).__init__()
        self.graph_wrapper = graph_wrapper
    
    def gen_features(self, raw_data):
        """tbd"""
        smiles, label = raw_data['smiles'], raw_data['label']
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            return None
        data = mol_to_graph_data(mol)
        data['label'] = label.reshape([-1])
        data['smiles'] = smiles
        return data

    def collate_fn(self, batch_data_list):
        """tbd"""
        g_list = []
        label_list = []
        for data in batch_data_list:
            g = pgl.graph.Graph(num_nodes = len(data['atom_type']),
                    edges = data['edges'],
                    node_feat = {
                        'atom_type': data['atom_type'].reshape([-1, 1]),
                        'chirality_tag': data['chirality_tag'].reshape([-1, 1]),
                    },
                    edge_feat ={
                        'bond_type': data['bond_type'].reshape([-1, 1]),
                        'bond_direction': data['bond_direction'].reshape([-1, 1]),
                    })
            g_list.append(g)
            label_list.append(data['label'])
        
        join_graph = pgl.graph.MultiGraph(g_list)
        feed_dict = self.graph_wrapper.to_feed(join_graph)

        batch_label = np.array(label_list)
        batch_label = ((batch_label + 1.0) / 2).astype('float32')
        batch_valid = (batch_label != 0.5).astype("float32")
        feed_dict['supervised_label'] = batch_label
        feed_dict['valid'] = batch_valid
        return feed_dict


def reset_idxes(G):
    """
    Resets node indices such that they are numbered from 0 to num_nodes - 1
    :param G:
    :return: copy of G with relabelled node indices, mapping
    """
    mapping = {}
    for new_idx, old_idx in enumerate(G.nodes()):
        mapping[old_idx] = new_idx
    new_G = nx.relabel_nodes(G, mapping, copy=True)
    return new_G, mapping


def graph_data_obj_to_nx_simple(data):
    """
    Converts graph Data object required by the pytorch geometric package to
    network x data object. NB: Uses simplified atom and bond features,
    and represent as indices. NB: possible issues with recapitulating relative
    stereochemistry since the edges in the nx object are unordered.
    :param data: pytorch geometric Data object
    :return: network x object
    """
    G = nx.Graph()

    # atoms
    # atom_features = data['node_feat']
    num_atoms = data['atom_type'].shape[0]
    for i in range(num_atoms):
        # atomic_num_idx, chirality_tag_idx = atom_features[i]
        G.add_node(i, 
                atom_num_idx=data['atom_type'][i], 
                chirality_tag_idx=data['chirality_tag'][i])

    # bonds
    edge_index = data['edges']
    # edge_attr = data['edge_feat']
    num_bonds = edge_index.shape[0]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[j, 0])
        end_idx = int(edge_index[j, 1])
        # bond_type_idx, bond_dir_idx = edge_attr[j]
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx, 
                    bond_type_idx=data['bond_type'][j],
                    bond_dir_idx=data['bond_direction'][j])

    return G


def nx_to_graph_data_obj_simple(G):
    """
    Converts nx graph to pytorch geometric Data object. Assume node indices
    are numbered from 0 to num_nodes - 1. NB: Uses simplified atom and bond
    features, and represent as indices. NB: possible issues with
    recapitulating relative stereochemistry since the edges in the nx
    object are unordered.
    :param G: nx graph obj
    :return: pytorch geometric Data object
    """
    # atoms
    atom_types = []
    chirality_tags = []
    for _, node in G.nodes(data=True):
        atom_types.append(node['atom_num_idx'])
        chirality_tags.append(node['chirality_tag_idx'])
    atom_types = np.array(atom_types)
    chirality_tags = np.array(chirality_tags)

    # bonds
    # num_bond_features = 2  # bond type, bond direction
    if len(G.edges()) > 0:  # mol has bonds
        edges = []
        bond_types = []
        bond_directions = []
        for i, j, edge in G.edges(data=True):
            # edge_feature = [edge['bond_type_idx'], edge['bond_dir_idx']]
            edges.append((i, j))
            bond_types.append(edge['bond_type_idx'])
            bond_directions.append(edge['bond_dir_idx'])
            edges.append((j, i))
            bond_types.append(edge['bond_type_idx'])
            bond_directions.append(edge['bond_dir_idx'])

        edges = np.array(edges)
        bond_types = np.array(bond_types)
        bond_directions = np.array(bond_directions)

    else:   # mol has no bonds
        edges = np.zeros((0, 2)).astype("int")
        bond_types = np.zeros((0,)).astype("int")
        bond_directions = np.zeros((0,)).astype("int")

    data = {
        'atom_type': atom_types,         # (N,)
        'chirality_tag': chirality_tags,    # (N,)
        'edges': edges,            # (E, 2)
        'bond_type': bond_types,    # (E,)
        'bond_direction': bond_directions,  # (E,)
    }
    return data


def transform_contextpred(data, k, l1, l2):
    """tbd"""
    assist_data = {}

    num_atoms = data['atom_type'].shape[0]
    root_idx = np.random.choice(range(num_atoms), 1)[0]

    G = graph_data_obj_to_nx_simple(data)  # same ordering as input data obj

    # Get k-hop subgraph rooted at specified atom idx
    substruct_node_idxes = nx.single_source_shortest_path_length(
            G, root_idx, k).keys()

    if len(substruct_node_idxes) == 0:
        return None
    substruct_G = G.subgraph(substruct_node_idxes)
    substruct_G, substruct_node_map = reset_idxes(substruct_G)  # need
    # to reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
    # make sense, since the node indices in data obj must start at 0
    substruct_data = nx_to_graph_data_obj_simple(substruct_G)
    assist_data['substruct_center_idx'] = np.array([substruct_node_map[root_idx]])  # need
    # to convert center idx from original graph node ordering to the
    # new substruct node ordering

    # Get subgraphs that is between l1 and l2 hops away from the root node
    l1_node_idxes = nx.single_source_shortest_path_length(
            G, root_idx, l1).keys()
    l2_node_idxes = nx.single_source_shortest_path_length(
            G, root_idx, l2).keys()
    context_node_idxes = set(l1_node_idxes).symmetric_difference(
            set(l2_node_idxes))
    if len(context_node_idxes) == 0:
        return None
    context_G = G.subgraph(context_node_idxes)
    context_G, context_node_map = reset_idxes(context_G)  # need to
    # reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
    # make sense, since the node indices in data obj must start at 0
    context_data = nx_to_graph_data_obj_simple(context_G)

    # Get indices of overlapping nodes between substruct and context,
    # WRT context ordering
    context_substruct_overlap_idxes = list(set(
        context_node_idxes).intersection(set(substruct_node_idxes)))
    if len(context_substruct_overlap_idxes) == 0:
        return None
    context_substruct_overlap_idxes_reorder = [
            context_node_map[old_idx] for old_idx in context_substruct_overlap_idxes]
    # need to convert the overlap node idxes, which is from the
    # original graph node ordering to the new context node ordering
    assist_data['context_overlap_idx'] = np.array(context_substruct_overlap_idxes_reorder)

    return substruct_data, context_data, assist_data


class PreGNNContextPredFeaturizer(Featurizer):
    """docstring for PreGNNContextPredFeaturizer"""
    def __init__(self, substruct_graph_wrapper, context_graph_wrapper, k, l1, l2):
        super(PreGNNContextPredFeaturizer, self).__init__()
        self.substruct_graph_wrapper = substruct_graph_wrapper
        self.context_graph_wrapper = context_graph_wrapper
        self.k = k
        self.l1 = l1
        self.l2 = l2
    
    def gen_features(self, raw_data):
        """tbd"""
        smiles = raw_data['smiles']
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            return None
        data = mol_to_graph_data(mol)
        new_data = {}
        transformed = transform_contextpred(data, self.k, self.l1, self.l2)
        if transformed is None:
            return None
        new_data['transformed'] = transformed
        new_data['smiles'] = smiles
        return new_data

    def collate_fn(self, batch_data_list):
        """tbd"""
        list_substruct_g = []
        list_context_g = []
        list_substruct_center_idx = []
        list_context_overlap_idx = []
        cum_substruct, cum_context = 0, 0
        for data in batch_data_list:
            substruct_data, context_data, assist_data = data['transformed']
            substruct_g = pgl.graph.Graph(num_nodes = len(substruct_data['atom_type']),
                    edges = substruct_data['edges'],
                    node_feat = {
                        'atom_type': substruct_data['atom_type'].reshape([-1, 1]),
                        'chirality_tag': substruct_data['chirality_tag'].reshape([-1, 1]),
                    },
                    edge_feat ={
                        'bond_type': substruct_data['bond_type'].reshape([-1, 1]),
                        'bond_direction': substruct_data['bond_direction'].reshape([-1, 1]),
                    })
            context_g = pgl.graph.Graph(num_nodes = len(context_data['atom_type']),
                    edges = context_data['edges'],
                    node_feat = {
                        'atom_type': context_data['atom_type'].reshape([-1, 1]),
                        'chirality_tag': context_data['chirality_tag'].reshape([-1, 1]),
                    },
                    edge_feat ={
                        'bond_type': context_data['bond_type'].reshape([-1, 1]),
                        'bond_direction': context_data['bond_direction'].reshape([-1, 1]),
                    })
            list_substruct_g.append(substruct_g)
            list_context_g.append(context_g)
            list_substruct_center_idx.append(assist_data['substruct_center_idx'] + cum_substruct)
            list_context_overlap_idx.append(assist_data['context_overlap_idx'] + cum_context)
            cum_substruct += len(substruct_data['atom_type'])
            cum_context += len(context_data['atom_type'])

        join_substruct_graph = pgl.graph.MultiGraph(list_substruct_g)
        join_context_graph = pgl.graph.MultiGraph(list_context_g)
        substruct_center_idx = np.concatenate(list_substruct_center_idx, 0)
        context_overlap_idx = np.concatenate(list_context_overlap_idx, 0)
        context_overlap_lod = np.append([0], np.cumsum([len(x) for x in list_context_overlap_idx]))
        batch_size = len(list_context_overlap_idx)
        context_cycle_index = np.append(np.arange(1, batch_size), [0])

        feed_dict = self.substruct_graph_wrapper.to_feed(join_substruct_graph)
        feed_dict.update(self.context_graph_wrapper.to_feed(join_context_graph))
        feed_dict['substruct_center_idx'] = np.reshape(substruct_center_idx, [-1, 1]).astype('int64')
        feed_dict['context_overlap_idx'] = np.reshape(context_overlap_idx, [-1, 1]).astype('int64')
        feed_dict['context_overlap_lod'] = np.reshape(context_overlap_lod, [1, -1]).astype('int32')
        feed_dict['context_cycle_index'] = np.reshape(context_cycle_index, [-1, 1]).astype('int64')
        return feed_dict

