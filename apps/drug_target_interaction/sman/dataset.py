# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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
This file implement the dataset for drug-target binding affinity prediction.
Adapted from https://github.com/PaddlePaddle/PGL/blob/main/examples/gin/Dataset.py
"""

import os
import sys
import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm

import pgl
from pgl.utils.logger import log


def random_split(dataset_size, split_ratio=0.9, seed=0, shuffle=True):
    """random splitter"""
    np.random.seed(seed)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(split_ratio * dataset_size)
    train_idx, valid_idx = indices[:split], indices[split:]
    log.info("train_set : test_set == %d : %d" %
             (len(train_idx), len(valid_idx)))
    return train_idx, valid_idx

def split_train_valid(data_path, dataset_name, seed=2020):
    
    train_filename = os.path.join(data_path, "{0}_train.pickle".format(dataset_name))
    train_filename_ = os.path.join(data_path, "{0}_train_.pickle".format(dataset_name))
    valid_filename = os.path.join(data_path, "{0}_valid.pickle".format(dataset_name))
    if os.path.isfile(valid_filename):
        return

    with open(train_filename, 'rb') as reader:
        data_drugs, data_Y = pickle.load(reader, encoding='iso-8859-1')
    
    train_idxs, valid_idxs = random_split(len(data_Y), split_ratio=0.9, seed=seed, shuffle=True)
    train_drugs = [data_drugs[i] for i in train_idxs]
    valid_drugs = [data_drugs[i] for i in valid_idxs]
    train_y = [data_Y[i] for i in train_idxs]
    valid_y = [data_Y[i] for i in valid_idxs]

    train_data = (train_drugs, train_y)
    valid_data = (valid_drugs, valid_y)
    with open(train_filename_, 'wb') as f:
        pickle.dump(train_data, f)
    with open(valid_filename, 'wb') as f:
        pickle.dump(valid_data, f)


class BaseDataset(object):
    """BaseDataset"""

    def __init__(self):
        pass

    def __getitem__(self, idx):
        """getitem"""
        raise NotImplementedError

    def __len__(self):
        """len"""
        raise NotImplementedError


class DrugDataset(BaseDataset):
    """Dataset for DTA
    """

    def __init__(self,
                 data_path,
                 dataset_name,
                 data_flag,
                 dist_dim=4,
                 self_loop=True,
                 use_identity=False):

        self.data_path = data_path
        self.dataset_name = dataset_name
        self.data_flag = data_flag
        self.self_loop = self_loop
        self.use_identity = use_identity
        self.dist_dim = dist_dim

        self.graph_list = []
        self.n2e_feed_list = []
        self.pk_list = []

        # global num
        self.num_graph = 0  # total graphs number
        self.n = 0  # total nodes number
        self.m = 0  # total edges number

        self.dim_nfeats = 0

        self._load_data()

    def __len__(self):
        """return the number of graphs"""
        return len(self.graph_list)

    def __getitem__(self, idx):
        """getitem"""
        return self.graph_list[idx], self.pk_list[idx], self.n2e_feed_list[idx]


    def _encode_dist_4(self, dist):
        dist_v = [0]*4
        if dist < 1 or dist >= 5:
            dist = min(4.9, max(1, dist))
        dist = int(dist - 1)
        # dist = min(dist, 7)
        dist_v[dist] = 1
        return dist_v
    
    def _encode_dist_np(self, dist):
        dist = np.clip(dist, 1.0, 4.999)
        dist = dist - 1
        interval = 4.0 / self.dist_dim
        dist = dist / interval
        dist = dist.astype('int32')
        dist_v = (np.arange(self.dist_dim)==dist[:, None]).astype(np.float32)
        return dist_v
    
    def _distance_uv(self, rela_pos):
        '''
        u_pos: [None, 3]
        v_pos: [None, 3]
        '''
        dist = np.sqrt(np.square(rela_pos).sum(axis=-1))
        dist_v = self._encode_dist_np(dist)
        # dist_v = np.array([self._encode_dist_4(d) for d in dist]).astype(np.float32)
        return dist, dist_v

    def _spatial_edge_feat(self, edges, coords, expand_e2e_edges, num_nodes):
        '''
        calculate spatial distance
        '''
        # distance
        edges = np.array(edges)
        u_pos, v_pos = coords[edges[:, 0]], coords[edges[:, 1]]
        rela_pos = u_pos - v_pos # edge feat: relative position vector with shape (None, 3)
        dist_scalar, dist_feat = self._distance_uv(rela_pos)

        return dist_feat


    def _node_edge_graph(self, num_nodes, node_feat, edges):
        '''
        expand edge: old_edge => node
        '''
        # new-edge
        expand_eidx_d = {}
        expand_eidx = num_nodes
        expand_e2n_edges = []
        # edges_dist_feat = []
        # n2e_feed = []
        for u, v in edges:
            expand_e2n_edges += [(expand_eidx, v)]
            expand_eidx_d[(u,v)] = expand_eidx
            # edges_dist_feat += [dist_v]
            # n2e_feed += [(u,v), dist_v]
            expand_eidx += 1
        
        # new-size, new-node_feat
        expand_size = expand_eidx
        padding_feat = np.zeros((len(edges), node_feat.shape[-1]))
        expand_node_feat = np.vstack([node_feat, padding_feat]).astype(np.float32)
        assert expand_node_feat.shape[0] == expand_size

        return expand_size, expand_node_feat, expand_e2n_edges, expand_eidx_d


    def _edge_edge_graph(self, edges, eidx_dict, num_nodes):
        '''
        (k,j) + (j,i) -> (kj, ji)
        '''
        assert self.self_loop == False

        expand_e2e_edges = []
        nx_g = nx.DiGraph(edges)
        for j in range(num_nodes):
            j_neigbors = nx_g[j]
            for i in j_neigbors:
                eidx_ji = eidx_dict[(j,i)]
                for k in j_neigbors:
                    if i == k:
                        continue
                    eidx_kj = eidx_dict[(k,j)]
                    expand_e2e_edges += [(eidx_kj, eidx_ji)]

        expand_size = num_nodes + len(eidx_dict)
        expand_node_feat = None
        return expand_size, expand_node_feat, expand_e2e_edges


    def _load_data(self):
        """Loads dataset
        """
        filename = os.path.join(self.data_path,
                                "{0}_{1}.pickle".format(self.dataset_name, self.data_flag))
        log.info("loading data from %s" % filename)

        with open(filename, 'rb') as reader:
            data_drugs, data_Y = pickle.load(reader, encoding='iso-8859-1')
        
        self.num_graph = len(data_drugs)
        self.pk_list = data_Y
        # self.pk_list = [5.0]*len(data_Y)

        for d_graph in tqdm(data_drugs):
            num_nodes, features, edges, coords = d_graph # int, ndarray, list, ndarray
            if self.self_loop:
                if [0, 0] in edges:
                    log.info('Warning: the graph need not to add self-loop which has been added.')
                    continue
                for i in range(num_nodes):
                    edges.append((i, i))

            features = np.array(features) if not self.use_identity else np.eye(num_nodes)
            features = features.astype(np.float32)
            edges = [(i, j) for i, j in edges]

            # 1. e2n_graph
            e2n_size, e2n_node_feat, e2n_edges, expand_eidx_d = self._node_edge_graph(num_nodes, features, edges)
            # 2. e2e_graph
            e2e_size, e2e_node_feat, e2e_edges = self._edge_edge_graph(edges, expand_eidx_d, num_nodes)
            # spatial features
            dist_feat = self._spatial_edge_feat(edges, coords, e2e_edges, num_nodes)
            self.n2e_feed_list.append((np.array(edges), dist_feat, num_nodes, len(edges))) # edges, feat, num_nodes, num_edges
            
            # g = pgl.graph.Graph(
            #             num_nodes=num_nodes,
            #             edges=edges,
            #             node_feat={'attr': features},
            #             edge_feat=edge_feat)
            g_e2n = pgl.graph.Graph(
                        num_nodes=e2n_size,
                        edges=e2n_edges,
                        node_feat={'attr': e2n_node_feat},
                        edge_feat={'dist': dist_feat})
            g_e2e = pgl.graph.Graph(
                        num_nodes=e2e_size,
                        edges=e2e_edges,
                        node_feat=None,
                        edge_feat=None)

            self.graph_list.append((g_e2n, g_e2e))

            # update statistics of graphs
            self.dim_nfeats = features.shape[-1]
            self.n += num_nodes
            self.m += len(edges)

        message = "finished loading data\n"
        message += """
                    num_graph: %d
                    node_features_dim: %d
                    use identity feat: %d
                    Avg. of pk values: %.2f
                    Avg. of #Nodes: %.2f
                    Avg. of #Edges: %.2f""" % (
                self.num_graph,
                self.dim_nfeats,
                self.use_identity,
                sum(self.pk_list) / len(self.pk_list),
                self.n / self.num_graph,
                self.m / self.num_graph,)
        log.info(message)
