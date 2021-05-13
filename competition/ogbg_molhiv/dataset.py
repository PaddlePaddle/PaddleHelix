# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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

import os
import sys
import json
import numpy as np
import glob
import argparse
from collections import OrderedDict, namedtuple

from ogb.graphproppred import GraphPropPredDataset

import pgl
from pgl.utils.data.dataset import Dataset, StreamDataset, HadoopDataset
from propeller import log

class MolDataset(Dataset):
    def __init__(self, config, raw_dataset, mode='train'):
        self.config = config
        self.raw_dataset = raw_dataset
        self.mode = mode

        log.info("preprocess graph data in %s" % self.__class__.__name__)
        self.graph_list = []

        log.info("loading mgf feature")
        mgf_feature = np.load(self.config.mgf_file)
        log.info(["the shape of mgf feature is: ", mgf_feature.shape])

        for i in range(len(self.raw_dataset)):
            # num_nodes, edge_index, node_feat, edge_feat, label
            graph, label = self.raw_dataset[i]
            num_nodes = graph['num_nodes']
            node_feat = graph['node_feat'].copy()
            edges = list(zip(graph["edge_index"][0], graph["edge_index"][1]))
            edge_feat = graph['edge_feat'].copy()

            new_graph = {}
            new_graph['num_nodes'] = num_nodes
            new_graph['node_feat'] = node_feat
            new_graph['edges'] = edges
            new_graph['edge_feat'] = edge_feat
            new_graph['mgf'] = mgf_feature[i, :].reshape(-1, )

            self.graph_list.append(new_graph)

    def __getitem__(self, idx):
        return self.graph_list[idx]

    def __len__(self):
        return len(self.graph_list)

        
class MgfCollateFn(object):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode

    def __call__(self, batch_data):
        feed_dict = {}
        num_nodes = []
        num_edges = []
        edges = []
        nfeat = []
        efeat = []
        mgf = []
        for graph_data in batch_data:
            num_nodes.append(graph_data['num_nodes'])
            n_e = len(graph_data['edges'])
            num_edges.append(n_e)
            if n_e != 0:
                edges.append(graph_data['edges'])
            nfeat.append(graph_data['node_feat'])
            efeat.append(graph_data["edge_feat"])
            mgf.append(graph_data['mgf'])

        feed_dict['num_nodes'] = np.array(num_nodes, dtype="int64")
        feed_dict['num_edges'] = np.array(num_edges, dtype="int64")
        feed_dict['edges'] = np.concatenate(edges).astype("int64")
        feed_dict['nfeat'] = np.concatenate(nfeat).astype("int64")
        feed_dict['efeat'] = np.concatenate(efeat).astype("int64")
        feed_dict['labels'] = np.array(mgf, dtype="float32")

        return feed_dict
