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
This is an implementation of pretrain gnns:
https://arxiv.org/abs/1905.12265
"""
import numpy as np

import paddle
import paddle.nn as nn
import pgl
from pgl.nn import GraphPool

from pahelix.networks.gnn_block import GIN
from pahelix.networks.compound_encoder import AtomEmbedding, BondEmbedding
from pahelix.utils.compound_tools import CompoundKit
from pahelix.networks.gnn_block import MeanPool, GraphNorm


class PretrainGNNModel(nn.Layer):
    """
    The basic GNN Model used in pretrain gnns.
    Args:
        model_config(dict): a dict of model configurations.
        name(str): the prefix of model params.
    """
    def __init__(self, model_config={}):
        super(PretrainGNNModel, self).__init__()

        self.embed_dim = model_config.get('embed_dim', 300)
        self.dropout_rate = model_config.get('dropout_rate', 0.5)
        self.norm_type = model_config.get('norm_type', 'batch_norm')
        self.graph_norm = model_config.get('graph_norm', False)
        self.residual = model_config.get('residual', False)
        self.layer_num = model_config.get('layer_num', 5)
        self.gnn_type = model_config.get('gnn_type', 'gin')
        self.JK = model_config.get('JK', 'last')
        self.readout = model_config.get('readout', 'mean')

        self.atom_names = model_config['atom_names']
        self.bond_names = model_config['bond_names']

        self.atom_embedding = AtomEmbedding(self.atom_names, self.embed_dim)
        self.bond_embedding_list = nn.LayerList()
        self.gnn_list = nn.LayerList()
        self.norm_list = nn.LayerList()
        self.graph_norm_list = nn.LayerList()
        self.dropout_list = nn.LayerList()
        for layer_id in range(self.layer_num):
            self.bond_embedding_list.append(BondEmbedding(self.bond_names, self.embed_dim))

            if self.gnn_type == 'gin':
                self.gnn_list.append(GIN(self.embed_dim))
            else:
                raise ValueError(self.gnn_type)

            if self.norm_type == 'batch_norm':
                self.norm_list.append(nn.BatchNorm1D(self.embed_dim))
            elif self.norm_type == 'layer_norm':
                self.norm_list.append(nn.LayerNorm(self.embed_dim))
            else:
                raise ValueError(self.norm_type)

            if self.graph_norm:
                self.graph_norm_list.append(GraphNorm())    # TODO: pgl.nn.GraphNorm not implemented in pgl==2.1.2

            self.dropout_list.append(nn.Dropout(self.dropout_rate))
        
        # TODO: use self-implemented MeanPool due to pgl bug.
        if self.readout == 'mean':
            self.graph_pool = MeanPool()
        else:
            self.graph_pool = pgl.nn.GraphPool(pool_type=self.readout)

        print('[PretrainGNNModel] embed_dim:%s' % self.embed_dim)
        print('[PretrainGNNModel] dropout_rate:%s' % self.dropout_rate)
        print('[PretrainGNNModel] norm_type:%s' % self.norm_type)
        print('[PretrainGNNModel] graph_norm:%s' % self.graph_norm)
        print('[PretrainGNNModel] residual:%s' % self.residual)
        print('[PretrainGNNModel] layer_num:%s' % self.layer_num)
        print('[PretrainGNNModel] gnn_type:%s' % self.gnn_type)
        print('[PretrainGNNModel] JK:%s' % self.JK)
        print('[PretrainGNNModel] readout:%s' % self.readout)
        print('[PretrainGNNModel] atom_names:%s' % str(self.atom_names))
        print('[PretrainGNNModel] bond_names:%s' % str(self.bond_names))

    @property
    def node_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    @property
    def graph_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    def forward(self, graph):
        """
        Build the network.
        """
        node_feat = self.atom_embedding(graph.node_feat)

        node_feat_list = [node_feat]
        for layer_id in range(self.layer_num):
            edge_features = self.bond_embedding_list[layer_id](graph.edge_feat)
            node_feat = self.gnn_list[layer_id](
                    graph, 
                    node_feat_list[layer_id], 
                    edge_features)
            node_feat = self.norm_list[layer_id](node_feat)
            if self.graph_norm:
                node_feat = self.graph_norm_list[layer_id](graph, node_feat)
            if layer_id < self.layer_num - 1:
                node_feat = nn.functional.relu(node_feat)
            node_feat = self.dropout_list[layer_id](node_feat)
            if self.residual:
                node_feat = node_feat + node_feat_list[layer_id]
            node_feat_list.append(node_feat)

        if self.JK == "sum":
            node_repr = paddle.sum(node_feat_list, axis=0)
        elif self.JK == "mean":
            node_repr = paddle.mean(node_feat_list, axis=0)
        elif self.JK == "last":
            node_repr = node_feat_list[-1]
        else:
            raise ValueError(self.JK)
        
        graph_repr = self.graph_pool(graph, node_repr)
        return node_repr, graph_repr


class AttrmaskModel(nn.Layer):
    """
    This is a pretraning model used by pretrain gnns for attribute mask training.
    Returns:
        loss: the loss variance of the model.
    """
    def __init__(self, model_config, compound_encoder):
        super(AttrmaskModel, self).__init__()

        self.compound_encoder = compound_encoder

        out_size = CompoundKit.get_atom_feature_size('atomic_num') + 3
        self.linear = nn.Linear(compound_encoder.node_dim, out_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, graphs, masked_node_indice, masked_node_labels):
        """
        Build the network.
        """
        node_repr, graph_repr = self.compound_encoder(graphs)
        masked_node_repr = paddle.gather(node_repr, masked_node_indice)
        logits = self.linear(masked_node_repr)
        loss = self.criterion(logits, masked_node_labels)
        return loss


class SupervisedModel(nn.Layer):
    """
    This is a pretraning model used by pretrain gnns for supervised training.
    Returns:
        self.loss: the loss variance of the model.
    """
    def __init__(self, model_config, compound_encoder):
        super(SupervisedModel, self).__init__()
        self.task_num = model_config['task_num']

        self.compound_encoder = compound_encoder

        self.linear = nn.Linear(compound_encoder.graph_dim, self.task_num)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, graphs, labels, valids):
        """
        Build the network.
        """
        node_repr, graph_repr = self.compound_encoder(graphs)
        logits = self.linear(graph_repr)
        loss = self.criterion(logits, labels)
        loss = paddle.sum(loss * valids) / paddle.sum(valids)
        return loss

