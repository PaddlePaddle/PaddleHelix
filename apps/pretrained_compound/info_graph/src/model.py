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
Graph-based models for compounds.
"""

import numpy as np

import paddle
import paddle.nn as nn
import pgl

from pahelix.networks.compound_encoder import AtomEmbedding, BondEmbedding
from pahelix.networks.gnn_block import MeanPool

from src.utils import get_positive_expectation, get_negative_expectation


class GINEncoder(nn.Layer):
    """
    | GIN Encoder for unsupervised InfoGraph.

    Public Functions:
        - ``forward``: forward to create the GIN compound encoder.
        - ``get_embeddings``: compute all the embeddings given dataset.
        - ``embedding_dim``: get dimension of the embedding.
    """
    def __init__(self, config):
        super(GINEncoder, self).__init__()
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.embed_dim = config['embed_dim']
        self.atom_type_num = config['atom_type_num']
        self.chirality_tag_num = config['chirality_tag_num']
        self.bond_type_num = config['bond_type_num']
        self.bond_direction_num = config['bond_direction_num']
        self.readout = config['readout']
        self.activation = config['activation']

        self.atom_names = config['atom_names']
        self.bond_names = config['bond_names']

        self.atom_embedding = AtomEmbedding(self.atom_names, self.embed_dim)
        self.gin_list = nn.LayerList()
        self.norm_list = nn.LayerList()

        for layer_id in range(self.num_layers):
            self.gin_list.append(
                pgl.nn.GINConv(self.embed_dim, self.embed_dim, activation=self.activation))
            self.norm_list.append(nn.BatchNorm1D(self.embed_dim))

        if self.readout == 'mean':
            self.graph_pool = MeanPool()
        else:
            self.graph_pool = pgl.nn.GraphPool(pool_type=self.readout)

    def forward(self, graph):
        """
        Build the network.
        """
        x = self.atom_embedding(graph.node_feat)
        x = paddle.squeeze(x, axis=1)

        patch_repr = []
        for i in range(self.num_layers):
            x = self.gin_list[i](graph, x)
            x = self.norm_list[i](x)
            patch_repr.append(x)  # $h_i^{(k)}$

        patch_summary = paddle.concat(patch_repr, axis=1)  # $h_{\phi}^i$
        patch_pool = [self.graph_pool(graph, x) for x in patch_repr]
        global_repr = paddle.concat(patch_pool, axis=1)
        return global_repr, patch_summary

    @property
    def embedding_dim(self):
        return self.num_layers * self.hidden_size


class FF(nn.Layer):
    """Feedforward network with linear shortcut for InfoGraph"""
    def __init__(self, in_size, hidden_size, num_layers=3):
        super(FF, self).__init__()
        layers = []
        for layer_id in range(num_layers):
            if layer_id == 0:
                layers.append(nn.Linear(in_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))

            layers.append(nn.ReLU())

        self.block = nn.Sequential(*layers)
        self.linear_shortcut = nn.Linear(in_size, hidden_size)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


class PriorDiscriminator(nn.Layer):
    """Prior discriminator for InfoGraph"""
    def __init__(self, in_size, hidden_size, num_layers=3):
        super(PriorDiscriminator, self).__init__()
        assert num_layers > 1

        layers = []
        for layer_id in range(num_layers):
            if layer_id == 0:
                layers.append(nn.Linear(in_size, hidden_size))
                layers.append(nn.ReLU())
            elif layer_id < num_layers - 1:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(hidden_size, 1))
        self.mlp = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.mlp(x)
        return self.sigmoid(x)


class InfoGraph(nn.Layer):
    """InfoGraph model.

    Args:
        config (dict): config dictionary of the GIN encoder.

    Returns:
        global_repr (Tensor): global-level representation of graph
        enc (Tensor): path-level representation of nodes

    Reference: InfoGraph: Unsupervised and Semi-supervised Graph-Level
    Representation Learning via Mutual Information Maximization
    """
    def __init__(self, config):
        super(InfoGraph, self).__init__()
        self.encoder = GINEncoder(config)
        dim = self.encoder.embedding_dim
        self.feedforward = FF(dim, dim)

    def forward(self, graph):
        global_repr, patch_summary = self.encoder(graph)
        g_enc = self.feedforward(global_repr)
        l_enc = self.feedforward(patch_summary)
        enc = paddle.matmul(l_enc, g_enc, transpose_y=True)
        return global_repr, enc


class InfoGraphCriterion(nn.Layer):
    """ Criterion of InfoGraph unspervised learning model
    via maximization of mutual information.
    """
    def __init__(self, config):
        super(InfoGraphCriterion, self).__init__()
        self.dim = config['hidden_size'] * config['num_layers']
        self.measure = config['measure']
        self.prior = config['prior']
        self.gamma = config['gamma']
        if self.prior:
            self.prior_discriminator = PriorDiscriminator(self.dim, self.dim)

    def forward(self, graph, global_repr, enc, pos_mask, neg_mask, prior=None):
        E_pos = get_positive_expectation(
            enc * pos_mask, self.measure, average=False)
        E_pos = paddle.sum(E_pos) / graph.num_nodes

        E_neg = get_negative_expectation(
            enc * neg_mask, self.measure, average=False)
        E_neg = paddle.sum(E_neg) / (graph.num_nodes * (graph.num_graph - 1))
        local_global_loss = E_neg - E_pos

        if self.prior:
            term_1 = paddle.mean(paddle.log(self.prior_discriminator(prior)))
            term_2 = paddle.mean(
                paddle.log(1.0 - self.prior_discriminator(global_repr)))
            prior_loss = - (term_1 + term_2) * self.gamma
        else:
            prior_loss = 0

        return local_global_loss + prior_loss
