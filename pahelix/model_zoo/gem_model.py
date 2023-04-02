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
This is an implementation of GeoGNN:
"""
import numpy as np

import paddle
import paddle.nn as nn
import pgl
from pgl.nn import GraphPool

from pahelix.networks.gnn_block import GIN
from pahelix.networks.compound_encoder import AtomEmbedding, BondEmbedding, \
        BondFloatRBF, BondAngleFloatRBF
from pahelix.utils.compound_tools import CompoundKit
from pahelix.networks.gnn_block import MeanPool, GraphNorm
from pahelix.networks.basic_block import MLP


class GeoGNNBlock(nn.Layer):
    """
    GeoGNN Block
    """
    def __init__(self, embed_dim, dropout_rate, last_act):
        super(GeoGNNBlock, self).__init__()

        self.embed_dim = embed_dim
        self.last_act = last_act

        self.gnn = GIN(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.graph_norm = GraphNorm()
        if last_act:
            self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, graph, node_hidden, edge_hidden):
        """tbd"""
        out = self.gnn(graph, node_hidden, edge_hidden)
        out = self.norm(out)
        out = self.graph_norm(graph, out)
        if self.last_act:
            out = self.act(out)
        out = self.dropout(out)
        out = out + node_hidden
        return out


class GeoGNNModel(nn.Layer):
    """
    The GeoGNN Model used in GEM.

    Args:
        model_config(dict): a dict of model configurations.
    """
    def __init__(self, model_config={}):
        super(GeoGNNModel, self).__init__()

        self.embed_dim = model_config.get('embed_dim', 32)
        self.dropout_rate = model_config.get('dropout_rate', 0.2)
        self.layer_num = model_config.get('layer_num', 8)
        self.readout = model_config.get('readout', 'mean')

        self.atom_names = model_config['atom_names']
        self.bond_names = model_config['bond_names']
        self.bond_float_names = model_config['bond_float_names']
        self.bond_angle_float_names = model_config['bond_angle_float_names']

        self.init_atom_embedding = AtomEmbedding(self.atom_names, self.embed_dim)
        self.init_bond_embedding = BondEmbedding(self.bond_names, self.embed_dim)
        self.init_bond_float_rbf = BondFloatRBF(self.bond_float_names, self.embed_dim)
        
        self.bond_embedding_list = nn.LayerList()
        self.bond_float_rbf_list = nn.LayerList()
        self.bond_angle_float_rbf_list = nn.LayerList()
        self.atom_bond_block_list = nn.LayerList()
        self.bond_angle_block_list = nn.LayerList()
        for layer_id in range(self.layer_num):
            self.bond_embedding_list.append(
                    BondEmbedding(self.bond_names, self.embed_dim))
            self.bond_float_rbf_list.append(
                    BondFloatRBF(self.bond_float_names, self.embed_dim))
            self.bond_angle_float_rbf_list.append(
                    BondAngleFloatRBF(self.bond_angle_float_names, self.embed_dim))
            self.atom_bond_block_list.append(
                    GeoGNNBlock(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1)))
            self.bond_angle_block_list.append(
                    GeoGNNBlock(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1)))
        
        # TODO: use self-implemented MeanPool due to pgl bug.
        if self.readout == 'mean':
            self.graph_pool = MeanPool()
        else:
            self.graph_pool = pgl.nn.GraphPool(pool_type=self.readout)

        print('[GeoGNNModel] embed_dim:%s' % self.embed_dim)
        print('[GeoGNNModel] dropout_rate:%s' % self.dropout_rate)
        print('[GeoGNNModel] layer_num:%s' % self.layer_num)
        print('[GeoGNNModel] readout:%s' % self.readout)
        print('[GeoGNNModel] atom_names:%s' % str(self.atom_names))
        print('[GeoGNNModel] bond_names:%s' % str(self.bond_names))
        print('[GeoGNNModel] bond_float_names:%s' % str(self.bond_float_names))
        print('[GeoGNNModel] bond_angle_float_names:%s' % str(self.bond_angle_float_names))

    @property
    def node_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    @property
    def graph_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    def forward(self, atom_bond_graph, bond_angle_graph):
        """
        Build the network.
        """
        node_hidden = self.init_atom_embedding(atom_bond_graph.node_feat)
        bond_embed = self.init_bond_embedding(atom_bond_graph.edge_feat)
        edge_hidden = bond_embed + self.init_bond_float_rbf(atom_bond_graph.edge_feat)

        node_hidden_list = [node_hidden]
        edge_hidden_list = [edge_hidden]
        for layer_id in range(self.layer_num):
            node_hidden = self.atom_bond_block_list[layer_id](
                    atom_bond_graph,
                    node_hidden_list[layer_id],
                    edge_hidden_list[layer_id])
            
            cur_edge_hidden = self.bond_embedding_list[layer_id](atom_bond_graph.edge_feat)
            cur_edge_hidden = cur_edge_hidden + self.bond_float_rbf_list[layer_id](atom_bond_graph.edge_feat)
            cur_angle_hidden = self.bond_angle_float_rbf_list[layer_id](bond_angle_graph.edge_feat)
            edge_hidden = self.bond_angle_block_list[layer_id](
                    bond_angle_graph,
                    cur_edge_hidden,
                    cur_angle_hidden)
            node_hidden_list.append(node_hidden)
            edge_hidden_list.append(edge_hidden)
        
        node_repr = node_hidden_list[-1]
        edge_repr = edge_hidden_list[-1]
        graph_repr = self.graph_pool(atom_bond_graph, node_repr)
        return node_repr, edge_repr, graph_repr


class GeoPredModel(nn.Layer):
    """tbd"""
    def __init__(self, model_config, compound_encoder):
        super(GeoPredModel, self).__init__()
        self.compound_encoder = compound_encoder
        
        self.hidden_size = model_config['hidden_size']
        self.dropout_rate = model_config['dropout_rate']
        self.act = model_config['act']
        self.pretrain_tasks = model_config['pretrain_tasks']
        
        # context mask
        if 'Cm' in self.pretrain_tasks:
            self.Cm_vocab = model_config['Cm_vocab']
            self.Cm_linear = nn.Linear(compound_encoder.embed_dim, self.Cm_vocab + 3)
            self.Cm_loss = nn.CrossEntropyLoss()
        # functinal group
        self.Fg_linear = nn.Linear(compound_encoder.embed_dim, model_config['Fg_size']) # 494
        self.Fg_loss = nn.BCEWithLogitsLoss()
        # bond angle with regression
        if 'Bar' in self.pretrain_tasks:
            self.Bar_mlp = MLP(2,
                    hidden_size=self.hidden_size,
                    act=self.act,
                    in_size=compound_encoder.embed_dim * 3,
                    out_size=1,
                    dropout_rate=self.dropout_rate)
            self.Bar_loss = nn.SmoothL1Loss()
        # bond length with regression
        if 'Blr' in self.pretrain_tasks:
            self.Blr_mlp = MLP(2,
                    hidden_size=self.hidden_size,
                    act=self.act,
                    in_size=compound_encoder.embed_dim * 2,
                    out_size=1,
                    dropout_rate=self.dropout_rate)
            self.Blr_loss = nn.SmoothL1Loss()
        # atom distance with classification
        if 'Adc' in self.pretrain_tasks:
            self.Adc_vocab = model_config['Adc_vocab']
            self.Adc_mlp = MLP(2,
                    hidden_size=self.hidden_size,
                    in_size=self.compound_encoder.embed_dim * 2,
                    act=self.act,
                    out_size=self.Adc_vocab + 3,
                    dropout_rate=self.dropout_rate)
            self.Adc_loss = nn.CrossEntropyLoss()

        print('[GeoPredModel] pretrain_tasks:%s' % str(self.pretrain_tasks))

    def _get_Cm_loss(self, feed_dict, node_repr):
        masked_node_repr = paddle.gather(node_repr, feed_dict['Cm_node_i'])
        logits = self.Cm_linear(masked_node_repr)
        loss = self.Cm_loss(logits, feed_dict['Cm_context_id'])
        return loss

    def _get_Fg_loss(self, feed_dict, graph_repr):
        fg_label = paddle.concat(
                [feed_dict['Fg_morgan'],
                feed_dict['Fg_daylight'],
                feed_dict['Fg_maccs']], 1)
        logits = self.Fg_linear(graph_repr)
        loss = self.Fg_loss(logits, fg_label)
        return loss

    def _get_Bar_loss(self, feed_dict, node_repr):
        node_i_repr = paddle.gather(node_repr, feed_dict['Ba_node_i'])
        node_j_repr = paddle.gather(node_repr, feed_dict['Ba_node_j'])
        node_k_repr = paddle.gather(node_repr, feed_dict['Ba_node_k'])
        node_ijk_repr = paddle.concat([node_i_repr, node_j_repr, node_k_repr], 1)
        pred = self.Bar_mlp(node_ijk_repr)
        loss = self.Bar_loss(pred, feed_dict['Ba_bond_angle'] / np.pi)
        return loss

    def _get_Blr_loss(self, feed_dict, node_repr):
        node_i_repr = paddle.gather(node_repr, feed_dict['Bl_node_i'])
        node_j_repr = paddle.gather(node_repr, feed_dict['Bl_node_j'])
        node_ij_repr = paddle.concat([node_i_repr, node_j_repr], 1)
        pred = self.Blr_mlp(node_ij_repr)
        loss = self.Blr_loss(pred, feed_dict['Bl_bond_length'])
        return loss

    def _get_Adc_loss(self, feed_dict, node_repr):
        node_i_repr = paddle.gather(node_repr, feed_dict['Ad_node_i'])
        node_j_repr = paddle.gather(node_repr, feed_dict['Ad_node_j'])
        node_ij_repr = paddle.concat([node_i_repr, node_j_repr], 1)
        logits = self.Adc_mlp.forward(node_ij_repr)
        atom_dist = paddle.clip(feed_dict['Ad_atom_dist'], 0.0, 20.0)
        atom_dist_id = paddle.cast(atom_dist / 20.0 * self.Adc_vocab, 'int64')
        loss = self.Adc_loss(logits, atom_dist_id)
        return loss

    def forward(self, graph_dict, feed_dict, return_subloss=False):
        """
        Build the network.
        """
        node_repr, edge_repr, graph_repr = self.compound_encoder.forward(
                graph_dict['atom_bond_graph'], graph_dict['bond_angle_graph'])
        masked_node_repr, masked_edge_repr, masked_graph_repr = self.compound_encoder.forward(
                graph_dict['masked_atom_bond_graph'], graph_dict['masked_bond_angle_graph'])

        sub_losses = {}
        if 'Cm' in self.pretrain_tasks:
            sub_losses['Cm_loss'] = self._get_Cm_loss(feed_dict, node_repr)
            sub_losses['Cm_loss'] += self._get_Cm_loss(feed_dict, masked_node_repr)
        if 'Fg' in self.pretrain_tasks:
            sub_losses['Fg_loss'] = self._get_Fg_loss(feed_dict, graph_repr)
            sub_losses['Fg_loss'] += self._get_Fg_loss(feed_dict, masked_graph_repr)
        if 'Bar' in self.pretrain_tasks:
            sub_losses['Bar_loss'] = self._get_Bar_loss(feed_dict, node_repr)
            sub_losses['Bar_loss'] += self._get_Bar_loss(feed_dict, masked_node_repr)
        if 'Blr' in self.pretrain_tasks:
            sub_losses['Blr_loss'] = self._get_Blr_loss(feed_dict, node_repr)
            sub_losses['Blr_loss'] += self._get_Blr_loss(feed_dict, masked_node_repr)
        if 'Adc' in self.pretrain_tasks:
            sub_losses['Adc_loss'] = self._get_Adc_loss(feed_dict, node_repr)
            sub_losses['Adc_loss'] += self._get_Adc_loss(feed_dict, masked_node_repr)

        loss = 0
        for name in sub_losses:
            loss += sub_losses[name]
        if return_subloss:
            return loss, sub_losses
        else:
            return loss