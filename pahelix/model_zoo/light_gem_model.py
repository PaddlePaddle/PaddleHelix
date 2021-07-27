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
This is an implementation of LiteGEM:
"""
import math

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed as dist

import pgl
import pgl.nn as gnn
from pgl.utils.logger import log

from pahelix.networks.compound_encoder import AtomEmbedding, AtomFloatEmbedding, BondEmbedding


def batch_norm_1d(num_channels):
    """tbd"""
    if dist.get_world_size() > 1:
        return nn.SyncBatchNorm.convert_sync_batchnorm(nn.BatchNorm1D(num_channels))
    else:
        return nn.BatchNorm1D(num_channels)

def norm_layer(norm_type, nc):
    """tbd"""
    # normalization layer 1d
    norm = norm_type.lower()
    if norm == 'batch':
        layer = batch_norm_1d(nc)
    elif norm == 'layer':
        layer = nn.LayerNorm(nc)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer


def act_layer(act_type, inplace=False, neg_slope=0.2, n_prelu=1):
    """tbd"""
    # activation layer
    act = act_type.lower()
    if act == 'relu':
        layer = nn.ReLU()
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'swish':
        layer = nn.Swish()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def Linear(input_size, hidden_size, with_bias=True):
    """tbd"""
    fan_in = input_size
    bias_bound = 1.0 / math.sqrt(fan_in)
    fc_bias_attr = paddle.ParamAttr(initializer=nn.initializer.Uniform(
	low=-bias_bound, high=bias_bound))

    negative_slope = math.sqrt(5)
    gain = math.sqrt(2.0 / (1 + negative_slope ** 2))
    std = gain / math.sqrt(fan_in)
    weight_bound = math.sqrt(3.0) * std
    fc_w_attr = paddle.ParamAttr(initializer=nn.initializer.Uniform(
	low=-weight_bound, high=weight_bound))

    if not with_bias:
        fc_bias_attr = False

    return nn.Linear(
        input_size, hidden_size, weight_attr=fc_w_attr, bias_attr=fc_bias_attr)


class MLP(paddle.nn.Sequential):
    """tbd"""
    def __init__(self, channels, act='swish', norm=None, bias=True, drop=0., last_lin=False):
        m = []

        for i in range(1, len(channels)):

            m.append(Linear(channels[i - 1], channels[i], bias))

            if (i == len(channels) - 1) and last_lin:
                pass
            else:
                if norm is not None and norm.lower() != 'none':
                    m.append(norm_layer(norm, channels[i]))
                if act is not None and act.lower() != 'none':
                    m.append(act_layer(act))
                if drop > 0:
                    m.append(nn.Dropout(drop))

        self.m = m
        super(MLP, self).__init__(*self.m)


class LiteGEMConv(paddle.nn.Layer):
    """tbd"""
    def __init__(self, config, with_efeat=True):
        super(LiteGEMConv, self).__init__()
        log.info("layer_type is %s" % self.__class__.__name__)
        self.config = config
        self.with_efeat = with_efeat

        self.aggr = self.config["aggr"]
        self.learn_t = self.config["learn_t"]
        self.learn_p = self.config["learn_p"]
        self.init_t = self.config["init_t"]
        self.init_p = self.config["init_p"]

        self.eps = 1e-7

        self.emb_dim = self.config["emb_dim"]

        if self.with_efeat:
            self.bond_encoder = BondEmbedding(self.config["bond_names"], self.emb_dim)

        self.concat = config["concat"]
        if self.concat:
            self.fc_concat = Linear(self.emb_dim * 3, self.emb_dim)

        assert self.aggr in ['softmax_sg', 'softmax', 'power']

        channels_list = [self.emb_dim]
        for i in range(1, self.config["mlp_layers"]):
            channels_list.append(self.emb_dim * 2)
        channels_list.append(self.emb_dim)

        self.mlp = MLP(channels_list,
                       norm=self.config["norm"],
                       last_lin=True)

        if self.learn_t and self.aggr == "softmax":
            self.t = self.create_parameter(
                shape=[1, ],
                dtype='float32',
                default_initializer=nn.initializer.Constant(value=self.init_t))
        else:
            self.t = self.init_t

        if self.learn_p:
            self.p = self.create_parameter(
                shape=[1, ],
                dtype='float32',
                default_initializer=nn.initializer.Constant(value=self.init_p))

    def send_func(self, src_feat, dst_feat, edge_feat):
        """tbd"""
        if self.with_efeat:
            if self.concat:
                h = paddle.concat([dst_feat['h'], src_feat['h'], edge_feat['e']], axis=1)
                h = self.fc_concat(h)
            else:
                h = src_feat["h"] + edge_feat["e"]
        else:
            h = src_feat["h"]
        msg = {"h": F.swish(h) + self.eps}
        return msg

    def recv_func(self, msg):
        """tbd"""
        if self.aggr == "softmax":
            alpha = msg.reduce_softmax(msg["h"] * self.t)
            out = msg['h'] * alpha
            out = msg.reduce_sum(out)
            return out
        elif self.aggr == "power":
            raise NotImplementedError

    def forward(self, graph, nfeat, efeat=None):
        """tbd"""
        if efeat is not None:
            if self.with_efeat:
                efeat = self.bond_encoder(efeat)

            msg = graph.send(src_feat={"h": nfeat},
                             dst_feat={"h": nfeat},
                             edge_feat={"e": efeat},
                             message_func=self.send_func)
        else:
            msg = graph.send(src_feat={"h": nfeat},
                             dst_feat={"h": nfeat},
                             message_func=self.send_func)

        out = graph.recv(msg=msg, reduce_func=self.recv_func)

        out = nfeat + out
        out = self.mlp(out)

        return out


class LiteGEM(paddle.nn.Layer):
    """tbd"""
    def __init__(self, config, with_efeat=True):
        super(LiteGEM, self).__init__()
        log.info("gnn_type is %s" % self.__class__.__name__)

        self.config = config
        self.with_efeat = with_efeat
        self.num_layers = config["num_layers"]
        self.drop_ratio = config["dropout_rate"]
        self.virtual_node = config["virtual_node"]
        self.emb_dim = config["emb_dim"]
        self.norm = config["norm"]
        self.num_tasks = config["num_tasks"]

        self.atom_names = config["atom_names"]
        self.atom_float_names = config["atom_float_names"]
        self.bond_names = config["bond_names"]
        self.gnns = paddle.nn.LayerList()
        self.norms = paddle.nn.LayerList()

        if self.virtual_node:
            log.info("using virtual node in %s" % self.__class__.__name__)
            self.mlp_virtualnode_list = paddle.nn.LayerList()

            self.virtualnode_embedding = self.create_parameter(
                shape=[1, self.emb_dim],
                dtype='float32',
                default_initializer=nn.initializer.Constant(value=0.0))

            for layer in range(self.num_layers - 1):
                self.mlp_virtualnode_list.append(MLP([self.emb_dim] * 3,
                                                       norm=self.norm))

        for layer in range(self.num_layers):
            self.gnns.append(LiteGEMConv(config, with_efeat=not self.with_efeat))
            self.norms.append(norm_layer(self.norm, self.emb_dim))
        
        self.atom_embedding = AtomEmbedding(self.atom_names, self.emb_dim)
        self.atom_float_embedding = AtomFloatEmbedding(self.atom_float_names, self.emb_dim)

        if self.with_efeat:
            self.init_bond_embedding = BondEmbedding(self.config["bond_names"], self.emb_dim)

        self.pool = gnn.GraphPool(pool_type="sum")

        if not self.config["graphnorm"]:
            self.gn = gnn.GraphNorm()
        
        hidden_size = self.emb_dim

        if self.config["clf_layers"] == 3:
            log.info("clf_layers is 3")
            self.graph_pred_linear = nn.Sequential(
                    Linear(hidden_size, hidden_size // 2),
                    batch_norm_1d(hidden_size // 2),
                    nn.Swish(),
                    Linear(hidden_size // 2, hidden_size // 4),
                    batch_norm_1d(hidden_size // 4),
                    nn.Swish(),
                    Linear(hidden_size // 4, self.num_tasks)
                    )
        elif self.config["clf_layers"] == 2:
            log.info("clf_layers is 2")
            self.graph_pred_linear = nn.Sequential(
                Linear(hidden_size, hidden_size // 2),
                batch_norm_1d(hidden_size // 2),
                nn.Swish(),
                Linear(hidden_size // 2, self.num_tasks)
            )
        else:
            self.graph_pred_linear = Linear(hidden_size, self.num_tasks)

    def forward(self, g):
        """tbd"""
        h = self.atom_embedding(g.node_feat)
        h += self.atom_float_embedding(g.node_feat)

        if self.virtual_node:
            virtualnode_embedding = self.virtualnode_embedding.expand(
                    [g.num_graph, self.virtualnode_embedding.shape[-1]])
            h = h + paddle.gather(virtualnode_embedding, g.graph_node_id)
            #  print("virt0: ", np.sum(h.numpy()))

        if self.with_efeat:
            edge_emb = self.init_bond_embedding(g.edge_feat)
        else:
            edge_emb = g.edge_feat

        h = self.gnns[0](g, h, edge_emb)
        if self.config["graphnorm"]:
            h = self.gn(g, h)

        #  print("h0: ", np.sum(h.numpy()))
        for layer in range(1, self.num_layers):
            h1 = self.norms[layer - 1](h)
            h2 = F.swish(h1)
            h2 = F.dropout(h2, p=self.drop_ratio, training=self.training)

            if self.virtual_node:
                virtualnode_embedding_temp = self.pool(g, h2) + virtualnode_embedding
                virtualnode_embedding = self.mlp_virtualnode_list[layer - 1](virtualnode_embedding_temp)
                virtualnode_embedding  = F.dropout(
                        virtualnode_embedding,
                        self.drop_ratio,
                        training=self.training)

                h2 = h2 + paddle.gather(virtualnode_embedding, g.graph_node_id)
                #  print("virt_h%s: " % (layer), np.sum(h2.numpy()))

            h = self.gnns[layer](g, h2, edge_emb) + h
            if self.config["graphnorm"]:
                h = self.gn(g, h)
            #  print("h%s: " % (layer), np.sum(h.numpy()))

        h = self.norms[self.num_layers - 1](h)
        h = F.dropout(h, p=self.drop_ratio, training=self.training)

        h_graph = self.pool(g, h)
        # return graph, node, edge representation
        return h_graph, h, edge_emb