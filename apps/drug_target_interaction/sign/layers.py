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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
from pgl.utils.helper import generate_segment_id_from_index
from pgl.utils import op
import pgl.math as math
from utils import generate_segment_id


class DenseLayer(nn.Layer):
    def __init__(self, in_dim, out_dim, activation=F.relu, bias=True):
        super(DenseLayer, self).__init__()
        self.activation = activation
        if not bias:
            self.fc = nn.Linear(in_dim, out_dim, bias_attr=False)
        else:
            self.fc = nn.Linear(in_dim, out_dim)
    
    def forward(self, input_feat):
        return self.activation(self.fc(input_feat))


class SpatialInputLayer(nn.Layer):
    """Implementation of Spatial Relation Embedding Module.
    """
    def __init__(self, hidden_dim, cut_dist, activation=F.relu):
        super(SpatialInputLayer, self).__init__()
        self.cut_dist = cut_dist
        self.dist_embedding_layer = nn.Embedding(int(cut_dist)-1, hidden_dim, sparse=True)
        self.dist_input_layer = DenseLayer(hidden_dim, hidden_dim, activation, bias=True)
    
    def forward(self, dist_feat):
        dist = paddle.clip(dist_feat.squeeze(), 1.0, self.cut_dist-1e-6).astype('int64') - 1
        eh_emb = self.dist_embedding_layer(dist)
        eh_emb = self.dist_input_layer(eh_emb)
        # eh_emb = paddle.cast(eh_emb, 'float64')
        return eh_emb


class Atom2BondLayer(nn.Layer):
    """Implementation of Node->Edge Aggregation Layer.
    """
    def __init__(self, atom_dim, bond_dim, activation=F.relu):
        super(Atom2BondLayer, self).__init__()
        in_dim = atom_dim * 2 + bond_dim
        self.fc_agg = DenseLayer(in_dim, bond_dim, activation=activation, bias=True)

    def agg_func(self, src_feat, dst_feat, edge_feat):
        h_src = src_feat['h']
        h_dst = dst_feat['h']
        h_agg = paddle.concat([h_src, h_dst, edge_feat['h']], axis=-1)
        return {'h': h_agg}

    def forward(self, g, atom_feat, edge_feat):
        msg = g.send(self.agg_func, src_feat={'h': atom_feat}, dst_feat={'h': atom_feat}, edge_feat={'h': edge_feat})
        bond_feat = msg['h']
        bond_feat = self.fc_agg(bond_feat)
        return bond_feat


class Bond2AtomLayer(nn.Layer):
    """Implementation of Distance-aware Edge->Node Aggregation Layer.
    """
    def __init__(self, bond_dim, atom_dim, hidden_dim, num_heads, dropout, merge='mean', activation=F.relu):
        super(Bond2AtomLayer, self).__init__()
        self.merge = merge
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.src_fc = nn.Linear(bond_dim, num_heads * hidden_dim)
        self.dst_fc = nn.Linear(atom_dim, num_heads * hidden_dim)
        self.edg_fc = nn.Linear(hidden_dim, num_heads * hidden_dim)
        self.weight_src = self.create_parameter(shape=[num_heads, hidden_dim])
        self.weight_dst = self.create_parameter(shape=[num_heads, hidden_dim])
        self.weight_edg = self.create_parameter(shape=[num_heads, hidden_dim])
        
        self.feat_drop = nn.Dropout(p=dropout)
        self.attn_drop = nn.Dropout(p=dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.activation = activation

    def attn_send_func(self, src_feat, dst_feat, edge_feat):
        alpha = src_feat["attn"] + dst_feat["attn"] + edge_feat['attn']
        alpha = self.leaky_relu(alpha)
        return {"alpha": alpha, "h": src_feat["h"]}
    
    def attn_recv_func(self, msg):
        alpha = msg.reduce_softmax(msg["alpha"])
        alpha = paddle.reshape(alpha, [-1, self.num_heads, 1])
        alpha = self.attn_drop(alpha)

        feature = msg["h"]
        feature = paddle.reshape(feature, [-1, self.num_heads, self.hidden_dim])
        feature = feature * alpha
        if self.merge == 'cat':
            feature = paddle.reshape(feature, [-1, self.num_heads * self.hidden_dim])
        if self.merge == 'mean':
            feature = paddle.mean(feature, axis=1)

        feature = msg.reduce(feature, pool_type="sum")
        return feature

    def forward(self, g, atom_feat, bond_feat, edge_feat):
        bond_feat = self.feat_drop(bond_feat)
        atom_feat = self.feat_drop(atom_feat)
        edge_feat = self.feat_drop(edge_feat)

        bond_feat = self.src_fc(bond_feat)
        atom_feat = self.dst_fc(atom_feat)
        edge_feat = self.edg_fc(edge_feat)
        bond_feat = paddle.reshape(bond_feat, [-1, self.num_heads, self.hidden_dim])
        atom_feat = paddle.reshape(atom_feat, [-1, self.num_heads, self.hidden_dim])
        edge_feat = paddle.reshape(edge_feat, [-1, self.num_heads, self.hidden_dim])

        attn_src = paddle.sum(bond_feat * self.weight_src, axis=-1)
        attn_dst = paddle.sum(atom_feat * self.weight_dst, axis=-1)
        attn_edg = paddle.sum(edge_feat * self.weight_edg, axis=-1)

        msg = g.send(self.attn_send_func,
                     src_feat={"attn": attn_src, "h": bond_feat},
                     dst_feat={"attn": attn_dst},
                     edge_feat={'attn': attn_edg})
        rst = g.recv(reduce_func=self.attn_recv_func, msg=msg)

        if self.activation:
            rst = self.activation(rst)
        return rst


class DomainAttentionLayer(nn.Layer):
    """Implementation of Angle Domain-speicific Attention Layer.
    """
    def __init__(self, bond_dim, hidden_dim, dropout, activation=F.relu):
        super(DomainAttentionLayer, self).__init__()
        self.attn_fc = nn.Linear(2 * bond_dim, hidden_dim)
        self.attn_out = nn.Linear(hidden_dim, 1, bias_attr=False)

        self.feat_drop = nn.Dropout(p=dropout)
        self.attn_drop = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.activation = activation
    
    def attn_send_func(self, src_feat, dst_feat, edge_feat):
        h_c = paddle.concat([src_feat['h'], src_feat['h']], axis=-1)
        h_c = self.attn_fc(h_c)
        h_c = self.tanh(h_c)
        h_s = self.attn_out(h_c)
        return {"alpha": h_s, "h": src_feat["h"]}
    
    def attn_recv_func(self, msg):
        alpha = msg.reduce_softmax(msg["alpha"])
        alpha = self.attn_drop(alpha) # [-1, 1]
        feature = msg["h"] # [-1, hidden_dim]
        feature = feature * alpha
        feature = msg.reduce(feature, pool_type="sum")
        return feature

    def forward(self, g, bond_feat):
        bond_feat = self.feat_drop(bond_feat)
        msg = g.send(self.attn_send_func,
                    src_feat={"h": bond_feat},
                    dst_feat={"h": bond_feat})
        rst = g.recv(reduce_func=self.attn_recv_func, msg=msg)
        if self.activation:
            rst = self.activation(rst)
        return rst

class Bond2BondLayer(nn.Layer):
    """Implementation of Angle-oriented Edge->Edge Aggregation Layer.
    """
    def __init__(self, bond_dim, hidden_dim, num_angle, dropout, merge='cat', activation=None):
        super(Bond2BondLayer, self).__init__()
        self.num_angle = num_angle
        self.hidden_dim = hidden_dim
        self.merge = merge
        self.conv_layer = nn.LayerList()
        for _ in range(num_angle):
            conv = DomainAttentionLayer(bond_dim, hidden_dim, dropout, activation=None)
            self.conv_layer.append(conv)
        self.activation = activation
    
    def forward(self, g_list, bond_feat):
        h_list = []
        for k in range(self.num_angle):
            h = self.conv_layer[k](g_list[k], bond_feat)
            h_list.append(h)

        if self.merge == 'cat':
            feat_h = paddle.concat(h_list, axis=-1)
        if self.merge == 'mean':
            feat_h = paddle.mean(paddle.stack(h_list, axis=-1), axis=1)
        if self.merge == 'sum':
            feat_h = paddle.sum(paddle.stack(h_list, axis=-1), axis=1)
        if self.merge == 'max':
            feat_h = paddle.max(paddle.stack(h_list, axis=-1), axis=1)
        if self.merge == 'cat_max':
            feat_h = paddle.stack(h_list, axis=-1)
            feat_max = paddle.max(feat_h, dim=1)[0]
            feat_max = paddle.reshape(feat_max, [-1, 1, self.hidden_dim])
            feat_h = paddle.reshape(feat_h * feat_max, [-1, self.num_angle * self.hidden_dim])

        if self.activation:
            feat_h = self.activation(feat_h)
        return feat_h
    

class PiPoolLayer(nn.Layer):
    """Implementation of Pairwise Interactive Pooling Layer.
    """
    def __init__(self, bond_dim, hidden_dim, num_angle):
        super(PiPoolLayer, self).__init__()
        self.bond_dim = bond_dim
        self.num_angle = num_angle
        self.num_type = 4 * 9
        fc_in_dim = num_angle * bond_dim
        self.fc_1 = DenseLayer(fc_in_dim, hidden_dim, activation=F.relu, bias=True)
        self.fc_2 = nn.Linear(hidden_dim, 1, bias_attr=False)
        self.softmax = nn.Softmax(axis=1)
    
    def forward(self, bond_types_batch, type_count_batch, bond_feat):
        """
        Input example:
            bond_types_batch: [0,0,2,0,1,2] + [0,0,2,0,1,2] + [2]
            type_count_batch: [[3, 3, 0], [1, 1, 0], [2, 2, 1]] # [num_type, batch_size]
        """
        bond_feat = self.fc_1(paddle.reshape(bond_feat, [-1, self.num_angle*self.bond_dim]))
        inter_mat_list =[]
        for type_i in range(self.num_type):
            type_i_index = paddle.masked_select(paddle.arange(len(bond_feat)), bond_types_batch==type_i)
            if paddle.sum(type_count_batch[type_i]) == 0:
                inter_mat_list.append(paddle.to_tensor(np.array([0.]*len(type_count_batch[type_i])), dtype='float32'))
                continue
            bond_feat_type_i = paddle.gather(bond_feat, type_i_index)
            graph_bond_index = op.get_index_from_counts(type_count_batch[type_i])
            # graph_bond_id = generate_segment_id_from_index(graph_bond_index)
            graph_bond_id = generate_segment_id(graph_bond_index)
            graph_feat_type_i = math.segment_pool(bond_feat_type_i, graph_bond_id, pool_type='sum')
            mat_flat_type_i = self.fc_2(graph_feat_type_i).squeeze(1)

            # print(graph_bond_id)
            # print(graph_bond_id.shape, graph_feat_type_i.shape, mat_flat_type_i.shape)
            my_pad = nn.Pad1D(padding=[0, len(type_count_batch[type_i])-len(mat_flat_type_i)], value=-1e9)
            mat_flat_type_i = my_pad(mat_flat_type_i)
            inter_mat_list.append(mat_flat_type_i)

        inter_mat_batch = paddle.stack(inter_mat_list, axis=1) # [batch_size, num_type]
        inter_mat_mask = paddle.ones_like(inter_mat_batch) * -1e9
        inter_mat_batch = paddle.where(type_count_batch.transpose([1, 0])>0, inter_mat_batch, inter_mat_mask)
        inter_mat_batch = self.softmax(inter_mat_batch)
        return inter_mat_batch


class OutputLayer(nn.Layer):
    """Implementation of Prediction Layer.
    """
    def __init__(self, atom_dim, hidden_dim_list):
        super(OutputLayer, self).__init__()
        self.pool = pgl.nn.GraphPool(pool_type='sum')
        self.mlp = nn.LayerList()
        for hidden_dim in hidden_dim_list:
            self.mlp.append(DenseLayer(atom_dim, hidden_dim, activation=F.relu))
            atom_dim = hidden_dim
        self.output_layer = nn.Linear(atom_dim, 1)
    
    def forward(self, g, atom_feat):
        graph_feat = self.pool(g, atom_feat)
        for layer in self.mlp:
            graph_feat = layer(graph_feat)
        output = self.output_layer(graph_feat)
        return output