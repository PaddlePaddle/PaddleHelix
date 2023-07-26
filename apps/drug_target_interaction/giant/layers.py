
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
import math as math_


class AngleRBF(nn.Layer):
    def __init__(self, K):
        super(AngleRBF, self).__init__()
        self.K = K
        self.mu = paddle.linspace(0., math_.pi, K).unsqueeze(0)
        self.beta = paddle.full((1, K), math_.pow((2 / K) * math_.pi, -2))
    
    def forward(self, r):
        batch_size = r.size
        local_r = paddle.expand(r, shape=[batch_size, self.K])
        g = paddle.exp(-self.beta.expand([batch_size, self.K]) * (local_r - self.mu.expand([batch_size, self.K]))**2)
        return g

class DenseLayer(nn.Layer):
    def __init__(self, in_dim, out_dim, activation=F.relu, bias=True):
        super(DenseLayer, self).__init__()
        self.activation = activation
        if not bias:
            self.fc = nn.Linear(in_dim, out_dim, bias_attr=False, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Orthogonal()))
        else:
            self.fc = nn.Linear(in_dim, out_dim, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Orthogonal()))
    
    def forward(self, input_feat):
        return self.activation(self.fc(input_feat))


class GemotryInputLayer(nn.Layer):
    """Implementation of Geometric Embedding Module.
    """
    def __init__(self, rbf_dim, hidden_dim, cut_dist, activation=F.relu):
        super(GemotryInputLayer, self).__init__()
        self.cut_dist = cut_dist
        self.dist_embedding_layer = nn.Embedding(int(cut_dist)-1, hidden_dim, sparse=True)
        self.dist_input_layer = DenseLayer(hidden_dim, hidden_dim, activation, bias=True)
        self.angle_rbf = AngleRBF(rbf_dim)
        self.angle_input_layer = DenseLayer(rbf_dim, hidden_dim, activation, bias=True)
    
    def forward(self, dist_feat, angle_feat_list):
        dist = paddle.clip(dist_feat.squeeze(), 1.0, self.cut_dist-1e-6).astype('int64') - 1
        dist_emb = self.dist_embedding_layer(dist)
        dist_emb = self.dist_input_layer(dist_emb)
        assert type(angle_feat_list) == list
        angle_emb_list = []
        for angle_feat in angle_feat_list:
            angle_feat = angle_feat * (np.pi/180)
            angle_emb = self.angle_rbf(angle_feat)
            angle_emb = self.angle_input_layer(angle_emb)
            angle_emb_list.append(angle_emb)
        return dist_emb, angle_emb_list


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

    def forward(self, g, bond_feat, angle_feat):
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
    def __init__(self, bond_dim, hidden_dim, num_angle, dropout, merge='cat', activation=F.relu):
        super(Bond2BondLayer, self).__init__()
        self.num_angle = num_angle
        self.hidden_dim = hidden_dim
        self.merge = merge
        self.conv_layer = nn.LayerList()
        for _ in range(num_angle):
            conv = AngleConv(bond_dim, hidden_dim, dropout, activation=F.relu)
            # conv = DomainAttentionLayer(bond_dim, hidden_dim, dropout, activation=None)
            self.conv_layer.append(conv)
        self.activation = activation
    
    def forward(self, g_list, bond_feat, angle_feat_list):
        h_list = []
        for k in range(self.num_angle):
            h = self.conv_layer[k](g_list[k], bond_feat, angle_feat_list[k])
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

        # if self.activation:
        #     feat_h = self.activation(feat_h)
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


class AngleConv(nn.Layer):
    """Implementation of triple-wise local aggregation for dihedral angle learning.
    """
    def __init__(self, bond_dim, hidden_dim, dropout, activation=F.relu):
        super(AngleConv, self).__init__()
        self.feat_drop = nn.Dropout(dropout)
        self.G = nn.Linear(hidden_dim, hidden_dim, bias_attr=False, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Orthogonal()))
        self.fc_src = DenseLayer(bond_dim, hidden_dim, activation, bias=True)
        self.fc_cat = DenseLayer(2 * bond_dim, hidden_dim, activation, bias=False)
        self.fc_dst = DenseLayer(bond_dim, hidden_dim, activation, bias=True)
        self.fc_bond_update = DenseLayer(2 * hidden_dim, hidden_dim, activation, bias=True)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
    
    def send_func(self, src_feat, dst_feat, edge_feat):
        combined_h = paddle.concat([src_feat["h"], edge_feat["pairs"]], axis=-1)
        combined_h = self.fc_cat(combined_h)
        combined_h, _ = self.gru(combined_h, edge_feat["angle"])
        return {"h": combined_h}

    def recv_func(self, msg):
        return msg.reduce(msg["h"], pool_type="sum")

    def forward(self, g, bond_feat, angle_feat):
        bond_feat = self.feat_drop(bond_feat)
        angle_feat = self.feat_drop(angle_feat)

        angle_h = self.G(angle_feat)
        bond_h_src = self.fc_src(bond_feat)
        pair_inds = g.edge_feat["pairs"]
        bond_h_pair = bond_h_src[pair_inds]

        msg = g.send(self.send_func,
                    src_feat={"h": bond_h_src},
                    edge_feat={"angle": angle_h, "pairs": bond_h_pair})
        rst = g.recv(reduce_func=self.recv_func, msg=msg)

        dst_h = self.fc_dst(bond_feat)
        m = paddle.concat([dst_h, rst], axis=-1)
        bond_h = self.fc_bond_update(m)
        return bond_h

class InteractiveOutputLayer(nn.Layer):
    """Implementation of Molecule-level Pairwise Interaction and Prediction Layer.
    """
    def __init__(self, hidden_dim, mlp_dims, num_cross_layer, num_inter_layer, dropout=0.2):
        super(InteractiveOutputLayer, self).__init__()
        self.bond_layer = Atom2BondLayer(hidden_dim, hidden_dim, F.relu)
        self.mol_pool = MolPoolLayer(hidden_dim, num_cross_layer, num_inter_layer, dropout)

        self.mlp = nn.LayerList()
        in_dim = hidden_dim
        for dim in mlp_dims:
            self.mlp.append(DenseLayer(in_dim, dim, activation=F.relu))
            in_dim = dim
        self.output_layer = nn.Linear(in_dim, 1)

    def forward(self, a2a_g, prot_g, liga_g, inter_g, atom_feat, dist_feat):
        bond_feat = self.bond_layer(a2a_g, atom_feat, dist_feat)
        graph_feat = self.mol_pool(prot_g, liga_g, inter_g, atom_feat, bond_feat)
        for layer in self.mlp:
            graph_feat = layer(graph_feat)
        output = self.output_layer(graph_feat)
        return output

def graph_edge_pool(g, feature, pool_type='sum'):
    graph_feat = math.segment_pool(feature, g.graph_edge_id, pool_type)
    return graph_feat

class MolPoolLayer(nn.Layer):
    def __init__(self, hidden_dim, num_cross_layer, num_inter_layer, dropout=0.2, alpha=0.7):
        super(MolPoolLayer, self).__init__()
        self.num_cross_layer = num_cross_layer
        self.num_inter_layer = num_inter_layer
        self.alpha = alpha

        self.readouts_prot = nn.LayerList()
        self.readouts_liga = nn.LayerList()
        for _ in range(num_cross_layer):
            self.readouts_prot.append(GlobalPool(hidden_dim, dropout))
            self.readouts_liga.append(GlobalPool(hidden_dim, dropout))

        self.readouts_bond = nn.LayerList()
        for _ in range(num_inter_layer):
            self.readouts_bond.append(GlobalPool(hidden_dim, dropout))

    def forward(self, prot_g, liga_g, inter_g, atom_feat, bond_feat):
        atom_feat_prot = atom_feat[prot_g.edges[:, 0]]
        atom_feat_liga = atom_feat[liga_g.edges[:, 0]]
        bond_feat_inter = bond_feat[inter_g.edges[:, 0]]
        g_feat_prot = graph_edge_pool(prot_g, atom_feat_prot)
        g_feat_liga = graph_edge_pool(liga_g, atom_feat_liga)

        for i in range(self.num_cross_layer):
            g_feat_liga_ = self.readouts_liga[i](liga_g, atom_feat_liga, g_feat_prot, g_feat_self=g_feat_liga)
            g_feat_prot_ = self.readouts_prot[i](prot_g, atom_feat_prot, g_feat_liga, g_feat_self=g_feat_prot)
            g_feat_liga = g_feat_liga_
            g_feat_prot = g_feat_prot_
        g_feat = self.alpha * g_feat_liga + (1 - self.alpha) * g_feat_prot

        for i in range(self.num_inter_layer):
            g_feat = self.readouts_bond[i](inter_g, bond_feat_inter, g_feat, g_feat_self=g_feat)
        return g_feat

class GlobalPool(nn.Layer):
    def __init__(self, hidden_dim, dropout):
        super(GlobalPool, self).__init__()
        self.compute_logits = nn.Sequential(
            nn.Linear(2 * hidden_dim, 1),
            nn.LeakyReLU()
        )
        self.project_nodes = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.pool = pgl.nn.GraphPool(pool_type='sum')
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
    
    def broadcast_graph_feat(self, g, g_feat):
        eids = g._graph_edge_index
        eids_ = paddle.concat([eids[1:], eids[-1:]])
        batch_num_edges = (eids_-eids)[:-1]
        h_list = []
        for i, k in enumerate(batch_num_edges):
            h_list += [g_feat[i].tile([k,1])]
        return paddle.concat(h_list)
    
    def forward(self, g, node_feat, g_feat, g_feat_self):
        g_feat_broad = self.broadcast_graph_feat(g, F.relu(g_feat))
        g_node_feat = paddle.concat([g_feat_broad, node_feat], axis=1)
        logits = self.compute_logits(g_node_feat)
        node_a = pgl.math.segment_softmax(logits, g.graph_edge_id)
        node_h = self.project_nodes(node_feat)
        # context = F.elu(self.pool(g, node_h * node_a))
        context = F.elu(graph_edge_pool(g, node_h * node_a))
        graph_h, _ = self.gru(context, g_feat_self)
        return graph_h