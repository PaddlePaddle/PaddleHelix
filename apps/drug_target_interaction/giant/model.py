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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
from layers import GemotryInputLayer, Atom2BondLayer, Bond2BondLayer, Bond2AtomLayer, PiPoolLayer, InteractiveOutputLayer, OutputLayer

class GIANT(nn.Layer):
    def __init__(self, args):
        super(GIANT, self).__init__()
        num_convs = args.num_convs
        mlp_dims = args.mlp_dims
        infeat_dim = args.infeat_dim
        hidden_dim = args.hidden_dim
        rbf_dim = args.rbf_dim
        self.num_convs = num_convs

        cut_dist = args.cut_dist
        num_angle = args.num_angle
        merge_b2b = args.merge_b2b
        merge_b2a = args.merge_b2a

        activation = args.activation
        num_heads = args.num_heads
        feat_drop = args.feat_drop

        self.input_layer = GemotryInputLayer(rbf_dim, hidden_dim, cut_dist, activation=F.relu)
        self.atom2bond_layers = nn.LayerList()
        self.bond2bond_layers = nn.LayerList()
        self.bond2atom_layers = nn.LayerList()
        for i in range(num_convs):
            if i == 0:
                atom_dim = infeat_dim
            else:
                atom_dim = hidden_dim * num_heads if 'cat' in merge_b2a else hidden_dim
            bond_dim = hidden_dim * num_angle if 'cat' in merge_b2b else hidden_dim

            self.atom2bond_layers.append(Atom2BondLayer(atom_dim, bond_dim=hidden_dim, activation=activation))
            self.bond2bond_layers.append(Bond2BondLayer(hidden_dim, hidden_dim, num_angle, feat_drop, merge=merge_b2b, activation=activation))
            self.bond2atom_layers.append(Bond2AtomLayer(bond_dim, atom_dim, hidden_dim, num_heads, feat_drop, merge=merge_b2a, activation=activation))

        self.pipool_layer = PiPoolLayer(hidden_dim, hidden_dim, num_angle)
        self.output_layer = InteractiveOutputLayer(hidden_dim, mlp_dims, args.num_cross_layer, args.num_inter_layer, dropout=0.2)
        # self.output_layer = OutputLayer(hidden_dim, mlp_dims)
     
    def forward(self, a2a_g, b2a_g, b2b_gl, mark_gl, bond_types, type_count):
        atom_feat = a2a_g.node_feat['feat']
        dist_feat = a2a_g.edge_feat['dist']
        atom_feat = paddle.cast(atom_feat, 'float32')
        dist_feat = paddle.cast(dist_feat, 'float32')

        atom_h = atom_feat
        angle_feat = [g.edge_feat['angle'] for g in b2b_gl]
        dist_h, angle_hl = self.input_layer(dist_feat, angle_feat)

        for i in range(self.num_convs):
            bond_h = self.atom2bond_layers[i](a2a_g, atom_h, dist_h)
            bond_h = self.bond2bond_layers[i](b2b_gl, bond_h, angle_hl)
            atom_h = self.bond2atom_layers[i](b2a_g, atom_h, bond_h, dist_h)

        inter_g, liga_g, prot_g = mark_gl
        pred_inter_mat = self.pipool_layer(bond_types, type_count, bond_h)
        pred_score = self.output_layer(a2a_g, prot_g, liga_g, inter_g, atom_h, dist_h)
        return pred_inter_mat, pred_score