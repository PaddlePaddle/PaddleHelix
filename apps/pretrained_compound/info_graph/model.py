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

import paddle.fluid as F
import paddle.fluid.layers as L
from pahelix.networks.gnn_block import gin_layer
import pgl

class GINEncoder(object):
    """GIN Encoder for unsupervised InfoGraph"""
    def __init__(self, config):
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.embed_dim = config['gw_emb_dim']
        self.atom_type_num = config['atom_type_num']
        self.chirality_tag_num = config['chirality_tag_num']
        self.bond_type_num = config['bond_type_num']
        self.bond_direction_num = config['bond_direction_num']

    def _atom_encoder(self, gw):
        embed_init = F.initializer.XavierInitializer(uniform=True)

        atom_type_embed = L.embedding(
                input=gw.node_feat['atom_type'],
                size=[self.atom_type_num, self.embed_dim],
                param_attr=F.ParamAttr(
                    name='embed_atom_type', initializer=embed_init))
        chirality_tag_embed = L.embedding(
                input=gw.node_feat['chirality_tag'],
                size=[self.chirality_tag_num, self.embed_dim],
                param_attr=F.ParamAttr(
                    name='embed_chirality_tag', initializer=embed_init))
        node_features = atom_type_embed + chirality_tag_embed
        return node_features

    def _bond_encoder(self, gw, name=""):
        embed_init = F.initializer.XavierInitializer(uniform=True)

        bond_type_embed = L.embedding(
                input=gw.edge_feat['bond_type'],
                size=[self.bond_type_num, self.embed_dim],
                param_attr=F.ParamAttr(
                    name="%s_embed_bond_type" % name, initializer=embed_init))
        bond_direction_embed = L.embedding(
                input=gw.edge_feat['bond_direction'],
                size=[self.bond_direction_num, self.embed_dim],
                param_attr=F.ParamAttr(
                    name="%s_embed_bond_direction" % name, initializer=embed_init))
        bond_features = bond_type_embed + bond_direction_embed
        return bond_features

    def forward(self, gw):
        x = self._atom_encoder(gw)
        patch_repr = []
        for i in range(self.num_layers):
            e = self._bond_encoder(gw, name='l%d'%i)
            x = gin_layer(gw, x, e, 'gin_%s' % i)
            x = L.batch_norm(
                x, param_attr=F.ParamAttr(name='batchnorm_%s' % i))
            patch_repr.append(x)  # $h_i^{(k)}$

        patch_summary = L.concat(patch_repr, axis=1)  # $h_{\phi}^i$
        patch_pool = [pgl.layers.graph_pooling(gw, x, 'sum')
                      for x in patch_repr]
        global_repr = L.concat(patch_pool, axis=1)
        return global_repr, patch_summary

    def get_embeddings(self, loader, exe, prog, graph_emb):
        emb_lst, ys = [], []
        for feed_dict in loader:
            y = feed_dict['label'].copy()
            del feed_dict['label']
            emb = exe.run(
                prog, feed=feed_dict, fetch_list=[graph_emb])[0]
            emb_lst.append(emb)
            ys.append(y)

        emb = np.concatenate(emb_lst, 0)
        y = np.concatenate(ys, 0)
        return emb, y

    @property
    def embedding_dim(self):
        return self.num_layers * self.hidden_size


class FF(object):
    """Feedforward network with linear shortcut for InfoGraph"""
    def __init__(self, hidden_size, num_layers=3):
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)

    def block(self, x):
        for i in range(self.num_layers):
            x = L.fc(x, self.hidden_size, act='relu')
        return x

    def linear_shortcut(self, x):
        return L.fc(x, self.hidden_size)


class PriorDiscriminator(object):
    """Prior discriminator for InfoGraph"""
    def __init__(self, hidden_size, num_layers=3):
        assert num_layers > 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        # Support multiple calls and share parameters
        for i in range(self.num_layers - 1):
            x = L.fc(
                x, self.hidden_size,
                act='relu',
                param_attr=F.ParamAttr(name='prior_fc_%s.w' % i),
                bias_attr=F.ParamAttr(
                    name='prior_fc_%s.b' % i,
                    initializer=F.initializer.Constant(value=0.0)))

        i = self.num_layers - 1
        x = L.fc(
            x, 1,
            act=None,
            param_attr=F.ParamAttr(name='prior_fc_%s.w' % i),
            bias_attr=F.ParamAttr(
                name='prior_fc_%s.b' % i,
                initializer=F.initializer.Constant(value=0.0)))
        return L.sigmoid(x)
