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
Basic Encoder for compound atom/bond features.
"""

import paddle
import paddle.nn as nn
import pgl

from pahelix.utils.compound_tools import CompoundKit


class AtomEmbedding(nn.Layer):
    """
    Atom Encoder
    """
    def __init__(self, atom_names, embed_dim):
        super(AtomEmbedding, self).__init__()
        self.atom_names = atom_names
        
        self.embed_list = nn.LayerList()
        for name in self.atom_names:
            embed = nn.Embedding(
                    CompoundKit.get_atom_feature_size(name) + 5,
                    embed_dim, 
                    weight_attr=nn.initializer.XavierUniform())
            self.embed_list.append(embed)

    def forward(self, node_features):
        """
        Args:
            node_features(dict of tensor):
        """
        out_embed = 0
        for i, name in enumerate(self.atom_names):
            out_embed += self.embed_list[i](node_features[name])
        return out_embed


class BondEmbedding(nn.Layer):
    """
    Bond Encoder
    """
    def __init__(self, bond_names, embed_dim):
        super(BondEmbedding, self).__init__()
        self.bond_names = bond_names
        
        self.embed_list = nn.LayerList()
        for name in self.bond_names:
            embed = nn.Embedding(
                    CompoundKit.get_bond_feature_size(name) + 5,
                    embed_dim, 
                    weight_attr=nn.initializer.XavierUniform())
            self.embed_list.append(embed)

    def forward(self, edge_features):
        """
        Args:
            edge_features(dict of tensor):
        """
        out_embed = 0
        for i, name in enumerate(self.bond_names):
            out_embed += self.embed_list[i](edge_features[name])
        return out_embed
