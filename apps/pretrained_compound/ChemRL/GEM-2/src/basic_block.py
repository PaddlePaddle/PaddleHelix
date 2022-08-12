#!/usr/bin/python                                                                                                
#-*-coding:utf-8-*- 
#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
basic paddle blocks
"""

import os
import paddle
import paddle.nn as nn


class MLP(nn.Layer):
    """
    hidden_size: int or list or tuple
    """
    def __init__(self, in_size, hidden_size, out_size):
        super(MLP, self).__init__()

        if isinstance(hidden_size, list) or isinstance(hidden_size, tuple):
            hidden_size_list = hidden_size
        else:
            hidden_size_list = [hidden_size]

        hidden_size_list = [in_size] + hidden_size_list
        layers = []
        for in_d, out_d in zip(hidden_size_list[:-1], hidden_size_list[1:]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.LayerNorm(out_d))
            layers.append(nn.GELU())
        layers.append(nn.Linear(out_d, out_size))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class RBF(nn.Layer):
    """
    Radial Basis Function
    """
    def __init__(self, centers, gamma, dtype='float32'):
        super(RBF, self).__init__()
        self.centers = paddle.to_tensor(centers, dtype=dtype)
        self.gamma = gamma
    
    def forward(self, x):
        """
        Args:
            x(tensor): (*).
        Returns:
            y(tensor): (*, n_centers)
        """
        x = paddle.unsqueeze(x, [-1])   # (*, 1)
        return paddle.exp(-self.gamma * paddle.square(x - self.centers))    # (*, n_center)
        
    
class LnDropWrapper(nn.Layer):
    """
    layer norm and dropout wrapper
    """
    def __init__(self, embed_dim, layer, dropout_rate):
        super(LnDropWrapper, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.layer = layer
        self.dropout_module = nn.Dropout(dropout_rate) 

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        outputs = self.layer(x, *args, **kwargs)
        if isinstance(outputs, tuple):
            x, *out = outputs
        else:
            x = outputs
            out = None
        x = self.dropout_module(x)

        if out is not None:
            return (x,) + tuple(out)
        else:
            return x


class IntEmbedding(nn.Layer):
    """
    Atom Encoder
    """
    def __init__(self, names, embed_dim, embed_params):
        super(IntEmbedding, self).__init__()
        self.names = names
        
        self.embed_dict = nn.LayerDict()
        for name in self.names:
            embed = nn.Embedding(
                    embed_params[name]['vocab_size'],
                    embed_dim, 
                    weight_attr=nn.initializer.XavierUniform())
            self.embed_dict[name] = embed

    def forward(self, input):
        """
        Args: 
            input(dict of tensor): node features.
        """
        out_embed = 0
        for name in self.names:
            out_embed += self.embed_dict[name](input[name])
        return out_embed


class RBFEmbedding(nn.Layer):
    """
    Atom Float Encoder
    """
    def __init__(self, names, embed_dim, rbf_params=None):
        super(RBFEmbedding, self).__init__()
        self.names = names
        
        self.rbf_dict = nn.LayerDict()
        self.linear_dict = nn.LayerDict()
        for name in self.names:
            centers = rbf_params[name]['centers']
            gamma = rbf_params[name]['gamma']
            self.rbf_dict[name] = RBF(centers, gamma)
            self.linear_dict[name] = nn.Linear(len(centers), embed_dim)

    def forward(self, feats):
        """
        Args: 
            feats(dict of tensor): node float features.
        """
        out_embed = 0
        for name in self.names:
            x = feats[name]
            rbf_x = self.rbf_dict[name](x)
            out_embed += self.linear_dict[name](rbf_x)
        return out_embed


def node_pooling(node_acts, node_mask, pool_type):
    """
    node_acts: (B, N, d)
    node_mask: (B, N), 0 for invalid

    res: (B, d)
    """
    masked_acts = node_acts * node_mask.unsqueeze([-1])     # (B, N, d)
    res = paddle.sum(masked_acts, axis=-2)  # (B, d)
    if pool_type.lower() == 'sum':
        return res
    elif pool_type.lower() == 'mean':
        node_num = paddle.sum(node_mask, axis=-1, keepdim=True) # (B, 1)
        return res / node_num   # (B, d)
    else:
        raise ValueError(pool_type)


def get_vec_norm(vec, keepdim=False):
    """
    vec: (*, d) -> (*, 1) or (*)
    """
    return paddle.sqrt(paddle.sum(paddle.square(vec), -1, keepdim=keepdim))


def get_angle(vec1, vec2, keepdim=False):
    """
    vec1: (*, d)
    vec2: (*, d)
    return:
        angle: (*, 1) or (*)
    """
    norm1 = get_vec_norm(vec1, keepdim=True)
    norm2 = get_vec_norm(vec2, keepdim=True)
    vec1 = vec1 / (norm1 + 1e-5)    # 1e-5: prevent numerical errors
    vec2 = vec2 / (norm2 + 1e-5)
    angle = paddle.acos(paddle.sum(vec1 * vec2, -1, keepdim=keepdim))
    return angle


def atom_pos_to_pair_dist(atom_pos):
    """
    atom_pos: (b, n, 3)
    return:
        dist: (b, n, n)
    """
    left = atom_pos.unsqueeze([2])  # (b, n, 1, 3)
    right = atom_pos.unsqueeze([1]) # (b, 1, n, 3)
    dist = get_vec_norm(left - right)  # (b, n, n)
    return dist


def atom_pos_to_triple_angle(atom_pos):
    """
    atom_pos: (b, n, 3)
    angle_i + angle_j == angle_k
    return:
        angle: (b, n, n, n)
    """
    i = atom_pos.unsqueeze([2, 3])  # (b, n, 1, 1, 3)
    j = atom_pos.unsqueeze([1, 3])  # (b, 1, n, 1, 3)
    k = atom_pos.unsqueeze([1, 2])  # (b, 1, 1, n, 3)
    vec_ij = i - j  # (b, n, n, 1, 3)
    vec_ik = i - k  # (b, n, 1, n, 3)
    vec_kj = k - j  # (b, 1, n, n, 3)
    angle_i = get_angle(vec_ij, vec_ik)   # (b, n, n, n)
    angle_k = get_angle(vec_ik, vec_kj)   # (b, n, n, n)
    angle_j = get_angle(vec_ij, vec_kj)   # (b, n, n, n)
    angle_dict = {
        'angle_i': angle_i,
        'angle_k': angle_k,
        'angle_j': angle_j,
    }
    return angle_dict
