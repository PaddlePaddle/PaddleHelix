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
Optimus
"""

import os
import math
import numpy as np
import paddle
import paddle.nn as nn
from paddle.distributed.fleet.utils import recompute

from pahelix.networks.compound_encoder import AtomEmbedding
from pahelix.utils.compound_tools import CompoundKit

from .basic_block import IntEmbedding, RBFEmbedding, atom_pos_to_pair_dist, atom_pos_to_triple_angle
from .paddle_utils import recompute_wrapper


class EmbeddingLayer(nn.Layer):
    """
    EmbeddingLayer
    """
    def __init__(self, model_config, global_config):
        super(EmbeddingLayer, self).__init__()
        self.model_config = model_config

        node_channel = global_config.node_channel
        pair_channel = global_config.pair_channel
        triple_channel = global_config.triple_channel
        
        embed_params = self._get_embed_params()
        rbf_params = self._get_rbf_params()
        self.atom_embedding = IntEmbedding(
                self.model_config.atom_names, node_channel, embed_params)
        self.bond_embedding = IntEmbedding(
                self.model_config.bond_names, pair_channel, embed_params)
        self.bond_float_rbf = RBFEmbedding(
                self.model_config.bond_float_names, pair_channel, rbf_params)
        self.triple_embedding = IntEmbedding(
                self.model_config.triple_names, triple_channel, embed_params)
        self.triple_float_rbf = RBFEmbedding(
                self.model_config.triple_float_names, triple_channel, rbf_params)
    
    def _get_embed_params(self):
        embed_params = {}
        # atom
        for name in self.model_config.atom_names:
            embed_params[name] = {
                'vocab_size': CompoundKit.get_atom_feature_size(name) + 5}
        # bond
        for name in self.model_config.bond_names:
            embed_params[name] = {
                'vocab_size': CompoundKit.get_bond_feature_size(name) + 5}
        # triple
        for name in ["hop_num_ij", "hop_num_ik", "hop_num_jk"]:
            embed_params[name] = {'vocab_size': 100}
        return embed_params

    def _get_rbf_params(self):
        rbf_params = {}
        for name in self.model_config.rbf_params:
            start, end, stride, gamma = self.model_config.rbf_params[name]
            rbf_params[name] = {
                'centers': np.arange(start, end, stride),
                'gamma': gamma,
            }
        return rbf_params

    def _create_bond_float_input(self, batch):
        if len(self.model_config.bond_float_names) == 0:
            return {}
        res = {
            'bond_length': atom_pos_to_pair_dist(batch['atom_pos'])}
        return res

    def _create_triple_input(self, batch):
        hop_num = batch['hop_num']
        return {
            'hop_num_ij': hop_num.unsqueeze(-1),
            'hop_num_ik': hop_num.unsqueeze(-2),
            'hop_num_jk': hop_num.unsqueeze(-3),
        }
    
    def _create_triple_float_input(self, batch):
        if len(self.model_config.triple_float_names) == 0:
            return {}
        res = {}
        res.update(atom_pos_to_triple_angle(batch['atom_pos']))
        return res
    
    def forward(self, batch):
        """
        node_feat: {feat1:(B, N), feat2:(B, N), ...} number of feat is the number of atom names
        edge_feat: {feat1: (B, N, N), feat2:(B, N, N)} number of feat is the number of bond names
        """
        node_acts = self.atom_embedding(batch)  # (B, N, D)

        pair_acts = self.bond_embedding(batch)  # (B, N, N, D)
        pair_acts += self.bond_float_rbf(self._create_bond_float_input(batch))

        triple_acts = self.triple_embedding(self._create_triple_input(batch))   # (B, N, N, D)
        triple_acts += self.triple_float_rbf(self._create_triple_float_input(batch))
        
        results = {
            'node_acts': node_acts,
            'pair_acts': pair_acts,
            'triple_acts': triple_acts,
        }
        return results
        

class FirstBodyAxialAttention(nn.Layer):
    """Compute self-attention over columns of a 2D input."""
    def __init__(self, model_config, global_config):
        super(FirstBodyAxialAttention, self).__init__()
        self.model_config = model_config

        node_channel = global_config.node_channel
        pair_channel = global_config.pair_channel

        self.use_pair_layer_norm = model_config.use_pair_layer_norm
        self.virtual_node = model_config.virtual_node
        self.num_head = model_config.num_head
        self.head_dim = node_channel // self.num_head

        self.node_ln = nn.LayerNorm(node_channel)
        if self.use_pair_layer_norm:
            self.pair_ln = nn.LayerNorm(pair_channel)

        self.q_proj = nn.Linear(node_channel, node_channel)
        self.k_proj = nn.Linear(node_channel, node_channel)
        self.v_proj = nn.Linear(node_channel, node_channel)
 
        self.k_e_proj = nn.Linear(pair_channel, node_channel)
        self.v_e_proj = nn.Linear(pair_channel, node_channel)
        self.add_mlp = nn.Linear(self.head_dim, self.head_dim)
        
        if self.virtual_node:
            self.virtual_node_mlp = nn.Linear(self.head_dim, self.head_dim)

        self.dropout = nn.Dropout(model_config.dropout_rate)
        self.out_proj = nn.Linear(node_channel, node_channel)

    def get_node_edge_attention(self, node_acts, pair_acts, pair_mask):
        """
        node_acts: (B, N, D)
        pair_acts: (B, N, N, D)
        pair_mask: (B, N, N)
        """
        B, N, D = paddle.shape(node_acts)

        H, d = self.num_head, self.head_dim

        q = self.q_proj(node_acts).reshape([B, N, H, d]).transpose([0, 2, 1, 3]) # (B, H, N, d)
        q *= (1 / d ** 0.5)
        k_n = self.k_proj(node_acts).reshape([B, N, H, d]).transpose([0, 2, 1, 3])  # (B, H, N, d)
        k_e = self.k_e_proj(pair_acts).reshape([B, N, N, H, d]).transpose([0, 3, 1, 2, 4]) #(B, H, N, N, d)

        q = q.unsqueeze([3]) # (B, H, N, 1, d)
        k_n = k_n.unsqueeze([2]) # (B, H, 1, N, d)
        k = k_n + k_e # (B, H, N, N, d)

        attn_weights = paddle.matmul(q, k, transpose_y=True)    # (B, H, N, 1, N)
        attn_weights = attn_weights.reshape([B, H, N, N])       # (B, H, N, N)
        attn_weights += (1 - pair_mask).unsqueeze([1]) * (-1e6)   # (B, N, N) -> (B, 1, N, N)
        attn_probs = paddle.nn.functional.softmax(attn_weights) # (B, H, N, N)
        attn_probs = attn_probs.reshape([B, H, N, 1, N])
        return attn_probs

    def get_attention_update(self, node_acts, pair_acts, attn_probs, node_mask):
        """
        node_acts: (B, N, D)
        pair_acts: (B, N, N, D)
        attn_probs: (B, H, N, 1, N)
        """
        B, N, D = paddle.shape(node_acts)
        H, d = self.num_head, self.head_dim

        v = self.v_proj(node_acts).reshape([B, N, H, d]).transpose([0, 2, 1, 3])  # (B, H, N, d)
        v_n = v.unsqueeze([3]) #(B, H, N, 1, d)
        v_r = v.reshape([B, H, 1, N, d])  
        v_e = self.v_e_proj(pair_acts).reshape([B, N, N, H, d]).transpose([0, 3, 1, 2, 4]) # (B, H, N, N, d)
        v_final = self.add_mlp(v_n + v_r + v_e) # (B, H, N, N, d)

        output = paddle.matmul(attn_probs, v_final) # (B, H, N, 1, d)
        output = output.squeeze(-2)     # (B, H, N, d)
        if self.virtual_node:
            if self.model_config.get('virt2', False):
                v_mask = node_mask.unsqueeze([1, 3])        # (B, 1, N, 1)
                v_node = paddle.sum(output * v_mask, -2)    # (B, H, d)
                v_node /= (paddle.sum(v_mask, -2) + 1e-6)
                v_node_repr = self.virtual_node_mlp(v_node) # (B, H, d)
                output += v_node_repr.unsqueeze(-2)         # (B, H, N, d)
            else:    
                v_mask = node_mask.unsqueeze([1, 2, 4])     # (B, 1, 1, N, 1)
                v_node = paddle.sum(v_final * v_mask, -2)   # (B, H, N, d)
                v_node /= (paddle.sum(v_mask, -2) + 1e-6)
                v_node_repr = self.virtual_node_mlp(v_node)
                output += v_node_repr
        output = output.transpose([0, 2, 1, 3]).reshape([B, N, D])
        output = self.out_proj(output)
        return output
    
    def forward(self, node_acts, pair_acts, node_mask, pair_mask):
        """
        node_acts: (B, N, D)
        node_mask: (B, N), 0 for padding, 1 for real values
        pair_mask: (B, N, N), 0 for padding, 1 for real values

        return:
            output: (B, N, D)
        """
        node_acts = self.node_ln(node_acts)
        if self.use_pair_layer_norm:
            pair_acts = self.pair_ln(pair_acts)

        attn_probs = self.get_node_edge_attention(node_acts, pair_acts, pair_mask)
        attn_probs = self.dropout(attn_probs)
        output = self.get_attention_update(node_acts, pair_acts, attn_probs, node_mask)
        return output


class FeedForwardNetwork(nn.Layer):
    """
    FFN for the transformer
    """
    def __init__(self, model_config, input_channel):
        super(FeedForwardNetwork, self).__init__()
        hidden_channel = input_channel * model_config.hidden_factor

        self.ln = nn.LayerNorm(input_channel)
        self.fc1 = nn.Linear(input_channel, hidden_channel)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(model_config.dropout_rate)
        self.fc2 = nn.Linear(hidden_channel, input_channel)

    def forward(self, x):
        x = self.ln(x)
        x = self.fc2(self.dropout(self.act(self.fc1(x))))
        return x


class Low2HighModule(nn.Layer):
    def __init__(self, model_config, global_config):
        super(Low2HighModule, self).__init__()
        node_channel = global_config.node_channel
        pair_channel = global_config.pair_channel
        inner_channel = model_config.inner_channel

        self.ln = nn.LayerNorm(node_channel)
        self.fc1 = nn.Linear(node_channel, inner_channel)
        self.fc2 = nn.Linear(node_channel, inner_channel)
        self.fc_act = nn.Linear(inner_channel * inner_channel, pair_channel)

    def forward(self, node_acts, node_mask):
        """
        node_acts: (B, N, D)
        node_mask: (B, N)
        return:
            act: (B, C, C, DP)
        """
        node_acts = self.ln(node_acts)
        node_mask = node_mask.unsqueeze(-1)    # (B, N, 1)
        left_act = (node_mask * self.fc1(node_acts)).unsqueeze(1).transpose([0, 2, 3, 1])   # (B, C, DI, R)
        right_act = (node_mask * self.fc2(node_acts)).unsqueeze(1).transpose([0, 1, 3, 2])  # (B, R, DI, C)
        B, C, DI, R = paddle.shape(left_act)
        left_act = left_act.reshape([B, C * DI, R])
        right_act = right_act.reshape([B, R, DI * C])
        act = paddle.matmul(left_act, right_act).reshape([B, C, DI, DI, C])   # (B, C, DI, DI, C)
        act = act.transpose([0, 1, 4, 2, 3]).reshape([B, C, C, DI * DI])
        act = self.fc_act(act)
        return act


class SecondBodyAxialAttentionWithAngle(nn.Layer):
    def __init__(self, model_config, global_config):
        super(SecondBodyAxialAttentionWithAngle, self).__init__()
        pair_channel = global_config.pair_channel
        triple_channel = global_config.triple_channel
        self.num_head = model_config.num_head
        self.head_dim = triple_channel // self.num_head

        self.ln = nn.LayerNorm(pair_channel)

        self.q_proj = nn.Linear(pair_channel, triple_channel)
        self.k_e_proj = nn.Linear(pair_channel, triple_channel)
        self.k_a_proj = nn.Linear(pair_channel, triple_channel)
        self.v_proj = nn.Linear(pair_channel, triple_channel)
        self.v_a_proj = nn.Linear(pair_channel, triple_channel)

        self.attn_dropout = nn.Dropout(model_config.dropout_rate)
        self.out_proj = nn.Linear(triple_channel, pair_channel)
        self.out_gate = nn.Linear(pair_channel, pair_channel)
    
    def forward(self, pair_acts, triple_acts, bias):
        """
        pair_acts: (B, N, N, D)
        triple_acts: (B, N, N, N, D)
        bias: (B, 1, N, N)
        """
        H, d = self.num_head, self.head_dim
        pair_acts = self.ln(pair_acts)

        q = self.q_proj(pair_acts).reshape(shape=[0, 0, 0, H, d])   # (B, N, N, H, d)
        q = q.transpose(perm=[0, 1, 3, 2, 4]) * (d ** (-0.5))       # (B, N, H, N, d)
        q = q.unsqueeze([-2])                           # (B, N, H, N, 1, d)

        k_e = self.k_e_proj(pair_acts).reshape(shape=[0, 0, 0, H, d])   # (B, N, N, H, d)
        k_e = k_e.transpose(perm=[0, 1, 3, 4, 2])       # (B, N, H, d, N)
        k_e = k_e.unsqueeze([-3])                       # (B, N, H, 1, d, N)
        k_a = self.k_a_proj(triple_acts).reshape(shape=[0, 0, 0, 0, H, d])  # (B, N, N, N, H, d)
        k_a = k_a.transpose(perm=[0, 1, 4, 2, 5, 3])    # (B, N, H, N, d, N)
        k = k_e + k_a                                   # (B, N, H, N, d, N) 

        v_e = self.v_proj(pair_acts).reshape(shape=[0, 0, 0, H, d])     # (B, N, N, H, d)
        v_a = self.v_a_proj(triple_acts).reshape(shape=[0, 0, 0, 0, H, d])  # (B, N, N, N, H, d)
        v_e = v_e.unsqueeze(2)
        v = v_a + v_e
        v = v.transpose(perm=[0, 1, 4, 2, 3, 5])        # (B, N, H, N, N, d)

        logits = paddle.matmul(q, k) + bias.unsqueeze([2, 4]) # (B, N, H, N, 1, N)
        weights = nn.functional.softmax(logits)
        weights = self.attn_dropout(weights)
        out = paddle.matmul(weights, v).squeeze(4).transpose([0, 1, 3, 2, 4])

        out = self.out_proj(out.reshape([0, 0, 0, H * d]))    # (B, N, N, D)
        gate = nn.functional.sigmoid(self.out_gate(pair_acts))
        out = out * gate
        return out


class SecondBodyAxialAttentionWithAngleBias(nn.Layer):
    def __init__(self, model_config, global_config):
        super(SecondBodyAxialAttentionWithAngleBias, self).__init__()
        pair_channel = global_config.pair_channel
        triple_channel = global_config.triple_channel
        self.num_head = model_config.num_head
        self.head_dim = triple_channel // self.num_head

        self.ln = nn.LayerNorm(pair_channel)

        self.q_proj = nn.Linear(pair_channel, triple_channel)
        self.k_proj = nn.Linear(pair_channel, triple_channel)
        self.v_proj = nn.Linear(pair_channel, triple_channel)
        self.a_proj = nn.Linear(pair_channel, self.num_head)

        self.attn_dropout = nn.Dropout(model_config.dropout_rate)
        self.out_proj = nn.Linear(triple_channel, pair_channel)
        self.out_gate = nn.Linear(pair_channel, pair_channel)
    
    def forward(self, pair_acts, triple_acts, bias):
        """
        pair_acts: (B, N, N, D)
        triple_acts: (B, N, N, N, D)
        bias: (B, N, N, N)
        """
        H, d = self.num_head, self.head_dim
        pair_acts = self.ln(pair_acts)

        q = self.q_proj(pair_acts).reshape(shape=[0, 0, 0, H, d])   # (B, N, N, H, d)
        q = q.transpose(perm=[0, 1, 3, 2, 4]) * (d ** (-0.5))       # (B, N, H, N, d)

        k = self.k_proj(pair_acts).reshape(shape=[0, 0, 0, H, d])   # (B, N, N, H, d)
        k = k.transpose(perm=[0, 1, 3, 2, 4])       # (B, N, H, N, d)

        a = self.a_proj(triple_acts)                # (B, N, N, N, H)
        a = a.transpose([0, 1, 4, 2, 3])

        v = self.v_proj(pair_acts).reshape(shape=[0, 0, 0, H, d])     # (B, N, N, H, d)
        v = v.transpose(perm=[0, 1, 3, 2, 4])       # (B, N, H, N, d)

        logits = paddle.matmul(q, k, transpose_y=True)    # (B, N, H, N, N)
        logits += a + bias.unsqueeze([2])
        weights = nn.functional.softmax(logits)
        weights = self.attn_dropout(weights)
        out = paddle.matmul(weights, v)     # (B, N, H, N, d)
        out = out.transpose([0, 1, 3, 2, 4]).reshape([0, 0, 0, H * d])  # (B, N, N, D)

        out = self.out_proj(out)    # (B, N, N, D)
        gate = nn.functional.sigmoid(self.out_gate(pair_acts))
        out = out * gate
        return out


class SecondBodyAxialAttention(nn.Layer):
    def __init__(self, model_config, global_config):
        super(SecondBodyAxialAttention, self).__init__()
        self.is_start = model_config.is_start

        if model_config.get('angle_as_bias', False):
            self.attn_mod = SecondBodyAxialAttentionWithAngleBias(model_config, global_config)
        else:
            self.attn_mod = SecondBodyAxialAttentionWithAngle(model_config, global_config)
    
    def forward(self, pair_acts, triple_acts, triple_mask):
        """
        pair_act: (B, N, N, D)
        triple_mask: (B, 1, N, N), 1 for valid
        """
        bias = (1 - triple_mask) * -1e9  # (B, 1, N, N)
        if self.is_start:
            pair_acts = self.attn_mod(pair_acts, triple_acts, bias)
        else:
            pair_acts = pair_acts.transpose([0, 2, 1, 3])
            pair_acts = self.attn_mod(pair_acts, triple_acts, bias)
            pair_acts = pair_acts.transpose([0, 2, 1, 3])
        return pair_acts


class OptimusBlock(nn.Layer):
    """
    Column version of the Axial Transformer
    """
    def __init__(self, model_config, global_config):
        super(OptimusBlock, self).__init__()

        node_channel = global_config.node_channel
        pair_channel = global_config.pair_channel
        
        ### node track
        self.first_body_axial_attention = FirstBodyAxialAttention(
                model_config.first_body_axial_attention, global_config)
        self.first_body_axial_attention_dropout = nn.Dropout(model_config.first_body_axial_attention_dropout)

        self.node_ffn = FeedForwardNetwork(
                model_config.node_ffn, node_channel)
        self.node_ffn_dropout = nn.Dropout(model_config.first_body_axial_attention_dropout)

        ### low2high
        self.low2high = Low2HighModule(
                model_config.low2high, global_config)
        self.low2high_dropout = nn.Dropout(model_config.pair_dropout_rate)

        ### pair track
        self.pair_before_ln = nn.LayerNorm(pair_channel)

        self.second_body_first_axis = SecondBodyAxialAttention(
                model_config.second_body_first_axis, global_config)
        self.second_body_first_axis_dropout = nn.Dropout(model_config.pair_dropout_rate)

        self.second_body_second_axis = SecondBodyAxialAttention(
                model_config.second_body_second_axis, global_config)
        self.second_body_second_axis_dropout = nn.Dropout(model_config.pair_dropout_rate)

        self.pair_ffn = FeedForwardNetwork(
                model_config.pair_ffn, pair_channel)
        self.pair_ffn_dropout = nn.Dropout(model_config.pair_dropout_rate)

    def forward(self, node_acts, pair_acts, triple_acts, mask_dict):
        """
        node_acts: (B, N, D)
        pair_acts: (B, N, N, D)
        node_mask: (B, N)
        pair_mask: (B, N, N)

        return:
            ffn_out: (B, R, C, D)
            ffn_out: (B, N, D)
            //row_attn: (B, H, C, C)
        """
        node_mask = mask_dict['node']
        pair_mask = mask_dict['pair']
        triple_mask = mask_dict['triple']

        # node track
        residual = self.first_body_axial_attention(node_acts, pair_acts, node_mask, pair_mask)
        node_acts += self.first_body_axial_attention_dropout(residual)

        residual = self.node_ffn(node_acts)
        node_acts += self.node_ffn_dropout(residual)

        # outer
        outer = self.low2high(node_acts, node_mask)
        pair_acts += self.low2high_dropout(outer)

        # pair track
        pair_acts = self.pair_before_ln(pair_acts)

        residual = self.second_body_first_axis(pair_acts, triple_acts, triple_mask)
        pair_acts += self.second_body_first_axis_dropout(residual)

        residual = self.second_body_second_axis(pair_acts, triple_acts, triple_mask)
        pair_acts += self.second_body_second_axis_dropout(residual)

        residual = self.pair_ffn(pair_acts)
        pair_acts += self.pair_ffn_dropout(residual)
        return node_acts, pair_acts
  

class Optimus(nn.Layer):
    """
    Optimus
    """
    def __init__(self, model_config):
        super(Optimus, self).__init__()
        self.model_config = model_config

        self.embedding_layer = EmbeddingLayer(model_config.embedding_layer, model_config)
        
        self.layer_norm_before = nn.LayerNorm(self.model_config.node_channel)
        self.dropout = nn.Dropout(p=self.model_config.init_dropout_rate)

        self.optimus_blocks = nn.LayerList()
        for i in range(self.model_config.optimus_block_num):
            block = OptimusBlock(self.model_config.optimus_block, self.model_config)
            self.optimus_blocks.append(block)
        
        self.apply(self.init_weights)
 
    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=0.02,
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12
    
    def _create_mask(self, batch):
        node_mask = batch["node_mask"]  # (B, N)

        pair_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)   # (B, N, N)
        if self.model_config.attention_max_hop_clip >= 0:
            pair_mask = paddle.logical_and(
                    paddle.cast(pair_mask, 'bool'), 
                    batch['hop_num'] <= self.model_config.attention_max_hop_clip)
            pair_mask = paddle.cast(pair_mask, 'float32')

        triple_mask = pair_mask.unsqueeze(-3)   # (B, 1, N, N)
        mask_dict = {
            'node': node_mask,
            'pair': pair_mask,
            'triple': triple_mask,
        }
        return mask_dict

    def forward(self, batch):
        """
        node_feat: {feat1:(B, N), feat2:(B, N), ...} number of feat is the number of atom names
        edge_feat: {feat1: (B, N, N), feat2:(B, N, N)} number of feat is the number of bond names
        node_mask: (B, N), 1 for atom, 0 for padding

        return:
            compound_out: (B, N, D)
            prot_out: (B, C, D)
            row_attn: [(B, H, C, C)]
        """
        embed_repr = self.embedding_layer(batch)
        node_acts = embed_repr['node_acts']
        pair_acts = embed_repr['pair_acts']
        triple_acts = embed_repr['triple_acts']

        mask_dict = self._create_mask(batch)
        node_acts = self.dropout(self.layer_norm_before(node_acts))
        for block_i, block in enumerate(self.optimus_blocks):
            node_acts, pair_acts = recompute_wrapper(block, 
                    node_acts, pair_acts, triple_acts, mask_dict,
                    is_recompute=self.training)
        results = {
            "node_acts": node_acts,
            "pair_acts": pair_acts,
        }
        return results
        