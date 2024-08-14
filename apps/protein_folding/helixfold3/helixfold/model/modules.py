#   Copyright (c) 2024 PaddleHelix Authors. All Rights Reserved.
#
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# International License (the "License");  you may not use this file  except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modules."""

import numpy as np
import paddle
import paddle.nn as nn
from paddle.distributed.fleet.utils import recompute
try:
    from paddle import _legacy_C_ops as _C_ops
except:
    from paddle import _C_ops

from helixfold.model.utils import subbatch, init_gate_linear, init_final_linear


def recompute_wrapper(func, *args, is_recompute=True):
    """Function wrapper for recompute"""
    if is_recompute:
        return recompute(func, *args)
    else:
        return func(*args)



class Dropout(nn.Layer):
    def __init__(self, p=0.5, axis=None, mode="upscale_in_train", name=None):
        super(Dropout, self).__init__()

        if not isinstance(p, (float, int)):
            raise TypeError("p argument should be a number")
        if p < 0 or p > 1:
            raise ValueError("p argument should between 0 and 1")

        mode = 'downgrade_in_infer' if mode == 'downscale_in_infer' else mode  #semantic transfer
        if mode not in ('downscale_in_infer', 'upscale_in_train'):
            raise ValueError(
                "mode argument should be 'downscale_in_infer' or 'upscale_in_train'"
            )

        if axis and not isinstance(axis, (int, list, tuple)):
            raise TypeError("datatype of axis argument should be int or list")

        self.p = p
        self.axis = axis
        self.mode = mode
        self.name = name

    def forward(self, input):
        # fast return for p == 0
        if self.p == 0:
            return input

        if self.axis == None: 
            out = nn.functional.dropout(input,
                            p=self.p,
                            axis=self.axis,
                            training=self.training,
                            mode=self.mode,
                            name=self.name)
        else:
            seed = None
            drop_axes = [self.axis] if isinstance(self.axis, int) else list(self.axis)
            if paddle.static.default_main_program().random_seed != 0:
                seed = paddle.static.default_main_program().random_seed

            out, mask = _C_ops.dropout_nd(input, 'dropout_prob', self.p, 'is_test',
                                                    not self.training, 'fix_seed', seed
                                                    is not None, 'seed',
                                                    seed if seed is not None else 0,
                                                    'dropout_implementation', self.mode, 'axis',
                                                    drop_axes)

        return out

    def extra_repr(self):
        name_str = ', name={}'.format(self.name) if self.name else ''
        return 'p={}, axis={}, mode={}{}'.format(self.p, self.axis, self.mode,
                                                 name_str)

class Attention(nn.Layer):
    """Multihead attention."""

    def __init__(self, config, global_config, q_dim, kv_dim, output_dim):
        super(Attention, self).__init__()
        self.config = config
        self.global_config = global_config

        num_head = self.config.num_head
        key_dim = self.config.get('key_dim', q_dim)
        value_dim = self.config.get('value_dim', kv_dim)

        self.fuse_attention = self.global_config.fuse_attention
        self.use_flash_attn = self.global_config.use_flash_attn
        self.merge_qkv = (q_dim == kv_dim)

        assert key_dim % num_head == 0
        assert value_dim % num_head == 0
        key_dim = key_dim // num_head
        value_dim = value_dim // num_head

        self.key_dim = key_dim
        self.value_dim = value_dim

        self.qkv_w = None
        self.query_w = None
        self.key_w = None
        self.value_w = None
        if self.merge_qkv and self.fuse_attention:
            self.qkv_w = paddle.create_parameter(
                [3, num_head, key_dim, q_dim], 'float32',
                default_initializer=nn.initializer.XavierUniform())
        else:
            self.query_w = paddle.create_parameter(
                [q_dim, num_head, key_dim], 'float32',
                default_initializer=nn.initializer.XavierUniform())
            self.key_w = paddle.create_parameter(
                [kv_dim, num_head, key_dim], 'float32',
                default_initializer=nn.initializer.XavierUniform())
            self.value_w = paddle.create_parameter(
                [kv_dim, num_head, value_dim], 'float32',
                default_initializer=nn.initializer.XavierUniform())

        self.gating_w = None
        self.gating_b = None
        if self.config.gating:
            self.gating_w = paddle.create_parameter(
                [q_dim, num_head, value_dim], 'float32',
                default_initializer=nn.initializer.Constant(0.0))
            self.gating_b = paddle.create_parameter(
                [num_head, value_dim], 'float32',
                default_initializer=nn.initializer.Constant(1.0))

        if self.global_config.zero_init:
            init = nn.initializer.Constant(0.0)
        else:
            init = nn.initializer.XavierUniform()

        self.output_w = paddle.create_parameter(
            [num_head, value_dim, output_dim], 'float32',
            default_initializer=init)
        self.output_b = paddle.create_parameter(
            [output_dim], 'float32',
            default_initializer=nn.initializer.Constant(0.0))

    def forward(self, q_data, m_data, bias, nonbatched_bias=None):
        """Builds Attention module.
        Arguments:
            q_data: A tensor of queries, shape [batch, row_size, N_queries, q_channels].
            m_data: A tensor of memories from which the keys and values are
                projected, shape [batch, row_size, N_keys, m_channels].
            bias: A bias for the attention, shape [batch, row_size, num_head, N_queries, N_keys].
            nonbatched_bias: Shared bias, shape [N_queries, N_keys].

        Returns:
            A float32 tensor of shape [batch_size, row_size, N_queries, output_dim].
        """
        if self.fuse_attention:
            if nonbatched_bias is not None:
                nonbatched_bias = paddle.unsqueeze(nonbatched_bias, axis=1)

            import paddle.incubate.nn.functional as F
            output = F.fused_gate_attention(
                query=q_data,
                key=m_data,
                query_weight=self.query_w,
                key_weight=self.key_w,
                value_weight=self.value_w,
                qkv_weight=self.qkv_w,
                gate_linear_weight=self.gating_w,
                gate_linear_bias=self.gating_b,
                out_linear_weight=self.output_w,
                out_linear_bias=self.output_b,
                nonbatched_bias=nonbatched_bias,
                attn_mask=bias,
                has_gating=self.config.gating,
                merge_qkv=self.merge_qkv,
                use_flash_attn=self.use_flash_attn,
            )
        else:
            c = self.key_dim ** (-0.5)
            q = paddle.einsum('nbqa,ahc->nbqhc', q_data, self.query_w) * c
            k = paddle.einsum('nbka,ahc->nbkhc', m_data, self.key_w)
            v = paddle.einsum('nbka,ahc->nbkhc', m_data, self.value_w)
            logits = paddle.einsum('nbqhc,nbkhc->nbhqk', q, k) + bias

            if nonbatched_bias is not None:
                logits += paddle.unsqueeze(nonbatched_bias, axis=1)

            weights = nn.functional.softmax(logits)
            weighted_avg = paddle.einsum('nbhqk,nbkhc->nbqhc', weights, v)

            if self.config.gating:
                gate_values = paddle.einsum('nbqc,chv->nbqhv', q_data,
                                            self.gating_w) + self.gating_b
                gate_values = nn.functional.sigmoid(gate_values)
                weighted_avg *= gate_values

            output = paddle.einsum('nbqhc,hco->nbqo', weighted_avg,
                                self.output_w) + self.output_b 
        return output


class OuterProductMean(nn.Layer):
    """Computes mean outer product.

    Jumper et al. (2021) Suppl. Alg. 10 "OuterProductMean"
    """

    def __init__(self, channel_num, config, global_config, is_extra_msa, name='outer_product_mean'):
        super(OuterProductMean, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        Linear = paddle.incubate.nn.FusedLinear if self.global_config.fuse_linear else paddle.nn.Linear

        if is_extra_msa:
            c_m = channel_num['extra_msa_channel']
        else:
            c_m = channel_num['msa_channel']

        self.layer_norm_input = nn.LayerNorm(c_m, name='layer_norm_input')
        self.left_projection = Linear(
            c_m, self.config.num_outer_channel, name='left_projection')
        self.right_projection = Linear(
            c_m, self.config.num_outer_channel, name='right_projection')

        if self.global_config.zero_init:
            init_w = nn.initializer.Constant(value=0.0)
        else:
            init_w = nn.initializer.KaimingNormal()

        self.output_w = paddle.create_parameter(
            [self.config.num_outer_channel, self.config.num_outer_channel, channel_num['pair_channel']],
            'float32', default_initializer=init_w)
        self.output_b = paddle.create_parameter(
            [channel_num['pair_channel']], 'float32',
            default_initializer=nn.initializer.Constant(value=0.0))

    def forward(self, act, mask):
        """Builds OuterProductMean module.

        Arguments:
        act: MSA representation, shape [batch, N_seq, N_res, c_m].
        mask: MSA mask, shape [batch, N_seq, N_res].

        Returns:
        Update to pair representation, shape [batch, N_res, N_res, c_z].
        """
        
        act = self.layer_norm_input(act)
        right_act = self.right_projection(act)
        
        left_act = self.left_projection(act)
        mask = paddle.unsqueeze(mask, axis=-1)

        left_act = mask * left_act
        
        epsilon = 1e-3
        norm = paddle.einsum('nabc,nadc->nbdc', mask, mask) + epsilon

        def fast_einsum(equation, left_act, right_act):
            assert equation == "nacb,nade->ndceb"
            tmp = paddle.matmul(
                x=paddle.reshape(right_act, [right_act.shape[0], right_act.shape[1], -1]),  # na(de)
                y=paddle.reshape(left_act, [left_act.shape[0], left_act.shape[1], -1]),     # na(cb)
                transpose_x=True,
                transpose_y=False)  # n(de)(cb)
            tmp = paddle.reshape(tmp, [left_act.shape[0], right_act.shape[2], right_act.shape[3], left_act.shape[2], left_act.shape[3]])
            out = paddle.transpose(tmp, perm=[0, 1, 3, 2, 4])
            return out

        def compute_chunk(left_act, right_act):
            # This is equivalent to
            #
            # act = jnp.einsum('abc,ade->dceb', left_act, right_act)
            # act = jnp.einsum('dceb,cef->bdf', act, output_w) + output_b
            #
            # but faster. maybe for subbatch inference?
            
            left_act = left_act.transpose([0, 1, 3, 2])
            act = fast_einsum('nacb,nade->ndceb', left_act, right_act)
            act = paddle.einsum('ndceb,cef->ndbf', act, self.output_w) + self.output_b
            return act.transpose([0, 2, 1, 3])

        if not self.training:
            # low memory mode using subbatch
            sb_chunk = subbatch(compute_chunk, [0], [2],
                               self.config.chunk_size, 1)
            act = sb_chunk(left_act, right_act)
        else:
            act = compute_chunk(left_act, right_act)

        act = act / norm

        return act


class TriangleAttention(nn.Layer):
    """Triangle Attention.

    Jumper et al. (2021) Suppl. Alg. 13 "TriangleAttentionStartingNode"
    Jumper et al. (2021) Suppl. Alg. 14 "TriangleAttentionEndingNode"
    """

    def __init__(self, channel_num, config, global_config, name='triangle_attention'):
        super(TriangleAttention, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        assert config.orientation in ['per_row', 'per_column']

        self.query_norm = nn.LayerNorm(channel_num['pair_channel'],
                                    name='query_norm')
        self.feat_2d_weights = paddle.create_parameter(
            [channel_num['pair_channel'], self.config.num_head], 'float32',
            default_initializer=nn.initializer.Normal(
                std=1. / np.sqrt(channel_num['pair_channel'])))

        self.attention = Attention(self.config, self.global_config,
                        channel_num['pair_channel'], channel_num['pair_channel'],
                        channel_num['pair_channel'])


    def forward(self, pair_act, pair_mask):
        """Builds TriangleAttention module.

        Arguments:
        pair_act: [batch, N_res, N_res, c_z] pair activations tensor
        pair_mask: [batch, N_res, N_res] mask of non-padded regions in the tensor.

        Returns:
        Update to pair_act, shape [batch, N_res, N_res, c_z].
        """
        if self.config.orientation == 'per_column':
            pair_act = pair_act.transpose([0, 2, 1, 3])
            pair_mask = pair_mask.transpose([0, 2, 1])

        bias = 1e9 * (pair_mask - 1.)
        bias = paddle.unsqueeze(bias, axis=[2, 3])

        pair_act = self.query_norm(pair_act)

        nonbatched_bias = paddle.einsum('bqkc,ch->bhqk', pair_act, self.feat_2d_weights)

        if not self.training:
            # low memory mode using subbatch
            sb_attn = subbatch(self.attention, [0, 1, 2], [1, 1, 1],
                               self.global_config.subbatch_size, 1, same_arg_idx={1: 0})
            pair_act = sb_attn(pair_act, pair_act, bias, nonbatched_bias)
        elif "train_subbatch_size" in self.global_config:
            sb_attn = subbatch(self.attention, [0, 1, 2], [1, 1, 1],
                               self.global_config.train_subbatch_size, 1, same_arg_idx={1: 0})
            pair_act = sb_attn(pair_act, pair_act, bias, nonbatched_bias)
        else:
            pair_act = self.attention(pair_act, pair_act, bias, nonbatched_bias)

        if self.config.orientation == 'per_column':
            pair_act = pair_act.transpose([0, 2, 1, 3])

        return pair_act


class TriangleMultiplication(nn.Layer):
    """Triangle multiplication layer ("outgoing" or "incoming").

    Jumper et al. (2021) Suppl. Alg. 11 "TriangleMultiplicationOutgoing"
    Jumper et al. (2021) Suppl. Alg. 12 "TriangleMultiplicationIncoming"
    """

    def __init__(self, channel_num, config, global_config, name='triangle_multiplication'):
        super(TriangleMultiplication, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        # Linear = paddle.incubate.nn.FusedLinear if self.global_config.fuse_linear else paddle.nn.Linear
        Linear = paddle.nn.Linear

        if self.config.get("fuse_projection_weights", False):
            self.left_norm_input = nn.LayerNorm(self.channel_num['pair_channel'], name='left_norm_input')
            self.projection = Linear(self.channel_num['pair_channel'],
                                    2 * self.config.num_intermediate_channel, name='projection')

            self.gate = Linear(self.channel_num['pair_channel'],
                                    2 * self.config.num_intermediate_channel, name='gate')
            init_gate_linear(self.gate)

            # line 4
            self.center_norm = nn.LayerNorm(self.config.num_intermediate_channel, name='center_norm')
        else:
            self.layer_norm_input = nn.LayerNorm(self.channel_num['pair_channel'], name='layer_norm_input')
            self.left_projection = Linear(self.channel_num['pair_channel'],
                                    self.config.num_intermediate_channel, name='left_projection')
            self.right_projection = Linear(self.channel_num['pair_channel'],
                                    self.config.num_intermediate_channel, name='right_projection')
            self.left_gate = Linear(self.channel_num['pair_channel'],
                                    self.config.num_intermediate_channel, name='left_gate')
            init_gate_linear(self.left_gate)
            self.right_gate = Linear(self.channel_num['pair_channel'],
                                    self.config.num_intermediate_channel, name='right_gate')
            init_gate_linear(self.right_gate)

            # line 4
            self.center_layer_norm = nn.LayerNorm(self.config.num_intermediate_channel, name='center_layer_norm')
        
        self.output_projection = Linear(self.config.num_intermediate_channel,
                                    self.channel_num['pair_channel'], name='output_projection')
        init_final_linear(self.output_projection)
        # line 3
        self.gating_linear = Linear(self.channel_num['pair_channel'],
                                    self.channel_num['pair_channel'], name='gating_linear')
        init_gate_linear(self.gating_linear)

    def forward(self, act, mask):
        """Builds TriangleMultiplication module.

        Arguments:
        act: Pair activations, shape [batch, N_res, N_res, c_z]
        mask: Pair mask, shape [batch, N_res, N_res].

        Returns:
        Outputs, same shape/type as act.
        """
        mask = paddle.unsqueeze(mask, axis=-1) # [batch, N_res, N_res, 1]

        if self.config.get("fuse_projection_weights", False):
            left_act = self.left_norm_input(act) # line 1

            # Both left and right projections are fused into projection.
            proj_act = mask * self.projection(left_act)

            # Both left + right gate are fused into gate_values.
            gate_values = nn.functional.sigmoid(self.gate(left_act))
            
            proj_act = proj_act * gate_values

            left_proj_act = proj_act[..., :self.config.num_intermediate_channel]
            right_proj_act = proj_act[..., self.config.num_intermediate_channel:]
        else:
            act = self.layer_norm_input(act) # line 1

            left_proj_act = mask * self.left_projection(act)
            right_proj_act = mask * self.right_projection(act)
            
            left_gate_values = nn.functional.sigmoid(self.left_gate(act))
            right_gate_values = nn.functional.sigmoid(self.right_gate(act))
            
            left_proj_act = left_proj_act * left_gate_values
            right_proj_act = right_proj_act * right_gate_values

        # "Outgoing" edges equation: 'ikc,jkc->ijc'
        # "Incoming" edges equation: 'kjc,kic->ijc'
        # Note on the Suppl. Alg. 11 & 12 notation:
        # For the "outgoing" edges, a = left_proj_act and b = right_proj_act
        # For the "incoming" edges, it's swapped:
        #   b = left_proj_act and a = right_proj_act
            
        if self.config.get("fuse_projection_weights", False):
            gate_values = nn.functional.sigmoid(self.gating_linear(left_act)) # line 3
        else:
            gate_values = nn.functional.sigmoid(self.gating_linear(act)) # line 3

        if self.config.equation == 'ikc,jkc->ijc':
            # Outgoing
            dim, out_idx = 1, 1
            equation = 'bikc,bjkc->bijc'
            
        elif  self.config.equation == 'kjc,kic->ijc':
            # Incoming
            dim, out_idx = 2, 2
            equation = 'bkjc,bkic->bijc'

        else:
            raise ValueError('unknown equation.')

        if not self.training:
            einsum_fn = subbatch(paddle.einsum, [1], [dim],
                                 self.global_config.subbatch_size, out_idx)
            act = einsum_fn(equation, left_proj_act, right_proj_act)
        elif "train_subbatch_size" in self.global_config:
            einsum_fn = subbatch(paddle.einsum, [1], [dim],
                                 self.global_config.train_subbatch_size, out_idx)
            act = einsum_fn(equation, left_proj_act, right_proj_act)
        else:
            # Outgoing equation = 'bikc,bjkc->bijc'
            # Incoming equation = 'bkjc,bkic->bijc'
            act = paddle.einsum(equation, left_proj_act, right_proj_act)

        if self.config.get("fuse_projection_weights", False):
            act = self.center_norm(act)
        else:
            act = self.center_layer_norm(act)
        act = self.output_projection(act)

        act = act * gate_values

        return act


def dgram_from_positions(positions, num_bins, min_bin, max_bin):
    lower_breaks = paddle.linspace(min_bin, max_bin, num_bins)
    lower_breaks = paddle.square(lower_breaks)
    upper_breaks = paddle.concat([lower_breaks[1:],
            paddle.full(shape=[1], fill_value=1e8, dtype='float32')])

    def _squared_difference(x, y):
        return paddle.square(x - y)

    dist2 = paddle.sum(
        _squared_difference(
            paddle.unsqueeze(positions, axis=-2),
            paddle.unsqueeze(positions, axis=-3)),
        axis=-1, keepdim=True)

    dgram = ((dist2 > lower_breaks.astype(dist2.dtype)).astype('float32') *
                (dist2 < upper_breaks.astype(dist2.dtype)).astype('float32'))
    return dgram
