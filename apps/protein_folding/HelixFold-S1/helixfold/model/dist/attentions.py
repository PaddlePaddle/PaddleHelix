import gc
import numpy as np
import paddle
import paddle.nn as nn

try:
    from paddle import _legacy_C_ops as _C_ops
except:
    from paddle import _C_ops

from ppfleetx.distributed.protein_folding import dap

from ppfleetx.models.protein_folding.common import (
    init_gate_linear,
    init_final_linear )
from helixfold.model.utils import subbatch


class Attention(nn.Layer):
    """Multihead attention."""

    def __init__(self, config, global_config, q_dim, kv_dim, output_dim):
        super(Attention, self).__init__()
        self.config = config
        self.global_config = global_config

        num_head = self.config.num_head
        key_dim = self.config.get('key_dim', q_dim)
        value_dim = self.config.get('value_dim', kv_dim)

        # TODO(GuoxiaWang): delete non fuse_attention related code on dcu
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
                [3, num_head, key_dim, q_dim],
                'float32',
                default_initializer=nn.initializer.XavierUniform())
        else:
            self.query_w = paddle.create_parameter(
                [q_dim, num_head, key_dim],
                'float32',
                default_initializer=nn.initializer.XavierUniform())
            self.key_w = paddle.create_parameter(
                [kv_dim, num_head, key_dim],
                'float32',
                default_initializer=nn.initializer.XavierUniform())
            self.value_w = paddle.create_parameter(
                [kv_dim, num_head, value_dim],
                'float32',
                default_initializer=nn.initializer.XavierUniform())

        self.gating_w = None
        self.gating_b = None
        if self.config.gating:
            self.gating_w = paddle.create_parameter(
                [q_dim, num_head, value_dim],
                'float32',
                default_initializer=nn.initializer.Constant(0.0))
            self.gating_b = paddle.create_parameter(
                [num_head, value_dim],
                'float32',
                default_initializer=nn.initializer.Constant(1.0))

        if self.global_config.zero_init:
            init = nn.initializer.Constant(0.0)
        else:
            init = nn.initializer.XavierUniform()

        self.output_w = paddle.create_parameter(
            [num_head, value_dim, output_dim],
            'float32',
            default_initializer=init)
        self.output_b = paddle.create_parameter(
            [output_dim],
            'float32',
            default_initializer=nn.initializer.Constant(0.0))

    def forward(self, q_data, m_data, bias, nonbatched_bias=None):
        """Builds Attention module.
        
        Args:
            q_data (float): A tensor of queries, shape [batch, row_size, N_queries, q_channels].
            m_data (float): A tensor of memories from which the keys and values are
                projected, shape [batch, row_size, N_keys, m_channels].
            bias (float): A bias for the attention, shape [batch, row_size, num_head, N_queries, N_keys].
            nonbatched_bias (float): Shared bias, shape [N_queries, N_keys].

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
                use_flash_attn=self.use_flash_attn, )
        else:
            c = self.key_dim**(-0.5)
            if not self.training and self.global_config.low_memory is True:
                eisum_sp2_gt2 = subbatch(paddle.einsum, [1], [2], self.global_config.subbatch_size, 2)
                eisum_sp2_gt3 = subbatch(paddle.einsum, [1], [2], self.global_config.subbatch_size, 3)
            else:
                eisum_sp2_gt2 = paddle.einsum
                eisum_sp2_gt3 = paddle.einsum
            q = eisum_sp2_gt2('nbqa,ahc->nbqhc', q_data, self.query_w) * c
            k = eisum_sp2_gt2('nbka,ahc->nbkhc', m_data, self.key_w)
            v = eisum_sp2_gt2('nbka,ahc->nbkhc', m_data, self.value_w)
            logits = eisum_sp2_gt3('nbqhc,nbkhc->nbhqk', q, k) + bias

            if nonbatched_bias is not None:
                logits += paddle.unsqueeze(nonbatched_bias, axis=1)

            weights = nn.functional.softmax(logits)
            weighted_avg = eisum_sp2_gt3('nbhqk,nbkhc->nbqhc', weights, v)

            if self.config.gating:
                gate_values = paddle.einsum('nbqc,chv->nbqhv', q_data,
                                            self.gating_w) + self.gating_b
                gate_values = nn.functional.sigmoid(gate_values)
                weighted_avg *= gate_values

            output = eisum_sp2_gt2('nbqhc,hco->nbqo', weighted_avg,
                                   self.output_w) + self.output_b
        return output


class TriangleAttention(nn.Layer):
    """Triangle Attention.

    Jumper et al. (2021) Suppl. Alg. 13 "TriangleAttentionStartingNode"
    Jumper et al. (2021) Suppl. Alg. 14 "TriangleAttentionEndingNode"
    """

    def __init__(self,
                 channel_num,
                 config,
                 global_config,
                 name='triangle_attention'):
        super(TriangleAttention, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        assert config.orientation in ['per_row', 'per_column']

        self.query_norm = nn.LayerNorm(
            channel_num['pair_channel'], name='query_norm')
        self.feat_2d_weights = paddle.create_parameter(
            [channel_num['pair_channel'], self.config.num_head],
            'float32',
            default_initializer=nn.initializer.Normal(
                std=1. / np.sqrt(channel_num['pair_channel'])))

        self.attention = Attention(
            self.config, self.global_config, channel_num['pair_channel'],
            channel_num['pair_channel'], channel_num['pair_channel'])

    def forward(self, pair_act, pair_mask):
        """Builds TriangleAttention module.

        Args:
            pair_act (float): [batch, N_res, N_res, c_z] pair activations tensor
            pair_mask (float): [batch, N_res, N_res] mask of non-padded regions in the tensor.

        Returns:
            Update to pair_act, shape [batch, N_res, N_res, c_z].
        """
        if self.config.orientation == 'per_column':
            pair_act = pair_act.transpose([0, 2, 1, 3])
            pair_mask = pair_mask.transpose([0, 2, 1])

        # [B, N_res//dap_size, N_res]
        bias = 1e9 * (pair_mask - 1.)
        # [B, N_res//dap_size, 1, 1, N_res]
        bias = paddle.unsqueeze(bias, axis=[2, 3])

        pair_act = self.query_norm(pair_act)

        # [B, N_res//dap_size, N_res, cz], [cz, head] => [B, head, N_res//dap_size, N_res]
        nonbatched_bias_before = paddle.einsum('bqkc,ch->bhqk', pair_act,
                                               self.feat_2d_weights)

        # # [B, head, N_res//dap_size, N_res] => [B, head, N_res, N_res]
        nonbatched_bias = dap.all_gather(paddle.assign(nonbatched_bias_before), axis=2)
        # if not self.training:
        if not self.training and self.global_config.low_memory is True:
            del nonbatched_bias_before
            gc.collect()
        nonbatched_bias = dap.all_gather_opp(paddle.assign(nonbatched_bias), axis=2)

        if not self.training:
            # low memory mode using subbatch
            sb_attn = subbatch(
                self.attention, [0, 1, 2], [1, 1, 1],
                self.global_config.subbatch_size,
                1,
                same_arg_idx={1: 0})
            pair_act = sb_attn(pair_act, pair_act, bias, nonbatched_bias)
        else:
            pair_act = self.attention(pair_act, pair_act, bias,
                                      nonbatched_bias)

        if self.config.orientation == 'per_column':
            pair_act = pair_act.transpose([0, 2, 1, 3])

        return pair_act


class TriangleMultiplication(nn.Layer):
    """Triangle multiplication layer ("outgoing" or "incoming").

    Jumper et al. (2021) Suppl. Alg. 11 "TriangleMultiplicationOutgoing"
    Jumper et al. (2021) Suppl. Alg. 12 "TriangleMultiplicationIncoming"
    """

    def __init__(self,
                 channel_num,
                 config,
                 global_config,
                 name='triangle_multiplication'):
        super(TriangleMultiplication, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        Linear = paddle.incubate.nn.FusedLinear if self.global_config.fuse_linear else paddle.nn.Linear

        self.layer_norm_input = nn.LayerNorm(
            self.channel_num['pair_channel'], name='layer_norm_input')
        self.left_projection = Linear(
            self.channel_num['pair_channel'],
            self.config.num_intermediate_channel,
            name='left_projection')
        self.right_projection = Linear(
            self.channel_num['pair_channel'],
            self.config.num_intermediate_channel,
            name='right_projection')
        self.left_gate = Linear(
            self.channel_num['pair_channel'],
            self.config.num_intermediate_channel,
            name='left_gate')
        init_gate_linear(self.left_gate)
        self.right_gate = Linear(
            self.channel_num['pair_channel'],
            self.config.num_intermediate_channel,
            name='right_gate')
        init_gate_linear(self.right_gate)

        # line 4
        self.center_layer_norm = nn.LayerNorm(
            self.config.num_intermediate_channel, name='center_layer_norm')
        self.output_projection = Linear(
            self.config.num_intermediate_channel,
            self.channel_num['pair_channel'],
            name='output_projection')
        init_final_linear(self.output_projection)
        # line 3
        self.gating_linear = Linear(
            self.channel_num['pair_channel'],
            self.channel_num['pair_channel'],
            name='output_projection')
        init_gate_linear(self.gating_linear)

    def forward(self, act, mask):
        """Builds TriangleMultiplication module.

        Args:
            act (float): Pair activations, shape [batch, N_res, N_res, c_z]
            mask (float): Pair mask, shape [batch, N_res, N_res].

        Returns:
            Outputs, same shape/type as act.
        """
        # Outgoing [batch, N_res//dap_size, N_res] => [batch, N_res//dap_size, N_res, 1]
        # Incoming [batch, N_res, N_res//dap_size] => [batch, N_res, N_res//dap_size, 1] 
        mask = paddle.unsqueeze(mask, axis=-1)  # [batch, N_res, N_res, 1]

        # Outgoing [B, N_res//dap_size, N_res, c_z]
        # Incoming [B, N_res, N_res//dap_size, c_z]
        act = self.layer_norm_input(act)  # line 1

        # if not self.training:
        if not self.training and self.global_config.low_memory is True:
            # Note(GuoxiaWang): using inplace version to save memory(low_mem=True).
            left_proj_act = self.left_gate(act)
            left_proj_act.sigmoid_()
            left_proj_act.multiply_(self.left_projection(act))
            left_proj_act.multiply_(mask)

            right_proj_act_before = self.right_gate(act)
            right_proj_act_before.sigmoid_()
            right_proj_act_before.multiply_(self.right_projection(act))
            right_proj_act_before.multiply_(mask)

        else:
            # Outgoing [B, N_res//dap_size, N_res, c_z] => [B, N_res//dap_size, N_res, num_intermediate_channel]
            # Incoming [B, N_res, N_res//dap_size, c_z] => [B, N_res, N_res//dap_size, num_intermediate_channel]
            left_proj_act = mask * self.left_projection(act)
            right_proj_act = mask * self.right_projection(act)

            # Outgoing [B, N_res//dap_size, N_res, c_z] => [B, N_res//dap_size, N_res, num_intermediate_channel]
            # Incoming [B, N_res, N_res//dap_size, c_z] => [B, N_res, N_res//dap_size, num_intermediate_channel]
            left_gate_values = nn.functional.sigmoid(self.left_gate(act))
            right_gate_values = nn.functional.sigmoid(self.right_gate(act))

            # Outgoing [B, N_res//dap_size, N_res, num_intermediate_channel]
            # Incoming [B, N_res, N_res//dap_size, num_intermediate_channel]
            left_proj_act = left_proj_act * left_gate_values
            right_proj_act_before = right_proj_act * right_gate_values

        # "Outgoing" edges equation: 'ikc,jkc->ijc'
        # "Incoming" edges equation: 'kjc,kic->ijc'
        # Note on the Suppl. Alg. 11 & 12 notation:
        # For the "outgoing" edges, a = left_proj_act and b = right_proj_act
        # For the "incoming" edges, it's swapped:
        #   b = left_proj_act and a = right_proj_act

        if self.config.equation == 'ikc,jkc->ijc':
            # Outgoing
            # [B, N_res//dap_size, N_res, num_intermediate_channel] => [B, N_res, N_res, num_intermediate_channel]
            right_proj_act = dap.all_gather(paddle.assign(right_proj_act_before), axis=1)
            # if not self.training:
            if not self.training and self.global_config.low_memory is True:
                del right_proj_act_before
                gc.collect()
        elif self.config.equation == 'kjc,kic->ijc':
            # Incoming
            # [B, N_res, N_res//dap_size, num_intermediate_channel] => [B, N_res, N_res, num_intermediate_channel]
            right_proj_act = dap.all_gather(paddle.assign(right_proj_act_before), axis=2)
            # if not self.training:
            if not self.training and self.global_config.low_memory is True:
                del right_proj_act_before
                gc.collect()
        else:
            raise ValueError('unknown equation.')

        # Outgoing [B, N_res//dap_size, N_res, c_z]
        # Incoming [B, N_res, N_res//dap_size, c_z]        

        # if not self.training:
        if not self.training and self.global_config.low_memory is True:
            gate_values = self.gating_linear(act).sigmoid_()  # line 3
        else:
            gate_values = nn.functional.sigmoid(
                self.gating_linear(act))  # line 3

        if self.config.equation == 'ikc,jkc->ijc':
            # Outgoing
            dim, out_idx = 1, 1
            equation = 'bikc,bjkc->bijc'

            # [B, N_res, N_res, num_intermediate_channel]
            right_proj_act_after = dap.all_gather_opp(paddle.assign(right_proj_act), axis=1)
        elif self.config.equation == 'kjc,kic->ijc':
            # Incoming
            dim, out_idx = 2, 2
            equation = 'bkjc,bkic->bijc'

            # [B, N_res, N_res, num_intermediate_channel]
            right_proj_act_after = dap.all_gather_opp(paddle.assign(right_proj_act), axis=2)
        else:
            raise ValueError('unknown equation.')

        if not self.training:
            einsum_fn = subbatch(paddle.einsum, [1], [dim],
                                 self.global_config.subbatch_size, out_idx)
            act = einsum_fn(equation, left_proj_act, right_proj_act_after)
        else:
            # Outgoing equation = 'bikc,bjkc->bijc'
            # [B, N_res//dap_size, N_res, num_intermediate_channel], [B, N_res, N_res, num_intermediate_channel]
            # => [B, N_res//dap_size, N_res, num_intermediate_channel]

            # Incoming equation = 'bkjc,bkic->bijc'
            # [B, N_res, N_res//dap_size, num_intermediate_channel], [B, N_res, N_res, num_intermediate_channel]
            # => [B, N_res, N_res//dap_size, num_intermediate_channel]
            act = paddle.einsum(equation, left_proj_act, right_proj_act_after)

        act = self.center_layer_norm(act)
        act = self.output_projection(act)

        if not self.training and self.global_config.low_memory is True:
            act.multiply_(gate_values)
            del gate_values, left_proj_act, right_proj_act_after
            gc.collect()
        else:
            act = act * gate_values

        return act
