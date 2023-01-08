from paddle import nn
import paddle
from tools import dap
import numpy as np
import functools
import numbers
import collections


def set_tensor_constant(tensor, constant):
    tensor.set_value(paddle.full_like(tensor, constant))


def init_gate_linear(linear):
    set_tensor_constant(linear.weight, 0)
    set_tensor_constant(linear.bias, 1)


def init_final_linear(linear):
    set_tensor_constant(linear.weight, 0)

# alternative way to reduce memory cost during Evoformer
def subbatch(f, arg_idx, dim, bs, out_idx):
    """ Converts a function to one that applies to subbatch of an input
    dimension.

    Args:
        f(Callable): original function.
        arg_idx([int]): indices of the inputs to be subbatched.
        dim([int]): index of the dimension to be subbatched.
        bs(int): subbatch size.
        out_idx(int): index of the output dimension that needs stacking

    Returns:
        converted function.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        assert len(arg_idx) == len(dim), f'Number of batching args and number of batching dims should match.'

        inps = [args[i] for i in arg_idx]
        dim_width = [inp.shape[d] for inp, d in zip(inps, dim)]
        assert len(set(dim_width)) == 1, f'Batch sizes should be kept equal.'

        inp_dim = {inp: d for inp, d in zip(inps, dim)}

        dim_width = dim_width[0]
        if dim_width < bs:
            return f(*args, **kwargs)

        outs = []
        for slice_at in np.arange(0, dim_width, bs):
            _args = []
            for i, inp in enumerate(args):
                if i in arg_idx:
                    inp = inp.slice([inp_dim[inp]], [slice_at], [slice_at + bs])
                _args.append(inp)
            outs.append(f(*_args, **kwargs))

        return paddle.concat(outs, out_idx)

    return wrapper


def mask_mean(mask:paddle.Tensor, value:paddle.Tensor, axis=None, drop_mask_channel=False, eps=1e-10):
    if drop_mask_channel:
        mask = mask[:, 0]

    mask_shape = mask.shape
    value_shape = value.shape
    assert len(mask_shape) == len(value_shape)

    if isinstance(axis, numbers.Integral):
        axis = [axis]
    elif axis is None:
        axis = list(range(len(mask_shape)))

    assert isinstance(axis, collections.Iterable), \
        'axis needs to be either an iterable, integer or "None"'

    broadcast_factor = 1.
    for axis_ in axis:
        value_size = value_shape[axis_]
        mask_size = mask_shape[axis_]
        if mask_size == 1:
            broadcast_factor *= value_size
        else:
            assert mask_size == value_size

    return (paddle.sum(mask * value, axis=axis) /
            (paddle.sum(mask, axis=axis) * broadcast_factor + eps))


def set_tensor_constant(tensor:paddle.Tensor, constant):
    tensor.set_value(paddle.full_like(tensor, constant))


def init_gate_linear(linear:nn.Linear):
    set_tensor_constant(linear.weight, 0)
    set_tensor_constant(linear.bias, 1)


class Attention(nn.Layer):
    """Multihead attention."""

    def __init__(self, config, global_config, q_dim, kv_dim, output_dim):
        super(Attention, self).__init__()
        self.config = config
        self.global_config = global_config

        num_head = self.config.num_head
        key_dim = self.config.get('key_dim', q_dim)
        value_dim = self.config.get('value_dim', kv_dim)

        assert key_dim % num_head == 0
        assert value_dim % num_head == 0
        key_dim = key_dim // num_head
        value_dim = value_dim // num_head

        self.key_dim = key_dim
        self.value_dim = value_dim

        self.query_w = paddle.create_parameter(
            [q_dim, num_head, key_dim], 'float32',
            default_initializer=nn.initializer.XavierUniform())
        self.key_w = paddle.create_parameter(
            [kv_dim, num_head, key_dim], 'float32',
            default_initializer=nn.initializer.XavierUniform())
        self.value_w = paddle.create_parameter(
            [kv_dim, num_head, value_dim], 'float32',
            default_initializer=nn.initializer.XavierUniform())

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
        c = self.key_dim ** (-0.5)
        q = paddle.einsum('nbqa,ahc->nbqhc', q_data, self.query_w) * c
        # q_data [1,48,5120,64]
        # self.query_w [64, 8, 8]
        k = paddle.einsum('nbka,ahc->nbkhc', m_data, self.key_w)
        v = paddle.einsum('nbka,ahc->nbkhc', m_data, self.value_w)
        logits = paddle.einsum('nbqhc,nbkhc->nbhqk', q, k) + bias # segment fault when input following test samples
        # q [1, 48, 5120, 8, 8]
        # k [1, 48, 5120, 8, 8]
        # bias [1, 48, 1, 1, 5120]
        
        if nonbatched_bias is not None:
            nonbatched_bias_after = dap.all_gather_opp(nonbatched_bias, axis=2)
            logits += paddle.unsqueeze(nonbatched_bias_after, axis=1)

        weights = nn.functional.softmax(logits)

        # by paddlepaddle team
        if weights.shape[-1] != v.shape[2]:
            v = paddle.tile(v, [1,1,weights.shape[-1], 1, 1])
        weighted_avg = paddle.einsum('nbhqk,nbkhc->nbqhc', weights, v)

        if self.config.gating:
            gate_values = paddle.einsum('nbqc,chv->nbqhv', q_data,
                                        self.gating_w) + self.gating_b
            gate_values = nn.functional.sigmoid(gate_values)
            weighted_avg *= gate_values

        output = paddle.einsum('nbqhc,hco->nbqo', weighted_avg,
                               self.output_w) + self.output_b
        return output


class GlobalAttention(nn.Layer):
    """Global attention.

    Jumper et al. (2021) Suppl. Alg. 19 "MSAColumnGlobalAttention" lines 2-7
    """

    def __init__(self, config, global_config, q_dim, kv_dim, output_dim):
        super(GlobalAttention, self).__init__()
        self.config = config
        self.global_config = global_config

        num_head = self.config.num_head
        key_dim = self.config.get('key_dim', q_dim)
        value_dim = self.config.get('value_dim', kv_dim)

        assert key_dim % num_head == 0
        assert value_dim % num_head == 0
        key_dim = key_dim // num_head
        value_dim = value_dim // num_head

        self.key_dim = key_dim
        self.value_dim = value_dim

        self.query_w = paddle.create_parameter(
            [q_dim, num_head, key_dim], 'float32',
            default_initializer=nn.initializer.XavierUniform())
        self.key_w = paddle.create_parameter(
            [kv_dim, key_dim], 'float32',
            default_initializer=nn.initializer.XavierUniform())
        self.value_w = paddle.create_parameter(
            [kv_dim, value_dim], 'float32',
            default_initializer=nn.initializer.XavierUniform())

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

    def forward(self, q_data, m_data, q_mask):
        k = paddle.einsum('nbka,ac->nbkc', m_data, self.key_w)
        v = paddle.einsum('nbka,ac->nbkc', m_data, self.value_w)

        # NOTE: differ from non-global version using q_avg for attn
        q_avg = mask_mean(q_mask, q_data, axis=2)
        c = self.key_dim ** (-0.5)
        q = paddle.einsum('nba,ahc->nbhc', q_avg, self.query_w) * c

        q_mask_ = paddle.unsqueeze(q_mask, axis=2)[..., 0]
        bias = 1e9 * (q_mask_ - 1.)

        logits = paddle.einsum('nbhc,nbkc->nbhk', q, k) + bias
        weights = nn.functional.softmax(logits)
        weighted_avg = paddle.einsum('nbhk,nbkc->nbhc', weights, v)

        if self.config.gating:
            gate_values = paddle.einsum('nbqc,chv->nbqhv', q_data,
                                        self.gating_w) + self.gating_b
            gate_values = nn.functional.sigmoid(gate_values)
            weighted_avg = paddle.unsqueeze(weighted_avg, axis=2)
            weighted_avg *= gate_values

            output = paddle.einsum('nbqhc,hco->nbqo', weighted_avg,
                                   self.output_w) + self.output_b
        else:
            output = paddle.einsum('nbhc,hco->nbo', weighted_avg,
                                   self.output_w) + self.output_b
            output = paddle.unsqueeze(output, axis=-1)

        return output


class MSAColumnAttention(nn.Layer):
    """MSA per-column attention.

    Jumper et al. (2021) Suppl. Alg. 8 "MSAColumnAttention"
    """

    def __init__(self, channel_num, config, global_config):
        super(MSAColumnAttention, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config
        self.bs = self.global_config.subbatch_size # 48
        assert config.orientation == 'per_column'

        msa_channel = channel_num['msa_channel']
        self.query_norm = nn.LayerNorm(msa_channel)
        self.attention = Attention(
            self.config, self.global_config,
            msa_channel, msa_channel, msa_channel)


    def subbatch_attention(self, q_mat:paddle.Tensor, m_mat:paddle.Tensor, bias:paddle.Tensor):
        arg_idx = [0,1,2]
        dim = [1,1,1]
        out_idx = 1
        #inps = [args[i] for i in arg_idx]
        inps = [q_mat, m_mat, bias]
        dim_width = [inp.shape[d] for inp, d in zip(inps, dim)]
        inp_dim = {inp: d for inp, d in zip(inps, dim)}
        dim_width = dim_width[0]
        if dim_width < self.bs:
            return self.attention(q_mat, m_mat, bias)

        outs = []
        for slice_at in np.arange(0, dim_width, self.bs): # use np.arange to escape the warning: for-range when cvt to static graph
            _args = []
            for i, inp in enumerate(inps):
                if i in arg_idx:
                    inp = inp.slice([inp_dim[inp]], [slice_at], [slice_at + self.bs])
                _args.append(inp)
            outs.append(self.attention(_args[0], _args[1], _args[2]))

        return paddle.concat(outs, out_idx)


    def forward(self, msa_act, msa_mask):
        # scatter if using dap, otherwise do nothing
        # [B, N_seq, N_res] => [B, N_seq, N_res//dap_size]
        msa_mask = dap.scatter(msa_mask, axis=2)
        msa_act = paddle.transpose(msa_act, [0, 2, 1, 3])
        msa_mask = paddle.transpose(msa_mask, [0, 2, 1])
        bias = 1e9 * (msa_mask - 1.)
        bias = paddle.unsqueeze(bias, axis=[2, 3])
        msa_act = self.query_norm(msa_act)
        
        msa_act = self.subbatch_attention(msa_act, msa_act, bias)
        # unit = self.bs
        # n_inps = msa_act.shape[0]
        # if msa_act.shape[1] < unit:
        #     msa_act = self.attention(msa_act, msa_act, bias)
        # else:
        #     for i_inp in range(n_inps):
        #         for i in range(msa_act.shape[1] // unit):
        #             q_sub_data = paddle.unsqueeze(msa_act[i_inp, unit*i:unit*(i+1)], axis=0)
        #             bias_sub = paddle.unsqueeze(bias[i_inp, unit*i:unit*(i+1)], axis=0)
        #             msa_act[i_inp, unit*i:unit*(i+1)] = self.attention(q_sub_data, q_sub_data, bias_sub)

        msa_act = paddle.transpose(msa_act, [0, 2, 1, 3])
        return msa_act


class Transition(nn.Layer):
    """Transition layer.

    Jumper et al. (2021) Suppl. Alg. 9 "MSATransition"
    Jumper et al. (2021) Suppl. Alg. 15 "PairTransition"
    """

    def __init__(self, channel_num, config, global_config, is_extra_msa,
                 transition_type):
        super(Transition, self).__init__()
        assert transition_type in ['msa_transition', 'pair_transition']
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config
        self.is_extra_msa = is_extra_msa
        self.transition_type = transition_type
        self.bs = self.global_config.subbatch_size # 48

        if transition_type == 'msa_transition' and is_extra_msa:
            in_dim = channel_num['extra_msa_channel']
        elif transition_type == 'msa_transition' and not is_extra_msa:
            in_dim = channel_num['msa_channel']
        elif transition_type == 'pair_transition':
            in_dim = channel_num['pair_channel']

        self.input_layer_norm = nn.LayerNorm(in_dim)
        self.transition1 = nn.Linear(
            in_dim, int(in_dim * self.config.num_intermediate_factor),
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.KaimingNormal()))

        if self.global_config.zero_init:
            last_init = nn.initializer.Constant(0.0)
        else:
            last_init = nn.initializer.TruncatedNormal()

        self.transition2 = nn.Linear(
            int(in_dim * self.config.num_intermediate_factor), in_dim,
            weight_attr=paddle.ParamAttr(initializer=last_init))
    
    def subbatch_transition(self, act:paddle.Tensor):
        arg_idx = [0]
        dim = [1]
        out_idx = 1
        #inps = [args[i] for i in arg_idx]
        inps = [act]
        dim_width = [inp.shape[d] for inp, d in zip(inps, dim)]
        inp_dim = {inp: d for inp, d in zip(inps, dim)}
        dim_width = dim_width[0]
        if dim_width < self.bs:
            return self.transition_module(act)
        outs = []
        for slice_at in np.arange(0, dim_width, self.bs): # use np.arange to escape the warning: for-range when cvt to static graph
            _args = []
            for i, inp in enumerate(inps):
                if i in arg_idx:
                    inp = inp.slice([inp_dim[inp]], [slice_at], [slice_at + self.bs])
                _args.append(inp)
            outs.append(self.transition_module(_args[0]))

        return paddle.concat(outs, out_idx)

    def transition_module(self, x):
            x = self.transition1(x)
            x = nn.functional.relu(x)
            x = self.transition2(x)
            return x

    def forward(self, act):  # edit by zjh@intel SMG 20220825
        act = self.input_layer_norm(act)

        # act = self.subbatch_transition(act) # [TODO] change slice appendage to slice on-site
        dim_width = act.shape[1]
        if dim_width < self.bs:
            act = self.transition_module(act)
        else:
            for i in np.arange(0, dim_width, self.bs): # use np.arange to escape the warning: for-range when cvt to static graph
                act[:, i:(i + self.bs)] = self.transition_module(act[:, i:(i + self.bs)])
        return act


class MSARowAttentionWithPairBias(nn.Layer):
    """MSA per-row attention biased by the pair representation.

    Jumper et al. (2021) Suppl. Alg. 7 "MSARowAttentionWithPairBias"
    """

    def __init__(self, channel_num, config, global_config, is_extra_msa):
        super(MSARowAttentionWithPairBias, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config
        self.is_extra_msa = is_extra_msa
        self.bs = self.global_config.subbatch_size
        assert config.orientation == 'per_row'

        if is_extra_msa:
            self.query_norm = nn.LayerNorm(channel_num['extra_msa_channel'])
        else:
            self.query_norm = nn.LayerNorm(channel_num['msa_channel'])

        self.feat_2d_norm = nn.LayerNorm(channel_num['pair_channel'])
        self.feat_2d_weights = paddle.create_parameter(
            [channel_num['pair_channel'], self.config.num_head], 'float32',
            default_initializer=nn.initializer.Normal(
                std=1. / np.sqrt(channel_num['pair_channel'])))

        if is_extra_msa:
            extra_msa_channel = channel_num['extra_msa_channel']
            self.attention = Attention(
                self.config, self.global_config,
                extra_msa_channel, extra_msa_channel, extra_msa_channel)
        else:
            msa_channel = channel_num['msa_channel']
            self.attention = Attention(
                self.config, self.global_config,
                msa_channel, msa_channel, msa_channel)

    def subbatch_attention(self, 
        q_mat:paddle.Tensor, 
        m_mat:paddle.Tensor, 
        bias:paddle.Tensor, 
        nonbatched_bias:paddle.Tensor
    ):
        arg_idx = [0,1,2]
        dim = [1,1,1]
        out_idx = 1
        #inps = [args[i] for i in arg_idx]
        inps = [q_mat, m_mat, bias]
        dim_width = [inp.shape[d] for inp, d in zip(inps, dim)]
        inp_dim = {inp: d for inp, d in zip(inps, dim)}
        dim_width = dim_width[0]
        if dim_width < self.bs:
            return self.attention(q_mat, m_mat, bias)

        outs = []
        for slice_at in np.arange(0, dim_width, self.bs): # use np.arange to escape the warning: for-range when cvt to static graph
            _args = []
            for i, inp in enumerate(inps):
                if i in arg_idx:
                    inp = inp.slice([inp_dim[inp]], [slice_at], [slice_at + self.bs])
                _args.append(inp)
            outs.append(self.attention(_args[0], _args[1], _args[2], nonbatched_bias))

        return paddle.concat(outs, out_idx)
    
    def forward(self, msa_act, msa_mask, pair_act):

        pair_act = self.feat_2d_norm(pair_act)
        
        # [B, N_res//dap_size, N_res, cz], [cz, head] => [B, head, N_res//dap_size, N_res]
        nonbatched_bias_before = paddle.einsum(
            'nqkc,ch->nhqk', pair_act, self.feat_2d_weights)
        
        # [B, head, N_res//dap_size, N_res] => [B, head, N_res, N_res]
        nonbatched_bias = dap.all_gather(nonbatched_bias_before, axis=2)

        # [B, N_seq, N_res] => [B, N_seq//dap_size, N_res]
        msa_mask = dap.scatter(msa_mask, axis=1)
        

        bias = 1e9 * (msa_mask - 1.)
        # [B, N_seq//dap_size, N_res] => [B, N_seq//dap_size, 1, 1, N_res]
        bias = paddle.unsqueeze(bias, axis=[2, 3])
        msa_act = self.query_norm(msa_act)

        # if not self.training:
        # low memory mode using subbatch
        # msa_act = self.subbatch_attention(msa_act, msa_act, bias, nonbatched_bias)
        
        unit = self.bs
        n_inps = msa_act.shape[0]
        if msa_act.shape[1] < unit:
            msa_act = self.attention(msa_act, msa_act, bias, nonbatched_bias)
        else:
            for i_inp in range(n_inps):
                for i in range(msa_act.shape[1] // unit):
                    q_sub_data = paddle.unsqueeze(msa_act[i_inp, unit*i:unit*(i+1)], axis=0)
                    bias_sub = paddle.unsqueeze(bias[i_inp, unit*i:unit*(i+1)], axis=0)
                    msa_act[i_inp, unit*i:unit*(i+1)] = self.attention(
                        q_sub_data, q_sub_data, bias_sub, nonbatched_bias)

        # msa_act = self.sliced_attention(msa_act, msa_act, bias, nonbatched_bias)
        # else:
        #     msa_act = self.attention(msa_act, msa_act, bias, nonbatched_bias)

        return msa_act


class MSAColumnGlobalAttention(nn.Layer):
    """MSA per-column global attention.

    Jumper et al. (2021) Suppl. Alg. 19 "MSAColumnGlobalAttention"
    """

    def __init__(self, channel_num, config, global_config):
        super(MSAColumnGlobalAttention, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config
        self.bs = self.global_config.subbatch_size
        assert config.orientation == 'per_column'

        extra_msa_channel = channel_num['extra_msa_channel']
        self.query_norm = nn.LayerNorm(extra_msa_channel)
        self.attention = GlobalAttention(
            self.config, self.global_config,
            extra_msa_channel, extra_msa_channel, extra_msa_channel)

    def subbatch_attention(self, msa_act1:paddle.Tensor, msa_act2:paddle.Tensor, msa_mask:paddle.Tensor):
        arg_idx = [0,1,2]
        dim = [1,1,1]
        out_idx = 1
        #inps = [args[i] for i in arg_idx]
        inps = [msa_act1, msa_act2, msa_mask]
        dim_width = [inp.shape[d] for inp, d in zip(inps, dim)]
        inp_dim = {inp: d for inp, d in zip(inps, dim)}
        dim_width = dim_width[0]
        if dim_width < self.bs:
            return self.attention(msa_act1, msa_act2, msa_mask)

        outs = []
        for slice_at in np.arange(0, dim_width, self.bs): # use np.arange to escape the warning: for-range when cvt to static graph
            _args = []
            for i, inp in enumerate(inps):
                if i in arg_idx:
                    inp = inp.slice([inp_dim[inp]], [slice_at], [slice_at + self.bs])
                _args.append(inp)
            outs.append(self.attention(_args[0], _args[1], _args[2]))

        return paddle.concat(outs, out_idx)

    def forward(self, msa_act, msa_mask):
        # scatter if using dap, otherwise do nothing
        # [B, N_seq, N_res] => [B, N_seq, N_res//dap_size]
        msa_mask = dap.scatter(msa_mask, axis=2)

        msa_act = paddle.transpose(msa_act, [0, 2, 1, 3])
        msa_mask = paddle.transpose(msa_mask, [0, 2, 1])

        bias = 1e9 * (msa_mask - 1.)
        bias = paddle.unsqueeze(bias, axis=[2, 3])

        msa_mask = paddle.unsqueeze(msa_mask, axis=-1)
        msa_act = self.query_norm(msa_act)

        if not self.training:
            # low memory mode using subbatch
            # sb_attn = subbatch(self.attention, [0, 1, 2], [1, 1, 1],
            #                    self.global_config.subbatch_size, 1)
            # msa_act = sb_attn(msa_act, msa_act, msa_mask)
            msa_act = self.subbatch_attention(msa_act, msa_act, msa_mask)
        else:
            msa_act = self.attention(msa_act, msa_act, msa_mask)

        msa_act = paddle.transpose(msa_act, [0, 2, 1, 3])
        return msa_act


class OuterProductMean(nn.Layer):
    """Computes mean outer product.

    Jumper et al. (2021) Suppl. Alg. 10 "OuterProductMean"
    """

    def __init__(self, channel_num, config, global_config, is_extra_msa, name='outer_product_mean'):
        super(OuterProductMean, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        if is_extra_msa:
            c_m = channel_num['extra_msa_channel']
        else:
            c_m = channel_num['msa_channel']

        self.layer_norm_input = nn.LayerNorm(c_m, name='layer_norm_input')
        self.left_projection = nn.Linear(
            c_m, self.config.num_outer_channel, name='left_projection')
        self.right_projection = nn.Linear(
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

    def compute_chunk(self, left_act, right_act):
        # This is equivalent to
        #
        # act = jnp.einsum('abc,ade->dceb', left_act, right_act)
        # act = jnp.einsum('dceb,cef->bdf', act, output_w) + output_b
        #
        # but faster. maybe for subbatch inference?
            
        # [B, N_seq, N_res//dap_size, num_outer_channel] => [B, N_seq, num_outer_channel, N_res//dap_size]
        left_act = left_act.transpose([0, 1, 3, 2])
        # wait if using async communication and dap, otherwise do nothing
        right_act_after = dap.all_gather_opp(right_act, axis=2)
        # [B, N_seq, num_outer_channel, N_res//dap_size], [B, N_seq, N_res, num_outer_channel]
        # => [B, N_res, num_outer_channel, num_outer_channel, N_res//dap_size]
        act = paddle.einsum('nacb,nade->ndceb', left_act, right_act_after)
        # [B, N_res, num_outer_channel, num_outer_channel, N_res//dap_size], [num_outer_channel, num_outer_channel, c_z]
        # => [B, N_res, N_res//dap_size, c_z]
        act = paddle.einsum('ndceb,cef->ndbf', act, self.output_w) + self.output_b
        # [B, N_res, N_res//dap_size, c_z] => [B, N_res//dap_size, N_res, c_z]
        return act.transpose([0, 2, 1, 3])

    def forward(self, act, mask):
        """Builds OuterProductMean module.

        Arguments:
        act: MSA representation, shape [batch, N_seq, N_res, c_m].
        mask: MSA mask, shape [batch, N_seq, N_res].

        Returns:
        Update to pair representation, shape [batch, N_res, N_res, c_z].
        """
        # [B, N_seq, N_res//dap_size, c_m]
        act = self.layer_norm_input(act)
        # [B, N_seq, N_res//dap_size, c_m] => [B, N_seq, N_res//dap_size, num_outer_channel]
        right_act_before = self.right_projection(act)
        # [B, N_seq, N_res//dap_size, num_outer_channel] => [B, N_seq, N_res, num_outer_channel]
        right_act = dap.all_gather(right_act_before, axis=2)
        
        # [B, N_seq, N_res//dap_size, c_m] => [B, N_seq, N_res//dap_size, num_outer_channel]
        left_act = self.left_projection(act)
        # [B, N_seq, N_res] => [B, N_seq, N_res, 1]
        mask = paddle.unsqueeze(mask, axis=-1)
        # [B, N_seq, N_res, 1] => [B, N_seq, N_res//dap_size, 1]
        mask_col = dap.scatter(mask, axis=2)
        left_act = mask_col * left_act
        
        # [B, N_seq, N_res//dap_size, 1], [B, N_seq, N_res, 1] => [B, N_res//dap_size, N_res, 1]
        epsilon = 1e-3
        norm = paddle.einsum('nabc,nadc->nbdc', mask_col, mask) + epsilon

        

        # if not self.training:
        #     # low memory mode using subbatch
        #     sb_chunk = subbatch(self.compute_chunk, [0], [2],
        #                        self.config.chunk_size, 1)
        #     act = sb_chunk(left_act, right_act)
        # else:
        act = self.compute_chunk(left_act, right_act)

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
        self.bs = self.global_config.subbatch_size

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

    def subbatch_attention(self, q_mat:paddle.Tensor, m_mat:paddle.Tensor, bias:paddle.Tensor, nonbatched_bias:paddle.Tensor):
        arg_idx = [0,1,2]
        dim = [1,1,1]
        out_idx = 1
        #inps = [args[i] for i in arg_idx]
        inps = [q_mat, m_mat, bias]
        dim_width = [inp.shape[d] for inp, d in zip(inps, dim)]
        inp_dim = {inp: d for inp, d in zip(inps, dim)}
        dim_width = dim_width[0]
        if dim_width < self.bs:
            return self.attention(q_mat, m_mat, bias)

        outs = []
        for slice_at in np.arange(0, dim_width, self.bs): # use np.arange to escape the warning: for-range when cvt to static graph
            _args = []
            for i, inp in enumerate(inps):
                if i in arg_idx:
                    inp = inp.slice([inp_dim[inp]], [slice_at], [slice_at + self.bs])
                _args.append(inp)
            outs.append(self.attention(_args[0], _args[1], _args[2], nonbatched_bias))

        return paddle.concat(outs, out_idx)

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

        # [B, N_res//dap_size, N_res]
        bias = 1e9 * (pair_mask - 1.)
        # [B, N_res//dap_size, 1, 1, N_res]
        bias = paddle.unsqueeze(bias, axis=[2, 3])

        pair_act = self.query_norm(pair_act)

        # [B, N_res//dap_size, N_res, cz], [cz, head] => [B, head, N_res//dap_size, N_res]
        nonbatched_bias_before = paddle.einsum('bqkc,ch->bhqk', pair_act, self.feat_2d_weights)
        
        # # [B, head, N_res//dap_size, N_res] => [B, head, N_res, N_res]
        nonbatched_bias = dap.all_gather(nonbatched_bias_before, axis=2)

        # pair_act = self.subbatch_attention(pair_act, pair_act, bias, nonbatched_bias)

        unit = self.bs
        n_inps = pair_act.shape[0]
        for i_inp in range(n_inps):
            for i in range(pair_act.shape[1] // unit):
                q_sub_data = paddle.unsqueeze(pair_act[i_inp, unit*i:unit*(i+1)], axis=0)
                bias_sub = paddle.unsqueeze(bias[i_inp, unit*i:unit*(i+1)], axis=0)
                pair_act[i_inp, unit*i:unit*(i+1)] = self.attention(
                    q_sub_data, q_sub_data, bias_sub, nonbatched_bias)
        
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

        self.layer_norm_input = nn.LayerNorm(self.channel_num['pair_channel'], name='layer_norm_input')
        self.left_projection = nn.Linear(self.channel_num['pair_channel'],
                                self.config.num_intermediate_channel, name='left_projection')
        self.right_projection = nn.Linear(self.channel_num['pair_channel'],
                                self.config.num_intermediate_channel, name='right_projection')
        self.left_gate = nn.Linear(self.channel_num['pair_channel'],
                                self.config.num_intermediate_channel, name='left_gate')
        init_gate_linear(self.left_gate)
        self.right_gate = nn.Linear(self.channel_num['pair_channel'],
                                self.config.num_intermediate_channel, name='right_gate')
        init_gate_linear(self.right_gate)

        # line 4
        self.center_layer_norm = nn.LayerNorm(self.config.num_intermediate_channel, name='center_layer_norm')
        self.output_projection = nn.Linear(self.config.num_intermediate_channel,
                                    self.channel_num['pair_channel'], name='output_projection')
        init_final_linear(self.output_projection)
        # line 3
        self.gating_linear = nn.Linear(self.channel_num['pair_channel'],
                                    self.channel_num['pair_channel'], name='output_projection')
        init_gate_linear(self.gating_linear)

    def forward(self, act, mask):
        """Builds TriangleMultiplication module.

        Arguments:
        act: Pair activations, shape [batch, N_res, N_res, c_z]
        mask: Pair mask, shape [batch, N_res, N_res].

        Returns:
        Outputs, same shape/type as act.
        """
        # Outgoing [batch, N_res//dap_size, N_res] => [batch, N_res//dap_size, N_res, 1]
        # Incoming [batch, N_res, N_res//dap_size] => [batch, N_res, N_res//dap_size, 1] 
        mask = paddle.unsqueeze(mask, axis=-1) # [batch, N_res, N_res, 1]

        # Outgoing [B, N_res//dap_size, N_res, c_z]
        # Incoming [B, N_res, N_res//dap_size, c_z]
        act = self.layer_norm_input(act) # line 1

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
            right_proj_act = dap.all_gather(right_proj_act_before, axis=1)
        elif  self.config.equation == 'kjc,kic->ijc':
            # Incoming
            # [B, N_res, N_res//dap_size, num_intermediate_channel] => [B, N_res, N_res, num_intermediate_channel]
            right_proj_act = dap.all_gather(right_proj_act_before, axis=2)
        else:
            raise ValueError('unknown equation.')
        
        
        # Outgoing [B, N_res//dap_size, N_res, c_z]
        # Incoming [B, N_res, N_res//dap_size, c_z]        
        gate_values = nn.functional.sigmoid(self.gating_linear(act)) # line 3

        if self.config.equation == 'ikc,jkc->ijc':
            # Outgoing
            dim, out_idx = 1, 1
            equation = 'bikc,bjkc->bijc'
            
            # [B, N_res, N_res, num_intermediate_channel]
            right_proj_act_after = dap.all_gather_opp(right_proj_act, axis=1)
        elif  self.config.equation == 'kjc,kic->ijc':
            # Incoming
            dim, out_idx = 2, 2
            equation = 'bkjc,bkic->bijc'
            
            # [B, N_res, N_res, num_intermediate_channel]
            right_proj_act_after = dap.all_gather_opp(right_proj_act, axis=2)
        else:
            raise ValueError('unknown equation.')

        # if not self.training:
        #     einsum_fn = subbatch(paddle.einsum, [1], [dim], self.global_config.subbatch_size, out_idx)
        #     act = einsum_fn(equation, left_proj_act, right_proj_act_after)
        # else:
            # Outgoing equation = 'bikc,bjkc->bijc'
            # [B, N_res//dap_size, N_res, num_intermediate_channel], [B, N_res, N_res, num_intermediate_channel]
            # => [B, N_res//dap_size, N_res, num_intermediate_channel]
            
            # Incoming equation = 'bkjc,bkic->bijc'
            # [B, N_res, N_res//dap_size, num_intermediate_channel], [B, N_res, N_res, num_intermediate_channel]
            # => [B, N_res, N_res//dap_size, num_intermediate_channel]
        act = paddle.einsum(equation, left_proj_act, right_proj_act_after)

        act = self.center_layer_norm(act)
        act = self.output_projection(act)

        act = act * gate_values

        return act


def dgram_from_positions(positions, num_bins, min_bin, max_bin):
    lower_breaks = paddle.linspace(min_bin, max_bin, num_bins)
    lower_breaks = paddle.square(lower_breaks)
    upper_breaks = paddle.concat([lower_breaks[1:],
                                    paddle.to_tensor([1e8], dtype='float32')])

    def _squared_difference(x, y):
        return paddle.square(x - y)

    dist2 = paddle.sum(
        _squared_difference(
            paddle.unsqueeze(positions, axis=-2),
            paddle.unsqueeze(positions, axis=-3)),
        axis=-1, keepdim=True)

    dgram = ((dist2 > lower_breaks).astype('float32') *
                (dist2 < upper_breaks).astype('float32'))
    return dgram
