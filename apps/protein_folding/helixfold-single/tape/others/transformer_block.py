import copy
import collections
import math
import numpy as np

import paddle
from paddle.nn.layer.common import Linear, Dropout
from paddle.nn.layer.norm import LayerNorm
from paddle.nn import functional as F
from paddle import tensor
from paddle.fluid import layers
from paddle.fluid.dygraph import Layer, LayerList
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.data_feeder import convert_dtype
from paddle.distributed.fleet.utils import recompute
from paddle import nn


__all__ = []

def recompute_wrapper(func, *args, is_recompute=True):
    """Function wrapper for recompute"""
    if is_recompute:
        return recompute(func, *args)
    else:
        return func(*args)

def _convert_param_attr_to_list(param_attr, n):
    if isinstance(param_attr, (list, tuple)):
        assert len(param_attr) == n, (
            "length of param_attr should be %d when it is a list/tuple" % n)
        param_attrs = []
        for attr in param_attr:
            if isinstance(attr, bool):
                if attr:
                    param_attrs.append(ParamAttr._to_attr(None))
                else:
                    param_attrs.append(False)
            else:
                param_attrs.append(ParamAttr._to_attr(attr))
        # param_attrs = [ParamAttr._to_attr(attr) for attr in param_attr]
    elif isinstance(param_attr, bool):
        param_attrs = []
        if param_attr:
            param_attrs = [ParamAttr._to_attr(None) for i in range(n)]
        else:
            param_attrs = [False] * n
    else:
        param_attrs = []
        attr = ParamAttr._to_attr(param_attr)
        for i in range(n):
            attr_i = copy.deepcopy(attr)
            if attr.name:
                attr_i.name = attr_i.name + "_" + str(i)
            param_attrs.append(attr_i)
    return param_attrs


def _convert_attention_mask(attn_mask, dtype):
    if attn_mask is not None and attn_mask.dtype != dtype:
        attn_mask_dtype = convert_dtype(attn_mask.dtype)
        if attn_mask_dtype == 'bool' or 'int' in attn_mask_dtype:
            attn_mask = (paddle.cast(attn_mask, dtype) - 1.0) * 1e9
        else:
            attn_mask = paddle.cast(attn_mask, dtype)
    return attn_mask


class MultiHeadAttention(Layer):
    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False,
                 weight_attr=None,
                 bias_attr=None):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.k_proj = Linear(
            self.kdim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.v_proj = Linear(
            self.vdim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.out_proj = Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)

    def _prepare_qkv(self, query, key, value, cache=None):
        q = self.q_proj(query)
        q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = tensor.transpose(x=q, perm=[0, 2, 1, 3])

        if isinstance(cache, self.StaticCache):
            # for encoder-decoder attention in inference and has cached
            k, v = cache.k, cache.v
        else:
            k, v = self.compute_kv(key, value)

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = tensor.concat([cache.k, k], axis=2)
            v = tensor.concat([cache.v, v], axis=2)
            cache = self.Cache(k, v)

        return (q, k, v) if cache is None else (q, k, v, cache)

    def compute_kv(self, key, value):
        k = self.k_proj(key)
        v = self.v_proj(value)
        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])
        return k, v

    def gen_cache(self, key, value=None, type=Cache):
        if type == MultiHeadAttention.StaticCache:  # static_kv
            k, v = self.compute_kv(key, value)
            return self.StaticCache(k, v)
        elif value is None:  # incremental_state
            k = layers.fill_constant_batch_size_like(
                input=key,
                shape=[-1, self.num_heads, 0, self.head_dim],
                dtype=key.dtype,
                value=0)
            v = layers.fill_constant_batch_size_like(
                input=key,
                shape=[-1, self.num_heads, 0, self.head_dim],
                dtype=key.dtype,
                value=0)
            return self.Cache(k, v)
        else:
            # incremental_state with initial value, mainly for usage like UniLM
            return self.Cache(key, value)

    def forward(self, query, key=None, value=None, attn_mask=None, cache=None):
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        if cache is None:
            q, k, v = self._prepare_qkv(query, key, value, cache)
        else:
            q, k, v, cache = self._prepare_qkv(query, key, value, cache)

        # scale dot product attention
        # TODO(guosheng): use tensor.matmul, however it doesn't support `alpha`
        product = layers.matmul(
            x=q, y=k, transpose_y=True, alpha=self.head_dim**-0.5)
        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask
        weights = F.softmax(product)
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode="upscale_in_train")

        out = tensor.matmul(weights, v)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if cache is not None:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)


class TransformerEncoderLayer(Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerEncoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 2)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 2)

        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0])
        self.linear1 = Linear(
            d_model, dim_feedforward, weight_attrs[1], bias_attr=bias_attrs[1])
        self.dropout = Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = Linear(
            dim_feedforward, d_model, weight_attrs[1], bias_attr=bias_attrs[1])
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)

    def forward(self, src, src_mask=None, cache=None):
        src_mask = _convert_attention_mask(src_mask, src.dtype)

        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        # Add cache for encoder for the usage like UniLM
        if cache is None:
            src = self.self_attn(src, src, src, src_mask)
        else:
            src, incremental_cache = self.self_attn(src, src, src, src_mask,
                                                    cache)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src if cache is None else (src, incremental_cache)

    def gen_cache(self, src):
        incremental_cache = self.self_attn.gen_cache(
            src, type=self.self_attn.Cache)
        return incremental_cache


class TransformerEncoder(Layer):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = LayerList([(encoder_layer if i == 0 else
                                  type(encoder_layer)(**encoder_layer._config))
                                 for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.checkpoints = []

    def forward(self, src, src_mask=None, cache=None):
        src_mask = _convert_attention_mask(src_mask, src.dtype)

        output = src
        for _, mod in enumerate(self.layers):
            if paddle.in_dynamic_mode():
                output = recompute_wrapper(mod, output, src_mask, is_recompute=self.training)
            else:
                output = mod(output, src_mask)
                self.checkpoints.append(output.name)

        if self.norm is not None:
            output = self.norm(output)

        return output 



class DisentangledSelfAttention(Layer):
    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False,
                 weight_attr=None,
                 bias_attr=None,
                 only_c2p=False):
        super(DisentangledSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.dropout = dropout
        self.need_weights = need_weights

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.query_proj = Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.key_proj = Linear(
            self.kdim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.value_proj = Linear(
            self.vdim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.out_proj = Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
        
        self.pos_key_proj = Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
        
        self.only_c2p = only_c2p

        if not only_c2p:
            self.pos_query_proj = Linear(
                embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
    
    def gather_4d(self, x, index):
        a = paddle.shape(index)[0]
        b = paddle.shape(index)[1]
        c = paddle.shape(index)[2]
        d = paddle.shape(index)[3]

        stack_0 = paddle.tile(paddle.arange(start=0, end=a, step=1, dtype="float32").reshape([a, 1]), [b * c * d]).reshape([a, b, c, d]).cast(index.dtype)
        stack_1 = paddle.tile(paddle.arange(start=0, end=b, step=1, dtype="float32").reshape([b, 1]), [a, 1, c * d]).reshape([a, b, c, d]).cast(index.dtype)
        stack_2 = paddle.tile(paddle.arange(start=0, end=c, step=1, dtype="float32").reshape([c, 1]), [a * b, 1, d]).reshape([a, b, c, d]).cast(index.dtype)
        
        new_index = paddle.stack([stack_0, stack_1, stack_2, index], axis=-1)
        gather_output = paddle.gather_nd(x=x, index=new_index)

        return gather_output


    def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        rel_embeddings = rel_embeddings.unsqueeze(0)
        pos_key_layer = self.pos_key_proj(rel_embeddings).reshape(shape=[0, 0, self.num_heads, self.head_dim]).transpose(perm=[0, 2, 1, 3])
        pos_key_layer = paddle.expand(pos_key_layer, shape=[paddle.shape(key_layer)[0], -1, -1, -1])

        score = 0
        # content->position
        scale = math.sqrt(pos_key_layer.shape[-1] * scale_factor)

        c2p_att = layers.matmul(x=query_layer, y=pos_key_layer, transpose_y=True)
        # Todos: max_pos_len(1024)/2
        attn_span = 512
        c2p_pos = paddle.clip(relative_pos + attn_span, 0, attn_span * 2 - 1)
        c2p_gather_idx = c2p_pos.unsqueeze(0).expand(shape=[paddle.shape(c2p_att)[0], paddle.shape(c2p_att)[1], -1, -1])
        c2p_att = self.gather_4d(c2p_att, index=c2p_gather_idx)
        score += paddle.scale(c2p_att, 1/scale)

        if not self.only_c2p:
            # position->content
            pos_query_layer = self.pos_query_proj(rel_embeddings).reshape(shape=[0, 0, self.num_heads, self.head_dim]).transpose(perm=[0, 2, 1, 3])
            pos_query_layer = paddle.expand(pos_query_layer, shape=[paddle.shape(query_layer)[0], -1, -1, -1])
            scale = math.sqrt(pos_query_layer.shape[-1] * scale_factor)
            p2c_pos = paddle.clip(-relative_pos + attn_span, 0, attn_span * 2 - 1)
            p2c_att = layers.matmul(x=key_layer, y=pos_query_layer, transpose_y=True)
            p2c_gather_idx = p2c_pos.unsqueeze(0).expand(shape=[paddle.shape(p2c_att)[0], paddle.shape(p2c_att)[1], -1, -1])
            p2c_att = self.gather_4d(p2c_att, index=p2c_gather_idx)
            score = paddle.add(score, paddle.scale(p2c_att, 1/scale))

        return score


    def forward(self, query, key=None, value=None, attn_mask=None, relative_pos=None, rel_embeddings=None, cache=None):
        key = query if key is None else key
        value = query if value is None else value

        query_layer = self.query_proj(query).reshape(shape=[0, 0, self.num_heads, self.head_dim]).transpose(perm=[0, 2, 1, 3])
        key_layer = self.key_proj(key).reshape(shape=[0, 0, self.num_heads, self.head_dim]).transpose(perm=[0, 2, 1, 3])
        value_layer = self.value_proj(value).reshape(shape=[0, 0, self.num_heads, self.head_dim]).transpose(perm=[0, 2, 1, 3])

        # paddle.static.Print(query_layer, message="query_layer.shape: ")

        scale_factor = 2 if self.only_c2p else 3
        scale = math.sqrt(query_layer.shape[-1] * scale_factor)
        attention_scores = paddle.scale(layers.matmul(x=query_layer, y=key_layer, transpose_y=True), 1/scale)
        rel_embeddings = F.dropout(rel_embeddings, p=0.1, training=self.training) 
        rel_att = self.disentangled_attention_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)
        attention_scores = paddle.add(attention_scores, rel_att)

        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = _convert_attention_mask(attn_mask, attention_scores.dtype)
            attention_scores = paddle.add(attention_scores, attn_mask)

        weights = F.softmax(attention_scores)
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode="upscale_in_train")

        out = tensor.matmul(weights, value_layer)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        results = {
            'output': out,
            'prob': weights
        }
        return results


class DeBERTaEncoderLayer(Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None,
                 only_c2p=False):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(DeBERTaEncoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 2)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 2)

        self.self_attn = DisentangledSelfAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0],
            only_c2p=only_c2p)
        self.linear1 = Linear(
            d_model, dim_feedforward, weight_attrs[1], bias_attr=bias_attrs[1])
        self.dropout = Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = Linear(
            dim_feedforward, d_model, weight_attrs[1], bias_attr=bias_attrs[1])
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)

    def forward(self, src, src_mask=None, relative_pos=None, rel_embeddings=None, return_weight=False):
        src_mask = _convert_attention_mask(src_mask, src.dtype)

        residual = src
        if self.normalize_before:
            src = self.norm1(src)

        attn_results = self.self_attn(src, src, src, src_mask, relative_pos, rel_embeddings)
        src = attn_results['output']

        src = paddle.add(residual, self.dropout1(src))
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = paddle.add(residual, self.dropout2(src))
        if not self.normalize_before:
            src = self.norm2(src)
        
        if return_weight:
            return src, attn_results['prob']
        else:
            return src

    def gen_cache(self, src):
        incremental_cache = self.self_attn.gen_cache(
            src, type=self.self_attn.Cache)
        return incremental_cache


class DeBERTaEncoder(Layer):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(DeBERTaEncoder, self).__init__()
        self.layers = LayerList([(encoder_layer if i == 0 else
                                  type(encoder_layer)(**encoder_layer._config))
                                 for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.checkpoints = []

    def forward(self, src, src_mask=None, relative_pos=None, rel_embeddings=None, return_last_n_weight=False):
        src_mask = _convert_attention_mask(src_mask, src.dtype)

        output = src
        attn_weight_list = []
        for i, mod in enumerate(self.layers):
            return_weight = (i >= self.num_layers - return_last_n_weight)
            if paddle.in_dynamic_mode():
                outputs = recompute_wrapper(mod, 
                        output, 
                        src_mask, 
                        relative_pos, 
                        rel_embeddings, 
                        return_weight, 
                        is_recompute=self.training)
            else:
                outputs = mod(
                        output, 
                        src_mask=src_mask, 
                        relative_pos=relative_pos, 
                        rel_embeddings=rel_embeddings,
                        return_weight=return_weight)
            if return_weight:
                output, attn_weight = outputs
                attn_weight_list.append(attn_weight)
            else:
                output = outputs
            if not paddle.in_dynamic_mode():
                self.checkpoints.append(output.name)

        if self.norm is not None:
            output = self.norm(output)

        results = {
            'output': output,
        }
        if len(attn_weight_list) > 0:
            results['attn_weight'] = paddle.concat(attn_weight_list, 1)    # (B, layer * H, L, L)
        return results


class RotaryPositionEmbedding(Layer):

    def __init__(self, dim, max_position_embeddings=512):
        super(RotaryPositionEmbedding, self).__init__()
        inv_freq = 1.0 / (10000**(
            paddle.arange(0, dim, 2, dtype=paddle.get_default_dtype()) / dim))
        t = paddle.arange(max_position_embeddings,
                          dtype=paddle.get_default_dtype())
        freqs = paddle.matmul(t.unsqueeze(1), inv_freq.unsqueeze(0))
        self.register_buffer("sin", freqs.sin(), persistable=False)
        self.register_buffer("cos", freqs.cos(), persistable=False)

    def forward(self, x, offset=0):
        # x shape [batch_size, num_heads, seqlen, head_dim]
        seqlen = paddle.shape(x)[-2]
        sin, cos = (
            self.sin[offset:offset + seqlen, :],
            self.cos[offset:offset + seqlen, :],
        )
        x1, x2 = x[..., 0::2], x[..., 1::2]
        # [cos_nθ, -sin_nθ] [x1]
        # [sin_nθ,  cos_nθ] [x2]
        # => [x1 * cos_nθ - x2 * sin_nθ, x1 * sin_nθ + x2 * cos_nθ]
        return paddle.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos],
                            axis=-1).flatten(-2, -1)


class MultiHeadAttentionWithRotary(Layer):

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False,
                 rotary_value=False,
                 max_position_embeddings=512):
        super(MultiHeadAttentionWithRotary, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.need_weights = need_weights
        self.rotary_value = rotary_value

        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(self.kdim, embed_dim)
        self.v_proj = nn.Linear(self.vdim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.rotary = RotaryPositionEmbedding(self.head_dim,
                                              max_position_embeddings)

    def forward(self, query, key=None, value=None, attn_mask=None, cache=None):
        key = query if key is None else key
        value = query if value is None else value

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = tensor.transpose(x=q, perm=[0, 2, 1, 3])
        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])

        q, k = self.rotary(q), self.rotary(k)
        if self.rotary_value:
            v = self.rotary(v)

        product = tensor.matmul(x=q, y=k, transpose_y=True) * self.scale
        if attn_mask is not None:
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask

        weights = F.softmax(product)
        weights = self.dropout(weights)
        out = tensor.matmul(weights, v)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if cache is not None:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)


class TransformerEncoderLayerWithRotary(TransformerEncoderLayer):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 rotary_value=False,
                 max_position_embeddings=512,
                 **kwargs):
        super(TransformerEncoderLayerWithRotary, self).__init__(d_model,
                         nhead,
                         dim_feedforward,
                         dropout=dropout,
                         activation=activation,
                         attn_dropout=attn_dropout,
                         act_dropout=act_dropout,
                         normalize_before=normalize_before)
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.self_attn = MultiHeadAttentionWithRotary(
            d_model,
            nhead,
            dropout=attn_dropout,
            rotary_value=rotary_value,
            max_position_embeddings=max_position_embeddings)
        self._config.update({
            "rotary_value": rotary_value,
            "max_position_embeddings": max_position_embeddings
        })


if __name__ == "__main__":
    d_model = 512
    n_head = 8
    dim_feedforward = 2048
    rotary_layer = TransformerEncoderLayerWithRotary(d_model, n_head, dim_feedforward)
    fake_data = paddle.rand(shape=[20, 342, 512])
    layer_output = rotary_layer(fake_data)