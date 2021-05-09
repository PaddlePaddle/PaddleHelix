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
Transformer block.
"""

import paddle.fluid as fluid
import paddle.fluid.layers as layers
from pahelix.networks.pre_post_process import pre_process_layer, post_process_layer


def multi_head_attention(queries,
                         keys,
                         values,
                         attn_bias,
                         d_key,
                         d_value,
                         d_model,
                         n_head=1,
                         dropout_rate=0.,
                         cache=None,
                         gather_idx=None,
                         store=False,
                         param_initializer=None,
                         lr=1.0,
                         name="multi_head_att",
                         is_test=False):
    """
    Multi-Head Attention. 

    Note that attn_bias is added to the logit before
    computing softmax activiation to mask certain selected positions so that
    they will not considered in attention weights.
    """
    keys = queries if keys is None else keys
    values = keys if values is None else values

    if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
        raise ValueError(
            "Inputs: quries, keys and values should all be 3-D tensors.")

    def __compute_qkv(queries, keys, values, n_head, d_key, d_value):
        """
        Add linear projection to queries, keys, and values.
        """
        q = layers.fc(input=queries,
                      size=d_key * n_head,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(
                          name=name + "_query_fc.w_0",
                          learning_rate=lr,
                          initializer=param_initializer),
                      bias_attr=name + "_query_fc.b_0")
        k = layers.fc(input=keys,
                      size=d_key * n_head,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(
                          name=name + "_key_fc.w_0",
                          learning_rate=lr,
                          initializer=param_initializer),
                      bias_attr=name + "_key_fc.b_0")
        v = layers.fc(input=values,
                      size=d_value * n_head,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(
                          name=name + "_value_fc.w_0",
                          learning_rate=lr,
                          initializer=param_initializer),
                      bias_attr=name + "_value_fc.b_0")
        return q, k, v

    def __split_heads(x, n_head):
        """
        Reshape the last dimension of inpunt tensor x so that it becomes two
        dimensions and then transpose. Specifically, input a tensor with shape
        [bs, max_sequence_length, n_head * hidden_dim] then output a tensor
        with shape [bs, n_head, max_sequence_length, hidden_dim].
        """
        hidden_size = x.shape[-1]
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        reshaped = layers.reshape(
            x=x, shape=[0, 0, n_head, hidden_size // n_head], inplace=True)

        # permuate the dimensions into:
        # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        return layers.transpose(x=reshaped, perm=[0, 2, 1, 3])

    def __combine_heads(x):
        """
        Transpose and then reshape the last two dimensions of inpunt tensor x
        so that it becomes one dimension, which is reverse to __split_heads.
        """
        if len(x.shape) == 3: return x
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = layers.transpose(x, perm=[0, 2, 1, 3])
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        return layers.reshape(
            x=trans_x,
            shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]],
            inplace=True)

    def scaled_dot_product_attention(q, k, v, attn_bias, d_key, dropout_rate, is_test=False):
        """
        Scaled Dot-Product Attention
        """
        scaled_q = layers.scale(x=q, scale=d_key ** -0.5)
        product = layers.matmul(x=scaled_q, y=k, transpose_y=True)
        if attn_bias:
            product += attn_bias
        weights = layers.softmax(product, use_cudnn=True)
        if dropout_rate:
            weights = layers.dropout(
                weights,
                dropout_prob=dropout_rate,
                dropout_implementation="upscale_in_train",
                is_test=is_test)
        out = layers.matmul(weights, v)
        return out

    q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)

    if cache is not None:  # use cache and concat time steps
        # Since the inplace reshape in __split_heads changes the shape of k and
        # v, which is the cache input for next time step, reshape the cache
        # input from the previous time step first.
        cache_k, cache_v = cache["k"], cache["v"]
        select_k = layers.gather(cache_k, index=gather_idx)
        select_v = layers.gather(cache_v, index=gather_idx)
        select_k = layers.reshape(select_k, shape=[0, 0, d_key * n_head])
        select_v = layers.reshape(select_v, shape=[0, 0, d_value * n_head])
        if store:
            k = layers.concat([select_k, k], axis=1) 
            v = layers.concat([select_v, v], axis=1)
            layers.assign(k, cache["k"])
            layers.assign(v, cache["v"])
        else:
            #k = select_k
            #v = select_v
            tmp_k = layers.concat([select_k, k[:, :1]], axis=1)
            tmp_v = layers.concat([select_v, v[:, :1]], axis=1)
            layers.assign(tmp_k, cache["k"])
            layers.assign(tmp_v, cache["v"])
            k = layers.concat([select_k, k], axis=1)
            v = layers.concat([select_v, v], axis=1)

    q = __split_heads(q, n_head)
    k = __split_heads(k, n_head)
    v = __split_heads(v, n_head)

    ctx_multiheads = scaled_dot_product_attention(q, k, v, attn_bias, d_key,
                                                  dropout_rate, is_test)

    out = __combine_heads(ctx_multiheads)

    # Project back to the model size.
    proj_out = layers.fc(input=out,
                         size=d_model,
                         num_flatten_dims=2,
                         param_attr=fluid.ParamAttr(
                             name=name + "_output_fc.w_0",
                             learning_rate=lr,
                             initializer=param_initializer),
                         bias_attr=name + "_output_fc.b_0")
    return proj_out


def positionwise_feed_forward(x,
                              d_inner_hid,
                              d_hid,
                              dropout_rate,
                              hidden_act,
                              num_flatten_dims=2,
                              param_initializer=None,
                              name="ffn",
                              is_test=False):
    """
    Position-wise Feed-Forward Networks.

    This module consists of two linear transformations with a ReLU activation
    in between, which is applied to each position separately and identically.
    """
    hidden = layers.fc(input=x,
                       size=d_inner_hid,
                       num_flatten_dims=num_flatten_dims,
                       act=hidden_act,
                       param_attr=fluid.ParamAttr(
                           name=name + "_fc_0.w_0",
                           initializer=param_initializer),
                       bias_attr=name + "_fc_0.b_0")
    if dropout_rate:
        hidden = layers.dropout(
            hidden,
            dropout_prob=dropout_rate,
            dropout_implementation="upscale_in_train",
            is_test=is_test)
    out = layers.fc(input=hidden,
                    size=d_hid,
                    num_flatten_dims=num_flatten_dims,
                    param_attr=fluid.ParamAttr(
                        name=name + "_fc_1.w_0", initializer=param_initializer),
                    bias_attr=name + "_fc_1.b_0")
    return out


def transformer_encoder_layer(input,
                  attn_bias,
                  n_head,
                  d_key,
                  d_value,
                  d_model,
                  d_inner_hid,
                  prepostprocess_dropout,
                  attention_dropout,
                  act_dropout,
                  hidden_act,
                  preprocess_cmd="n",
                  postprocess_cmd="da",
                  param_initializer=None,
                  name="",
                  epsilon=1e-5,
                  cache=None,
                  gather_idx=None,
                  store=False,
                  is_test=False):
    """
    The encoder layers that can be stacked to form a deep encoder.
    
    This module consits of a multi-head (self) attention followed by
    position-wise feed-forward networks and both the two components companied
    with the pre_process_layer / post_process_layer to add residual connection,
    layer normalization and droput.
    """
    attn_output = multi_head_attention(
        pre_process_layer(
            input,
            preprocess_cmd,
            prepostprocess_dropout,
            epsilon=epsilon,
            name=name + "_pre_att",
            is_test=is_test),
        None,
        None,
        attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        attention_dropout,
        param_initializer=param_initializer,
        name=name + "_multi_head_att",
        cache=cache,
        gather_idx=gather_idx,
        store=store,
        is_test=is_test)
    attn_output = post_process_layer(
        input,
        attn_output,
        postprocess_cmd,
        prepostprocess_dropout,
        name=name + "_post_att",
        epsilon=epsilon,
        is_test=is_test)
    ffd_output = positionwise_feed_forward(
        pre_process_layer(
            attn_output,
            preprocess_cmd,
            prepostprocess_dropout,
            epsilon=epsilon,
            name=name + "_pre_ffn",
            is_test=is_test),
        d_inner_hid,
        d_model,
        act_dropout,
        hidden_act,
        param_initializer=param_initializer,
        name=name + "_ffn",
        is_test=is_test)
    ffd_output = post_process_layer(
        attn_output,
        ffd_output,
        postprocess_cmd,
        prepostprocess_dropout,
        name=name + "_post_ffn",
        epsilon=epsilon,
        is_test=is_test)
    return ffd_output, [attn_output, ffd_output]


def transformer_encoder(enc_input,
            attn_bias,
            n_layer,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            act_dropout,
            hidden_act,
            preprocess_cmd="n",
            postprocess_cmd="da",
            param_initializer=None,
            name="",
            epsilon=1e-5,
            n_layer_per_block=1,
            param_share="normal",
            caches=None,
            gather_idx=None,
            store=False,
            is_test=False):
    """
    The encoder is composed of a stack of identical layers returned by calling
    transformer_encoder_layer.
    """
    checkpoints = []
    names = []
    if param_share == "inner_share":
        for _ in range(n_layer // n_layer_per_block):
            for i in range(n_layer_per_block):
                names.append(name + "_layer_" + str(i))
    else:
        for i in range(n_layer // n_layer_per_block):
            for _ in range(n_layer_per_block):
                names.append(name + "_layer_" + str(i))

    for i in range(n_layer):
        enc_output, cps = transformer_encoder_layer(
            enc_input,
            attn_bias,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            act_dropout,
            hidden_act,
            preprocess_cmd,
            postprocess_cmd,
            param_initializer=param_initializer,
            epsilon=epsilon,
            name=names[i],
            cache=caches[i] if caches is not None else None,
            gather_idx=gather_idx,
            store=store,
            is_test=is_test)
        checkpoints.extend(cps)
        enc_input = enc_output
    enc_output = pre_process_layer(
        enc_output, preprocess_cmd, prepostprocess_dropout, name="post_encoder", epsilon=epsilon, is_test=is_test)

    return enc_output, checkpoints
