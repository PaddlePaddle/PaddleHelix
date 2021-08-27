# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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
Doc String
"""
import math
import paddle
import paddle.fluid as F
import paddle.fluid.layers as L

from pgl import message_passing
from pgl.utils import paddle_helper

def linear(X, hidden_size, name, with_bias=True, init_type=None):
    """fluid.layers.fc with different init_type
    """
    
    if init_type == 'gcn':
        fc_w_attr = F.ParamAttr(initializer=F.initializer.XavierInitializer(),
                                name="%s_w" % name)
        fc_bias_attr = F.ParamAttr(initializer=F.initializer.ConstantInitializer(0.0), 
                                name="%s_b" % name)
    else:
        fan_in = X.shape[-1]
        bias_bound = 1.0 / math.sqrt(fan_in)
        init_b = F.initializer.UniformInitializer(low=-bias_bound, high=bias_bound)
        fc_bias_attr = F.ParamAttr(initializer=init_b, name="%s_b" % name)

        negative_slope = math.sqrt(5)
        gain = math.sqrt(2.0 / (1 + negative_slope ** 2))
        std = gain / math.sqrt(fan_in)
        weight_bound = math.sqrt(3.0) * std
        init_w = F.initializer.UniformInitializer(low=-weight_bound, high=weight_bound)
        fc_w_attr = F.ParamAttr(initializer=init_w, name="%s_w" % name)
    
    if not with_bias:
        fc_bias_attr = False
        
    output = L.fc(X,
        hidden_size,
        param_attr=fc_w_attr,
        name=name,
        bias_attr=fc_bias_attr)
    return output


def layer_norm(feature, name=""):
    lay_norm_attr=F.ParamAttr(
            name="attr_%s" % name,
            initializer=F.initializer.ConstantInitializer(value=1))
    lay_norm_bias=F.ParamAttr(
            name="bias_%s" % name,
            initializer=F.initializer.ConstantInitializer(value=0))

    feature = L.layer_norm(feature, 
                           param_attr=lay_norm_attr,
                           bias_attr=lay_norm_bias)

    return feature



def gin_layer(gw, node_features, edge_features, hidden_size, act, name):
    def send_func(src_feat, dst_feat, edge_feat):
        """Send"""
        return src_feat["h"] + edge_feat["h"]

    epsilon = L.create_parameter(
        shape=[1, 1],
        dtype="float32",
        attr=F.ParamAttr(name="%s_eps" % name),
        default_initializer=F.initializer.ConstantInitializer(value=0.0))

    msg = gw.send(
        send_func,
        nfeat_list=[("h", node_features)],
        efeat_list=[("h", edge_features)])

    node_feat = gw.recv(msg, "sum") + node_features * (epsilon + 1.0)

    return node_feat

def gen_layer(gw, nfeat, efeat, hidden_size, name):
    def _send_func(src_feat, dst_feat, edge_feat):
        h = src_feat['h'] + edge_feat['h']
        h = L.relu(h)
        return h

    def _recv_func(msg):
        return L.sequence_pool(msg, "sum")

    beta = L.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=
                F.initializer.ConstantInitializer(value=1.0),
            name=name + '_beta')

    # message passing
    msg = gw.send(_send_func, nfeat_list=[("h", nfeat)], efeat_list=[("h", efeat)])
    output = gw.recv(msg, message_passing.softmax_agg(beta))

    output = message_passing.msg_norm(nfeat, output, name)
    output = nfeat + output

    output = mlp(output, [hidden_size*2, hidden_size], norm="layer_norm", name=name)

    return output

def mlp(X, layer_dims, norm="layer_norm", name=""):
    for idx, dim in enumerate(layer_dims):
        X = linear(X, dim, "%s_%s" % (name, idx))
        if norm == "layer_norm":
            X = layer_norm(X, "norm_%s_%s" % (name, idx))

    return X

def graph_transformer(
        gw,
        feature,
        edge_feature,
        hidden_size,
        name, 
        num_heads=4,
        attn_drop=False,
        concat=True,
        skip_feat=True,
        gate=False,
        norm=True, 
        relu=True, 
        is_test=False):
    """Implementation of graph Transformer from UniMP

    This is an implementation of the paper Unified Massage Passing Model for Semi-Supervised Classification
    (https://arxiv.org/abs/2009.03509).

    Args:
        name: Granph Transformer layer names.
        
        gw: Graph wrapper object (:code:`StaticGraphWrapper` or :code:`GraphWrapper`)

        feature: A tensor with shape (num_nodes, feature_size).

        hidden_size: The hidden size for graph transformer.

        num_heads: The head number in graph transformer.

        attn_drop: Dropout rate for attention.
        
        edge_feature: A tensor with shape (num_edges, feature_size).
        
        concat: Reshape the output (num_nodes, num_heads, hidden_size) by concat (num_nodes, hidden_size * num_heads) or mean (num_nodes, hidden_size)
        
        skip_feat: Whether use skip connect
        
        gate: Whether add skip_feat and output up with gate weight
        
        norm: Whether use layer_norm for output
        
        relu: Whether use relu activation for output

        is_test: Whether in test phrase.

    Return:
        A tensor with shape (num_nodes, hidden_size * num_heads) or (num_nodes, hidden_size)
    """
    def send_attention(src_feat, dst_feat, edge_feat):
        if edge_feat is None or not edge_feat:
            output = src_feat["k_h"] * dst_feat["q_h"]
            output = L.reduce_sum(output, -1)
            output = output / (hidden_size ** 0.5)
            return {"alpha": output, "v": src_feat["v_h"]}   # batch x h     batch x h x feat
        else:
            edge_feat = edge_feat["edge"]
            edge_feat = L.reshape(edge_feat, [-1, num_heads, hidden_size])
            output = (src_feat["k_h"] + edge_feat) * dst_feat["q_h"]
            output = L.reduce_sum(output, -1)
            output = output / (hidden_size ** 0.5)
            return {"alpha": output, "v": (src_feat["v_h"] + edge_feat)}   # batch x h     batch x h x feat
    
    class Reduce_attention():
        def __init__(self,):
            self.alpha = None
        def __call__(self, msg):
            alpha = msg["alpha"]  # lod-tensor (batch_size, num_heads)
            if attn_drop:
                old_h = alpha
                dropout = F.data(name='attn_drop', shape=[1], dtype="int64")
                u = L.uniform_random(shape=L.cast(L.shape(alpha)[:1], 'int64'), min=0., max=1.)
                keeped = L.cast(u > dropout, dtype="float32")
                self_attn_mask = L.scale(x=keeped, scale=10000.0, bias=-1.0, bias_after_scale=False)
                n_head_self_attn_mask = L.stack( x=[self_attn_mask] * num_heads, axis=1)
                n_head_self_attn_mask.stop_gradient = True
                alpha = n_head_self_attn_mask+ alpha
                alpha = L.lod_reset(alpha, old_h)

            h = msg["v"]
            alpha = paddle_helper.sequence_softmax(alpha)
            self.alpha = alpha
            old_h = h
            h = h * alpha
            h = L.lod_reset(h, old_h)
            h = L.sequence_pool(h, "sum")
            if concat:
                h = L.reshape(h, [-1, num_heads * hidden_size])
            else:
                h = L.reduce_mean(h, dim=1)
            return h
    reduce_attention = Reduce_attention()
    
    q = linear(feature, hidden_size * num_heads, name=name + '_q_weight', init_type='gcn')
    k = linear(feature, hidden_size * num_heads, name=name + '_k_weight', init_type='gcn')
    v = linear(feature, hidden_size * num_heads, name=name + '_v_weight', init_type='gcn')
    
    
    reshape_q = L.reshape(q, [-1, num_heads, hidden_size])
    reshape_k = L.reshape(k, [-1, num_heads, hidden_size])
    reshape_v = L.reshape(v, [-1, num_heads, hidden_size])

    msg = gw.send(
        send_attention,
        nfeat_list=[("q_h", reshape_q), ("k_h", reshape_k),
                    ("v_h", reshape_v)],
        efeat_list=[('edge', edge_feature)])
    out_feat = gw.recv(msg, reduce_attention)
    
    if skip_feat:
        if concat:
            skip_feature = linear(feature, hidden_size * num_heads, name=name + '_skip_weight', init_type='lin')
        else:
            skip_feature = linear(feature, hidden_size, name=name + '_skip_weight', init_type='lin')
            
        if gate:
            temp_output = L.concat([skip_feature, out_feat, out_feat - skip_feature], axis=-1)
            gate_f = L.sigmoid(linear(temp_output, 1, name=name + '_gate_weight', init_type='lin'))
            out_feat = skip_feature * gate_f + out_feat * (1 - gate_f)
        else:
            out_feat = skip_feature + out_feat
            
    if norm:
        out_feat = layer_norm(out_feat, name="ln_%s" % name)

    if relu:
        out_feat = L.relu(out_feat)
    
    return out_feat

def graph_linformer(
        gw,
        feature,
        edge_feature,
        hidden_size,
        name, 
        num_heads=4,
        attn_drop=False,
        concat=True,
        skip_feat=True,
        gate=False,
        norm=True, 
        relu=True, 
        k_hop=2,
        is_test=False):
    """Implementation of graph Transformer from UniMP

    This is an implementation of the paper Unified Massage Passing Model for Semi-Supervised Classification
    (https://arxiv.org/abs/2009.03509).

    Args:
        name: Granph Transformer layer names.
        
        gw: Graph wrapper object (:code:`StaticGraphWrapper` or :code:`GraphWrapper`)

        feature: A tensor with shape (num_nodes, feature_size).

        hidden_size: The hidden size for graph transformer.

        num_heads: The head number in graph transformer.

        attn_drop: Dropout rate for attention.
        
        edge_feature: A tensor with shape (num_edges, feature_size).
        num_heads: 8
        
        concat: Reshape the output (num_nodes, num_heads, hidden_size) by concat (num_nodes, hidden_size * num_heads) or mean (num_nodes, hidden_size)
        
        skip_feat: Whether use skip connect
        
        gate: Whether add skip_feat and output up with gate weight
        
        norm: Whether use layer_norm for output
        
        relu: Whether use relu activation for output

        is_test: Whether in test phrase.

    Return:
        A tensor with shape (num_nodes, hidden_size * num_heads) or (num_nodes, hidden_size)
    """
    def send_attention(src_feat, dst_feat, edge_feat):
        if edge_feat is None or not edge_feat:
            k_h = L.elu(L.reshape(src_feat["k_h"], [-1, num_heads, hidden_size, 1])) + 1
            v_h = dst_feat["v_h"]
        else:
            edge_feat = edge_feat["edge"]
            edge_feat = L.reshape(edge_feat, [-1, num_heads, hidden_size])
            k_h = L.elu(src_feat["k_h"] + edge_feat) + 1
            v_h = dst_feat["v_h"] + edge_feat
        k_h = L.reshape(k_h, [-1, num_heads, hidden_size, 1])

        v_h = L.reshape(v_h, [-1, num_heads, hidden_size, 1])
        sum_kTv = L.matmul(k_h, v_h, transpose_y=True)
        sum_k = L.reshape(k_h, [-1, num_heads * hidden_size])
        sum_kTv = L.reshape(sum_kTv, [-1, num_heads * hidden_size * hidden_size])
        
        return {"sum_k": sum_k, "sum_kTv": sum_kTv} 

    def send_copy(src_feat, dst_feat, edge_feat):
        return src_feat

    def reduce_sum(msg):
        return L.sequence_pool(msg, "sum")
   
    q = L.elu(linear(feature, hidden_size * num_heads, name=name + '_q_weight', init_type='gcn')) + 1
    k = linear(feature, hidden_size * num_heads, name=name + '_k_weight', init_type='gcn')
    v = linear(feature, hidden_size * num_heads, name=name + '_v_weight', init_type='gcn')
    
    
    reshape_q = L.reshape(q, [-1, num_heads, 1, hidden_size])
    reshape_k = L.reshape(k, [-1, num_heads, hidden_size])
    reshape_v = L.reshape(v, [-1, num_heads, hidden_size])

    msg = gw.send(
        send_attention,
        nfeat_list=[("k_h", reshape_k),
                    ("v_h", reshape_v)],
        efeat_list=[('edge', edge_feature)])

    sum_k = gw.recv(msg["sum_k"], reduce_sum)
    sum_kTv = gw.recv(msg["sum_kTv"], reduce_sum)

    for i in range(1, k_hop):
        msg = gw.send(send_copy, nfeat_list=[("sum_k", sum_k), ("sum_kTv", sum_kTv)])            
        sum_k = gw.recv(msg["sum_k"], reduce_sum)
        sum_kTv = gw.recv(msg["sum_kTv"], reduce_sum)
        # sum_k: [-1, num_heads * hidden_size]
        # sum_kTv: [-1, num_heads * hidden_size * hidden_size]
    sum_k = L.reshape(sum_k, [-1, num_heads, 1, hidden_size])
    sum_kTv = L.reshape(sum_kTv, [-1, num_heads, hidden_size, hidden_size])
    out_feat = L.reshape(L.matmul(reshape_q, sum_kTv), [-1, num_heads, hidden_size]) / L.reduce_sum(reshape_q * sum_k, -1)
    if concat:
        out_feat = L.reshape(out_feat, [-1, num_heads * hidden_size])
    else:
        out_feat = L.reduce_mean(out_feat, dim=1)

    
    if skip_feat:
        if concat:
            skip_feature = linear(feature, hidden_size * num_heads, name=name + '_skip_weight', init_type='lin')
        else:
            skip_feature = linear(feature, hidden_size, name=name + '_skip_weight', init_type='lin')
            
        if gate:
            temp_output = L.concat([skip_feature, out_feat, out_feat - skip_feature], axis=-1)
            gate_f = L.sigmoid(linear(temp_output, 1, name=name + '_gate_weight', init_type='lin'))
            out_feat = skip_feature * gate_f + out_feat * (1 - gate_f)
        else:
            out_feat = skip_feature + out_feat
            
    if norm:
        out_feat = layer_norm(out_feat, name="ln_%s" % name)

    if relu:
        out_feat = L.relu(out_feat)
    
    return out_feat
