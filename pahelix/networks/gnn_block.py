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
blocks for Graph Neural Network (GNN)
Adapted from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/model.py
"""

import os
import re
import time
import logging
from random import random
from functools import reduce, partial

import numpy as np
import multiprocessing

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import pgl
from pgl.graph_wrapper import GraphWrapper
from pgl.layers.conv import gcn, gat
from pgl.utils import paddle_helper


def mean_recv(feat):
    """average pooling"""
    return layers.sequence_pool(feat, pool_type="average")


def sum_recv(feat):
    """sum pooling"""
    return layers.sequence_pool(feat, pool_type="sum")


def max_recv(feat):
    """max pooling"""
    return layers.sequence_pool(feat, pool_type="max")


def gcn_layer(gw, feature, edge_features, act, name):
    """
    Implementation of graph convolutional neural networks (GCN)
    
    Args:
        gw(GraphWrapper): pgl graph wrapper object.
        feature(tensor): node features with shape (num_nodes, feature_size).
        edge_features(tensor): edges features with shape (num_edges, feature_size).
        hidden_size(int): the hidden size for gcn.
        act(int): the activation for the output.
        name(int): the prefix of layer param names.
    """
    def send_func(src_feat, dst_feat, edge_feat):
        """send function"""
        return src_feat["h"] + edge_feat["h"]

    size = feature.shape[-1]

    msg = gw.send(send_func,
            nfeat_list=[("h", feature)],
            efeat_list=[("h", edge_features)])

    output = gw.recv(msg, mean_recv)
    output = layers.fc(output,
            size=size,
            bias_attr=False,
            param_attr=fluid.ParamAttr(name=name))

    bias = layers.create_parameter(
            shape=[size],
            dtype='float32',
            is_bias=True,
            name=name + '_bias')
    output = layers.elementwise_add(output, bias, act=act)
    return output


def gat_layer(
        gw,
        feature,
        edge_features,
        hidden_size,
        act,
        name,
        num_heads=1,
        feat_drop=0.1,
        attn_drop=0.1,
        is_test=False):
    """
    Implementation of graph attention networks (GAT)
    
    Args:
        gw(GraphWrapper): pgl graph wrapper object.
        feature(tensor): node features with shape (num_nodes, feature_size).
        edge_features(tensor): edges features with shape (num_edges, feature_size).
        hidden_size(int): the hidden size for gcn.
        act(str): the activation for the output.
        name(str): the prefix of layer param names.
        num_heads(int): the head number in gat.
        feat_drop: dropout rate for the :attr:`feature`.
        attn_drop: dropout rate for the attention.
        is_test: whether in test phrase.
    """
    def send_attention(src_feat, dst_feat, edge_feat):
        """send function"""
        output = src_feat["left_a"] + dst_feat["right_a"]
        output = layers.leaky_relu(
                output, alpha=0.2)  # (num_edges, num_heads)
        return {"alpha": output, "h": src_feat["h"] + edge_feat["h"]}

    def reduce_attention(msg):
        """reduce function"""
        alpha = msg["alpha"]  # lod-tensor (batch_size, seq_len, num_heads)
        h = msg["h"]
        alpha = paddle_helper.sequence_softmax(alpha)
        old_h = h
        h = layers.reshape(h, [-1, num_heads, hidden_size])
        alpha = layers.reshape(alpha, [-1, num_heads, 1])
        if attn_drop > 1e-15:
            alpha = layers.dropout(
                    alpha,
                    dropout_prob=attn_drop,
                    is_test=is_test,
                    dropout_implementation="upscale_in_train")
        h = h * alpha
        h = layers.reshape(h, [-1, num_heads * hidden_size])
        h = layers.lod_reset(h, old_h)
        return layers.sequence_pool(h, "sum")

    if feat_drop > 1e-15:
        feature = layers.dropout(
                feature,
                dropout_prob=feat_drop,
                is_test=is_test,
                dropout_implementation='upscale_in_train')

    ft = layers.fc(feature,
            hidden_size * num_heads,
            bias_attr=False,
            param_attr=fluid.ParamAttr(name=name + '_weight'))
    left_a = layers.create_parameter(
            shape=[num_heads, hidden_size],
            dtype='float32',
            name=name + '_gat_l_A')
    right_a = layers.create_parameter(
            shape=[num_heads, hidden_size],
            dtype='float32',
            name=name + '_gat_r_A')
    reshape_ft = layers.reshape(ft, [-1, num_heads, hidden_size])
    left_a_value = layers.reduce_sum(reshape_ft * left_a, -1)
    right_a_value = layers.reduce_sum(reshape_ft * right_a, -1)

    msg = gw.send(
            send_attention,
            nfeat_list=[("h", ft), ("left_a", left_a_value),
                        ("right_a", right_a_value)],
            efeat_list=[("h", edge_features)])
    output = gw.recv(msg, reduce_attention)
    bias = layers.create_parameter(
            shape=[hidden_size * num_heads],
            dtype='float32',
            is_bias=True,
            name=name + '_bias')
    bias.stop_gradient = True
    output = layers.elementwise_add(output, bias, act=act)
    return output


def gin_layer(gw, node_features, edge_features, name):
    """
    Implementation of Graph Isomorphism Network (GIN) layer.
    
    Args:
        gw(GraphWrapper): pgl graph wrapper object.
        node_features(tensor): node features with shape (num_nodes, feature_size).
        edge_features(tensor): edges features with shape (num_edges, feature_size).
        name(str): the prefix of layer param names.
    """
    def send_func(src_feat, dst_feat, edge_feat):
        """send function"""
        return src_feat["h"] + edge_feat["h"]

    msg = gw.send(send_func,
            nfeat_list=[("h", node_features)],
            efeat_list=[("h", edge_features)])

    node_feat = gw.recv(msg, "sum")

    dim = node_feat.shape[-1]
    node_feat = layers.fc(
            node_feat,
            size=dim * 2,
            name="%s_fc_%s" % (name, 0),
            act="relu")

    node_feat = layers.fc(
            node_feat,
            size=dim,
            name="%s_fc_%s" % (name, 1),
            act=None)

    return node_feat

