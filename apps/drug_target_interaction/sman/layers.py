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
This file implement some layers for S-MAN.
"""
import pgl
import paddle.fluid as fluid
import paddle.fluid.layers as L
from pgl.utils import paddle_helper
import numpy as np


def graph_pooling(node_feat, graph_lod, pool_type='sum'):
    """graph pooling layers for nodes"""
    node_feat = L.lod_reset(node_feat, graph_lod)
    graph_feat = L.sequence_pool(node_feat, pool_type)
    return graph_feat


def spatial_embedding(dist_feat, dist_feat_order, embed_size):
    """
    **Spatial Embedding Layer**
    This function can encode the one-hot feature into the embedding representation.
    Args:
        dist_feat(Variable): The input one-hot distance feature for the edges of node-node graph, and the data type is float32 or float64.
        dist_feat_order(Variable): The input one-hot distance feature in the order of edge-edge matrix, and the data type is float32 or float64.
        embed_size(int): The embedding size parameter for encoding.
    Returns:
        (Variable, Variable): The tuple of distance features after spatial embedding.
    """
    dist_dim = dist_feat.shape[-1]
    dist_w = L.create_parameter(
                shape=[dist_dim, embed_size],
                dtype='float32',
                name='spgat_dist_w')
    dist_feat = L.matmul(dist_feat, dist_w)
    if dist_feat_order:
        dist_feat_order = L.matmul(dist_feat_order, dist_w)
    return dist_feat, dist_feat_order


def aggregate_edges_from_nodes(node_edge_feat, dist_feat, srcs, dsts):
    """
    ** Node-to-Edge Aggregation Layer **
    This function can aggregate the two node features and spatial features to update the edge embedding.
    Args:
        node_edge_feat(Variable): A tensor with shape (num_nodes + num_edges, feature_size).
        dist_feat(Variable): The ispatial distance feature for the edges of node-node graph, the shape is (num_edges, embedding_size).
        srcs(Variable): Source indices of edges with shape (num_edges, 1) to gather source features.
        dsts(Variable): Target indices of edges with shape (num_edges, 1) to gather target features.
    Returns:
        Variable: The updated edge features after aggregating embeddings of nodes.
    """
    hidden_size = node_edge_feat.shape[-1]
    src_feat = L.gather(node_edge_feat, srcs)
    dst_feat = L.gather(node_edge_feat, dsts)
    feat_h = L.concat([src_feat, dst_feat, dist_feat], axis=-1)
    feat_h = L.fc(input=feat_h, size=hidden_size, act="relu")
    return feat_h


def aggregate_edges_from_edges(gw, node_feat, hidden_size, name):
    """The gat function can aggregate the edge-neighbors of edge to update the edfe embedding."""
    node_edge_feat = gat(gw,
                        node_feat,
                        hidden_size,
                        dist_feat=None,
                        activation="relu",
                        name=name,
                        num_heads=4,
                        feat_drop=0.2,
                        attn_drop=0.2,
                        is_test=False)
    return node_edge_feat


def aggregate_nodes_from_edges(gw, node_feat, edge_feat, hidden_size, name):
    """The sgat function can aggregate the edge-neighbors of node to update the node embedding."""
    node_edge_feat = sgat(gw,
                        node_feat,
                        edge_feat,
                        hidden_size,
                        name=name,
                        activation='relu',
                        num_heads=4,
                        feat_drop=0.2,
                        attn_drop=0.2,
                        is_test=False)
    return node_edge_feat


def concat_node_edge_feat(node_feat, edge_feat, nlod, elod):
    """
    This function can concat node features and edge features to form the node-edge feature matrix.
    Args:
        node_feat(Variable): A tensor of node features with shape (num_nodes, feature_size).
        edge_feat(Variable): A tensor of edge features with shape (num_edges, feature_size).
        nlod(Variable): Graph Lod Index for Node Items for Concat.
        elod(Variable): Graph Lod Index for Edge Items for Concat.
    Returns:
        Variable: The updated node-edge feature matrix with shape (num_nodes + num_edges, feature_size).
    """
    node_feat_lod = L.lod_reset(node_feat, nlod)
    edge_feat_lod = L.lod_reset(edge_feat, elod)
    node_edge_feat_lod = L.sequence_concat([node_feat_lod, edge_feat_lod])
    return node_edge_feat_lod


def SpatialConv(e2n_gw, e2e_gw, srcs, dsts, node_edge_feat, \
		dist_feat_order, dist_feat, nids, eids, nlod, elod, hidden_size, name):
    """
    ** Spatial Graph Convolution Layer **
    The API implements the function of the spatial graph convolution layer for molecular graph。

    Args:
        e2n_gw(GraphWrapper): A graph wrapper for edge-to-node graph.
        e2e_gw(GraphWrapper): A graph wrapper for edge-to-edge graph.
        srcs(Variable): Source indices of edges with shape (num_edges, 1).
        dsts(Variable): Target indices of edges with shape (num_edges, 1).
        node_edge_feat(Variable): A tensor of node-edge feature matrix with shape (num_nodes + num_edges, feature_size).
        dist_feat_order(Variable): The spatial distance feature in the order of edge-edge matrix, the shape is (num_edges, embedding_size).
        dist_feat(Variable): The ispatial distance feature for the edges of node-node graph, the shape is (num_edges, embedding_size).

        nids(Variable): The indices of node items, the shape is (num_nodes, 1).
        eids(Variable): The indices of egde items, the shape is (num_edges, 1).
        nlod(Variable): Graph Lod Index for Node Items.
        elod(Variable): Graph Lod Index for Edge Items.
        name(str): The name of layer.

    Returns:
        Variable: The updated node-edge feature matrix with shape (num_nodes + num_edges, feature_size).
    """
    # step1. update edge features
    node_feat = L.gather(node_edge_feat, nids)
    edge_feat = aggregate_edges_from_nodes(node_edge_feat, dist_feat_order, srcs, dsts)
    node_edge_feat_lod = concat_node_edge_feat(node_feat, edge_feat, nlod, elod)
    node_edge_feat = aggregate_edges_from_edges(e2e_gw, node_edge_feat_lod, hidden_size, name + '_ee')

    # step2. update node features
    edge_feat = L.gather(node_edge_feat, eids)
    node_edge_feat_lod = concat_node_edge_feat(node_feat, edge_feat, nlod, elod)
    node_edge_feat = aggregate_nodes_from_edges(e2n_gw, node_edge_feat_lod, dist_feat, hidden_size, name + '_en')

    # update node-edge feature matrix
    node_feat = L.gather(node_edge_feat, nids)
    node_edge_feat = concat_node_edge_feat(node_feat, edge_feat, nlod, elod)
    return node_edge_feat


def sgat(gw,
        node_feat,
        edge_feat,
        hidden_size,
        name,
        activation='relu',
        combine='mean',
        num_heads=4,
        feat_drop=0.2,
        attn_drop=0.2,
        is_test=False):
    """
    The sgat function can aggregate the edge-neighbors of node to update the node embedding.
    Adapted from https://github.com/PaddlePaddle/PGL/blob/main/pgl/layers/conv.py.
    Args:
        gw(GraphWrapper): A graph wrapper for edge-node graph.
        node_feat(Variable): A tensor of node-edge features with shape (num_nodes + num_nodes, feature_size).
        edge_feat(Variable): A tensor of spatial distance features with shape (num_edges, feature_size).
        combine(str): The choice of combining multi-head embeddings. It can be mean, max or dense.

        hidden_size: The hidden size for gat.
        activation: The activation for the output.
        name: Gat layer names.
        num_heads: The head number in gat.
        feat_drop: Dropout rate for feature.
        attn_drop: Dropout rate for attention.
        is_test: Whether in test phrase.
    Returns:
        Variable: The updated node-edge feature matrix with shape (num_nodes + num_edges, feature_size).
    """

    def send_attention(src_feat, dst_feat, edge_feat):
        output = src_feat["left_a"] + dst_feat["right_a"]
        if 'edge_a' in edge_feat:
            output += edge_feat["edge_a"]
        output = L.leaky_relu(
            output, alpha=0.2)  # (num_edges, num_heads)
        return {"alpha": output, "h": src_feat["h"]}

    def reduce_attention(msg):
        alpha = msg["alpha"]  # lod-tensor (batch_size, seq_len, num_heads)
        h = msg["h"]
        alpha = paddle_helper.sequence_softmax(alpha)
        old_h = h
        h = L.reshape(h, [-1, num_heads, hidden_size])
        alpha = L.reshape(alpha, [-1, num_heads, 1])
        if attn_drop > 1e-15:
            alpha = L.dropout(
                alpha,
                dropout_prob=attn_drop,
                is_test=is_test,
                dropout_implementation="upscale_in_train")
        h = h * alpha
        h = L.reshape(h, [-1, num_heads * hidden_size])
        h = L.lod_reset(h, old_h)
        return L.sequence_pool(h, "sum")

    if feat_drop > 1e-15:
        node_feat = L.dropout(
                node_feat,
                dropout_prob=feat_drop,
                is_test=is_test,
                dropout_implementation='upscale_in_train')
        edge_feat = L.dropout(
                edge_feat,
                dropout_prob=feat_drop,
                is_test=is_test,
                dropout_implementation='upscale_in_train') 

    ft = L.fc(node_feat,
                         hidden_size * num_heads,
                         bias_attr=False,
                         param_attr=fluid.ParamAttr(name=name + '_weight'))
    left_a = L.create_parameter(
        shape=[num_heads, hidden_size],
        dtype='float32',
        name=name + '_gat_l_A')
    right_a = L.create_parameter(
        shape=[num_heads, hidden_size],
        dtype='float32',
        name=name + '_gat_r_A')
    reshape_ft = L.reshape(ft, [-1, num_heads, hidden_size])
    left_a_value = L.reduce_sum(reshape_ft * left_a, -1)
    right_a_value = L.reduce_sum(reshape_ft * right_a, -1)

    fd = L.fc(edge_feat,
            size=hidden_size * num_heads,
            bias_attr=False,
            param_attr=fluid.ParamAttr(name=name + '_fc_eW'))
    edge_a = L.create_parameter(
            shape=[num_heads, hidden_size],
            dtype='float32',
            name=name + '_gat_d_A')
    fd = L.reshape(fd, [-1, num_heads, hidden_size])
    edge_a_value = L.reduce_sum(fd * edge_a, -1)
    efeat_list = [('edge_a', edge_a_value)]
        
    msg = gw.send(
        send_attention,
        nfeat_list=[("h", ft), ("left_a", left_a_value),
                    ("right_a", right_a_value)], efeat_list=efeat_list)
    output = gw.recv(msg, reduce_attention)
    
    if combine == 'mean':
        output = L.reshape(output, [-1, num_heads, hidden_size])
        output = L.reduce_mean(output, dim=1)
        num_heads = 1
    if combine == 'max':
        output = L.reshape(output, [-1, num_heads, hidden_size])
        output = L.reduce_max(output, dim=1)
        num_heads = 1
    if combine == 'dense':
        output = L.fc(output, hidden_size, bias_attr=False, param_attr=fluid.ParamAttr(name=name + '_dense_combine'))
        num_heads = 1

    bias = L.create_parameter(
        shape=[hidden_size * num_heads],
        dtype='float32',
        is_bias=True,
        name=name + '_bias')
    bias.stop_gradient = True
    output = L.elementwise_add(output, bias, act=activation)
    return output


def gat(gw,
        feature,
        hidden_size,
        activation,
        name,
        dist_feat=None,
        num_heads=4,
        feat_drop=0.2,
        attn_drop=0.2,
        is_test=False):
    """Implementation of graph attention networks (GAT)
    Adapted from https://github.com/PaddlePaddle/PGL/blob/main/pgl/layers/conv.py.
    """

    def send_attention(src_feat, dst_feat, edge_feat):
        output = src_feat["left_a"] + dst_feat["right_a"]
        if 'dist_a' in edge_feat:
            output += edge_feat["dist_a"]
        output = L.leaky_relu(
            output, alpha=0.2)  # (num_edges, num_heads)
        return {"alpha": output, "h": src_feat["h"]}

    def reduce_attention(msg):
        alpha = msg["alpha"]  # lod-tensor (batch_size, seq_len, num_heads)
        h = msg["h"]
        alpha = paddle_helper.sequence_softmax(alpha)
        old_h = h
        h = L.reshape(h, [-1, num_heads, hidden_size])
        alpha = L.reshape(alpha, [-1, num_heads, 1])
        if attn_drop > 1e-15:
            alpha = L.dropout(
                alpha,
                dropout_prob=attn_drop,
                is_test=is_test,
                dropout_implementation="upscale_in_train")
        h = h * alpha
        h = L.reshape(h, [-1, num_heads * hidden_size])
        h = L.lod_reset(h, old_h)
        return L.sequence_pool(h, "sum")

    if feat_drop > 1e-15:
        feature = L.dropout(
            feature,
            dropout_prob=feat_drop,
            is_test=is_test,
            dropout_implementation='upscale_in_train')
        if dist_feat:
           dist_feat = L.dropout(
                       dist_feat,
                       dropout_prob=feat_drop,
                       is_test=is_test,
                       dropout_implementation='upscale_in_train') 

    ft = L.fc(feature,
                         hidden_size * num_heads,
                         bias_attr=False,
                         param_attr=fluid.ParamAttr(name=name + '_weight'))
    left_a = L.create_parameter(
        shape=[num_heads, hidden_size],
        dtype='float32',
        name=name + '_gat_l_A')
    right_a = L.create_parameter(
        shape=[num_heads, hidden_size],
        dtype='float32',
        name=name + '_gat_r_A')
    reshape_ft = L.reshape(ft, [-1, num_heads, hidden_size])
    left_a_value = L.reduce_sum(reshape_ft * left_a, -1)
    right_a_value = L.reduce_sum(reshape_ft * right_a, -1)
    efeat_list = []

    if dist_feat:
        fd = L.fc(dist_feat,
                  size=hidden_size * num_heads,
                  bias_attr=False,
                  param_attr=fluid.ParamAttr(name=name + '_fc_eW'))
        dist_a = L.create_parameter(
            shape=[num_heads, hidden_size],
            dtype='float32',
            name=name + '_gat_d_A')
        fd = L.reshape(fd, [-1, num_heads, hidden_size])
        dist_a_value = L.reduce_sum(fd * dist_a, -1)
        efeat_list = [('dist_a', dist_a_value)]
        
    msg = gw.send(
        send_attention,
        nfeat_list=[("h", ft), ("left_a", left_a_value),
                    ("right_a", right_a_value)], efeat_list=efeat_list)
    output = gw.recv(msg, reduce_attention)

    
    output = L.reshape(output, [-1, num_heads, hidden_size])
    output = L.reduce_mean(output, dim=1)
    num_heads = 1

    bias = L.create_parameter(
        shape=[hidden_size * num_heads],
        dtype='float32',
        is_bias=True,
        name=name + '_bias')
    bias.stop_gradient = True
    output = L.elementwise_add(output, bias, act=activation)
    return output
