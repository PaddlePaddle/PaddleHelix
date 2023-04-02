#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
| Blocks for Graph Neural Network (GNN)
| Adapted from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/model.py
"""


import paddle
import paddle.nn as nn
import pgl


class GraphNorm(nn.Layer):
    """Implementation of graph normalization. Each node features is divied by sqrt(num_nodes) per graphs.
    
    Args:
        graph: the graph object from (:code:`Graph`)
        feature: A tensor with shape (num_nodes, feature_size).

    Return:
        A tensor with shape (num_nodes, hidden_size)

    References:

    [1] BENCHMARKING GRAPH NEURAL NETWORKS. https://arxiv.org/abs/2003.00982

    """

    def __init__(self):
        super(GraphNorm, self).__init__()
        self.graph_pool = pgl.nn.GraphPool(pool_type="sum")

    def forward(self, graph, feature):
        """graph norm"""
        nodes = paddle.ones(shape=[graph.num_nodes, 1], dtype="float32")
        norm = self.graph_pool(graph, nodes)
        norm = paddle.sqrt(norm)
        norm = paddle.gather(norm, graph.graph_node_id)
        return feature / norm


class MeanPool(nn.Layer):
    """
    TODO: temporary class due to pgl mean pooling
    """
    def __init__(self):
        super().__init__()
        self.graph_pool = pgl.nn.GraphPool(pool_type="sum")

    def forward(self, graph, node_feat):
        """
        mean pooling
        """
        sum_pooled = self.graph_pool(graph, node_feat)
        ones_sum_pooled = self.graph_pool(
            graph,
            paddle.ones_like(node_feat, dtype="float32"))
        pooled = sum_pooled / ones_sum_pooled
        return pooled


class GIN(nn.Layer):
    """
    Implementation of Graph Isomorphism Network (GIN) layer with edge features
    """
    def __init__(self, hidden_size):
        super(GIN, self).__init__()
        self.mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size))

    def forward(self, graph, node_feat, edge_feat):
        """
        Args:
            node_feat(tensor): node features with shape (num_nodes, feature_size).
            edge_feat(tensor): edges features with shape (num_edges, feature_size).
        """
        def _send_func(src_feat, dst_feat, edge_feat):
            x = src_feat['h'] + edge_feat['h']
            return {'h': x}

        def _recv_func(msg):
            return msg.reduce_sum(msg['h'])

        msg = graph.send(
                message_func=_send_func,
                node_feat={'h': node_feat},
                edge_feat={'h': edge_feat})
        node_feat = graph.recv(reduce_func=_recv_func, msg=msg)
        node_feat = self.mlp(node_feat)
        return node_feat
