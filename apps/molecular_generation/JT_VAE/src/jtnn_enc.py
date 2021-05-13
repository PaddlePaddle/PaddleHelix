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
"""tree encoder"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from src.mol_tree import Vocab, MolTree
from src.nnutils import index_select_ND


class JTNNEncoder(nn.Layer):
    """Tree encodee layer"""

    def __init__(self, hidden_size, depth, embedding):
        super(JTNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.embedding = embedding
        self.outputNN = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU()
        )
        self.GRU = GraphGRU(hidden_size, hidden_size, depth=depth)

    def forward(self, fnode, fmess, node_graph, mess_graph, scope):
        """Forward
        Args:
            fnode(list): nodes ids.
            fmess(list): nodes order in tensorize_nodes process.
            node_graph(list): precursors of node_i.
            mess_graph(list): precursors of node_i in message(i->j).
            scope(list): record nodes number of each tree in a batch.
        Returns:
            tree vectors and message.
        """
        fnode = paddle.to_tensor(fnode)
        fmess = paddle.to_tensor(fmess)
        node_graph = paddle.to_tensor(node_graph)
        mess_graph = paddle.to_tensor(mess_graph)
        messages = paddle.zeros([mess_graph.shape[0], self.hidden_size])

        fnode = self.embedding(fnode)
        fmess = index_select_ND(fnode, 0, fmess)
        messages = self.GRU(messages, fmess, mess_graph)

        mess_nei = index_select_ND(messages, 0, node_graph)
        node_vecs = paddle.concat([fnode, paddle.sum(mess_nei, axis=1)], axis=-1)
        node_vecs = self.outputNN(node_vecs)

        batch_vecs = []
        for st, le in scope:
            cur_vecs = node_vecs[st]
            batch_vecs.append(cur_vecs)

        tree_vecs = paddle.stack(batch_vecs, axis=0)
        return tree_vecs, messages

    @staticmethod
    def tensorize(tree_batch):
        """tensorize"""
        node_batch = []
        scope = []
        for tree in tree_batch:
            scope.append((len(node_batch), len(tree.nodes)))
            node_batch.extend(tree.nodes)

        return JTNNEncoder.tensorize_nodes(node_batch, scope)

    @staticmethod
    def tensorize_nodes(node_batch, scope):
        """tensorize_nodes.
        Args:
            node_batch(list): nodes in a batch.
            scope: record nodes number of each tree in a batch.
        Returns:
            fnode: nodes ids.
            fmess: nodes order in tensorize_nodes process.
            node_graph: precursors of node_i.
            mess_graph: precursors of node_i in message(i->j).
            scope: record nodes number of each tree in a batch.
            mess_dict: message order.
        """
        messages, mess_dict = [None], {}
        fnode = []
        for x in node_batch:
            fnode.append(x.wid)
            for y in x.neighbors:
                mess_dict[(x.idx, y.idx)] = len(messages)
                messages.append((x, y))

        node_graph = [[] for i in range(len(node_batch))]
        mess_graph = [[] for i in range(len(messages))]
        fmess = [0] * len(messages)

        for x, y in messages[1:]:
            mid1 = mess_dict[(x.idx, y.idx)]
            fmess[mid1] = x.idx
            node_graph[y.idx].append(mid1)
            for z in y.neighbors:
                if z.idx == x.idx: continue
                mid2 = mess_dict[(y.idx, z.idx)]
                mess_graph[mid2].append(mid1)

        max_len = max([len(t) for t in node_graph] + [1])
        for t in node_graph:
            pad_len = max_len - len(t)
            t.extend([0] * pad_len)

        max_len = max([len(t) for t in mess_graph] + [1])
        for t in mess_graph:
            pad_len = max_len - len(t)
            t.extend([0] * pad_len)

        return (fnode, fmess, node_graph, mess_graph, scope), mess_dict


class GraphGRU(nn.Layer):
    """tbd"""

    def __init__(self, input_size, hidden_size, depth):
        super(GraphGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth

        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size, hidden_size, bias_attr=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, h, x, mess_graph):
        """tbd"""
        mask = paddle.ones([h.shape[0], 1])
        mask[0] = 0
        for it in range(self.depth):
            h_nei = index_select_ND(h, 0, mess_graph)
            sum_h = paddle.sum(h_nei, axis=1)
            z_input = paddle.concat([x, sum_h], axis=1)
            z = F.sigmoid(self.W_z(z_input))

            r_1 = paddle.reshape(self.W_r(x), shape=[-1, 1, self.hidden_size])
            r_2 = self.U_r(h_nei)
            r = F.sigmoid(r_1 + r_2)

            gated_h = r * h_nei
            sum_gated_h = paddle.sum(gated_h, axis=1)
            h_input = paddle.concat([x, sum_gated_h], axis=1)
            pre_h = F.tanh(self.W_h(h_input))
            h = (1.0 - z) * sum_h + z * pre_h
            h = h * mask
        return h


