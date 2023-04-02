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
"""tree decoder"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from src.mol_tree import Vocab, MolTree, MolTreeNode
from src.nnutils import GRU
from src.chemutils import enum_assemble, set_atommap

MAX_NB = 15
MAX_DECODE_LEN = 100


class JTNNDecoder(nn.Layer):
    """Tree decoder layer"""

    def __init__(self, vocab, hidden_size, latent_size, embedding):
        super(JTNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab
        self.embedding = embedding
        latent_size = int(latent_size)
        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias_attr=False)
        self.W_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)

        self.W = nn.Linear(hidden_size + latent_size, hidden_size)

        self.U = nn.Linear(hidden_size + latent_size, hidden_size)
        self.U_i = nn.Linear(2 * hidden_size, hidden_size)

        self.W_o = nn.Linear(hidden_size, self.vocab_size)
        self.U_o = nn.Linear(hidden_size, 1)

        self.pred_loss = nn.CrossEntropyLoss(reduction='sum')
        self.stop_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def aggregate(self, hiddens, contexts, x_tree_vecs, mode):
        """Aggregate"""
        if mode == 'word':
            V, V_o = self.W, self.W_o
        elif mode == 'stop':
            V, V_o = self.U, self.U_o
        else:
            raise ValueError('aggregate mode is wrong')

        tree_contexts = paddle.index_select(axis=0, index=contexts, x=x_tree_vecs)
        input_vec = paddle.concat([hiddens, tree_contexts], axis=-1)
        output_vec = F.relu(V(input_vec))
        return V_o(output_vec)

    def forward(self, mol_batch, x_tree_vecs):
        """Tree decoding in training
        Args:
            mol_batch(list): mol objects in a batch.
            x_tree_vecs(tensor): tree latent representation.
        Returns:
            pred_loss: label prediction loss.
            stop_loss: topological prediction loss.
            pred_acc: label prediction accuracy.
            stop_acc: topological prediction accuracy.
        """
        pred_hiddens, pred_contexts, pred_targets = [], [], []
        stop_hiddens, stop_contexts, stop_targets = [], [], []
        traces = []

        for mol_tree in mol_batch:
            s = []
            dfs(s, mol_tree.nodes[0], -1)
            traces.append(s)
            for node in mol_tree.nodes:
                node.neighbors = []

        batch_size = len(mol_batch)

        pred_hiddens.append(paddle.zeros([len(mol_batch), self.hidden_size]))
        pred_targets.extend([mol_tree.nodes[0].wid for mol_tree in mol_batch])
        pred_contexts.append(paddle.to_tensor(list(range(batch_size))))

        max_iter = max([len(tr) for tr in traces])
        padding = paddle.zeros([self.hidden_size])
        padding.stop_gradient = False
        h = {}

        for t in range(max_iter):
            prop_list = []
            batch_list = []
            for i, plist in enumerate(traces):
                if t < len(plist):
                    prop_list.append(plist[t])
                    batch_list.append(i)

            cur_x = []
            cur_h_nei, cur_o_nei = [], []

            for node_x, real_y, _ in prop_list:
                cur_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors if node_y.idx != real_y.idx]
                pad_len = MAX_NB - len(cur_nei)
                cur_h_nei.extend(cur_nei)
                cur_h_nei.extend([padding] * pad_len)

                cur_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
                pad_len = MAX_NB - len(cur_nei)
                cur_o_nei.extend(cur_nei)
                cur_o_nei.extend([padding] * pad_len)

                cur_x.append(node_x.wid)

            cur_x = paddle.to_tensor(cur_x)
            cur_x = self.embedding(cur_x)

            cur_h_nei = paddle.reshape(paddle.stack(cur_h_nei, axis=0), shape=[-1, MAX_NB, self.hidden_size])
            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)

            cur_o_nei = paddle.reshape(paddle.stack(cur_o_nei, axis=0), shape=[-1, MAX_NB, self.hidden_size])
            cur_o = paddle.sum(cur_o_nei, axis=1)

            pred_target, pred_list = [], []
            stop_target = []
            for i, m in enumerate(prop_list):
                node_x, node_y, direction = m
                x, y = node_x.idx, node_y.idx
                h[(x, y)] = new_h[i]
                node_y.neighbors.append(node_x)
                if direction == 1:
                    pred_target.append(node_y.wid)
                    pred_list.append(i)
                stop_target.append(direction)

            cur_batch = paddle.to_tensor((batch_list))
            stop_hidden = paddle.concat([cur_x, cur_o], axis=1)
            stop_hiddens.append(stop_hidden)
            stop_contexts.append(cur_batch)
            stop_targets.extend(stop_target)

            if len(pred_list) > 0:
                batch_list = [batch_list[i] for i in pred_list]
                cur_batch = paddle.to_tensor(batch_list)
                pred_contexts.append(cur_batch)

                cur_pred = paddle.to_tensor(pred_list)
                pred_hiddens.append(paddle.index_select(axis=0, index=cur_pred, x=new_h))
                pred_targets.extend(pred_target)

        cur_x, cur_o_nei = [], []
        for mol_tree in mol_batch:
            node_x = mol_tree.nodes[0]
            cur_x.append(node_x.wid)
            cur_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
            pad_len = MAX_NB - len(cur_nei)
            cur_o_nei.extend(cur_nei)
            cur_o_nei.extend([padding] * pad_len)

        cur_x = paddle.to_tensor(cur_x)
        cur_x = self.embedding(cur_x)
        cur_o_nei = paddle.reshape(paddle.stack(cur_o_nei, axis=0), shape=[-1, MAX_NB, self.hidden_size])
        cur_o = paddle.sum(cur_o_nei, axis=1)

        stop_hidden = paddle.concat([cur_x, cur_o], axis=1)
        stop_hiddens.append(stop_hidden)
        stop_contexts.append(paddle.to_tensor(list(range(batch_size))))
        stop_targets.extend([0] * len(mol_batch))

        pred_contexts = paddle.concat(pred_contexts, axis=0)
        pred_hiddens = paddle.concat(pred_hiddens, axis=0)
        pred_scores = self.aggregate(pred_hiddens, pred_contexts, x_tree_vecs, 'word')
        pred_targets = paddle.to_tensor(pred_targets)

        pred_loss = self.pred_loss(pred_scores, pred_targets) / len(mol_batch)
        preds = paddle.argmax(pred_scores, axis=1)
        pred_acc = paddle.equal(preds, pred_targets).astype('float32')
        pred_acc = paddle.sum(pred_acc) / pred_targets.size

        stop_contexts = paddle.concat(stop_contexts, axis=0)
        stop_hiddens = paddle.concat(stop_hiddens, axis=0)
        stop_hiddens = F.relu(self.U_i(stop_hiddens))
        stop_scores = self.aggregate(stop_hiddens, stop_contexts, x_tree_vecs, 'stop')
        stop_scores = stop_scores.squeeze(-1)
        stop_targets = paddle.to_tensor(stop_targets).astype('float32')

        stop_loss = self.stop_loss(stop_scores, stop_targets) / len(mol_batch)
        stops = paddle.greater_equal(stop_scores, paddle.ones(shape=[1])).astype('float32')
        stop_acc = paddle.equal(stops, stop_targets).astype('float32')
        stop_acc = paddle.sum(stop_acc) / stop_targets.size
        return {'pred_loss': pred_loss,
                'stop_loss': stop_loss,
                'pred_acc': float(pred_acc.numpy()),
                'stop_acc': float(stop_acc.numpy())}

    def decode(self, x_tree_vecs, prob_decode):
        """
        Decode tree structre from tree latent space.
        Args:
            x_tree_mess(tensor): tree latent represenation.
            prob_decode(bool): using bernoulli distribution in tree decode if prob_decode=true.
        Returns:
            root node and all nodes.
        """
        assert x_tree_vecs.shape[0] == 1
        stack = []
        init_hiddens = paddle.zeros([1, self.hidden_size])
        zero_pad = paddle.zeros([1, 1, self.hidden_size])
        contexts = paddle.zeros([1]).astype('int64')

        root_score = self.aggregate(init_hiddens, contexts, x_tree_vecs, 'word')
        root_wid = paddle.argmax(root_score, axis=1)
        root_wid = int(root_wid.numpy())

        root = MolTreeNode(self.vocab.get_smiles(root_wid))
        root.wid = root_wid
        root.idx = 0
        stack.append((root, self.vocab.get_slots(root.wid)))

        all_nodes = [root]
        h = {}
        for step in range(MAX_DECODE_LEN):
            node_x, fa_slot = stack[-1]
            cur_h_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
            if len(cur_h_nei) > 0:
                cur_h_nei = paddle.reshape(paddle.stack(cur_h_nei, axis=0), shape=[1, -1, self.hidden_size])
            else:
                cur_h_nei = zero_pad

            cur_x = paddle.to_tensor([node_x.wid])
            cur_x = self.embedding(cur_x)
            cur_h = paddle.sum(cur_h_nei, axis=1)
            stop_hiddens = paddle.concat([cur_x, cur_h], axis=1)
            stop_hiddens = F.relu(self.U_i(stop_hiddens))
            stop_score = self.aggregate(stop_hiddens, contexts, x_tree_vecs, 'stop')

            if prob_decode:
                backtrack = (paddle.bernoulli(F.sigmoid(stop_score)).item() == 0)
            else:
                backtrack = (float(stop_score.numpy()) < 0)

            if not backtrack:
                new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                pred_score = self.aggregate(new_h, contexts, x_tree_vecs, 'word')

                if prob_decode:
                    sort_wid = paddle.multinomial(F.softmax(pred_score, axis=1).squeeze(), 5)
                else:
                    sort_wid = paddle.argsort(
                        pred_score, axis=1, descending=True)
                    sort_wid = sort_wid.squeeze()

                next_wid = None
                for wid in sort_wid[:5]:
                    slots = self.vocab.get_slots(wid)
                    node_y = MolTreeNode(self.vocab.get_smiles(wid))
                    if have_slots(fa_slot, slots) and can_assemble(node_x, node_y):
                        next_wid = wid
                        next_slots = slots
                        break

                if next_wid is None:
                    backtrack = True
                else:
                    node_y = MolTreeNode(self.vocab.get_smiles(next_wid))
                    node_y.wid = int(next_wid.numpy())
                    node_y.idx = len(all_nodes)
                    node_y.neighbors.append(node_x)
                    h[(node_x.idx, node_y.idx)] = new_h[0]
                    stack.append((node_y, next_slots))
                    all_nodes.append(node_y)

            if backtrack:
                if len(stack) == 1:
                    break

                node_fa, _ = stack[-2]
                cur_h_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors if node_y.idx != node_fa.idx]
                if len(cur_h_nei) > 0:
                    cur_h_nei = paddle.reshape(paddle.stack(cur_h_nei, axis=0), shape=[1, -1, self.hidden_size])
                else:
                    cur_h_nei = zero_pad

                new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                h[(node_x.idx, node_fa.idx)] = new_h[0]
                node_fa.neighbors.append(node_x)
                stack.pop()

        return root, all_nodes


def dfs(stack, x, fa_idx):
    """dfs"""
    for y in x.neighbors:
        if y.idx == fa_idx:
            continue
        stack.append((x, y, 1))
        dfs(stack, y, x.idx)
        stack.append((y, x, 0))


def have_slots(fa_slots, ch_slots):
    """have slots"""
    if len(fa_slots) > 2 and len(ch_slots) > 2:
        return True
    matches = []
    for i, s1 in enumerate(fa_slots):
        a1, c1, h1 = s1
        for j, s2 in enumerate(ch_slots):
            a2, c2, h2 = s2
            if a1 == a2 and c1 == c2 and (a1 != "C" or h1 + h2 >= 4):
                matches.append((i, j))

    if len(matches) == 0: return False

    fa_match, ch_match = zip(*matches)
    if len(set(fa_match)) == 1 and 1 < len(fa_slots) <= 2:
        fa_slots.pop(fa_match[0])
    if len(set(ch_match)) == 1 and 1 < len(ch_slots) <= 2:
        ch_slots.pop(ch_match[0])

    return True


def can_assemble(node_x, node_y):
    """assemble candidate node """
    node_x.nid = 1
    node_x.is_leaf = False
    set_atommap(node_x.mol, node_x.nid)

    neis = node_x.neighbors + [node_y]
    for i, nei in enumerate(neis):
        nei.nid = i + 2
        nei.is_leaf = (len(nei.neighbors) <= 1)
        if nei.is_leaf:
            set_atommap(nei.mol, 0)
        else:
            set_atommap(nei.mol, nei.nid)

    neighbors = [nei for nei in neis if nei.mol.GetNumAtoms() > 1]
    neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
    singletons = [nei for nei in neis if nei.mol.GetNumAtoms() == 1]
    neighbors = singletons + neighbors
    cands, aroma_scores = enum_assemble(node_x, neighbors, [], [])
    return len(cands) > 0

