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
DTA model
"""

import paddle
import paddle.nn as nn
import pgl

from pahelix.networks.compound_encoder import AtomEmbedding
from pahelix.utils.protein_tools import ProteinTokenizer
from pahelix.networks.gnn_block import MeanPool


class CompoundGNNModel(nn.Layer):
    """
    | CompoundGNNModel, implementation of the variant GNN models in paper
        ``GraphDTA: Predicting drug-target binding affinity with graph neural networks``.

    Public Functions:
        - ``forward``: forward to create the compound representation.
    """
    def __init__(self, config):
        super(CompoundGNNModel, self).__init__()
        self.hidden_size = config['hidden_size']
        self.embed_dim = config['embed_dim']
        self.output_dim = config['output_dim']
        self.dropout_rate = config['dropout_rate']
        self.layer_num = config['layer_num']
        self.gnn_type = config['gnn_type']

        self.gat_nheads = config.get('gat_nheads', 10)
        self.activation = config.get('activation', 'relu')
        self.atomic_numeric_feat_dim = config.get(
            'atomic_numeric_feat_dim', 28)

        self.atom_names = config['atom_names']
        self.bond_names = config['bond_names']

        self.atom_embedding = AtomEmbedding(self.atom_names, self.embed_dim)

        self.gnn_layers = nn.LayerList()
        if self.gnn_type == 'gcn':
            for layer_id in range(self.layer_num):
                self.gnn_layers.append(pgl.nn.GCNConv(
                    self._get_in_size(layer_id),
                    self.hidden_size,
                    activation=self.activation))

            self.graph_pool = pgl.nn.GraphPool(pool_type='max')
            self.fc = nn.Linear(self.hidden_size, self.output_dim)

        elif self.gnn_type == 'gat':
            for layer_id in range(self.layer_num):
                self.gnn_layers.append(pgl.nn.GATConv(
                    self._get_in_size(layer_id, self.gat_nheads),
                    self.hidden_size,
                    activation=self.activation,
                    num_heads=self.gat_nheads,
                    feat_drop=self.dropout_rate,
                    attn_drop=self.dropout_rate))

            self.graph_pool = pgl.nn.GraphPool(pool_type='max')
            in_size = self.hidden_size * self.gat_nheads
            self.fc = nn.Linear(in_size, self.output_dim)

        elif self.gnn_type == 'gin':
            for layer_id in range(self.layer_num):
                self.gnn_layers.append(pgl.nn.GINConv(
                    self._get_in_size(layer_id),
                    self.hidden_size,
                    activation=self.activation))
                self.gnn_layers.append(
                    nn.BatchNorm1D(self.hidden_size))

            self.graph_pool = pgl.nn.GraphPool(pool_type='sum')
            self.fc = nn.Linear(self.hidden_size, self.output_dim)

        elif self.gnn_type == "gat_gcn":
            self.gnn_layers.append(pgl.nn.GATConv(
                self._get_in_size(0),
                self.hidden_size,
                activation=self.activation,
                num_heads=self.gat_nheads,
                feat_drop=0.0,
                attn_drop=0.0))

            self.gnn_layers.append(pgl.nn.GCNConv(
                self.hidden_size * self.gat_nheads,
                self.hidden_size * self.gat_nheads,
                activation=self.activation))

            self.graph_max_pool = pgl.nn.GraphPool(pool_type='max')
            self.graph_avg_pool = MeanPool()

            dim = self.hidden_size * self.gat_nheads * 2
            self.fc1 = nn.Linear(dim, 1500)
            self.act1 = nn.ReLU()

            self.fc2 = nn.Linear(1500, self.output_dim)

        self.dropout = nn.Dropout(p=self.dropout_rate)

    def _get_in_size(self, layer_id, gat_heads=None):
        in_size = self.embed_dim + self.atomic_numeric_feat_dim
        gat_heads = 1 if gat_heads is None else gat_heads
        if layer_id > 0:
            in_size = self.hidden_size * gat_heads
        return in_size

    def _mol_encoder(self, graph):
        x = self.atom_embedding(graph.node_feat)
        x = paddle.squeeze(x, axis=1)
        x = paddle.concat([x, graph.node_feat['atom_numeric_feat']], axis=1)
        return x

    def forward(self, graph):
        """Forward function.

        Args:
            graph (pgl.Graph): a PGL Graph instance.
        """
        feat = self._mol_encoder(graph)
        for i in range(len(self.gnn_layers)):
            if isinstance(self.gnn_layers[i], nn.BatchNorm1D):
                feat = self.gnn_layers[i](feat)
            else:
                feat = self.gnn_layers[i](graph, feat)

        if self.gnn_type == 'gat_gcn':
            x1 = self.graph_max_pool(graph, feat)
            x2 = self.graph_avg_pool(graph, feat)
            feat = paddle.concat([x1, x2], axis=1)
            feat = self.dropout(self.act1(self.fc1(feat)))
            feat = self.fc2(feat)
        else:
            feat = self.graph_pool(graph, feat)
            feat = self.dropout(self.fc(feat))

        return feat


class ProteinSequenceModel(nn.Layer):
    """
    | ProteinSequenceModel, implementation of Conv1D model for protein representation.

    Public Functions:
        - ``forward``: forward to create protein sequence representation.
    """
    def __init__(self, config):
        super(ProteinSequenceModel, self).__init__()
        self.config = config
        self.output_dim = config['output_dim']
        self.embed_dim = config['embed_dim']
        self.max_protein_len = config['max_protein_len']
        self.vocab_size = len(ProteinTokenizer.vocab)

        self.num_filters = config.get('num_filters', 32)
        self.pool_type = config.get('pool_type', 'mean')
        self.initializer_range = config.get('initializer_range', 0.02)

        self.protein_embeddings =nn.Embedding(
            self.vocab_size, self.embed_dim,
            weight_attr=nn.initializer.TruncatedNormal(
                std=self.initializer_range))

        self.conv1d = nn.Conv1D(
            self.embed_dim, self.num_filters,
            kernel_size=8, padding='SAME', data_format='NLC')

        if self.max_protein_len < 0:
            self.fc = nn.Linear(self.num_filters, self.output_dim)
        else:
            self.fc = nn.Linear(self.num_filters * self.max_protein_len, self.output_dim)

    def forward(self, token, mask):
        """Forward.

        Args:
            token (Tensor): a tensor that represents the amino acid sequence as IDs.
            mask (Tensor): a tensor that marks whether the position is a valid amino acid or a padding.
        """
        token_emb = self.protein_embeddings(token)
        feat = self.conv1d(token_emb)

        if self.max_protein_len < 0:
            # average pooling
            feat = feat * paddle.unsqueeze(mask, 2)
            feat = paddle.sum(feat, axis=1) / paddle.sum(mask, 1, keepdim=True)
        else:
            feat = paddle.reshape(feat, [-1, self.max_protein_len * self.num_filters])

        feat = self.fc(feat)
        return feat


class DTAModel(nn.Layer):
    """
    | DTAModel, implementation of the network architecture in GraphDTA.

    Public Functions:
        - ``forward``: forward.
    """
    def __init__(self, config):
        super(DTAModel, self).__init__()
        self.dropout_rate = config['dropout_rate']

        self.compound_model = CompoundGNNModel(config['compound'])
        self.protein_model = ProteinSequenceModel(config['protein'])
        self.fc1 = nn.Linear(self.compound_model.output_dim + self.protein_model.output_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, graph, protein_token, protein_mask):
        """Forward function.

        Args:
            graph (pgl.Graph): a PGL Graph instance.
            protein_token (Tensor): a tensor that represents the amino acid sequence as IDs.
            protein_mask (Tensor): a tensor that marks whether the position is a valid amino acid or a padding.
        """
        compound_repr = self.compound_model(graph)
        protein_repr = self.protein_model(protein_token, protein_mask)
        compound_protein = paddle.concat(
            [compound_repr, protein_repr], axis=1)

        h = self.dropout(self.act(self.fc1(compound_protein)))
        h = self.dropout(self.act(self.fc2(h)))
        pred = self.fc3(h)
        return pred


class DTAModelCriterion(nn.Layer):
    """
    | DTAModelCriterion, implementation of MSE loss for DTA model.

    Public Functions:
        - ``forward``: forward function.
    """
    def __init__(self):
        super(DTAModelCriterion, self).__init__()

    def forward(self, pred, label):
        """Forward function.

        Args:
            pred (Tensor): affinity predictions, i.e. output from DTAModel.
            label (Tensor): affinity label.
        """
        loss = nn.functional.square_error_cost(pred, label)
        loss = paddle.mean(loss)
        return loss
