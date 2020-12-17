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

from paddle import fluid
import pgl
from pgl.graph_wrapper import GraphWrapper

from pahelix.networks.pre_post_process import pre_process_layer
from pahelix.utils.protein_tools import ProteinTokenizer
from pahelix.utils.compound_tools import CompoundConstants
from pahelix.model_zoo.pretrain_gnns_model import PretrainGNNModel


class CompoundGNNModel(object):
    """
    | CompoundGNNModel, implementation of the variant GNN models in paper
        ``GraphDTA: Predicting drug-target binding affinity with graph neural networks``.

    Public Functions:
        - ``forward``: forward to create the compound representation.
    """
    def __init__(self, model_config, name=''):
        self.name = name

        self.hidden_size = model_config['hidden_size']
        self.embed_dim = model_config['embed_dim']
        self.output_dim = model_config['output_dim']
        self.dropout_rate = model_config['dropout_rate']
        self.layer_num = model_config['layer_num']
        self.gnn_type = model_config['gnn_type']

        self.gat_nheads = model_config.get('gat_nheads', 10)
        self.atom_type_num = model_config.get(
                'atom_type_num', len(CompoundConstants.atom_num_list) + 2)

    def _mol_encoder(self, graph_wrapper, name=""):
        embed_init = fluid.initializer.XavierInitializer(uniform=True)

        atom_type_embed = fluid.layers.embedding(
                input=graph_wrapper.node_feat['atom_type'],
                size=[self.atom_type_num, self.embed_dim],
                param_attr=fluid.ParamAttr(
                    name="%s_embed_atom_type" % name, initializer=embed_init))
        node_features = fluid.layers.concat(
            [atom_type_embed, graph_wrapper.node_feat['atom_numeric_feat']], axis=1)
        return node_features

    def _gnn_forward(self, graph_wrapper):
        """GCN, GAT, or GIN"""
        node_features = self._mol_encoder(graph_wrapper, name=self.name)

        features_list = [node_features]
        for layer in range(self.layer_num):
            if self.gnn_type == "gcn":
                feat = pgl.layers.gcn(
                        graph_wrapper,
                        features_list[layer],
                        self.hidden_size,
                        "relu",
                        "%s_layer%s_gcn" % (self.name, layer))
            elif self.gnn_type == "gat":
                feat = pgl.layers.gat(
                        graph_wrapper,
                        features_list[layer],
                        self.hidden_size,
                        "relu",
                        "%s_layer%s_gat" % (self.name, layer),
                        num_heads=self.gat_nheads,
                        feat_drop=self.dropout_rate,
                        attn_drop=self.dropout_rate)
            elif self.gnn_type == "gin":
                feat = pgl.layers.gin(
                        graph_wrapper,
                        features_list[layer],
                        self.hidden_size,
                        "relu",
                        "%s_layer%s_gin" % (self.name, layer))
                feat = fluid.layers.batch_norm(
                    feat,
                    param_attr=fluid.ParamAttr(
                        name="%s_layer%s_batchnorm" % (self.name, layer)))

            features_list.append(feat)

        graph_pooling_map = {
            "gin": "sum",
            "gcn": "max",
            "gat": "max"
        }
        feat = pgl.layers.graph_pooling(
            graph_wrapper, features_list[-1], graph_pooling_map[self.gnn_type])
        feat = fluid.layers.fc(
            feat, self.output_dim, name="%s_fc_out" % self.name, act="relu")
        feat = fluid.layers.dropout(
            feat, self.dropout_rate, dropout_implementation="upscale_in_train")
        return feat

    def _hybrid_gnn_forward(self, graph_wrapper):
        """GAT + GCN"""
        node_features = self._mol_encoder(graph_wrapper, name=self.name)

        feat = pgl.layers.gat(
                graph_wrapper,
                node_features,
                node_features.shape[1],
                "relu",
                "%s_layer_gat" % self.name,
                num_heads=self.gat_nheads,
                feat_drop=0.0,
                attn_drop=0.0)

        feat = pgl.layers.gcn(
                graph_wrapper,
                feat,
                node_features.shape[1] * self.gat_nheads,
                "relu",
                "%s_layer_gcn" % self.name)

        feat = fluid.layers.concat([
            pgl.layers.graph_pooling(graph_wrapper, feat, "max"),
            pgl.layers.graph_pooling(graph_wrapper, feat, "average")
        ], axis=1)

        feat = fluid.layers.fc(
            feat, 1500, act="relu", name="%s_fc1" % self.name)
        feat = fluid.layers.dropout(
            feat, self.dropout_rate, dropout_implementation="upscale_in_train")
        feat = fluid.layers.fc(feat, self.output_dim, name="%s_fc2" % self.name)
        return feat

    def forward(self, graph_wrapper):
        if self.gnn_type == "gat_gcn":
            return self._hybrid_gnn_forward(graph_wrapper)
        else:
            return self._gnn_forward(graph_wrapper)


class ProteinSequenceModel(object):
    """
    | ProteinSequenceModel, implementation of Conv1D model for protein representation.

    Public Functions:
        - ``forward``: forward to create protein sequence representation.
    """
    def __init__(self, model_config, name=''):
        self.name = name
        self.model_config = model_config
        self.output_dim = model_config['output_dim']
        self._fill_default_model_params()

        self.embed_dim = model_config['embed_dim']
        self.param_initializer = fluid.initializer.TruncatedNormal(
                scale=self.model_config['initializer_range'])

    def _fill_default_model_params(self):
        if 'initializer_range' not in self.model_config:
            self.model_config['initializer_range'] = 0.02
        if 'num_filters' not in self.model_config:
            self.model_config['num_filters'] = 32
        if 'vocab_size' not in self.model_config:
            self.model_config['vocab_size'] = len(ProteinTokenizer.vocab)
        if 'pool_type' not in self.model_config:
            self.model_config['pool_type'] = 'average'

    def forward(self, token):
        """Forward.

        Args:
            token (Variable): data variable that represents the amino acid sequence as IDs.
        """
        token_emb = fluid.layers.embedding(
                input=token,
                param_attr=fluid.ParamAttr(name='%s_token_emb' % self.name, initializer=self.param_initializer),
                size=[self.model_config['vocab_size'], self.embed_dim],
                padding_idx=0,
                is_sparse=False)
        feat = fluid.layers.sequence_conv(
            input=token_emb,
            num_filters=self.model_config['num_filters'],
            filter_size=8, name="%s_conv1d" % self.name)

        if self.model_config['max_protein_len'] < 0:
            feat = fluid.layers.sequence_pool(
                feat, pool_type=self.model_config['pool_type'])
        else:
            dim = self.model_config['max_protein_len'] * \
                self.model_config['num_filters']
            feat = fluid.layers.reshape(feat, [-1, dim])

        feat = fluid.layers.fc(
            feat, self.output_dim, name="%s_fc" % self.name)
        return feat


class DTAModel(object):
    """
    | DTAModel, implementation of the network architecture in GraphDTA.

    Public Functions:
        - ``forward``: forward.
        - ``train``: build the network, predictor and loss.
        - ``inference``: build the network and predictor.
    """
    def __init__(self,
                 model_config,
                 use_pretrained_compound_gnns=False):
        self.model_config = model_config
        self.use_pretrained_compound_gnns = use_pretrained_compound_gnns

        dim = CompoundConstants.atomic_numeric_feat_dim
        self.compound_graph_wrapper = GraphWrapper(
            name="compound_graph",
            node_feat=[
                ('atom_type', [None, 1], "int64"),
                ('chirality_tag', [None, 1], "int64"),
                ('atom_numeric_feat', [None, dim], "float32")],
            edge_feat=[
                ('bond_type', [None, 1], "int64"),
                ('bond_direction', [None, 1], "int64")
            ])

        protein_token = fluid.layers.data(name='protein_token', shape=[None, 1], dtype='int64')
        protein_token_lod = fluid.layers.data(name='protein_token_lod', shape=[None], dtype='int32')
        self.protein_token = fluid.layers.lod_reset(protein_token, y=protein_token_lod)

        if use_pretrained_compound_gnns:
            self.compound_model = PretrainGNNModel(model_config['compound'], name='gnn')  # TODO: update the name to 'compound'
        else:
            self.compound_model = CompoundGNNModel(model_config['compound'], name='compound')

        self.protein_model = ProteinSequenceModel(model_config['protein'], name='protein')

    def forward(self):
        compound_repr = self.compound_model.forward(self.compound_graph_wrapper)
        if self.use_pretrained_compound_gnns:
            compound_repr = pgl.layers.graph_pooling(
                self.compound_graph_wrapper, compound_repr, 'average')

        protein_sequence_repr = self.protein_model.forward(self.protein_token)
        compound_protein = fluid.layers.concat(
            [compound_repr, protein_sequence_repr], axis=1)

        h = fluid.layers.fc(compound_protein, 1024, act='relu')
        h = fluid.layers.dropout(h, self.model_config['dropout_rate'],
                                 dropout_implementation='upscale_in_train')
        h = fluid.layers.fc(h, 256, act='relu')
        h = fluid.layers.dropout(h, self.model_config['dropout_rate'],
                                 dropout_implementation='upscale_in_train')
        pred = fluid.layers.fc(h, 1)
        return pred

    def train(self):
        label = fluid.layers.data(name="label", dtype='float32', shape=[None, 1])
        pred = self.forward()
        loss = fluid.layers.square_error_cost(input=pred, label=label)
        loss = fluid.layers.mean(loss)

        self.pred = pred
        self.loss = loss

    def inference(self):
        self.pred = self.forward()
