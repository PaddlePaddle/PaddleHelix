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
This is an implementation of pretrain gnns:
https://arxiv.org/abs/1905.12265
"""

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import pgl
from pgl.graph_wrapper import GraphWrapper

from pahelix.networks.gnn_block import gcn_layer, gat_layer, gin_layer
from pahelix.utils.compound_tools import CompoundConstants


class PretrainGNNModel(object):
    """
    The basic GNN Model used in pretrain gnns.
    

    Args:
        model_config(dict): a dict of model configurations.
        name(str): the prefix of model params.
    """
    def __init__(self, model_config={}, name=''):
        self.name = name

        self.hidden_size = model_config.get('hidden_size', 256)
        self.embed_dim = model_config.get('embed_dim', 300)
        self.dropout_rate = model_config.get('dropout_rate', 0.5)
        self.norm_type = model_config.get('norm_type', 'batch_norm')
        self.graph_norm = model_config.get('graph_norm', False)
        self.residual = model_config.get('residual', False)
        self.layer_num = model_config.get('layer_num', 5)
        self.gnn_type = model_config.get('gnn_type', 'gin')
        self.JK = model_config.get('JK', 'last')

        self.atom_type_num = model_config.get(
                'atom_type_num', len(CompoundConstants.atom_num_list) + 2)
        self.chirality_tag_num = model_config.get(
                'chirality_tag_num', len(CompoundConstants.chiral_type_list) + 1)
        self.bond_type_num = model_config.get(
                'bond_type_num', len(CompoundConstants.bond_type_list) + 1)
        self.bond_direction_num = model_config.get(
                'bond_direction_num', len(CompoundConstants.bond_dir_list) + 1)
        self.embedding_trainable = model_config.get('embedding_trainable', True)

    def _mol_encoder(self, graph_wrapper, name=""):
        embed_init = fluid.initializer.XavierInitializer(uniform=True)

        atom_type_embed = layers.embedding(
                input=graph_wrapper.node_feat['atom_type'],
                size=[self.atom_type_num, self.embed_dim],
                param_attr=fluid.ParamAttr(
                    name="%s_embed_atom_type" % name,
                    initializer=embed_init,
                    trainable=self.embedding_trainable))
        chirality_tag_embed = layers.embedding(
                input=graph_wrapper.node_feat['chirality_tag'],
                size=[self.chirality_tag_num, self.embed_dim],
                param_attr=fluid.ParamAttr(
                    name="%s_embed_chirality_tag" % name,
                    initializer=embed_init,
                    trainable=self.embedding_trainable))
        node_features = atom_type_embed + chirality_tag_embed
        return node_features

    def _bond_encoder(self, graph_wrapper, name=""):
        embed_init = fluid.initializer.XavierInitializer(uniform=True)

        bond_type_embed = layers.embedding(
                input=graph_wrapper.edge_feat['bond_type'],
                size=[self.bond_type_num, self.embed_dim],
                param_attr=fluid.ParamAttr(
                    name="%s_embed_bond_type" % name,
                    initializer=embed_init,
                    trainable=self.embedding_trainable))
        bond_direction_embed = layers.embedding(
                input=graph_wrapper.edge_feat['bond_direction'],
                size=[self.bond_direction_num, self.embed_dim],
                param_attr=fluid.ParamAttr(
                    name="%s_embed_bond_direction" % name,
                    initializer=embed_init,
                    trainable=self.embedding_trainable))
        bond_features = bond_type_embed + bond_direction_embed
        return bond_features

    def forward(self, graph_wrapper, is_test=False):
        """
        Build the network.
        """
        node_features = self._mol_encoder(graph_wrapper, name=self.name)

        features_list = [node_features]
        for layer in range(self.layer_num):
            edge_features = self._bond_encoder(
                    graph_wrapper, 
                    name='%s_layer%s' % (self.name, layer))
            if self.gnn_type == "gcn":
                feat = gcn_layer(
                        graph_wrapper,
                        features_list[layer],
                        edge_features,
                        act="relu",
                        name="%s_layer%s_gcn" % (self.name, layer))
            elif self.gnn_type == "gat":
                feat = gat_layer(
                        graph_wrapper, 
                        features_list[layer],
                        edge_features,
                        self.embed_dim,
                        act="relu",
                        name="%s_layer%s_gat" % (self.name, layer))
            else:
                feat = gin_layer(
                        graph_wrapper,
                        features_list[layer],
                        edge_features,
                        name="%s_layer%s_gin" % (self.name, layer))

            if self.norm_type == 'batch_norm':
                feat = layers.batch_norm(
                        feat, 
                        param_attr=fluid.ParamAttr(
                            name="%s_layer%s_batch_norm_scale" % (self.name, layer),
                            initializer=fluid.initializer.Constant(1.0)),
                        bias_attr=fluid.ParamAttr(
                            name="%s_layer%s_batch_norm_bias" % (self.name, layer),
                            initializer=fluid.initializer.Constant(0.0)),
                        moving_mean_name="%s_layer%s_batch_norm_moving_avearage" % (self.name, layer),
                        moving_variance_name="%s_layer%s_batch_norm_moving_variance" % (self.name, layer),
                        is_test=is_test)
            elif self.norm_type == 'layer_norm':
                feat = layers.layer_norm(
                        feat, 
                        param_attr=fluid.ParamAttr(
                            name="%s_layer%s_layer_norm_scale" % (self.name, layer),
                            initializer=fluid.initializer.Constant(1.0)),
                        bias_attr=fluid.ParamAttr(
                            name="%s_layer%s_layer_norm_bias" % (self.name, layer),
                            initializer=fluid.initializer.Constant(0.0)))
            else:
                raise ValueError('%s not supported.' % self.norm_type)

            if self.graph_norm:
                feat = pgl.layers.graph_norm(graph_wrapper, feat)

            if layer < self.layer_num - 1:
                feat = layers.relu(feat)
            feat = layers.dropout(
                    feat,
                    self.dropout_rate,
                    dropout_implementation="upscale_in_train",
                    is_test=is_test)

            # residual
            if self.residual:
                feat = feat + features_list[layer]

            features_list.append(feat)

        if self.JK == "sum":
            node_repr = layers.reduce_sum(features_list, axis=0)
        elif self.JK == "mean":
            node_repr = layers.reduce_mean(features_list, axis=0)
        elif self.JK == "last":
            node_repr = features_list[-1]
        else:
            node_repr = features_list[-1]
        return node_repr


class PreGNNAttrmaskModel(object):
    """
    This is a pretraning model used by pretrain gnns for unsupervised training. 
    It randomly mask the atom_type of some nodes and use the masked atom_type 
    as the predicting target. 

    Returns:
        self.graph_wrapper: pgl graph_wrapper object for the input compound graph.
        self.loss: the loss variance of the model.
    """
    def __init__(self, model_config):
        self.gnn_model = PretrainGNNModel(model_config, name='gnn')

    def forward(self, is_test=False):
        """
        Build the network.
        """
        graph_wrapper = GraphWrapper(name="graph",
                node_feat=[
                    ('atom_type', [None, 1], "int64"), 
                    ('chirality_tag', [None, 1], "int64")],
                edge_feat=[
                    ('bond_type', [None, 1], "int64"),
                    ('bond_direction', [None, 1], "int64")])
        masked_node_indice = layers.data(name="masked_node_indice", shape=[-1, 1], dtype="int64")
        masked_node_label = layers.data(name="masked_node_label", shape=[-1, 1], dtype="int64")

        node_repr = self.gnn_model.forward(graph_wrapper, is_test=is_test)
        masked_node_repr = layers.gather(node_repr, masked_node_indice)
        logits = layers.fc(masked_node_repr, 
                size=len(CompoundConstants.atom_num_list),
                name="masked_node_logits")

        loss, pred = layers.softmax_with_cross_entropy(
                logits, masked_node_label, return_softmax=True)
        loss = layers.reduce_mean(loss)
        acc = layers.accuracy(pred, masked_node_label)

        self.graph_wrapper = graph_wrapper
        self.loss = loss


class PreGNNContextpredModel(object):
    """
    This is a pretraning model used by pretrain gnns for unsupervised training. For 
    a given node, it builds the substructure that corresponds to k hop neighbours 
    rooted at the node, and the context substructures that corresponds to the subgraph 
    that is between l1 and l2 hops away from the node. Then the feature space of 
    the subtructure and the context substructures are required to be close.

    Returns:
        self.substruct_graph_wrapper: pgl graph_wrapper object for the input substruct graph.
        self.context_graph_wrapper: pgl graph_wrapper object for the input context graph.
        self.loss: the loss variance of the model.
    """
    def __init__(self, model_config):
        self.context_pooling = model_config['context_pooling']

        # set up models, one for pre-training and one for context embeddings
        self.substruct_model = PretrainGNNModel(model_config, name='gnn')
        self.context_model = PretrainGNNModel(model_config, name='context_gnn')

    def forward(self, is_test=False):
        """
        Build the network.
        """
        substruct_graph_wrapper = GraphWrapper(name="graph",
                node_feat=[
                    ('atom_type', [None, 1], "int64"), 
                    ('chirality_tag', [None, 1], "int64")],
                edge_feat=[
                    ('bond_type', [None, 1], "int64"),
                    ('bond_direction', [None, 1], "int64")])
        context_graph_wrapper = GraphWrapper(name="context_graph",
                node_feat=[
                    ('atom_type', [None, 1], "int64"), 
                    ('chirality_tag', [None, 1], "int64")],
                edge_feat=[
                    ('bond_type', [None, 1], "int64"),
                    ('bond_direction', [None, 1], "int64")])
        substruct_center_idx = layers.data(name="substruct_center_idx", shape=[-1, 1], dtype="int64")
        context_overlap_idx = layers.data(name="context_overlap_idx", shape=[-1, 1], dtype="int64")
        context_overlap_lod = layers.data(name="context_overlap_lod", shape=[1, -1], dtype="int32")
        context_cycle_index = layers.data(name="context_cycle_index", shape=[-1, 1], dtype="int64")

        substruct_node_repr = self.substruct_model.forward(substruct_graph_wrapper, is_test=is_test)
        substruct_repr = layers.gather(substruct_node_repr, substruct_center_idx)

        context_node_repr = self.context_model.forward(context_graph_wrapper, is_test=is_test)
        context_overlap_repr = layers.gather(context_node_repr, context_overlap_idx)
        context_repr = layers.sequence_pool(
                layers.lod_reset(context_overlap_repr, context_overlap_lod), self.context_pooling)
        neg_context_repr = layers.gather(context_repr, context_cycle_index)

        pred_pos = layers.reduce_sum(substruct_repr * context_repr, 1)
        pred_neg = layers.reduce_sum(substruct_repr * neg_context_repr, 1)
        label_pos = pred_pos * 0.0 + 1.0
        label_pos.stop_gradient = True
        label_neg = pred_neg * 0.0
        label_neg.stop_gradient = True

        loss = layers.sigmoid_cross_entropy_with_logits(x=pred_pos, label=label_pos) \
                + layers.sigmoid_cross_entropy_with_logits(x=pred_neg, label=label_neg)
        loss = layers.reduce_mean(loss)

        self.substruct_graph_wrapper = substruct_graph_wrapper
        self.context_graph_wrapper = context_graph_wrapper
        self.loss = loss


class PreGNNSupervisedModel(object):
    """
    This is a pretraning model used by pretrain gnns for supervised training.

    Returns:
        self.graph_wrapper: pgl graph_wrapper object for the input compound graph.
        self.loss: the loss variance of the model.
    """
    def __init__(self, model_config):
        self.task_num = model_config['task_num']
        self.pool_type = model_config['pool_type']

        self.gnn_model = PretrainGNNModel(model_config, name='gnn')

    def forward(self, is_test=False):
        """
        Build the network.
        """
        graph_wrapper = GraphWrapper(name="graph",
                node_feat=[
                    ('atom_type', [None, 1], "int64"), 
                    ('chirality_tag', [None, 1], "int64")],
                edge_feat=[
                    ('bond_type', [None, 1], "int64"),
                    ('bond_direction', [None, 1], "int64")])
        supervised_label = layers.data(name="supervised_label", shape=[None, self.task_num], dtype="float32")
        valid = layers.data("valid", shape=[None, self.task_num], dtype="float32")

        node_repr = self.gnn_model.forward(graph_wrapper, is_test=is_test)
        graph_repr = pgl.layers.graph_pooling(graph_wrapper, node_repr, self.pool_type)
        logits = layers.fc(graph_repr, size=self.task_num, name="pretrain_supervised_fc")

        loss = layers.sigmoid_cross_entropy_with_logits(x=logits, label=supervised_label)
        loss = layers.reduce_sum(loss * valid) / layers.reduce_sum(valid)

        self.graph_wrapper = graph_wrapper
        self.loss = loss
