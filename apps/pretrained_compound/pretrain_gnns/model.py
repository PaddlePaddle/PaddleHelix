#!/usr/bin/python                                                                                                
#-*-coding:utf-8-*- 
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
gnn network
"""

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import pgl
from pgl.graph_wrapper import GraphWrapper

from pahelix.model_zoo import PretrainGNNModel


class DownstreamModel(object):
    """
    Docstring for DownstreamModel,

    it is an supervised GNN model which predicts the tasks shown in num_tasks,pool_type and so on.

    """
    def __init__(self, model_config):
        self.num_tasks = model_config['num_tasks']
        self.pool_type = model_config.get('pool_type', 'average')

        self.gnn_model = PretrainGNNModel(model_config, name='gnn')

    def forward(self, is_test=False):
        """
        Define the forward function,set the parameter layer options.

        Graph wrapper creates a graph data holders that attributes and features 
        in the graph are :code:`fluid.layers.data`.And we provide interface :
        code:`to_feed` to help converting :code:`Graph`data into :code:`feed_dict`.

        Args:
            name: The graph data prefix,here is graph

            node_feat: A list of tuples that decribe the details of node
                    feature tenosr. Each tuple must be (name, shape, dtype)
                    and the first dimension of the shape must be set unknown
                    (-1 or None) or we can easily use :code:`Graph.node_feat_info()`
                    to get the node_feat settings.

            edge_feat: A list of tuples that decribe the details of edge
                    feature tenosr. Each tuple mush be (name, shape, dtype)
                    and the first dimension of the shape must be set unknown
                    (-1 or None) or we can easily use :code:`Graph.edge_feat_info()`
                    to get the edge_feat settings.

        Returns:
            logits: the model prediction.
     
        """
        graph_wrapper = GraphWrapper(name="graph",
                node_feat=[
                    ('atom_type', [None, 1], "int64"), 
                    ('chirality_tag', [None, 1], "int64")],
                edge_feat=[
                    ('bond_type', [None, 1], "int64"),
                    ('bond_direction', [None, 1], "int64")])
        node_repr = self.gnn_model.forward(graph_wrapper, is_test=is_test)
        graph_repr = pgl.layers.graph_pooling(graph_wrapper, node_repr, self.pool_type)
        logits = layers.fc(graph_repr, size=self.num_tasks, name="finetune_fc")
        return graph_wrapper, logits

    def train(self, is_test=False):
        """
        Used for train/test with labels and train loss.
        """
        graph_wrapper, logits = self.forward(is_test=is_test)

        finetune_label = layers.data(name="finetune_label", shape=[None, self.num_tasks], dtype="float32")
        valid = layers.data("valid", shape=[None, self.num_tasks], dtype="float32")
        loss = layers.sigmoid_cross_entropy_with_logits(x=logits, label=finetune_label)
        loss = layers.reduce_sum(loss * valid) / layers.reduce_sum(valid)
        pred = layers.sigmoid(logits)

        self.graph_wrapper = graph_wrapper
        self.loss = loss
        self.pred = pred
        self.finetune_label = finetune_label

    def inference(self):
        """
        Used for inference with labels.
        """
        graph_wrapper, logits = self.forward(is_test=True)
        pred = layers.sigmoid(logits)
        self.graph_wrapper = graph_wrapper
        self.pred = pred

