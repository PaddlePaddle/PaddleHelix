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
import paddle.nn as nn
import pgl

from pahelix.networks.basic_block import MLP


class DownstreamModel(nn.Layer):
    """
    Docstring for DownstreamModel,it is an supervised 
    GNN model which predicts the tasks shown in num_tasks and so on.
    """
    def __init__(self, model_config, compound_encoder):
        super(DownstreamModel, self).__init__()
        self.num_tasks = model_config['num_tasks']

        self.compound_encoder = compound_encoder
        self.mlp = MLP(
                model_config['layer_num'],
                in_size=compound_encoder.graph_dim,
                hidden_size=model_config['hidden_size'],
                out_size=self.num_tasks,
                act=model_config['act'],
                dropout_rate=model_config['dropout_rate'])
        self.out_act = nn.Sigmoid()

    def forward(self, graphs):
        """
        Define the forward function,set the parameter layer options.compound_encoder 
        creates a graph data holders that attributes and features in the graph.
        Returns:
            pred: the model prediction.
        """
        node_repr, graph_repr = self.compound_encoder(graphs)
        logits = self.mlp(graph_repr)
        pred = self.out_act(logits)
        return pred

