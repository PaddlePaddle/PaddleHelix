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
This file implement the S-MAN model for drug-target binding affinity prediction.
"""

import numpy as np

import paddle.fluid as fluid
import paddle.fluid.layers as fl
import pgl
import layers


class SMANModel(object):
    """
    **Spatial-aware Molecular Graph Attention Network for DTA Prediction**
        Args:
            args: Configure information for model training.
            e2n_gw(GraphWrapper): A graph wrapper for edge-to-node graph.
            e2e_gw(GraphWrapper): A graph wrapper for edge-to-edge graph.
            n_output(int, optional): The size of output.
                    Default: 1.
            is_test(bool, optional): If running in testing mode or not
                    Default: True.
    """
    def __init__(self, args, e2n_gw, e2e_gw, n_output=1, is_test=False):
        self.args = args
        self.num_layers = self.args.num_layers
        self.hidden_size = self.args.hid_dim
        self.pool_type = self.args.pool_type
        self.dropout_prob = self.args.drop
        self.dist_dim = self.args.dist_dim
        self.n_output = n_output
        self.is_test = is_test

        self.e2n_gw = e2n_gw
        self.e2e_gw = e2e_gw
        # self.edges = fl.data(name="edges", shape=[None, 2], dtype="int32")
        self.edges_dist = fl.data(name="edges_dist", shape=[None, self.dist_dim], dtype="float32")
        self.nids = fl.data(name="nids", shape=[None, 1], dtype="int32")
        self.eids = fl.data(name="eids", shape=[None, 1], dtype="int32")
        self.srcs = fl.data(name="srcs", shape=[None, 1], dtype="int32")
        self.dsts = fl.data(name="dsts", shape=[None, 1], dtype="int32")
        self.node_lod = fl.data(name="node_lod", shape=[None, ], dtype="int32")
        self.edge_lod = fl.data(name="edge_lod", shape=[None, ], dtype="int32")
        self.pk = fl.data(name="pk", shape=[None, 1], dtype="float32")

    def forward(self):
        """forward"""
        dist_feat_order = self.edges_dist
        dist_feat = self.e2n_gw.edge_feat['dist']
        dist_feat, dist_feat_order = layers.spatial_embedding(dist_feat, dist_feat_order, self.hidden_size)

        node_edge_feat = self.e2n_gw.node_feat["attr"]
        feat_size = node_edge_feat.shape[-1]

        for i in range(self.num_layers):
            out_size = self.hidden_size if i == self.num_layers+1 else feat_size
            feat_h = layers.SpatialConv(self.e2n_gw, self.e2e_gw, self.srcs, self.dsts, node_edge_feat, dist_feat_order, dist_feat,
                                        self.nids, self.eids, self.node_lod, self.edge_lod, out_size, name="layer_%s" % (i))
            node_edge_feat = feat_h

        node_feat = fl.gather(node_edge_feat, self.nids)
        pooled_h = layers.graph_pooling(node_feat, self.node_lod, self.pool_type)

        output = fl.fc(pooled_h, size=self.hidden_size*4, act='relu')
        output = fl.dropout(output, self.dropout_prob, dropout_implementation="upscale_in_train")
        output = fl.fc(output, size=self.hidden_size*2, act='relu')
        output = fl.dropout(output, self.dropout_prob, dropout_implementation="upscale_in_train")
        output = fl.fc(output, size=self.hidden_size*1, act='relu')
        output = fl.dropout(output, self.dropout_prob, dropout_implementation="upscale_in_train")
        self.output = fl.fc(output, size=self.n_output, act=None)
        
        # calculate loss
        self.loss = fl.mse_loss(self.output, self.pk)
        self.loss = fl.reduce_mean(self.loss)