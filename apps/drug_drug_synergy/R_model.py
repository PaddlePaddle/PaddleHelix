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
drug drug synergy model.
"""
import sys
import os
import re
import  paddle
from paddle.optimizer import Adam
import pgl
import paddle.nn.functional as F
import paddle.fluid.layers as L
import paddle.fluid.dygraph as D
import paddle.fluid as fluid
from pgl import heter_graph
from pgl.nn import functional as GF

import networkx as nx

import pandas as pd
import numpy as np

def Decagon_norm(graph, feature, edges):
    """
    Relation Graph Neural Network degree normalization method
    """
    D1 = np.zeros((graph.nodes.shape[0], graph.nodes.shape[0]))
    D2 = np.zeros((graph.nodes.shape[0], graph.nodes.shape[0]))
    D3 = np.zeros((graph.nodes.shape[0], graph.nodes.shape[0]))
    r = 1
    for t in edges.keys():
        for edge in edges[t]:
            i = edge[0]
            j = edge[1]
            #A[i,j] = 1
            if r == 1:
                D1[i, j] = 1
            
            elif r == 2:
                D2[i, j] = 1
            
            else:
                D3[i, j] = 1            
        r += 1

    diag1 = np.sum(D1, axis = 1)
    diag2 = np.sum(D2, axis = 1)
    diag3 = np.sum(D3, axis = 1)
    Diag = [diag1, diag2, diag3]

    D = np.identity(graph.nodes.shape[0])
    for i in enumerate(edges.keys()):
        for p in edges[i[1]]:
            D[p[0], p[1]] = Diag[i[0]][p[0]] * Diag[i[0]][p[1]]
    
    D = paddle.to_tensor(np.sum(D, axis = 1).astype('float32'))
    D = paddle.reshape(D, [D.shape[0], 1])
    return D


class RGCNConv(paddle.nn.Layer):
    """
    | RGCNConv, implementatopn of Relation-GCN layer

    Public Functions:
        - ``forward``: forward to output the graph node representation``.
    """
    def __init__(self, in_dim, out_dim, etypes, num_bases=0, act='relu', norm=True):
        super(RGCNConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.etypes = etypes
        self.num_rels = len(self.etypes)
        self.num_bases = num_bases

        if self.num_bases <= 0 or self.num_bases >= self.num_rels:
            self.num_bases = self.num_rels
        
        self.weight = self.create_parameter(
                        shape=[self.num_bases, self.in_dim, self.out_dim])
        if self.num_bases < self.num_rels:
            self.w_comp = self.create_parameter(
                shape=[self.num_rels, self.num_bases]
            )
        self.act = act
        self.norm = norm
    def forward(self, graph, feat):
        """Forward
        Args:
            graph: hetergeneous graph built by pgl.HeterGraph.
            inputs: node features/representation from graph/previous layer.
        """
        if self.num_bases < self.num_rels:
            weight = paddle.transpose(self.weight, perm=[1, 0, 2])
            weight = paddle.matmul(self.w_comp, weight)
            weight = paddle.transpose(weight, perm=[1, 0, 2])
        else:
            weight = self.weight

        def send_func(src_feat, dst_feat, edge_feat):
            """
            send function
            """
            return src_feat

        def recv_func(msg):
            """
            receive function
            """
            return msg.reduce_mean(msg['h'])

        feat_list = []
             
        for idx, etype in enumerate(self.etypes):
            sub_g = graph[graph.edge_types[idx]]
            sub_g.tensor()
            if self.norm:
                norm = GF.degree_norm(sub_g)
            feat = feat * norm
            w = weight[idx, :, :].squeeze()
            h = paddle.matmul(feat, w)
            msg = sub_g.send(send_func, src_feat={'h':h})
            h = sub_g.recv(recv_func, msg)
            feat_list.append(h)
        h = paddle.stack(feat_list, axis=0)
        h = paddle.sum(h, axis=0)
        if self.act == 'relu':
            Act = paddle.nn.ReLU()
            h = Act(h)
        else:
            Act = paddle.nn.Sigmoid()
            h = Act(h)

        return h

class BilinearDecoder(paddle.nn.Layer):
    """
    | Bilinear_decoder, implementation of bilinear transform to desired shape.

    Public Functions:
        - ``forward``: forward to output with original shape``.
    """
    def __init__(self, input_size1, input_size2, output_size):
        super(BilinearDecoder, self).__init__()

        self.input_size1 = input_size1
        self.input_size2 = input_size2
        self.output_size = output_size
        self.w = np.random.random((self.output_size, self.input_size1, self.input_size2)).astype('float32')    

        self.bilinear = paddle.nn.Bilinear(in1_features=self.input_size1,
                                          in2_features=self.input_size2,
                                          out_features=self.output_size)
    def forward(self, inputs):
        """Forward"""
        x = paddle.to_tensor(inputs)
        output = self.bilinear(x, x)

        return output

class DDs(paddle.nn.Layer):
    """
    | DDs, implementatopn of the network architecture using R-GCN.

    Public Functions:
        - ``forward``: forward``.
    """
    def __init__(self, num_nodes, input_size, etypes, num_bases, outputsize):
        super(DDs, self).__init__()
        self.num_nodes = num_nodes
        self.input_size = input_size
        self.num_outsize = [1280, 640, 128, 48, outputsize]
        self.nfeat = self.create_parameter([self.num_nodes, self.input_size])
        self.etypes = etypes
        self.num_bases = num_bases
        self.layer1 = RGCNConv(self.input_size, self.num_outsize[0], self.etypes, 8, act='relu')
        self.layer2 = RGCNConv(self.num_outsize[0], self.num_outsize[1], self.etypes, 8, act='relu')
        self.layer3 = RGCNConv(self.num_outsize[1], self.num_outsize[2], self.etypes, 8, act='relu')
        self.layer4 = RGCNConv(self.num_outsize[2], self.num_outsize[3], self.etypes, 8, act='sigmoid')
        
        self.decoder = BilinearDecoder(self.num_outsize[-2], self.num_outsize[-2], self.num_outsize[-1])
    
    def forward(self, graph, inputs):
        """Forward
        Args:
            graph: hetergeneous graph built by pgl.HeterGraph.
            inputs: node features/representation from graph/previous layer.
        """
        h = self.layer1(graph, inputs)
        h = self.layer2(graph, h)
        h = self.layer3(graph, h)
        h = self.layer4(graph, h)
        h = self.decoder(h)
        h = paddle.nn.functional.sigmoid(h)

        return h.astype('float32')

#Negative Sampling training
#r = 1:2
def negative_Sampling(label):
    """mask negtive samples
       """
    valid = np.zeros((label.shape[0], label.shape[0]))
    num_pos = np.sum(label == 1.0)
    num_neg = num_pos * 2

    valid[np.where(label == 1.)] =1
    neg_pos = np.where(label == -1.)
    val_neg_idx = np.random.choice(len(neg_pos[0]), num_neg)
    valid[(neg_pos[0][val_neg_idx], neg_pos[1][val_neg_idx])] = -1

    return paddle.to_tensor(valid.astype('float32'))