#-*- coding: utf-8 -*-
import sys
import os
import argparse
import traceback
import re
import io
import json
import yaml
import time 
import logging
from tqdm import tqdm
import numpy as np
from collections import namedtuple

import pgl
from pgl.utils import paddle_helper
from pgl.graph_wrapper import BatchGraphWrapper
from propeller import log
import propeller.paddle as propeller
import paddle.fluid as F
import paddle.fluid.layers as L

import models.layers as GNNlayers
from models.mol_encoder import AtomEncoder, BondEncoder

class GNNModel(object):
    def __init__(self, config, gw):
        self.config = config
        self.gw = gw
        self._build_model(gw)

    def _build_model(self, gw):
        self.atom_encoder = AtomEncoder(name="atom", emb_dim=self.config.embed_dim)
        self.bond_encoder = BondEncoder(name="bond", emb_dim=self.config.embed_dim)

        nfeat = self.atom_encoder(gw.node_feat['nfeat'])
        efeat = self.bond_encoder(gw.edge_feat['efeat'])

        feature_list = [nfeat]
        for layer in range(self.config.num_layers):
            if layer == self.config.num_layers - 1:
                act = None
            else:
                act = 'leaky_relu'

            feature = getattr(GNNlayers, self.config.layer_type)(
                    gw,
                    feature_list[layer],
                    efeat,
                    self.config.hidden_size,
                    act,
                    name="%s_%s" % (self.config.layer_type, layer))

            feature = self.mlp(feature, name="mlp_%s" % layer)
            feature = feature + feature_list[layer]
            feature_list.append(feature)

        self.feature_list = feature_list

    def get_node_repr(self):
        if self.config.JK == "last":
            return self.feature_list[-1]
        elif self.config.JK == "mean":
            return L.reduce_mean(self.feature_list, axis=0)
        else:
            return L.reduce_sum(self.feature_list, axis=0)

    def get_pooled_repr(self):
        feature = pgl.layers.graph_pooling(self.gw, 
                                           self.feature_list[-1], 
                                           self.config.graph_pool_type)
        return feature

    def mlp(self, features, name):
        h = features
        dim = features.shape[-1]
        dim_list = [dim * 2, dim]
        for i in range(2):
            h = GNNlayers.linear(h, dim_list[i], "%s_%s" % (name, i))
            h = GNNlayers.layer_norm(h, "norm_%s_%s" % (name, i))
            h = pgl.layers.graph_norm(self.gw, h)
            h = L.relu(h)
        return h

class DeeperGCNModel(object):
    def __init__(self, config, gw):
        self.model_name = str(self.__class__.__name__)
        self.config = config
        self.gw = gw
        self._build_model(gw)

    def _build_model(self, gw):
        self.atom_encoder = AtomEncoder(name="atom", emb_dim=self.config.embed_dim)
        self.bond_encoder = BondEncoder(name="bond", emb_dim=self.config.embed_dim)
        nfeat = self.atom_encoder(gw.node_feat['nfeat'])
        efeat = self.bond_encoder(gw.edge_feat['efeat'])

        feature = GNNlayers.gen_layer(gw, nfeat, efeat, 
                self.config.hidden_size, name="_gen_conv_0")

        for layer in range(1, self.config.num_layers):

            # LN/BN->ReLU->GraphConv->Res
            old_feature = feature

            # 1. Layer Norm
            if self.config.norm == "layer_norm" or self.config.norm is None:
                feature = GNNlayers.layer_norm(feature, "norm_%s_%s" % (self.model_name, layer))

            # 2. ReLU
            feature = L.relu(feature)

            #3. dropout
            feature = L.dropout(feature, 
                    dropout_prob=self.config.dropout_rate,
                    dropout_implementation="upscale_in_train")

            #4 gen_conv
            #  feature = pgl.layers.gen_conv(gw, feature,
            #          name="%s_gen_conv_%d" % (self.model_name, layer), beta=beta)
            feature = GNNlayers.gen_layer(gw, feature, efeat, 
                self.config.hidden_size, name="_gen_conv_%s" % layer)
            
            #5 res
            feature = feature + old_feature

            if self.config.repool == "repool":
                pooled_feat = pgl.layers.graph_pooling(self.gw, feature, "average")
                #  L.Print(self.gw._graph_lod, message="graph_lod")
                segment_idx = paddle_helper.lod2segment_ids(
                        self.gw._graph_lod, self.gw.num_graph)
                #  L.Print(segment_idx, message="segment_idx")
                unpooled_feat = L.gather(pooled_feat, segment_idx, overwrite=False)
                feature = feature + unpooled_feat

        # final layer: LN + relu + droput
        if self.config.norm == "layer_norm" or self.config.norm is None:
            feature = GNNlayers.layer_norm(feature,
                    "norm_scale_%s_%s" % (self.model_name, self.config.num_layers))
        #  feature = L.relu(feature)
        feature = L.dropout(feature, 
                dropout_prob=self.config.dropout_rate,
                dropout_implementation="upscale_in_train")

        if self.config.appnp:
            feature = pgl.layers.appnp(gw, feature, alpha=self.config.alpha, k_hop=5)

        if self.config.GN:
            feature = pgl.layers.graph_norm(gw, feature)

        self.feature = feature

    def get_node_repr(self):
        return self.feature

    def get_pooled_repr(self):
        feature = pgl.layers.graph_pooling(self.gw, 
                                           self.feature,
                                           self.config.graph_pool_type)
        return feature

    def get_mgf_repr(self):
        feature = GNNlayers.linear(self.feature, 2048, 'mgf_fc')
        feature = L.softmax(feature)
        feature = pgl.layers.graph_pooling(self.gw, feature, "sum")
        return feature

    def get_maccs_repr(self):
        feature = GNNlayers.linear(self.feature, 167, 'maccs_fc')
        feature = L.softmax(feature) # for every node
        feature = pgl.layers.graph_pooling(self.gw, feature, "sum")
        return feature


class GraphTransformerModel(object):
    def __init__(self, config, gw):
        self.model_name = str(self.__class__.__name__)
        self.config = config
        self.gw = gw
        self._build_model(gw)

    def _build_model(self, gw):
        self.atom_encoder = AtomEncoder(name="atom", emb_dim=self.config.embed_dim)
        self.bond_encoder = BondEncoder(name="bond", emb_dim=self.config.embed_dim)
        nfeat = self.atom_encoder(gw.node_feat['nfeat'])
        efeat = self.bond_encoder(gw.edge_feat['efeat'])

        feature_list = [nfeat]
        for layer in range(self.config.num_layers):

            feature = GNNlayers.graph_transformer(
                    gw,
                    feature_list[layer],
                    efeat,
                    self.config.hidden_size // self.config.num_heads,
                    name="%s_%s" % (self.config.layer_type, layer),
                    num_heads=self.config.num_heads)

            feature_list.append(feature)

        self.feature_list = feature_list

    def get_node_repr(self):
        if self.config.JK == "last":
            return self.feature_list[-1]
        elif self.config.JK == "mean":
            return L.reduce_mean(self.feature_list, axis=0)
        else:
            return L.reduce_sum(self.feature_list, axis=0)

    def get_pooled_repr(self):
        feature = pgl.layers.graph_pooling(self.gw, 
                                           self.feature_list[-1], 
                                           self.config.graph_pool_type)
        return feature


