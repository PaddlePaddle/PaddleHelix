# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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
from functools import partial

import pgl
from pgl.utils import paddle_helper
from pgl.graph_wrapper import BatchGraphWrapper
from propeller import log
from propeller.types import RunMode
import propeller.paddle as propeller
import paddle.fluid as F
import paddle.fluid.layers as L
from ogb.utils.features import allowable_features

import models.gnn_model as GM
import models.layers as GNNlayers

class MgfModel(propeller.train.Model):
    def __init__(self, pretrained_model_config, mode, run_config):
        self.hparam = pretrained_model_config
        self.mode = mode
        self.run_config = run_config

    def forward(self, features):
        gw = BatchGraphWrapper(features['num_nodes'],
                               features['num_edges'],
                               features['edges'],
                               node_feats={'nfeat': features['nfeat']},
                               edge_feats={'efeat': features['efeat']})

        self.gnn_model = getattr(GM, self.hparam.pt_model_type)(self.hparam, gw)
        feature = self.gnn_model.get_mgf_repr()

        return [feature]

    def loss(self, predictions, label):
        logits = predictions[0]

        #  loss = L.reduce_mean((logits - label)**2)
        loss = L.sigmoid_cross_entropy_with_logits(logits, label)
        loss = L.reduce_mean(loss)

        return loss

    def backward(self, loss):
        optimizer = F.optimizer.Adam(learning_rate=self.run_config.lr)
        optimizer.minimize(loss)

    def metrics(self, predictions, label):
        result = {}
        logits = predictions[0]
        result['mse'] = propeller.metrics.MSE(label, logits)

        return result

