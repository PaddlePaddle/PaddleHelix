#!/usr/bin/python                                                                                                
#-*-coding:utf-8-*- 
#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
transformer network
"""

import os
from copy import deepcopy
import paddle
import paddle.nn as nn
import pgl

from .basic_block import MLP, node_pooling, atom_pos_to_pair_dist
from .optimus import Optimus


class PairDistDiffHead(nn.Layer):
    def __init__(self, model_config, encoder_config):
        super(PairDistDiffHead, self).__init__()
        self.model_config = model_config

        self.linear = nn.Linear(encoder_config.pair_channel, model_config.num_bins)
        self.criterion = nn.CrossEntropyLoss()

    def _dist_to_bin(self, dist):
        stride = float(self.model_config.bin_end - self.model_config.bin_start) / self.model_config.num_bins
        bin_id = paddle.cast((dist - self.model_config.bin_start) / stride, 'int64')
        bin_id = paddle.clip(bin_id, 0, self.model_config.num_bins - 1)
        return bin_id

    def forward(self, batch, encoder_results):
        if not 'epoch_id' in batch or batch['epoch_id'] >= self.model_config.pretrain_steps:
            return {}
        if self.model_config.loss_scale == 0:
            return {}

        pair_pred = self.linear(encoder_results['pair_acts'])
        pair_dist_diff = atom_pos_to_pair_dist(batch['rdkit_atom_pos']) \
                - atom_pos_to_pair_dist(batch['raw_atom_pos'])
        pair_label = self._dist_to_bin(pair_dist_diff)
        loss = self.criterion(pair_pred, pair_label) * self.model_config.loss_scale
        results = {
            'loss': loss,
        }
        return results


class PropertyRegrHead(nn.Layer):
    def __init__(self, model_config, encoder_config):
        super(PropertyRegrHead, self).__init__()
        self.model_config = model_config

        self.label_mean = paddle.to_tensor(self.model_config.label_mean)
        self.label_std = paddle.to_tensor(self.model_config.label_std)

        input_channel = encoder_config.node_channel
        self.node_acts_ln = nn.LayerNorm(input_channel)
        self.mlp = MLP(
                in_size=input_channel,
                hidden_size=self.model_config.hidden_size,
                out_size=self.model_config.output_size)

        loss_dict = {
            'l1loss': nn.L1Loss(),
        }
        self.criterion = loss_dict[self.model_config.loss_type]

    def _get_scaled_label(self, x):
        return (x - self.label_mean) / (self.label_std + 1e-5)

    def _get_unscaled_pred(self, x):
        return x * (self.label_std + 1e-5) + self.label_mean

    def forward(self, batch, encoder_results):
        node_acts = encoder_results['node_acts']
        node_acts = self.node_acts_ln(node_acts)
        if self.model_config.get('use_segment_pool', False):
            compound_repr = pgl.math.segment_pool(
                    node_acts, batch["pool_node_id"], self.model_config.pool_type)
        else:
            compound_repr = node_pooling(
                    node_acts, batch["node_mask"], self.model_config.pool_type)
        scaled_pred = self.mlp(compound_repr)

        scaled_label = self._get_scaled_label(batch['label'])
        loss = self.criterion(scaled_pred, scaled_label)
        pred = self._get_unscaled_pred(scaled_pred)
        results = {
            'pred': pred,
            'loss': loss,
        }
        return results


class MolRegressionModel(nn.Layer):
    def __init__(self, model_config, encoder_config):
        super(MolRegressionModel, self).__init__()
        self.model_config = deepcopy(model_config.model)
        self.encoder_config = deepcopy(encoder_config)
        
        if self.model_config.encoder_type == 'optimus':
            self.encoder = Optimus(self.encoder_config)
        else:
            raise ValueError(self.model_config.encoder_type)

        head_dict = {
            "pair_dist_diff": PairDistDiffHead,
            "property_regr": PropertyRegrHead,
        }
        self.heads = nn.LayerDict()
        for name in self.model_config.heads:
            self.heads[name] = head_dict[name](
                    self.model_config.heads[name], self.encoder_config) 

    def forward(self, batch):
        if self.model_config.atom_pos_source == 'rdkit3d':
            batch['atom_pos'] = batch['rdkit_atom_pos']
        encoder_results = self.encoder(batch)

        results = {}
        total_loss = 0
        for name in self.model_config.heads:
            results[name] = self.heads[name](batch, encoder_results)
            if len(results[name]) > 0:
                total_loss += results[name]['loss']
        results['pred'] = results['property_regr']['pred']
        results['loss'] = total_loss
        return results

