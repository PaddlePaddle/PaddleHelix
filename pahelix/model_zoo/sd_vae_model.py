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
This is an implementation of SD VAE 
"""
from __future__ import print_function
from past.builtins import range

import os
import sys
import numpy as np
import math
import random

import paddle
import paddle.nn.functional as F
import paddle.nn as nn
import pdb

sys.path.append('../mol_common')
from mol_util import rule_ranges, terminal_idxes, DECISION_DIM
from cmd_args import cmd_args
from paddle_initializer import weights_init


class StateDecoder(nn.Layer):
    """encoder

    Args:
        max_len: the maximun length of input sequemce
        latent_dim: the dimension of latent space of encoder
        rnn_type: the rnn type
    """
    def __init__(self, max_len, latent_dim, rnn_type):
        super(StateDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len

        self.z_to_latent = nn.Linear(self.latent_dim, self.latent_dim)
        if rnn_type == 'gru':
            self.gru = nn.GRU(self.latent_dim, 501, 3)
        else:
            raise NotImplementedError

        self.decoded_logits = nn.Linear(501, DECISION_DIM)
        weights_init(self)

    def forward(self, z, n_steps=None):
        """
        encoder forward
        """
        if n_steps is None:
            n_steps = self.max_len
            
        h = self.z_to_latent(z)
        h = F.relu(h)
        
        rep_h = paddle.expand(h.unsqueeze(1), shape = [z.unsqueeze(1).shape[0], n_steps, z.unsqueeze(1).shape[2]]) # repeat along time steps

        out, _ = self.gru(rep_h) # run multi-layer gru

        logits = self.decoded_logits(out)
        logits = paddle.transpose(logits, (1, 0, 2))

        return logits


class PerpCalculator(nn.Layer):
    """loss type

    Args:
        true_binary: one-hot, with size=time_steps x bsize x DECISION_DIM
        rule_masks: binary tensor, with size=time_steps x bsize x DECISION_DIM
        raw_logits: real tensor, with size=time_steps x bsize x DECISION_DIM
    """
    def __init__(self):
        super(PerpCalculator, self).__init__()

    def forward(self, true_binary, rule_masks, raw_logits):
        """
        forward
        """
        if cmd_args.loss_type == 'binary':
            exp_pred = paddle.exp(raw_logits) * rule_masks

            norm = paddle.sum(exp_pred, axis=2, keepdim=True)
            prob = paddle.divide(exp_pred, norm)

            return F.binary_cross_entropy(prob, true_binary) * cmd_args.max_decode_steps

        if cmd_args.loss_type == 'perplexity':
            my_perp_loss = MyPerpLoss()
            return my_perp_loss(true_binary, rule_masks, raw_logits)

        if cmd_args.loss_type == 'vanilla':
            exp_pred = paddle.exp(raw_logits) * rule_masks + 1e-30
            norm = paddle.sum(exp_pred, 2, keepdim=True)
            prob = paddle.divide(exp_pred, norm)

            ll = paddle.abs(paddle.sum(true_binary * prob, 2))
            mask = 1 - rule_masks[:, :, -1]
            logll = mask * paddle.log(ll)

            loss = -paddle.sum(logll) / true_binary.shape[1]
            
            return loss
        print('unknown loss type %s' % cmd_args.loss_type)
        raise NotImplementedError


class MyPerpLoss(nn.Layer):
    """perplexity loss
    """
    def __init__(self):
        super(MyPerpLoss, self).__init__()
        
    def forward(self, true_binary, rule_masks, input_logits):
        """
        tbd
        """
        b = paddle.max(input_logits, 2, keepdim=True)[0]
        raw_logits = input_logits - b
        exp_pred = paddle.exp(raw_logits) * rule_masks + 1e-30
        
        norm = paddle.sum(exp_pred, 2, keepdim=True)
        prob = paddle.divide(exp_pred, norm)
        
        ll = paddle.abs(paddle.sum(true_binary * prob, 2))
        
        mask = 1 - rule_masks[:, :, -1]

        logll = mask * paddle.log(ll)

        loss = -paddle.sum(logll) / true_binary.shape[1]
               
        return loss


class CNNEncoder(nn.Layer):
    """the encoder

    Args:
        max_len: the maximum length of input 
        latent_dim: the dimension of latent space of encoder
    """
    def __init__(self, max_len, latent_dim):
        super(CNNEncoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.max_len = max_len

        self.conv1 = nn.Conv1D(DECISION_DIM, 9, 9)
        self.conv2 = nn.Conv1D(9, 9, 9)
        self.conv3 = nn.Conv1D(9, 10, 11)

        self.last_conv_size = max_len - 9 + 1 - 9 + 1 - 11 + 1
        self.w1 = nn.Linear(self.last_conv_size * 10, 435)
        self.mean_w = nn.Linear(435, latent_dim)
        self.log_var_w = nn.Linear(435, latent_dim)
        
        weights_init(self)
        
    def forward(self, x):
        """
        encoder forward
        """

        batch_input = x

        h1 = self.conv1(batch_input)
        h1 = F.relu(h1)        
        h2 = self.conv2(h1)
        h2 = F.relu(h2)
        h3 = self.conv3(h2)
        h3 = F.relu(h3)

        flatten = paddle.reshape(h3, shape=[batch_input.shape[0], -1])
        
        h = self.w1(flatten)
        h = F.relu(h)

        z_mean = self.mean_w(h)
        z_log_var = self.log_var_w(h)
        
        return (z_mean, z_log_var)


def get_encoder(model_config):
    """
    get the encoder
    """
    if model_config['encoder_type'] == 'cnn':
        return CNNEncoder(max_len=model_config['max_decode_steps'], latent_dim=model_config['latent_dim'])
    else:
        raise ValueError('unknown encoder type %s' % model_config['encoder_type'])


class MolVAE(nn.Layer):
    """The Mol VAE model

    Args:
        model_config: the model parameters
    """
    def __init__(self, model_config):
        super(MolVAE, self).__init__()
        print('using vae')
        self.model_config = model_config
        self.latent_dim = self.model_config['latent_dim']
        self.encoder = get_encoder(self.model_config)
        self.state_decoder = StateDecoder(max_len=self.model_config['max_decode_steps'], \
                            latent_dim=self.model_config['latent_dim'], rnn_type=self.model_config['rnn_type'])
        self.perp_calc = PerpCalculator()

    def reparameterize(self, mu, logvar):
        """
        reparameterize trick
        """
        eps = paddle.normal(mean=0, std=self.model_config['eps_std'], shape=mu.shape)            
        return mu + (logvar / 2).exp() * eps   

    def forward(self, x_inputs, true_binary, rule_masks):    
        """
        MOL VAE forward
        """    
        z_mean, z_log_var = self.encoder(x_inputs)

        z = self.reparameterize(z_mean, z_log_var)

        raw_logits = self.state_decoder(z)
        perplexity = self.perp_calc(true_binary, rule_masks, raw_logits)

        kl_loss = 0.5 * (z_log_var.exp() + z_mean ** 2 - 1 - z_log_var).sum(1).mean()

        return perplexity, kl_loss

