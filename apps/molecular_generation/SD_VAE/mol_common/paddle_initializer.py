#!/usr/bin/python3                                                                                                
#-*-coding:utf-8-*- 
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
paddle initializer
"""

from __future__ import print_function


import os
import sys
import numpy as np

import paddle
import paddle.nn.functional as F
import paddle.nn as nn
import pdb

def glorot_uniform(t):
    """
    tbd
    """
    if len(t.shape) == 2:
        fan_in, fan_out = t.shape
    elif len(t.shape) == 3:
        # out_ch, in_ch, kernel for Conv 1
        fan_in = t.shape[1] * t.shape[2]
        fan_out = t.shape[0] * t.shape[2]
    else:
        fan_in = np.prod(t.shape)
        fan_out = np.prod(t.shape)

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    weight = paddle.uniform(shape=t.shape, min=-limit, max=limit)
    t.set_value(weight)
                        

def orthogonal(shape, gain=1.0):
    """
    tbd
    """
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are "
                           "supported.")

    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return gain * q
                

def orthogonal_gru(t):
    """
    tbd
    """
    assert len(t.shape) == 2
    assert t.shape[0] == 3 * t.shape[1]
    hidden_dim = t.shape[1]

    x0 = orthogonal([hidden_dim, hidden_dim])
    x1 = orthogonal([hidden_dim, hidden_dim])
    x2 = orthogonal([hidden_dim, hidden_dim])

    x0 = paddle.to_tensor(x0, dtype='float32')
    x1 = paddle.to_tensor(x1, dtype='float32')
    x2 = paddle.to_tensor(x2, dtype='float32')

    t.set_value(paddle.concat([x0, x1, x2], 0))
    

def weights_init(m):
    """
    tbd
    """
    for p in m.sublayers():
        if isinstance(p, nn.Conv1D):
            # bias initialization is 0
            # initialize weight
            glorot_uniform(p.weight)
            print('a Conv1d inited')
        if isinstance(p, nn.Linear):
            glorot_uniform(p.weight)
            print('a Linear inited')
        if isinstance(p, nn.GRU):
            for k in range(p.num_layers):
                
                bias_ih = getattr(p, 'bias_ih_l%d' % k)
            
                getattr(p, 'bias_ih_l%d' % k).set_value(paddle.zeros(shape=getattr(p, 'bias_ih_l%d' % k).shape))
                getattr(p, 'bias_hh_l%d' % k).set_value(paddle.zeros(shape=getattr(p, 'bias_hh_l%d' % k).shape))
                glorot_uniform(getattr(p, 'weight_ih_l%d' % k))            
                orthogonal_gru(getattr(p, 'weight_hh_l%d' % k))
                print('a GRU inited')
            