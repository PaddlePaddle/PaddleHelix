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

"""Exponential Moving Average"""

import numpy as np
import paddle
from paddle import nn


class ExponentialMovingAverage(object):
    """ExponentialMovingAverage"""
    def __init__(self, model, decay):
        self._model = model
        self._decay = decay
        self._shadow = {}
        self._back_up = None

        self._is_registered = False

    def _named_parameters(self):
        for name, param in self._model.named_parameters():
            if not param.stop_gradient:
                yield name, param

    @property
    def is_registered(self):
        return self._is_registered

    def register(self):
        """copy current paramters"""
        for name, param in self._named_parameters():
            self._shadow[name] = param.numpy()
        self._is_registered = True
        print('[EMA] registered.')

    def update(self):
        """called after optimizer.step()"""
        if not self.is_registered:
            return

        for name, param in self._named_parameters():
            self._shadow[name] = self._decay * self._shadow[name] + \
                    (1 - self._decay) * param.numpy()

    def apply(self):
        """apply ema params to the model"""
        if not self.is_registered:
            return

        origin_params = {}
        for name, param in self._named_parameters():
            origin_params[name] = param.numpy()
            param.set_value(self._shadow[name])
        self._back_up = origin_params

    def restore(self):
        """restore params"""
        if not self.is_registered:
            return

        for name, param in self._named_parameters():
            param.set_value(self._back_up[name])

    def Print(self):
        for name in self._shadow:
            print(name, self._shadow[name])



class ExponentialMovingAverage2(object):
    """ExponentialMovingAverage2"""
    def __init__(self, model, decay):
        self._model = model
        self._decay = decay
        self._param_vec = {}
        self._back_up = None

        self._is_registered = False

    def _named_parameters(self):
        for name, param in self._model.named_parameters():
            if not param.stop_gradient:
                yield name, param

    def _get_param_vec_from_model(self):
        param_vec = []
        for name, param in self._named_parameters():
            param_vec.append(param.flatten())
        param_vec = paddle.concat(param_vec, 0)
        return param_vec.detach()
    
    def _map_param_vec(self, param_vec):
        start = 0
        param_dict = {}
        for name, param in self._named_parameters():
            length = np.prod(param.shape)
            param_dict[name] = param_vec[start: start + length].reshape(param.shape)
            start += length
        return param_dict

    @property
    def is_registered(self):
        return self._is_registered

    def register(self):
        """copy current paramters"""
        self._param_vec = self._get_param_vec_from_model()
        self._is_registered = True
        print('[EMA] registered.')

    def update(self):
        """called after optimizer.step()"""
        if not self.is_registered:
            return

        self._param_vec = self._decay * self._param_vec + \
                (1 - self._decay) * self._get_param_vec_from_model()

    def apply(self):
        """apply ema params to the model"""
        if not self.is_registered:
            return

        param_dict = self._map_param_vec(self._param_vec)
        back_up = {}
        for name, param in self._named_parameters():
            back_up[name] = param.detach()
            param.set_value(param_dict[name])
        self._back_up = back_up

    def restore(self):
        """restore params"""
        if not self.is_registered:
            return

        for name, param in self._named_parameters():
            param.set_value(self._back_up[name])

