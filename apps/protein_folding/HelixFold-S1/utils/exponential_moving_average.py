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

"""EMA."""

import paddle
import numpy as np


class ExponentialMovingAverage():
    """
    Exponential Moving Average.
    """

    def __init__(self, param_groups, decay, thres_steps=True):
        self._param_groups = param_groups
        self._decay = decay
        self._thres_steps = thres_steps
        self._shadow = {}
        self._backup = {}

    def register(self):
        """Register."""
        self._update_step = 0
        for group in self._param_groups:
            for p in group['params']:
                if p.stop_gradient is True:
                    continue
                self._shadow[id(p)] = p.numpy()

    def update(self):
        """Update params."""
        decay = min(self._decay, (1 + self._update_step) / (
            10 + self._update_step)) if self._thres_steps else self._decay

        for group in self._param_groups:
            for p in group['params']:
                if p.stop_gradient is True:
                    continue
                new_val = p.numpy()
                old_val = self._shadow[id(p)]
                new_average = decay * old_val + (1 - decay) * new_val
                self._shadow[id(p)] = new_average
                
        self._update_step += 1
        return decay

    def apply_shadow(self):
        """Apply shadow params."""
        for group in self._param_groups:
            for p in group['params']:
                if p.stop_gradient is True:
                    continue
                assert id(p) in self._shadow

                self._backup[id(p)] = p.numpy()
                p.set_value(self._shadow[id(p)])

    def restore(self):
        """Restore params."""
        for group in self._param_groups:
            for p in group['params']:
                if p.stop_gradient is True:
                    continue
                assert id(p) in self._backup
                p.set_value(self._backup[id(p)])

        self._backup = {}


class EMA(object):
    """
    Exponential Moving Average.
    """

    def __init__(self, param_groups, decay, thres_steps=True):
        self._param_groups = param_groups
        self._decay = decay
        self._thres_steps = thres_steps
        self._shadow = {}
        self._backup = {}

    @paddle.no_grad()
    def register(self):
        """Register."""
        self._update_step = 0

        for group in self._param_groups:
            for p in group['params']:
                if p.stop_gradient is True:
                    continue
                self._shadow[id(p)] = paddle.zeros_like(p, dtype="float32")
                self._shadow[id(p)].set_value(p.astype("float32"))

    @paddle.no_grad()
    def update(self):
        """Update params."""
        decay = min(self._decay, (1 + self._update_step) / (
            10 + self._update_step)) if self._thres_steps else self._decay

        for group in self._param_groups:
            for p in group['params']:
                if p.stop_gradient is True:
                    continue
                new_val = p.detach().clone()
                old_val = self._shadow[id(p)]
                new_average = decay * old_val + (1 - decay) * new_val.astype("float32")
                self._shadow[id(p)] = new_average

        self._update_step += 1
        return decay

    @paddle.no_grad()
    def apply_shadow(self):
        """Apply shadow params."""

        for group in self._param_groups:
            for p in group['params']:
                if p.stop_gradient is True:
                    continue
                assert id(p) in self._shadow

                self._backup[id(p)] = p.detach().clone()
                if p.dtype == paddle.bfloat16:
                    p.set_value(self._shadow[id(p)].astype(paddle.bfloat16))
                else:
                    p.set_value(self._shadow[id(p)])

    @paddle.no_grad()
    def restore(self):
        """Restore params."""

        for group in self._param_groups:
            for p in group['params']:
                if p.stop_gradient is True:
                    continue
                assert id(p) in self._backup
                p.set_value(self._backup[id(p)])
        self._backup = {}