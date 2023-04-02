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
Involution Block
"""

import paddle
from paddle import nn
import paddle.nn.functional as F
import numpy as np
import math


class Involution2D(nn.Layer):
    """
    Involution module.

    Args:
        in_channel: The channel size of input.
        out_channel: The channel size of output.
        sigma_mapping: Sigma mapping.
        kernel_size: Kernel size.
        stride: Stride size.
        groups: Group size.
        reduce_ratio: The ratio of reduce.
        dilation: The dilation size.
        padding: The padding size.

    Returns:
        output: Tbe output of Involution2D block.

    References:

    [1] Involution: Inverting the Inherence of Convolution for Visual Recognition. https://arxiv.org/abs/2103.06255

    """
    def __init__(self, in_channel, out_channel, sigma_mapping=None, kernel_size=7, stride=1,
                 groups=1, reduce_ratio=1, dilation=1, padding=3):
        """
        Initialization
        """
        super(Involution2D, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.reduce_ratio = reduce_ratio
        self.dilation = dilation
        self.padding = padding
        self.sigma_mapping = nn.Sequential(
            nn.BatchNorm2D(num_features=self.out_channel // self.reduce_ratio),
            nn.ReLU()
        )
        self.initial_mapping = nn.Conv2D(in_channels=self.in_channel, out_channels=self.out_channel,
                                        kernel_size=1, stride=1, padding=0)
        self.o_mapping = nn.AvgPool2D(kernel_size=self.stride, stride=self.stride)
        self.reduce_mapping = nn.Conv2D(in_channels=self.in_channel, out_channels=self.out_channel // self.reduce_ratio,
                                    kernel_size=1, stride=1, padding=0)
        self.span_mapping = nn.Conv2D(in_channels=self.out_channel // self.reduce_ratio,
                                    out_channels=self.kernel_size * self.kernel_size * self.groups,
                                    kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Involution block
        """
        batch_size, _, height, width = x.shape

        temp_mapping = self.initial_mapping(x)
        input_unfolded = F.unfold(temp_mapping, self.kernel_size, strides=self.stride,
                                paddings=self.padding, dilations=self.dilation)
        input_unfolded = input_unfolded.view(batch_size, self.groups, self.out_channel // self.groups,
                                            self.kernel_size * self.kernel_size, height, width)
        kernel = self.span_mapping(self.sigma_mapping(self.reduce_mapping(self.o_mapping(x))))
        kernel = kernel.view(batch_size, self.groups, self.kernel_size * self.kernel_size, height, width).unsqueeze(2)
        
        output = paddle.sum(kernel * input_unfolded, axis=3).view(batch_size, -1, height, width)
        return output