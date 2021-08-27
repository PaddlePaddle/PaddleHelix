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
Resnet block.
"""

import paddle.fluid as fluid
import paddle.fluid.layers as layers


def resnet_encoder(
        input,
        hidden_size,
        n_layer=1,
        filter_size=3,
        act='gelu',
        epsilon=1e-6,
        param_initializer=None,
        name="resnet"):
    """
    The encoder is composed of a stack of resnet layers.

    Args:
        input: The input of resnet encoder.
        hidden_size: The hidden size of resnet.
        n_layer: The number of resnet layers.
        act: The activation function.
        param_initializer: The parameter initializer for resnet encoder.
        name: The prefix of the parameters' name in resnet encoder.

    Returns:
        hidden: The hidden units of resnet encoder.
        checkpoints: The checkpoints for recompute mechanism.
    """
    checkpoints = []
    for i in range(n_layer):
        hidden = fluid.layers.sequence_conv(
                input=input,
                num_filters=hidden_size,
                filter_size=filter_size,
                param_attr=fluid.ParamAttr(
                        name='%s_%d_0_fc.w_0' % (name, i),
                        initializer=param_initializer),
                bias_attr=fluid.ParamAttr(name='%s_%d_0_fc.b_0' % (name, i)))
        hidden = fluid.layers.layer_norm(
                hidden,
                begin_norm_axis=len(hidden.shape) - 1,
                param_attr=fluid.ParamAttr(
                        name='%s_%d_0_layer_norm_scale' % (name, i),
                        initializer=fluid.initializer.Constant(1.)),
                bias_attr=fluid.ParamAttr(
                        name='%s_%d_0_layer_norm_bias' % (name, i),
                        initializer=fluid.initializer.Constant(0.)),
                epsilon=epsilon,
                act=act)
        checkpoints.append(hidden)
        hidden = fluid.layers.sequence_conv(
                input=input,
                num_filters=hidden_size,
                filter_size=filter_size,
                param_attr=fluid.ParamAttr(
                        name='%s_%d_1_fc.w_0' % (name, i),
                        initializer=param_initializer),
                bias_attr=fluid.ParamAttr(name='%s_%d_1_fc.b_0' % (name, i)))
        hidden = fluid.layers.layer_norm(
                hidden,
                begin_norm_axis=len(hidden.shape) - 1,
                param_attr=fluid.ParamAttr(
                        name='%s_%d_1_layer_norm_scale' % (name, i),
                        initializer=fluid.initializer.Constant(1.)),
                bias_attr=fluid.ParamAttr(
                        name='%s_%d_1_layer_norm_bias' % (name, i),
                        initializer=fluid.initializer.Constant(0.)),
                epsilon=epsilon,
                act=act)
        hidden = fluid.layers.elementwise_add(hidden, input)
        checkpoints.append(hidden)

        input = hidden

    return hidden, checkpoints
  
