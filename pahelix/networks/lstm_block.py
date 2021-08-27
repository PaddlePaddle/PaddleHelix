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
Lstm block.
"""

import paddle.fluid as fluid
import paddle.fluid.layers as layers


def lstm_encoder(
        input,
        hidden_size,
        n_layer=1,
        is_bidirectory=True,
        param_initializer=None,
        name="lstm"):
    """
    The encoder is composed of a stack of lstm layers.

    Args:
        input: The input of lstm encoder.
        hidden_size: The hidden size of lstm.
        n_layer: The number of lstm layers.
        is_bidirectory: True if the lstm is bidirectory.
        param_initializer: The parameter initializer for lstm encoder.
        name: The prefix of the parameters' name in lstm encoder.

    Returns:
        hidden: The hidden units of lstm encoder.
        checkpoints: The checkpoints for recompute mechanism.
    """
    checkpoints = []
    f_lstm_input = input
    for i in range(n_layer):
        f_lstm_input = fluid.layers.fc(
                input=f_lstm_input,
                param_attr=fluid.ParamAttr(
                        name='%s_forward_%d_fc.w_0' % (name, i),
                        initializer=param_initializer),
                bias_attr=fluid.ParamAttr(name='%s_forward_%d_fc.b_0' % (name, i)),
                size=hidden_size * 4)
        f_hidden, f_cell = fluid.layers.dynamic_lstm(
                input=f_lstm_input,
                size=hidden_size * 4,
                param_attr=fluid.ParamAttr(
                        name='%s_forward_%d.w_0' % (name, i),
                        initializer=param_initializer),
                bias_attr=fluid.ParamAttr(name='%s_forward_%d.b_0' % (name, i)))
        f_lstm_input = f_hidden
        checkpoints.append(f_lstm_input)

    if is_bidirectory:
        r_lstm_input = input
        for i in range(n_layer):
            r_lstm_input = fluid.layers.fc(
                    input=r_lstm_input,
                    param_attr=fluid.ParamAttr(
                            name='%s_backward_%d_fc.w_0' % (name, i),
                            initializer=param_initializer),
                    bias_attr=fluid.ParamAttr(name='%s_backward_%d_fc.b_0' % (name, i)),
                    size=hidden_size * 4)
            r_hidden, r_cell = fluid.layers.dynamic_lstm(
                    input=r_lstm_input,
                    size=hidden_size * 4,
                    is_reverse=True,
                    param_attr=fluid.ParamAttr(
                            name='%s_backward_%d.w_0' % (name, i),
                            initializer=param_initializer),
                    bias_attr=fluid.ParamAttr(name='%s_backward_%d.b_0' % (name, i)))
            r_lstm_input = r_hidden
            checkpoints.append(r_lstm_input)
        
        hidden = fluid.layers.concat([f_lstm_input, r_lstm_input], axis=1)
    else:
        hidden = f_lstm_input

    return hidden, checkpoints
  
