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
Sequence-based models for protein.
"""

import math
import numpy
import paddle
import paddle.fluid as fluid
from pahelix.model_zoo.protein_sequence_model import ProteinSequenceModel
from pahelix.utils.protein_tools import ProteinTokenizer

class TAPEModel:
    """
    | TAPEModel, implementation of the methods in paper ``Evaluating Protein Transfer Learning with TAPE``.

    Public Functions:
        - ``forward``: forward.
        - ``cal_loss``: calculate the loss of the network.
        - ``get_fetch_list``: get the fetch_list for results.

    """
    def __init__(self, model_config={}, name=''):
        self.model_type = model_config['model_type']

        self.param_initializer = fluid.initializer.TruncatedNormal(
                scale=model_config.get('initializer_range', 0.02))
        self.model_name = name
        self.task = model_config.get('task', 'pretrain')
        self.hidden_size = model_config.get('hidden_size', 512)
        self.epsilon = model_config.get('epsilon', 1e-5)
        self.vocab_size = model_config.get('vocab_size', len(ProteinTokenizer.vocab))
        if self.task in ['seq_classification', 'classification']:
            if 'class_num' not in model_config:
                raise RuntimeError('No class_num in model_config for %s' % self.task)
            else:
                self.class_num = model_config['class_num']

        self.protein_model = ProteinSequenceModel(model_config, name='protein')

    def _pretrain_task(self, hidden, pooled_hidden, checkpoints):
        output_hidden = fluid.layers.fc(
                input=hidden,
                param_attr=fluid.ParamAttr(
                        name='%s_output_0_fc.w_0' % self.model_name,
                        initializer=self.param_initializer),
                bias_attr=fluid.ParamAttr(name='%s_output_0_fc.b_0' % self.model_name),
                size=self.hidden_size,
                act='gelu')
        output_hidden = fluid.layers.layer_norm(
                output_hidden,
                begin_norm_axis=len(output_hidden.shape) - 1,
                param_attr=fluid.ParamAttr(
                        name='%s_output_0_layer_norm_scale' % self.model_name,
                        initializer=fluid.initializer.Constant(1.)),
                bias_attr=fluid.ParamAttr(
                        name='%s_output_0_layer_norm_bias' % self.model_name,
                        initializer=fluid.initializer.Constant(0.)),
                epsilon=self.epsilon)
        checkpoints.append(output_hidden)
        self.pred = fluid.layers.fc(
                input=output_hidden,
                param_attr=fluid.ParamAttr(
                        name='%s_output_1_fc.w_0' % self.model_name,
                        initializer=self.param_initializer),
                bias_attr=fluid.ParamAttr(name='%s_output_1_fc.b_0' % self.model_name),
                size=self.vocab_size,
                act='softmax')
        checkpoints.append(self.pred)
    
    def _pretrain_task_loss(self):
        self.label = fluid.layers.data(name='label', shape=[None, 1], dtype='int64', lod_level=1)
        self.input_list.append(self.label)
        self.cross_entropy = fluid.layers.cross_entropy(
                input=self.pred, label=self.label, ignore_index=-1)
        self.loss = fluid.layers.mean(self.cross_entropy)

    def _seq_classification_task(self, hidden, pooled_hidden, checkpoints):
        output_hidden = fluid.layers.sequence_conv(
                input=hidden,
                num_filters=self.hidden_size,
                filter_size=5,
                param_attr=fluid.ParamAttr(
                        name='%s_output_0_conv.w_0' % self.model_name,
                        initializer=self.param_initializer),
                bias_attr=fluid.ParamAttr(name='%s_output_0_conv.b_0' % self.model_name),
                act='relu')
        checkpoints.append(output_hidden)
        self.pred = fluid.layers.sequence_conv(
                input=output_hidden,
                num_filters=self.class_num,
                filter_size=3,
                param_attr=fluid.ParamAttr(
                        name='%s_output_1_conv.w_0' % self.model_name,
                        initializer=self.param_initializer),
                bias_attr=fluid.ParamAttr(name='%s_output_1_conv.b_0' % self.model_name),
                act='softmax')
        checkpoints.append(self.pred)
        
    def _seq_classification_task_loss(self):
        self.label = fluid.layers.data(name='label', shape=[None, 1], dtype='int64', lod_level=1)
        self.input_list.append(self.label)
        self.cross_entropy = fluid.layers.cross_entropy(input=self.pred, label=self.label, ignore_index=-1)
        self.loss = fluid.layers.mean(self.cross_entropy)

    def _classification_task(self, hidden, pooled_hidden, checkpoints):
        output_hidden = fluid.layers.fc(
                input=pooled_hidden,
                param_attr=fluid.ParamAttr(
                        name='%s_output_0_fc.w_0' % self.model_name,
                        initializer=self.param_initializer),
                bias_attr=fluid.ParamAttr(name='%s_output_0_fc.b_0' % self.model_name),
                size=self.hidden_size,
                act='relu')
        checkpoints.append(output_hidden)
        self.pred = fluid.layers.fc(
                input=output_hidden,
                param_attr=fluid.ParamAttr(
                        name='%s_output_1_fc.w_0' % self.model_name,
                        initializer=self.param_initializer),
                bias_attr=fluid.ParamAttr(name='%s_output_1_fc.b_0' % self.model_name),
                size=self.class_num,
                act='softmax')
        checkpoints.append(self.pred)
    
    def _classification_task_loss(self):
        self.label = fluid.layers.data(name='label', shape=[None, 1], dtype='int64', lod_level=1)
        self.input_list.append(self.label)
        self.cross_entropy = fluid.layers.cross_entropy(input=self.pred, label=self.label)
        self.loss = fluid.layers.mean(self.cross_entropy)

    def _regression_task(self, hidden, pooled_hidden, checkpoints):
        output_hidden = fluid.layers.fc(
                input=pooled_hidden,
                param_attr=fluid.ParamAttr(
                        name='%s_output_0_fc.w_0' % self.model_name,
                        initializer=self.param_initializer),
                bias_attr=fluid.ParamAttr(name='%s_output_0_fc.b_0' % self.model_name),
                size=self.hidden_size,
                act='relu')
        checkpoints.append(output_hidden)
        self.pred = fluid.layers.fc(
                input=output_hidden,
                param_attr=fluid.ParamAttr(
                        name='%s_output_1_fc.w_0' % self.model_name,
                        initializer=self.param_initializer),
                bias_attr=fluid.ParamAttr(name='%s_output_1_fc.b_0' % self.model_name),
                size=1,
                act=None)
        checkpoints.append(self.pred)

    def _regression_task_loss(self):
        self.label = fluid.layers.data(name='label', shape=[None, 1], dtype='float32', lod_level=1)
        self.input_list.append(self.label)
        loss = fluid.layers.square_error_cost(input=self.pred, label=self.label)
        self.loss = fluid.layers.mean(loss)

    def forward(self, is_test):
        """Forward.

        Args:
            is_test(bool): whether is test mode.

        """
        protein_token = fluid.layers.data(name='protein_token', shape=[None, 1], dtype='int64', lod_level=1)
        protein_pos = fluid.layers.data(name='protein_pos', shape=[None, 1], dtype='int64', lod_level=1)
        self.input_list = [protein_token, protein_pos]

        self.protein_inputs = {'token': protein_token, 'pos': protein_pos}

        hidden, pooled_hidden, checkpoints = self.protein_model.forward(self.protein_inputs, is_test)

        if self.task == 'pretrain':
            self._pretrain_task(hidden, pooled_hidden, checkpoints)
        elif self.task == 'seq_classification':
            self._seq_classification_task(hidden, pooled_hidden, checkpoints)
        elif self.task == 'classification':
            self._classification_task(hidden, pooled_hidden, checkpoints)
        elif self.task == 'regression':
            self._regression_task(hidden, pooled_hidden, checkpoints)
        else:
            raise ValueError('Task %s is unsupported.' % self.task)

        self.checkpoints = checkpoints

    def cal_loss(self):
        """Calculate the loss according to the task.
        """
        if self.task == 'pretrain':
            self._pretrain_task_loss()
        elif self.task == 'seq_classification':
            self._seq_classification_task_loss()
        elif self.task == 'classification':
            self._classification_task_loss()
        elif self.task == 'regression':
            self._regression_task_loss()
        else:
            raise ValueError('Task %s is unsupported.' % self.task)

    def get_fetch_list(self, is_inference=False):
        """Get the fetch list according the task.

        Args:
            is_inference(bool): whether is inference mode.
        
        Returns:
            fetch_ist: fetch_list.
        """
        if is_inference:
            if self.task == 'pretrain':
                return [self.pred]
            else:
                return [self.pred]
        else:
            if self.task == 'pretrain':
                return [self.pred, self.label, self.cross_entropy, self.loss]
            else:
                return [self.pred, self.label, self.loss]

