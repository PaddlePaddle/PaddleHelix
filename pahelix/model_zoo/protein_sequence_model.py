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
from pahelix.utils.protein_tools import ProteinTokenizer
from pahelix.networks.lstm_block import lstm_encoder
from pahelix.networks.transformer_block import transformer_encoder
from pahelix.networks.pre_post_process import pre_process_layer
from pahelix.networks.resnet_block import resnet_encoder

class ProteinSequenceModel:
    """
    ProteinSequenceModel
    """
    def __init__(self, model_config, name=''):
        self.model_name = name
        self.model_type = model_config.get('model_type', 'transformer')
        self.hidden_size = model_config.get('hidden_size', 512)
        self.param_initializer = fluid.initializer.TruncatedNormal(
                scale=model_config.get('initializer_range', 0.02))
        self.dropout_rate = model_config.get('dropout', 0.1)
        self.epsilon = model_config.get('epsilon', 1e-5)
        self.pre_encoder_cmd = model_config.get('pre_encoder_cmd', 'nd')
        self.preprocess_cmd = model_config.get('preprocess_cmd', '')
        self.postprocess_cmd = model_config.get('postprocess_cmd', 'dan')
        self.vocab_size = model_config.get('vocab_size', len(ProteinTokenizer.vocab))
        self.max_len = model_config.get('max_len', 8192)

        if self.model_type == 'transformer':
            self.layer_num = model_config.get('layer_num', 4)
            self.head_num = model_config.get('head_num', 4)
            self.pool_type = model_config.get('pool_type', 'first')
        elif self.model_type == 'lstm':
            self.layer_num = model_config.get('layer_num', 3)
            self.pool_type = model_config.get('pool_type', 'average')
        elif self.model_type == 'resnet':
            self.layer_num = model_config.get('layer_num', 5)
            self.filter_size = model_config.get('filter_size', 3)
            self.pool_type = model_config.get('pool_type', 'average')
        else:
            raise ValueError('%s not supported.' % self.model_type)

    def _prepare_emb(self, inputs, is_test):
        token = inputs['token']
        token_emb = fluid.layers.embedding(
                input=token,
                param_attr=fluid.ParamAttr(name='%s_token_emb' % self.model_name, initializer=self.param_initializer),
                size=[self.vocab_size, self.hidden_size],
                padding_idx=0,
                is_sparse=False)
        if self.model_type in ['transformer', 'resnet']:
            pos = inputs['pos']
            pos_emb = fluid.layers.embedding(
                    input=pos,
                    param_attr=fluid.ParamAttr(name='%s_pos_emb' % self.model_name, initializer=self.param_initializer),
                    size=[self.max_len, self.hidden_size],
                    is_sparse=False)
            features = fluid.layers.elementwise_add(token_emb, pos_emb)
        else:
            features = token_emb

        features = pre_process_layer(
                features,
                self.pre_encoder_cmd,
                self.dropout_rate,
                name='%s_pre_encoder' % self.model_name,
                epsilon=self.epsilon,
                is_test=is_test)

        return features

    def _transformer(self, input, is_test):
        pad_value = fluid.layers.assign(input=numpy.array([0.0], dtype=numpy.float32))
        input_pad, input_len = fluid.layers.sequence_pad(
                input, pad_value=pad_value)
        transformer_in = input_pad

        layer_num = self.layer_num
        head_num = self.head_num
        transformer_out, checkpoints = transformer_encoder(
            enc_input=transformer_in,
            attn_bias=None,
            n_layer=layer_num,
            n_head=head_num,
            d_key=self.hidden_size // head_num, 
            d_value=self.hidden_size // head_num,
            d_model=self.hidden_size,
            d_inner_hid=self.hidden_size * 4,
            prepostprocess_dropout=self.dropout_rate,
            attention_dropout=self.dropout_rate,
            act_dropout=self.dropout_rate,
            hidden_act='gelu',
            preprocess_cmd=self.preprocess_cmd,
            postprocess_cmd=self.postprocess_cmd,
            param_initializer=self.param_initializer,
            epsilon=self.epsilon,
            name='%s_transformer' % self.model_name,
            is_test=is_test
        )

        hidden = fluid.layers.sequence_unpad(transformer_out, length=input_len)
        pooled_hidden = fluid.layers.fc(
                input=fluid.layers.sequence_pool(hidden, pool_type='first'),
                param_attr=fluid.ParamAttr(
                        name='%s_seq_pool_fc.w_0' % self.model_name,
                        initializer=self.param_initializer),
                bias_attr=fluid.ParamAttr(name='%s_seq_pool_fc.b_0' % self.model_name),
                size=self.hidden_size,
                act='tanh')

        return hidden, pooled_hidden, checkpoints

    def _lstm(self, input, is_test):
        hidden, checkpoints = lstm_encoder(
                input=input,
                hidden_size=self.hidden_size,
                n_layer=self.layer_num,
                is_bidirectory=True,
                param_initializer=self.param_initializer,
                name='%s_lstm' % self.model_name)
        pooled_hidden = fluid.layers.fc(
                input=fluid.layers.sequence_pool(hidden, pool_type=self.pool_type),
                param_attr=fluid.ParamAttr(
                        name='%s_seq_pool_fc.w_0' % self.model_name,
                        initializer=self.param_initializer),
                bias_attr=fluid.ParamAttr(name='%s_seq_pool_fc.b_0' % self.model_name),
                size=self.hidden_size,
                act='tanh')
        return hidden, pooled_hidden, checkpoints

    def _resnet(self, input, is_test):
        hidden, checkpoints = resnet_encoder(
                input=input,
                hidden_size=self.hidden_size,
                n_layer=self.layer_num,
                filter_size=self.filter_size,
                act='gelu',
                epsilon=self.epsilon,
                param_initializer=self.param_initializer,
                name='%s_resnet' % self.model_name)
        pooled_hidden = fluid.layers.fc(
                input=fluid.layers.sequence_pool(hidden, pool_type=self.pool_type),
                param_attr=fluid.ParamAttr(
                        name='%s_seq_pool_fc.w_0' % self.model_name,
                        initializer=self.param_initializer),
                bias_attr=fluid.ParamAttr(name='%s_seq_pool_fc.b_0' % self.model_name),
                size=self.hidden_size,
                act='tanh')
        return hidden, pooled_hidden, checkpoints

    def forward(self, inputs, is_test):
        """
        Forward.
        """
        features = self._prepare_emb(inputs, is_test)
        checkpoints = [features]
        
        if self.model_type == 'transformer':
            hidden, pooled_hidden, temp_checkpoints = self._transformer(features, is_test)
        elif self.model_type == 'lstm':
            hidden, pooled_hidden, temp_checkpoints = self._lstm(features, is_test)
        elif self.model_type == 'resnet':
            hidden, pooled_hidden, temp_checkpoints = self._resnet(features, is_test)
        else:
            print('Invalid model_type')
            return

        checkpoints.extend(temp_checkpoints)

        return hidden, pooled_hidden, checkpoints
