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
import numpy as np
import paddle
import paddle.nn as nn
import paddle.fluid as fluid
import paddle.nn.functional as F
from .protein_tools import ProteinTokenizer
from .transformer_block import TransformerEncoder, TransformerEncoderLayer
from .transformer_block import DeBERTaEncoder, DeBERTaEncoderLayer
from .transformer_block import TransformerEncoderLayerWithRotary


class TransformerEncoderModel(nn.Layer):
    def __init__(self,
                 vocab_size,
                 hidden_size=256,
                 feedforward_size=2048,
                 n_layers=12,
                 n_heads=8,
                 padding_idx=0,
                 dropout_rate=0.1):
        super(TransformerEncoderModel, self).__init__()
        self.padding_idx = padding_idx
        self.token_embedding = nn.Embedding(vocab_size,
                                            hidden_size,
                                            padding_idx=padding_idx)
        # todo: max_len + 1
        max_pos_len = 1025
        self.pos_embedding = nn.Embedding(max_pos_len,
                                          hidden_size,
                                          padding_idx=padding_idx)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.transformer_encoder_layer = TransformerEncoderLayer(hidden_size, n_heads, 
                                                                 dim_feedforward=feedforward_size, 
                                                                 dropout=0.1, activation='gelu', 
                                                                 attn_dropout=0.1, act_dropout=0,
                                                                 normalize_before=True)
        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, n_layers, 
                                                      norm=nn.LayerNorm(hidden_size))

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.apply(self.init_weights)
        self.checkpoints = []

    def forward(self, input, pos):
        attention_mask = paddle.unsqueeze(
            (pos == self.padding_idx).astype("float32") * -1e9,
            axis=[1, 2])
        # attention_mask = None
        token_embed = self.token_embedding(input)
        pos_embed = self.pos_embedding(pos)
        embed = token_embed + pos_embed
        embed = self.layer_norm(embed)
        embed = self.dropout(embed)
        encoder_output = self.transformer_encoder(embed, attention_mask)

        if not paddle.in_dynamic_mode():
            self.checkpoints.extend(self.transformer_encoder.checkpoints)

        return encoder_output

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=0.02,
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


class RotaryEncoderModel(nn.Layer):
    def __init__(self,
                 vocab_size,
                 hidden_size=256,
                 feedforward_size=2048,
                 n_layers=12,
                 n_heads=8,
                 padding_idx=0,
                 dropout_rate=0.1):
        super(RotaryEncoderModel, self).__init__()
        self.padding_idx = padding_idx
        self.token_embedding = nn.Embedding(vocab_size,
                                            hidden_size,
                                            padding_idx=padding_idx)
        # todo: max_len + 1
        max_pos_len = 1025
        self.pos_embedding = nn.Embedding(max_pos_len,
                                          hidden_size,
                                          padding_idx=padding_idx)

        self.dropout = nn.Dropout(p=dropout_rate)
        rotary_encoder_layer = TransformerEncoderLayerWithRotary(hidden_size, n_heads, 
                                                                 dim_feedforward=feedforward_size, 
                                                                 dropout=0.1, activation='gelu', 
                                                                 attn_dropout=0.1, act_dropout=0,
                                                                 normalize_before=True,
                                                                 rotary_value=True,
                                                                 max_position_embeddings=max_pos_len)
        self.rotary_encoder = TransformerEncoder(rotary_encoder_layer, n_layers, 
                                                      norm=nn.LayerNorm(hidden_size))

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.apply(self.init_weights)
        self.checkpoints = []

    def forward(self, input, pos):
        attention_mask = paddle.unsqueeze(
            (pos == self.padding_idx).astype("float32") * -1e9,
            axis=[1, 2])
        # attention_mask = None
        token_embed = self.token_embedding(input)
        pos_embed = self.pos_embedding(pos)
        embed = token_embed + pos_embed
        embed = self.layer_norm(embed)
        embed = self.dropout(embed)
        encoder_output = self.rotary_encoder(embed, attention_mask)

        if not paddle.in_dynamic_mode():
            self.checkpoints.extend(self.transformer_encoder.checkpoints)

        return encoder_output

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=0.02,
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


class DeBERTaEncoderModel(nn.Layer):
    def __init__(self,
                 vocab_size,
                 hidden_size=512,
                 intermediate_size=2048,
                 n_layers=20,
                 n_heads=8,
                 padding_idx=0,
                 dropout_rate=0.1,
                 only_c2p=False):
        super(DeBERTaEncoderModel, self).__init__()
        self.padding_idx = padding_idx
        self.token_embedding = nn.Embedding(vocab_size,
                                            hidden_size,
                                            padding_idx=padding_idx)

        max_pos_len = 2048
        self.rel_embeddings = nn.Embedding(max_pos_len, hidden_size).weight

        self.dropout = nn.Dropout(p=dropout_rate)
        deberta_encoder_layer = DeBERTaEncoderLayer(hidden_size, n_heads, dim_feedforward=intermediate_size, 
                                                         dropout=0.1, activation='gelu', attn_dropout=0.1, 
                                                         act_dropout=0, normalize_before=True, only_c2p=only_c2p)
        # Todos: 返回倒数第二层DeBERTaEncoderLayer的输出作为1D蛋白质表示
        self.deberta_encoder = DeBERTaEncoder(deberta_encoder_layer, n_layers, norm=nn.LayerNorm(hidden_size))
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.apply(self.init_weights)
        self.checkpoints = []
    
    def get_relative_pos(self, hidden_states):
        query_size = paddle.shape(hidden_states)[-2]
        key_size = paddle.shape(hidden_states)[-2]

        q_ids = paddle.arange(0, query_size)
        k_ids = paddle.arange(0, key_size)
        rel_pos_ids = paddle.subtract(q_ids.unsqueeze(-1), paddle.tile(k_ids, (query_size, 1)))
        rel_pos_ids = paddle.cast(rel_pos_ids, paddle.int32)
        rel_pos_ids = rel_pos_ids.unsqueeze(0)
        return rel_pos_ids

    def forward(self, input, pos, return_representations=False, return_last_n_weight=0):
        attention_mask = paddle.unsqueeze(
            (pos == self.padding_idx).astype("float32") * -1e9,
            axis=[1, 2])
        # attention_mask = None
        token_embed = self.token_embedding(input)
        embed = self.layer_norm(token_embed)
        embed = self.dropout(embed)

        relative_pos = self.get_relative_pos(embed)
        rel_embeddings = self.rel_embeddings

        results = self.deberta_encoder(
                embed, attention_mask, relative_pos, rel_embeddings, 
                return_last_n_weight=return_last_n_weight)

        if not paddle.in_dynamic_mode():
            self.checkpoints.extend(self.deberta_encoder.checkpoints)

        if return_representations:
            return results
        return results['output']

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=0.02,
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


class PretrainTaskModel(nn.Layer):
    def __init__(self,
                 class_num,
                 model_config,
                 encoder_model):
        super(PretrainTaskModel, self).__init__()

        model_type = model_config['model_type']
        in_channels = model_config.get('hidden_size', 512)

        self.conv_decoder = nn.Sequential(
            nn.Conv1D(in_channels=in_channels,
                      out_channels=128,
                      kernel_size=5,
                      padding="same",
                      data_format="NLC"),
            nn.ReLU(),
            nn.Conv1D(in_channels=128,
                      out_channels=class_num,
                      kernel_size=3,
                      padding="same",
                      data_format="NLC"),
        )
        self.encoder_model = encoder_model
        self.checkpoints = []

    def forward(self, input, pos):
        encoder_output = self.encoder_model(input, pos)
        decoder_output = self.conv_decoder(encoder_output)
        if not paddle.in_dynamic_mode():
            self.checkpoints.extend(self.encoder_model.checkpoints)

        return decoder_output


class ProteinEncoderModel(nn.Layer):
    """
    ProteinSequenceModel
    """
    def __init__(self, model_config, name=''):
        super(ProteinEncoderModel, self).__init__()
        self.model_name = name
        self.checkpoints = []

        n_layers = model_config.get('layer_num', 8)
        n_heads = model_config.get('head_num', 8)
        hidden_size = model_config.get('hidden_size', 512)
        intermediate_size = model_config.get('intermediate_size', 2048)
        only_c2p = model_config.get('only_c2p', False)

        model_type = model_config.get('model_type', 'lstm')
        if model_type == "transformer":
            self.encoder_model = TransformerEncoderModel(vocab_size=len(ProteinTokenizer.vocab),
                                                         hidden_size=hidden_size,
                                                         feedforward_size=intermediate_size,
                                                         n_layers=n_layers,
                                                         n_heads=n_heads)
        elif model_type == "deberta":
            self.encoder_model = DeBERTaEncoderModel(vocab_size=len(ProteinTokenizer.vocab), 
                                                     hidden_size=hidden_size,
                                                     intermediate_size=intermediate_size,
                                                     n_layers=n_layers,
                                                     n_heads=n_heads,
                                                     only_c2p=only_c2p)
        elif model_type == "rotary":
            self.encoder_model = RotaryEncoderModel(vocab_size=len(ProteinTokenizer.vocab), 
                                                     hidden_size=hidden_size,
                                                     feedforward_size=intermediate_size,
                                                     n_layers=n_layers,
                                                     n_heads=n_heads)

    def forward(self, input, pos):
        encoder_output = self.encoder_model(input, pos)
        if not paddle.in_dynamic_mode():
            self.checkpoints.extend(self.encoder_model.checkpoints)
        return encoder_output


class ProteinModel(nn.Layer):
    """
    ProteinModel
    """
    def __init__(self, encoder_model, model_config):
        super(ProteinModel, self).__init__()
        self.checkpoints = []
        task = model_config.get('task', 'pretrain')
        if task == 'pretrain':
            self.model = PretrainTaskModel(class_num=len(ProteinTokenizer.vocab), 
                                           model_config=model_config, 
                                           encoder_model=encoder_model)
        
        self.criterion = paddle.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, batch):
        pred = self.model(batch['masked_sequence'], batch['position'])
        if not paddle.in_dynamic_mode():
            self.checkpoints.extend(self.model.checkpoints)

        label = batch['label'].reshape([-1, 1])
        pred = pred.reshape([-1, pred.shape[-1]])
        loss = self.criterion(pred, label)
        results = {
            'pred': pred,
            'loss': loss,
        }
        return results


