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
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from pahelix.utils.protein_tools import ProteinTokenizer


class LstmEncoderModel(nn.Layer):
    """
    LstmEncoderModel
    """
    def __init__(self,
                vocab_size,
                emb_dim=128,
                hidden_size=1024,
                n_layers=3,
                padding_idx=0,
                epsilon=1e-5,
                dropout_rate=0.1):
        """
        __init__
        """
        super(LstmEncoderModel, self).__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size,
                                    emb_dim,
                                    padding_idx=padding_idx)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.lstm_encoder = nn.LSTM(emb_dim,
                            hidden_size,
                            num_layers=n_layers,
                            direction="bidirectional")

    def forward(self, input, pos):
        """
        forward
        """
        token_embed = self.embedding(input)
        encoder_output, _ = self.lstm_encoder(token_embed)
        encoder_output = self.dropout(encoder_output)
        
        return encoder_output

class ResnetBasicBlock(nn.Layer):
    """
    ResnetBasicBlock
    """
    def __init__(self,
                 inplanes=256,
                 planes=256,
                 kernel_size=9,
                 dilation=1,
                 dropout_rate=0.1):
        super(ResnetBasicBlock, self).__init__()
        self.conv1 = nn.Conv1D(in_channels=inplanes, out_channels=planes, kernel_size=kernel_size, dilation=dilation, \
                               padding="same", data_format="NLC", weight_attr=nn.initializer.KaimingNormal())
        self.bn1 = nn.BatchNorm1D(planes, data_format="NLC")
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Conv1D(in_channels=planes, out_channels=planes, kernel_size=kernel_size, dilation=dilation, \
                               padding="same", data_format="NLC", weight_attr=nn.initializer.KaimingNormal())
        self.bn2 = nn.BatchNorm1D(planes, data_format="NLC")
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(p=dropout_rate)
    
    def forward(self, x):
        """
        forward
        """
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gelu2(out)
        out = self.dropout2(out)
        out += identity
        return out


class ResnetEncoderModel(nn.Layer):
    """
    ResnetEncoderModel
    """
    def __init__(self,
                 vocab_size,
                 emb_dim=128,
                 hidden_size=256,
                 kernel_size=9,
                 n_layers=35,
                 padding_idx=0,
                 dropout_rate=0.1,
                 epsilon=1e-6):
        super(ResnetEncoderModel, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.token_embedding = nn.Embedding(vocab_size,
                                            emb_dim,
                                            padding_idx=padding_idx)
        max_pos_len = 3000
        self.pos_embedding = nn.Embedding(max_pos_len,
                                          emb_dim,
                                          padding_idx=padding_idx)

        self.layer_norm = nn.BatchNorm1D(emb_dim, data_format="NLC")
        self.dropout = nn.Dropout(dropout_rate)

        self.padded_conv = nn.Sequential(
            nn.Conv1D(in_channels=emb_dim, out_channels=hidden_size, kernel_size=kernel_size, padding="same", \
                      data_format="NLC", weight_attr=nn.initializer.KaimingNormal()),
            nn.BatchNorm1D(hidden_size, data_format="NLC"),
            nn.GELU(),
            nn.Dropout(p=dropout_rate)
        )
        self.residual_block_1 = ResnetBasicBlock(inplanes=hidden_size, planes=hidden_size, kernel_size=kernel_size, dropout_rate=dropout_rate)
        self.residual_block_n = nn.Sequential()
        for i in range(1, n_layers):
            self.residual_block_n.add_sublayer("residual_block_%d" % i, \
                ResnetBasicBlock(inplanes=hidden_size, planes=hidden_size, kernel_size=kernel_size, dilation=2, dropout_rate=dropout_rate))
        
        self.apply(self.init_weights)
    
    def forward(self, input, pos):
        """
        forward
        """
        token_embed = self.token_embedding(input)
        token_embed = token_embed * math.sqrt(self.hidden_size)
        pos_embed = self.pos_embedding(pos)
        embed = token_embed + pos_embed
        embed = self.layer_norm(embed)
        embed = self.dropout(embed)

        inputs = self.padded_conv(embed)
        inputs = self.residual_block_1(inputs)
        encoder_output = self.residual_block_n(inputs)
        
        return encoder_output
    
    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding, nn.BatchNorm1D)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=0.02,
                        shape=layer.weight.shape))


class TransformerEncoderModel(nn.Layer):
    """
    TransformerEncoderModel
    """
    def __init__(self,
                 vocab_size,
                 emb_dim=512,
                 hidden_size=512,
                 n_layers=8,
                 n_heads=8,
                 padding_idx=0,
                 dropout_rate=0.1):
        """
        __init__
        """
        super(TransformerEncoderModel, self).__init__()
        self.padding_idx = padding_idx
        self.token_embedding = nn.Embedding(vocab_size,
                                            emb_dim,
                                            padding_idx=padding_idx)
        max_pos_len = 3000
        self.pos_embedding = nn.Embedding(max_pos_len,
                                          emb_dim,
                                          padding_idx=padding_idx)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(emb_dim, n_heads, dim_feedforward=hidden_size * 4, \
                                                dropout=0.1, activation='gelu', attn_dropout=0.1, act_dropout=0)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, n_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.apply(self.init_weights)
    
    def forward(self, input, pos):
        """
        forward
        """
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


class PretrainTaskModel(nn.Layer):
    """
    PretrainTaskModel
    """
    def __init__(self,
                 class_num,
                 model_config,
                 encoder_model):
        """
        __init__
        """
        super(PretrainTaskModel, self).__init__()

        model_type = model_config.get('model_type', 'transformer')
        hidden_size = model_config.get('hidden_size', 512)
        in_channels = hidden_size * 2 if model_type == 'lstm' else hidden_size

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
    
    def forward(self, input, pos):
        """
        forward
        """
        encoder_output = self.encoder_model(input, pos)
        decoder_output = self.conv_decoder(encoder_output)

        return decoder_output


class SeqClassificationTaskModel(nn.Layer):
    """
    SeqClassificationTaskModel
    """
    def __init__(self,
                 class_num,
                 model_config,
                 encoder_model):
        """
        __init__
        """
        super(SeqClassificationTaskModel, self).__init__()

        model_type = model_config.get('model_type', 'transformer')
        hidden_size = model_config.get('hidden_size', 512)
        in_channels = hidden_size * 2 if model_type == 'lstm' else hidden_size

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
    
    def forward(self, input, pos):
        """
        forward
        """
        encoder_output = self.encoder_model(input, pos)
        decoder_output = self.conv_decoder(encoder_output)

        return decoder_output


class ClassificationTaskModel(nn.Layer):
    """
    ClassificationTaskModel
    """
    def __init__(self,
                 class_num,
                 model_config,
                 encoder_model):
        """
        __init__
        """
        super(ClassificationTaskModel, self).__init__()
        model_type = model_config.get('model_type', 'transformer')
        hidden_size = model_config.get('hidden_size', 512)
        in_channels = hidden_size * 2 if model_type == 'lstm' else hidden_size

        self.fc_decoder = nn.Sequential(
            nn.Linear(in_features=in_channels,
                      out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,
                      out_features=class_num)
        )
        self.encoder_model = encoder_model
    
    def forward(self, input, pos):
        """
        forward
        """
        encoder_output = self.encoder_model(input, pos)
        encoder_output = encoder_output[:, 0, :]
        decoder_output = self.fc_decoder(encoder_output)
    
        return decoder_output


class RegressionTaskModel(nn.Layer):
    """
    RegressionTaskModel
    """
    def __init__(self,
                 model_config,
                 encoder_model):
        """
        __init__
        """
        super(RegressionTaskModel, self).__init__()
        model_type = model_config.get('model_type', 'transformer')
        hidden_size = model_config.get('hidden_size', 512)
        in_channels = hidden_size * 2 if model_type == 'lstm' else hidden_size

        self.fc_decoder = nn.Sequential(
            nn.Linear(in_features=in_channels,
                      out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size,
                      out_features=1)
        )
        self.encoder_model = encoder_model
    
    def forward(self, input, pos):
        """
        forward
        """
        encoder_output = self.encoder_model(input, pos)
        encoder_output = encoder_output[:, 0, :]
        decoder_output = self.fc_decoder(encoder_output)

        return decoder_output


class ProteinEncoderModel(nn.Layer):
    """
    ProteinSequenceModel
    """
    def __init__(self, model_config, name=''):
        """
        tbd
        """
        super(ProteinEncoderModel, self).__init__()
        self.model_name = name
        model_type = model_config.get('model_type', 'transformer')
        emb_dim = model_config.get('emb_dim', 512)
        hidden_size = model_config.get('hidden_size', 512)
        n_layers = model_config.get('n_layers', 8)

        if model_type == "lstm":
            self.encoder_model = LstmEncoderModel(vocab_size=len(ProteinTokenizer.vocab),
                                                  emb_dim=emb_dim,
                                                  hidden_size=hidden_size,
                                                  n_layers=n_layers)
        elif model_type == "transformer":
            n_heads = model_config.get('n_heads', 8)
            self.encoder_model = TransformerEncoderModel(vocab_size=len(ProteinTokenizer.vocab),
                                                         emb_dim=emb_dim,
                                                         hidden_size=hidden_size,
                                                         n_layers=n_layers,
                                                         n_heads=n_heads)
        elif model_type == "resnet":
            kernel_size = model_config.get('kernel_size', 9)
            self.encoder_model = ResnetEncoderModel(vocab_size=len(ProteinTokenizer.vocab),
                                                    emb_dim=emb_dim,
                                                    hidden_size=hidden_size,
                                                    kernel_size=kernel_size,
                                                    n_layers=n_layers)
    
    def forward(self, input, pos):
        """
        forward
        """
        encoder_output = self.encoder_model(input, pos)
        return encoder_output


class ProteinModel(nn.Layer):
    """
    ProteinModel
    """
    def __init__(self, encoder_model, model_config):
        """
        __init__
        """
        super(ProteinModel, self).__init__()
        task = model_config.get('task', 'pretrain')
        if task == 'pretrain':
            self.model = PretrainTaskModel(class_num=len(ProteinTokenizer.vocab), \
                                           model_config=model_config, encoder_model=encoder_model)
        elif task == 'seq_classification':
            class_num = model_config.get('class_num', 3)
            self.model = SeqClassificationTaskModel(class_num=class_num, \
                                                    model_config=model_config, encoder_model=encoder_model)
        elif task == 'classification':
            class_num = model_config.get('class_num', 3)
            self.model = ClassificationTaskModel(class_num=class_num, \
                                                 model_config=model_config, encoder_model=encoder_model)
        elif task == 'regression':
            self.model = RegressionTaskModel(model_config=model_config, encoder_model=encoder_model)
    
    def forward(self, input, pos):
        """
        forward
        """
        output = self.model(input, pos)
        return output
        

class ProteinCriterion(object):
    """
    ProteinCriterion
    """
    def __init__(self, model_config):
        """
        __init__
        """
        super(ProteinCriterion, self).__init__()
        task = model_config.get('task', 'pretrain')
        if task == 'pretrain':
            self.criterion = paddle.nn.CrossEntropyLoss(ignore_index=-1)
        elif task == 'seq_classification':
            self.criterion = paddle.nn.CrossEntropyLoss(ignore_index=-1)
        elif task == 'classification':
            self.criterion = paddle.nn.CrossEntropyLoss(ignore_index=-1)
        elif task == 'regression':
            self.criterion = paddle.nn.MSELoss()
    
    def cal_loss(self, pred, label):
        """
        cal_loss
        """
        loss = self.criterion(pred, label)
        return loss

