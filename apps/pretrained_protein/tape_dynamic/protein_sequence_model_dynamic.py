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
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class LstmSeqClassificationModel(nn.Layer):
    """
    Lstm model for seq classification task.
    """

    def __init__(self,
                 vocab_size,
                 num_class,
                 emb_dim=512,
                 hidden_size=512,
                 n_lstm_layer=3,
                 is_bidirectory=True,
                 padding_idx=0,
                 epsilon=1e-5,
                 dropout_rate=0.1):
        """Init model

        Args:
            vocab_size (int): vocab size.
            num_class (int): num of classes.
            emb_dim (int, optional): embedding dimmension. Defaults to 512.
            hidden_size (int, optional): hidden size. Defaults to 512.
            n_lstm_layer (int, optional): number of lstm layer. Defaults to 3.
            is_bidirectory (bool, optional): use bidirect lstm. Defaults to True.
            padding_idx (int, optional): padding index. Defaults to 0.
            epsilon (float, optional): epsilon. Defaults to 1e-5.
            dropout_rate (float, optional): dropout rate. Defaults to 0.1.
        """
        super(LstmSeqClassificationModel, self).__init__()
        self.padding_idx = padding_idx
        self.embedder = nn.Embedding(vocab_size,
                                     emb_dim,
                                     padding_idx=padding_idx)
        self.layer_norm = nn.LayerNorm(normalized_shape=emb_dim,
                                       epsilon=epsilon)
        self.dropout = nn.Dropout(p=dropout_rate)
        direction = 'bidirectional' if is_bidirectory else 'forward'
        self.lstm_encoder = nn.LSTM(emb_dim,
                                    hidden_size,
                                    num_layers=n_lstm_layer,
                                    direction=direction)
        # kernel_size = (5, hidden_size * 2) if is_bidirectory else (5, hidden_size)

        in_channels = hidden_size * 2 if is_bidirectory else hidden_size
        self.conv_encoder = nn.Conv1D(in_channels=in_channels,
                                      out_channels=hidden_size,
                                      kernel_size=5,
                                      padding=2)
        self.output_layer = nn.Conv1D(in_channels=hidden_size,
                                      out_channels=num_class,
                                      kernel_size=3,
                                      padding=1)

    def _prepare_emb(self, tokens):
        """prepare emb"""
        embedded_text = self.embedder(tokens)
        embedded_text = self.layer_norm(embedded_text)
        embedded_text = self.dropout(embedded_text)
        return embedded_text

    def _seq_classification_task(self, lstm_output):
        """calc seq class loss"""
        conv_out = self.conv_encoder(lstm_output)
        conv_out = F.relu(conv_out)
        # Shape: (batch_size, num_class, num_tokens, )
        logits = self.output_layer(conv_out).transpose(perm=(0, 2, 1))
        return logits

    def forward(self, tokens, seq_lens):
        """model forward"""
        embedded_text = self._prepare_emb(tokens)
        lstm_output, (last_hidden,
                      last_cell) = self.lstm_encoder(embedded_text)
        # Shape: (batch_size, hidden_size, num_tokens) = (N, C, L)
        lstm_output = lstm_output.transpose(perm=(0, 2, 1))
        # Shape: (batch_size, out_channels, num_tokens)
        logits = self._seq_classification_task(lstm_output)
        return logits


class TransformerEncoder(nn.Layer):
    """TransformerEncoder"""

    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None):
        """TransformerEncoder"""
        super(TransformerEncoder, self).__init__()

        if isinstance(bias_attr, (list, tuple)):
            if len(bias_attr) == 1:
                encoder_bias_attr = [bias_attr[0]] * 2
            elif len(bias_attr) == 2:
                encoder_bias_attr = bias_attr
            elif len(bias_attr) == 3:
                encoder_bias_attr = [bias_attr[0], bias_attr[-1]]
            else:
                assert False, (
                    "length of bias_attr should be 1 or 2 or 3 when it is a list/tuple"
                )
        else:
            encoder_bias_attr = bias_attr

        if isinstance(weight_attr, (list, tuple)):
            if len(weight_attr) == 1:
                encoder_weight_attr = [weight_attr[0]] * 2
            elif len(weight_attr) == 2:
                encoder_weight_attr = weight_attr
            elif len(weight_attr) == 3:
                encoder_weight_attr = [weight_attr[0], weight_attr[-1]]
            else:
                assert False, (
                    "length of weight_attr should be 1 or 2 or 3 when it is a list/tuple"
                )
        else:
            encoder_weight_attr = weight_attr

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, attn_dropout,
            act_dropout, normalize_before, encoder_weight_attr,
            encoder_bias_attr)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers,
                                             encoder_norm)

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, src_mask=None):
        r"""
        Applies a Transformer model on the inputs.
        Parameters:
            src (Tensor): The input of Transformer encoder. It is a tensor
                with shape `[batch_size, source_length, d_model]`. The data type
                should be float32 or float64.
        """
        memory = self.encoder(src, src_mask=src_mask)
        return memory


class TransformerSeqClassificationModel(nn.Layer):
    """Transformer model for seq classification task
    """

    def __init__(self,
                 vocab_size,
                 num_class,
                 emb_dim=512,
                 hidden_size=512,
                 head_num=4,
                 layer_num=4,
                 padding_idx=0,
                 epsilon=1e-5,
                 dropout_rate=0.1):
        """Init model

        Args:
            vocab_size (int): vocab size.
            num_class (int): num classes.
            emb_dim (int, optional): embeding dimension. Defaults to 512.
            hidden_size (int, optional): hidden size. Defaults to 512.
            head_num (int, optional): head_num. Defaults to 4.
            layer_num (int, optional): layer num. Defaults to 4.
            padding_idx (int, optional): padding index. Defaults to 0.
            epsilon (float, optional): epsilon. Defaults to 1e-5.
            dropout_rate (float, optional): dropout rate. Defaults to 0.1.
        """
        super(TransformerSeqClassificationModel, self).__init__()
        self.padding_idx = padding_idx
        self.embedder = nn.Embedding(vocab_size,
                                     emb_dim,
                                     padding_idx=padding_idx)
        self.layer_norm = nn.LayerNorm(normalized_shape=emb_dim,
                                       epsilon=epsilon)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.transformer_encoder = TransformerEncoder(
            d_model=emb_dim,
            nhead=head_num,
            num_encoder_layers=layer_num,
            dim_feedforward=hidden_size * 4,
            dropout=dropout_rate,
            attn_dropout=dropout_rate,
            act_dropout=dropout_rate,
        )

        in_channels = hidden_size
        self.conv_encoder = nn.Conv1D(in_channels=in_channels,
                                      out_channels=hidden_size,
                                      kernel_size=5,
                                      padding=2)
        self.output_layer = nn.Conv1D(in_channels=hidden_size,
                                      out_channels=num_class,
                                      kernel_size=3,
                                      padding=1)

    def _prepare_emb(self, tokens):
        """Get emebedding"""
        embedded_text = self.embedder(tokens)
        embedded_text = self.layer_norm(embedded_text)
        embedded_text = self.dropout(embedded_text)
        return embedded_text

    def _seq_classification_task(self, lstm_output):
        """Calc sequence classification loss"""
        conv_out = self.conv_encoder(lstm_output)
        conv_out = F.relu(conv_out)
        # Shape: (batch_size, num_class, num_tokens, )
        logits = self.output_layer(conv_out).transpose(perm=(0, 2, 1))
        return logits

    def forward(self, tokens, seq_lens):
        """Forward function

        Args:
            tokens (Tensor): 
            seq_lens (Tensor): sequence length of inputs.

        Returns:
            Tensor: the logits results.
        """
        # model
        embedded_text = self._prepare_emb(tokens)
        output = self.transformer_encoder(embedded_text)
        logits = self._seq_classification_task(
            output.transpose(perm=(0, 2, 1)))

        return logits
