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

"""MolTrans backbone model."""

from helper import utils
import paddle
from paddle import nn
import paddle.nn.functional as F
import numpy as np
import math

# Set seed for reproduction
paddle.seed(2)
np.random.seed(3)


class MolTransModel(nn.Sequential):
    """MolTrans model.

    Args:
        d: Input drug.
        t: Input target.
        d_masking: Input drug maskings.
        t_masking: Input target maskings.
    
    Returns:
        res: Prediction results.
    """
    def __init__(self, model_config):
        """Initialization."""
        super(MolTransModel, self).__init__()
        # Basic config
        self.model_config = model_config
        self.drug_max_seq = model_config['drug_max_seq']
        self.target_max_seq = model_config['target_max_seq']
        self.emb_size = model_config['emb_size']
        self.dropout_ratio = model_config['dropout_ratio']
        self.input_drug_dim = model_config['input_drug_dim']
        self.input_target_dim = model_config['input_target_dim']
        self.layer_size = model_config['layer_size']
        self.gpus = 1

        # Model config
        self.interm_size = model_config['interm_size']
        self.num_attention_heads = model_config['num_attention_heads']
        self.attention_dropout_ratio = model_config['attention_dropout_ratio']
        self.hidden_dropout_ratio = model_config['hidden_dropout_ratio']
        self.flatten_dim = model_config['flatten_dim']
        self.hidden_size = model_config['emb_size']

        # Enhanced embeddings
        self.drug_emb = EnhancedEmbedding(self.input_drug_dim, self.emb_size, self.drug_max_seq, self.dropout_ratio)
        self.target_emb = EnhancedEmbedding(self.input_target_dim, self.emb_size, self.target_max_seq,
                                             self.dropout_ratio)
        # Encoder module
        self.encoder = EncoderModule(self.layer_size, self.hidden_size, self.interm_size, self.num_attention_heads,
                                      self.attention_dropout_ratio, self.hidden_dropout_ratio)
        # Cross information
        self.interaction_cnn = nn.Conv2D(1, 3, 3, padding=1) # Conv2D

        # Decoder module
        self.decoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),

            # nn.LayerNorm(512),
            LayerNorm(512),
            # nn.BatchNorm(512),
            nn.Linear(512, 64),
            nn.ReLU(),

            # nn.LayerNorm(64),
            LayerNorm(64),
            # nn.BatchNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, d, t, d_masking, t_masking):
        """MolTrans pipeline."""
        tempd_masking = d_masking.unsqueeze(1).unsqueeze(2)
        tempt_masking = t_masking.unsqueeze(1).unsqueeze(2)

        tempd_masking = (1.0 - tempd_masking) * -10000.0
        tempt_masking = (1.0 - tempt_masking) * -10000.0

        d_embedding = self.drug_emb(d)
        t_embedding = self.target_emb(t)
        
        d_encoder = self.encoder(d_embedding.float(), tempd_masking.float())
        t_encoder = self.encoder(t_embedding.float(), tempt_masking.float())

        drug_res = paddle.unsqueeze(d_encoder, 2).repeat(1, 1, self.target_max_seq, 1)
        target_res = paddle.unsqueeze(t_encoder, 1).repeat(1, self.drug_max_seq, 1, 1)

        i_score = drug_res * target_res

        i_scoreT = i_score.view(int(i_score.shape[0] / self.gpus), -1, self.drug_max_seq, self.target_max_seq)
        i_scoreT = paddle.sum(i_scoreT, axis=1)
        i_scoreT = paddle.unsqueeze(i_scoreT, 1)
        i_scoreT = F.dropout(i_scoreT, p=self.dropout_ratio)

        i_scoreT = self.interaction_cnn(i_scoreT)
        i_res = i_scoreT.view(int(i_scoreT.shape[0] / self.gpus), -1)
        res = self.decoder(i_res)

        return res


class EnhancedEmbedding(nn.Layer):
    """Enhanced embeddings of drug and target.

    Args:
        input_id: Raw input of drug or target.
    
    Returns:
        embedding: Output embedding results.
    """
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_ratio):
        """Initialization."""
        super(EnhancedEmbedding, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, input_id):
        """Embeddings."""
        seq_len = input_id.size(1)
        position_id = paddle.arange(seq_len, dtype="int64")
        position_id = position_id.unsqueeze(0).expand_as(input_id)

        word_embeddings = self.word_embedding(input_id)
        position_embeddings = self.position_embedding(position_id)

        embedding = word_embeddings + position_embeddings
        embedding = self.LayerNorm(embedding)
        embedding = self.dropout(embedding)
        return embedding


class LayerNorm(nn.Layer):
    """Customized LayerNorm.

    Args:
        x: Input Tensor.

    Returns:
        res: Output Tensor via LayerNorm.
    """
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        """Initialization."""
        super(LayerNorm, self).__init__()
        self.beta = paddle.create_parameter(shape=[hidden_size], dtype="float32",
            default_initializer = nn.initializer.Assign(paddle.zeros([hidden_size], "float32")))
        self.gamma = paddle.create_parameter(shape=[hidden_size], dtype="float32",
            default_initializer = nn.initializer.Assign(paddle.ones([hidden_size], "float32")))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        """LayerNorm."""
        v = x.mean(-1, keepdim=True)
        s = (x - v).pow(2).mean(-1, keepdim=True)
        x = (x - v) / paddle.sqrt(s + self.variance_epsilon)
        res = self.gamma * x + self.beta
        return res


class EncoderModule(nn.Layer):
    """Encoder module with multiple layers.

    Args:
        input_emb: Input embeddings.
        attention_mask: Input maskings.

    Returns:
        module_output: Temporary outputs.
    """
    def __init__(self, layer_size, hidden_size, interm_size, num_attention_heads,
                 attention_dropout_ratio, hidden_dropout_ratio):
        """Initialization."""
        super(EncoderModule, self).__init__()
        module = Encoder(hidden_size, interm_size, num_attention_heads, attention_dropout_ratio, hidden_dropout_ratio)
        self.module = nn.LayerList([module for _ in range(layer_size)])

    def forward(self, input_emb, attention_mask, output_all_encoded_layers=True):
        """Multiple encoders."""
        for layer_module in self.module:
            module_output = layer_module(input_emb, attention_mask)

        return module_output

  
class Encoder(nn.Layer):
    """Encoder module.

    Args:
        input_emb: Input embeddings.
        attention_mask: Input maskings.

    Returns:
        module_output: Temporary outputs.
    """
    def __init__(self, hidden_size, interm_size, num_attention_heads, attention_dropout_ratio, hidden_dropout_ratio):
        """Initialization."""
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_dropout_ratio, hidden_dropout_ratio)
        self.latent = LatentModule(hidden_size, interm_size)
        self.output = Output(interm_size, hidden_size, hidden_dropout_ratio)

    def forward(self, input_emb, attention_mask):
        """Encoder block."""
        attention_temp = self.attention(input_emb, attention_mask)
        latent_temp = self.latent(attention_temp)
        module_output = self.output(latent_temp, attention_temp)
        return module_output


class LatentModule(nn.Layer):
    """Intermediate Layer.

    Args:
        hidden_states: Temporary latent representation.

    Returns:
        module_output: Temporary outputs.
    """
    def __init__(self, hidden_size, interm_size):
        """Initialization."""
        super(LatentModule, self).__init__()
        self.connecter = nn.Linear(hidden_size, interm_size)

    def forward(self, hidden_states):
        """Latent block."""
        hidden_states = self.connecter(hidden_states)
        module_output = F.relu(hidden_states)
        return module_output


class Output(nn.Layer):
    """Output Layer.

    Args:
        hidden_states: Temporary latent representation.
        input_tensor: Input temporary Tensor.

    Returns:
        module_output: Temporary outputs.
    """
    def __init__(self, interm_size, hidden_size, hidden_dropout_ratio):
        """Initialization."""
        super(Output, self).__init__()
        self.connecter = nn.Linear(interm_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_ratio)

    def forward(self, hidden_states, input_tensor):
        """Output block."""
        hidden_states = self.connecter(hidden_states)
        hidden_states = self.dropout(hidden_states)
        module_output = self.LayerNorm(hidden_states + input_tensor)
        return module_output


class SelfOutput(nn.Layer):
    """Self-Output Layer.

    Args:
        hidden_states: Temporary latent representation.
        input_tensor: Input temporary Tensor.

    Returns:
        module_output: Temporary outputs.
    """
    def __init__(self, hidden_size, hidden_dropout_ratio):
        """Initialization."""
        super(SelfOutput, self).__init__()
        self.connecter = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_ratio)

    def forward(self, hidden_states, input_tensor):
        """Self-output block."""
        hidden_states = self.connecter(hidden_states)
        hidden_states = self.dropout(hidden_states)
        module_output = self.LayerNorm(hidden_states + input_tensor)
        return module_output


class Attention(nn.Layer):
    """Attention.

    Args:
        input_emb: Input embeddings.
        attention_mask: Input maskings.

    Returns:
        module_output: Temporary outputs.
    """
    def __init__(self, hidden_size, num_attention_heads, attention_dropout_ratio, hidden_dropout_ratio):
        """Initialization."""
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_dropout_ratio)
        self.output = SelfOutput(hidden_size, hidden_dropout_ratio)

    def forward(self, input_emb, attention_mask):
        """Attention block."""
        attention_output = self.self(input_emb, attention_mask)
        module_output = self.output(attention_output, input_emb)
        return module_output


class SelfAttention(nn.Layer):
    """Self-Attention.

    Args:
        hidden_states: Temporary latent representation.
        attention_mask: Input maskings.

    Returns:
        attention_map: Output attention map.
    """
    def __init__(self, hidden_size, num_attention_heads, attention_dropout_ratio):
        """Initialization."""
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                 "The hidden size (%d) is not a product of the number of attention heads (%d)" %
                 (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.head_size

        self.q = nn.Linear(hidden_size, self.all_head_size)
        self.k = nn.Linear(hidden_size, self.all_head_size)
        self.v = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_dropout_ratio)

    def score_transpose(self, x):
        """Score transpose."""
        temp = x.size()[:-1] + [self.num_attention_heads, self.head_size]
        x = x.view(*temp)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        """Self-Attention block."""
        temp_q = self.q(hidden_states)
        temp_k = self.k(hidden_states)
        temp_v = self.v(hidden_states)

        q_layer = self.score_transpose(temp_q)
        k_layer = self.score_transpose(temp_k)
        v_layer = self.score_transpose(temp_v)

        attention_score = paddle.matmul(q_layer, k_layer.transpose(-1, -2))
        attention_score = attention_score / math.sqrt(self.head_size)
        attention_score = attention_score + attention_mask

        attention_prob = nn.Softmax(axis=-1)(attention_score)
        attention_prob = self.dropout(attention_prob)

        attention_layer = paddle.matmul(attention_prob, v_layer)
        attention_layer = attention_layer.permute(0, 2, 1, 3).contiguous()

        temp_attention_layer = attention_layer.size()[:-2] + [self.all_head_size]
        attention_map = attention_layer.view(*temp_attention_layer)
        return attention_map