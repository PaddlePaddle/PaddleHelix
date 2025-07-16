#   Copyright (c) 2024 PaddleHelix Authors. All Rights Reserved.
#
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# International License (the "License");  you may not use this file  except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rotary Positional Embeddings (RoPE)"""

from typing import Optional

import paddle
import paddle.nn as nn


class ChainIDEmbedding(nn.Layer):
    def __init__(self, channel, max_vocab_size=256):
        super(ChainIDEmbedding, self).__init__()
        self.channel = channel

        self.entity_id_embed = nn.Embedding(max_vocab_size, channel)
        self.asym_id_embed = nn.Embedding(max_vocab_size, channel)

    def forward(self, batch):
        """
        Args:
            entity_id: [B, L]
            asym_id: [B, L]
        """
        embed = self.entity_id_embed(batch['entity_id'])
        embed += self.asym_id_embed(batch['asym_id'])
        return embed


class RotaryPositionalEmbeddings(nn.Layer):
    """
    Rotary Positional Embeddings (RoPE)
    https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()

    def _rope_init(self):
        # Calculate theta = 1 / (base ** (2i / dim))
        theta = 1.0 / (
            self.base ** (paddle.arange(0, self.dim, 2, dtype='float32') / self.dim)
        )
        seq_idx = paddle.arange(self.max_seq_len, dtype=theta.dtype)
        idx_theta = paddle.einsum('i,j->ij', seq_idx, theta)
        # [max_seq_len, dim // 2, 2]
        self.cache = paddle.stack([paddle.cos(idx_theta), paddle.sin(idx_theta)], axis=-1)
        
    def forward(self, x: paddle.Tensor, input_pos: Optional[paddle.Tensor] = None) -> paddle.Tensor:
        """
        Args:
            x (Tensor): input tensor with shape [*, s, d]
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied
        """
        s, d = x.shape[-2:]

        rope_cache = (
            self.cache[:s] if input_pos is None else paddle.gather(self.cache, input_pos)
        )   # (s, d // 2, 2)
        xshaped = x.reshape(list(x.shape)[:-2] + [s, d // 2, 2])    # (*, s, d // 2, 2)
        x_out = paddle.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            axis=-1,
        )   # [*, s, d // 2, 2]
        x_out = paddle.flatten(x_out, start_axis=-2) # [*, s, d]
        return x_out


if __name__ == '__main__':
    seq_len = 128
    batch_size = 4
    num_heads = 8
    head_dim = 64

    x = paddle.randn((batch_size, num_heads, seq_len, head_dim))
    rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=512)
    x_rope = rope(x)

    print(x_rope.shape)  # Output: [4, 8, 128, 64]
