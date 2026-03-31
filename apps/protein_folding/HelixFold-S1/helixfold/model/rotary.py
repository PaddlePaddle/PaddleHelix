from typing import Optional

import paddle
import paddle.nn as nn


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
