from typing import Optional, Tuple

import paddle
from paddle import Tensor, nn


class GatedEquivariantBlock(nn.Layer):
    r"""Applies a gated equivariant operation
    to scalar features and vector features from the
    `"Equivariant message passing for the prediction
    of tensorial properties and molecular spectra"
    <https://arxiv.org/abs/2102.03150>`_ paper.

    Args:
        hidden_channels (int): The number of hidden channels
            in the node embeddings.
        out_channels (int): The number of output channels.
        intermediate_channels (int or None): The number of channels
            in the intermediate layer,
            or None to use the same number as 'hidden_channels'.
        scalar_activation (bool): Whether to apply
            a scalar activation function to the output node features.
    """

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        intermediate_channels: Optional[int] = None,
        scalar_activation: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias_attr=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias_attr=False)

        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            nn.Silu(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = nn.Silu() if scalar_activation else None


    def forward(self, x: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Applies a gated equivariant operation
        to node features and vector features.

        Args:
            x (paddle.Tensor): The scalar features of the nodes.
            v (paddle.Tensor): The vector features of the nodes.

        Returns:
            x (paddle.Tensor): The updated scalar features of the nodes.
            v (paddle.Tensor): The updated vector features of the nodes.
        """
        vec1 = paddle.norm(self.vec1_proj(v), axis=-2)
        vec2 = self.vec2_proj(v)

        x = paddle.concat([x, vec1], axis=-1)
        x, v = paddle.split(self.update_net(x), 2, axis=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v


class EquivariantScalar(nn.Layer):
    r"""Computes final scalar outputs based on
    node features and vector features.

    Args:
        hidden_channels (int): The number of hidden channels
            in the node embeddings.
    """

    def __init__(self, hidden_channels: int):
        super(EquivariantScalar, self).__init__()
        self.output_network = nn.LayerList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(
                    hidden_channels // 2,
                    1,
                    scalar_activation=False,
                ),
            ]
        )

    def pre_reduce(self, x: Tensor, v: Tensor) -> Tensor:
        r"""Computes the final scalar outputs.

        Args:
            x (paddle.Tensor): The scalar features of the nodes.
            v (paddle.Tensor): The vector features of the nodes.

        Returns:
            out (paddle.Tensor): The final scalar outputs of the nodes.
        """
        for layer in self.output_network:
            x, v = layer(x, v)

        return x + v.sum() * 0
