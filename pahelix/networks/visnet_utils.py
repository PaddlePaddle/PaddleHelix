import math
from typing import Optional, Tuple

import numpy as np
import paddle
import paddle.nn.functional as F
import pgl
from paddle import Tensor, nn


def radius(x, y, r, max_num_neighbors):
    assert x.dim() == 2 and y.dim() == 2, "Input must be 2-D tensor"
    assert x.shape[1] == y.shape[1], "Input dimensions must match"

    row = paddle.full((y.shape[0] * max_num_neighbors,), -1, dtype="int64")
    col = paddle.full((y.shape[0] * max_num_neighbors,), -1, dtype="int64")

    count = 0
    for n_y in range(y.shape[0]):
        for n_x in range(x.shape[0]):
            dist = paddle.sum((x[n_x, :] - y[n_y, :]) ** 2)
            if dist < r:
                row[n_y * max_num_neighbors + count] = n_y
                col[n_y * max_num_neighbors + count] = n_x
                count += 1
            if count >= max_num_neighbors:
                break

    mask = row != -1
    return paddle.stack([row[mask], col[mask]], axis=0)


def batch_radius_graph(x, r, batch=None, loop=False, max_num_neighbors=32):
    assert x.dim() == 2, "Input must be 2-D tensor"

    if batch is None:
        batch = paddle.zeros([x.shape[0]], dtype="int64")

    unique_batches = paddle.unique(batch)
    num_batches = unique_batches.shape[0]

    all_edges = []

    for batch_id in range(num_batches):
        batch_mask = batch == batch_id
        batch_x = x[batch_mask]

        edge_index = radius(
            batch_x, batch_x, r, max_num_neighbors if loop else max_num_neighbors + 1
        )
        if not loop:
            mask = edge_index[0, :] != edge_index[1, :]
            edge_index = edge_index[:, mask]

        # Adjust the node indices for the current batch
        edge_index += batch_id * batch_x.shape[0]

        all_edges.append(edge_index)

    return paddle.concat(all_edges, axis=1)


class CosineCutoff(nn.Layer):
    r"""Appies a cosine cutoff to the input distances.
    
    .. math::
        \text{cutoffs} =
        \begin{cases}
        0.5 * (\cos(\frac{\text{distances} * \pi}{\text{cutoff}}) + 1.0),
        & \text{if } \text{distances} < \text{cutoff} \\
        0, & \text{otherwise}
        \end{cases}
        
    Args:
        cutoff (float): A scalar that determines the point
            at which the cutoff is applied.
    """

    def __init__(self, cutoff: float):
        super(CosineCutoff, self).__init__()
        self.cutoff = cutoff

    def forward(self, distances: Tensor):
        r"""Applies a cosine cutoff to the input distances.
        Args:
            distances (paddle.Tensor): A tensor of distances.
        Returns:
            cutoffs (paddle.Tensor): A tensor where the cosine function
                has been applied to the distances,
                but any values that exceed the cutoff are set to 0.
        """
        cutoffs = 0.5 * (paddle.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).astype("float32")
        return cutoffs


class ExpNormalSmearing(nn.Layer):
    r"""Applies exponential normal smearing to the input distances.

    .. math::
        \text{smeared\_dist} = \text{CosineCutoff}(\text{dist})
        * e^{-\beta * (e^{\alpha * (-\text{dist})} - \text{means})^2}

    Args:
        cutoff (float): A scalar that determines the point
            at which the cutoff is applied.
        num_rbf (int): The number of radial basis functions.
        trainable (bool): If True, the means and betas of the RBFs
            are trainable parameters.
    """

    def __init__(self, cutoff: float = 5.0, num_rbf: int = 128, trainable: bool = True):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.add_parameter(
                "means",
                self.create_parameter(
                    shape=means.shape, default_initializer=nn.initializer.Assign(means)
                ),
            )
            self.add_parameter(
                "betas",
                self.create_parameter(
                    shape=means.shape, default_initializer=nn.initializer.Assign(betas)
                ),
            )
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self) -> Tuple[Tensor, Tensor]:
        r"""Initializes the means and betas
        for the radial basis functions.
        Returns:
            means, betas (Tuple[paddle.Tensor, paddle.Tensor]): The
                initialized means and betas.
        """
        start_value = paddle.exp(paddle.to_tensor(-self.cutoff))
        means = paddle.linspace(start_value, 1, self.num_rbf)
        betas = paddle.to_tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def forward(self, dist: Tensor) -> Tensor:
        r"""Applies the exponential normal smearing
        to the input distance.

        Args:
            dist (paddle.Tensor): A tensor of distances.

        Returns:
            smeared_dist (paddle.Tensor): The smeared distances.
        """
        dist = dist.unsqueeze(-1)
        smeared_dist = self.cutoff_fn(dist) * paddle.exp(
            -self.betas * (paddle.exp(self.alpha * (-dist)) - self.means) ** 2
        )
        return smeared_dist


class Sphere(nn.Layer):
    r"""Computes spherical harmonics of the input data.

    This module computes the spherical harmonics up
    to a given degree `lmax` for the input tensor of 3D vectors.
    The vectors are assumed to be given in Cartesian coordinates.
    See `Wikipedia
    <https://en.wikipedia.org/wiki/Table_of_spherical_harmonics>`_
    for mathematical details.

    Args:
        lmax (int): The maximum degree of the spherical harmonics.
    """

    def __init__(self, lmax: int = 2):
        super(Sphere, self).__init__()
        self.lmax = lmax

    def forward(self, edge_vec: Tensor) -> Tensor:
        r"""Computes the spherical harmonics of the input tensor.

        Args:
            edge_vec (paddle.Tensor): A tensor of 3D vectors.

        Returns:
            edge_sh (paddle.Tensor): The spherical harmonics
                of the input tensor.
        """
        edge_sh = self._spherical_harmonics(
            self.lmax, edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2]
        )
        return edge_sh

    @staticmethod
    def _spherical_harmonics(lmax: int, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        r"""Computes the spherical harmonics
        up to degree `lmax` of the input vectors.

        Args:
            lmax (int): The maximum degree of the spherical harmonics.
            x (paddle.Tensor): The x coordinates of the vectors.
            y (paddle.Tensor): The y coordinates of the vectors.
            z (paddle.Tensor): The z coordinates of the vectors.

        Returns:
            sh (paddle.Tensor): The spherical harmonics of the input vectors.
        """

        sh_1_0, sh_1_1, sh_1_2 = x, y, z

        if lmax == 1:
            return paddle.stack([sh_1_0, sh_1_1, sh_1_2], axis=-1)

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

        if lmax == 2:
            return paddle.stack(
                [sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4],
                axis=-1,
            )


class VecLayerNorm(nn.Layer):
    r"""Applies layer normalization to the input data.

    This module applies a custom layer normalization to a tensor of vectors.
    The normalization can either be "max_min" normalization,
    or no normalization.

    Args:
        hidden_channels (int): The number of hidden channels in the input.
        trainable (bool): If True, the normalization weights
            are trainable parameters.
        norm_type (str): The type of normalization to apply.
            Can be "max_min" or "none".
    """

    def __init__(
        self, hidden_channels: int, trainable: bool, norm_type: str = "max_min"
    ):
        super(VecLayerNorm, self).__init__()

        self.hidden_channels = hidden_channels
        self.eps = 1e-12

        weight = paddle.ones(self.hidden_channels)
        if trainable:
            self.add_parameter(
                "weight",
                self.create_parameter(
                    shape=weight.shape,
                    default_initializer=nn.initializer.Assign(weight),
                ),
            )
        else:
            self.register_buffer("weight", weight)

        if norm_type == "max_min":
            self.norm = self.max_min_norm
        else:
            self.norm = self.none_norm

    def none_norm(self, vec: Tensor) -> Tensor:
        r"""Applies no normalization to the input tensor.

        Args:
            vec (paddle.Tensor): The input tensor.

        Returns:
            vec (paddle.Tensor): The same input tensor.
        """
        return vec

    def max_min_norm(self, vec: Tensor) -> Tensor:
        r"""Applies max-min normalization to the input tensor.

        .. math::
            \text{dist} = ||\text{vec}||_2
            \text{direct} = \frac{\text{vec}}{\text{dist}}
            \text{max\_val} = \max(\text{dist})
            \text{min\_val} = \min(\text{dist})
            \text{delta} = \text{max\_val} - \text{min\_val}
            \text{dist} = \frac{\text{dist} - \text{min\_val}}{\text{delta}}
            \text{normed\_vec} = \max(0, \text{dist}) \cdot \text{direct}

        Args:
            vec (paddle.Tensor): The input tensor.

        Returns:
            normed_vec (paddle.Tensor): The normalized tensor.
        """
        dist = paddle.norm(vec, axis=1, keepdim=True)

        if (dist == 0).all():
            return paddle.zeros_like(vec)

        dist = paddle.clip(dist, min=self.eps)
        direct = vec / dist

        max_val, _ = paddle.max(dist, axis=-1)
        min_val, _ = paddle.min(dist, axis=-1)
        delta = (max_val - min_val).view(-1)
        delta = paddle.where(delta == 0, paddle.ones_like(delta), delta)
        dist = (dist - min_val.view(-1, 1, 1)) / delta.view(-1, 1, 1)

        return F.relu(dist) * direct

    def forward(self, vec: Tensor) -> Tensor:
        r"""Applies the layer normalization to the input tensor.

        Args:
            vec (paddle.Tensor): The input tensor.

        Returns:
            normed_vec (paddle.Tensor): The normalized tensor.
        """
        if vec.shape[1] == 3:
            vec = self.norm(vec)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        elif vec.shape[1] == 8:
            vec1, vec2 = paddle.split(vec, [3, 5], axis=1)
            vec1 = self.norm(vec1)
            vec2 = self.norm(vec2)
            vec = paddle.concat([vec1, vec2], axis=1)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError("VecLayerNorm only support 3 or 8 channels")


class Distance(nn.Layer):
    r"""Computes the pairwise distances between atoms in a molecule.

    This module computes the pairwise distances between atoms in a molecule,
    represented by their positions `pos`.
    The distances are computed only between points
    that are within a certain cutoff radius.

    Args:
        cutoff (float): The cutoff radius beyond
            which distances are not computed.
        max_num_neighbors (int): The maximum number of neighbors
            considered for each point.
        loop (bool): Whether self-loops are included.
    """

    def __init__(self, cutoff: float, max_num_neighbors: int = 32, loop: bool = True):
        super(Distance, self).__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.loop = loop

    def forward(self, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Computes the pairwise distances between atoms in the molecule.

        Args:
            pos (paddle.Tensor): The positions of the atoms
                in the molecule.
            batch (paddle.Tensor): A batch vector,
                which assigns each node to a specific example.

        Returns:
            edge_index (paddle.Tensor): The indices of the edges
                in the graph.
            edge_weight (paddle.Tensor): The distances
                between connected nodes.
            edge_vec (paddle.Tensor): The vector differences
                between connected nodes.
        """
        edge_index = batch_radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            loop=self.loop,
            max_num_neighbors=self.max_num_neighbors,
        )
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        if self.loop:
            mask = edge_index[0] != edge_index[1]
            edge_weight = paddle.zeros(edge_vec.shape[0])
            edge_weight[mask] = paddle.norm(edge_vec[mask], axis=-1)
        else:
            edge_weight = paddle.norm(edge_vec, axis=-1)

        return edge_index, edge_weight, edge_vec


class NeighborEmbedding(nn.Layer):
    r"""The `NeighborEmbedding` module from the
    `"Enhancing geometric representations for molecules
    with equivariant vector-scalar interactive message passing"
    <https://arxiv.org/pdf/2210.16518.pdf>`_ paper.

    Args:
        hidden_channels (int): The number of hidden channels
            in the node embeddings.
        num_rbf (int): The number of radial basis functions.
        cutoff (float): The cutoff distance.
        max_z (int): The maximum atomic numbers.
    """

    def __init__(
        self, hidden_channels: int, num_rbf: int, cutoff: float, max_z: int = 100
    ):
        super(NeighborEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff)

    def forward(
        self,
        z: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        r"""Computes the neighborhood embedding of the nodes in the graph.

        Args:
            z (paddle.Tensor): The atomic numbers.
            x (paddle.Tensor): The node features.
            edge_index (paddle.Tensor): The indices of the edges.
            edge_weight (paddle.Tensor): The weights of the edges.
            edge_attr (paddle.Tensor): The edge features.

        Returns:
            x_neighbors (paddle.Tensor): The neighborhood embeddings
                of the nodes.
        """
        mask = edge_index[0] != edge_index[1]

        if not mask.all():
            edge_index = edge_index.T[mask].T
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.unsqueeze(-1)

        x_neighbors = self.embedding(z)

        graph = pgl.Graph(
            edges=edge_index.T,
            node_feat={
                "x": x,
                "z": z,
            },
            edge_feat={
                "W": W,
            },
        )

        def _send_func(src_feat, dst_feat, edge_feat):
            x_j = src_feat["x"]
            W = edge_feat["W"]

            return {"x": x_j * W}

        def _recv_func(msg: pgl.Message):
            x = msg["x"]
            return msg.reduce(x, pool_type="sum")

        msg = graph.send(
            message_func=_send_func,
            node_feat={
                "x": x_neighbors,
            },
            edge_feat={
                "W": W,
            },
        )

        x_neighbors = graph.recv(reduce_func=_recv_func, msg=msg)
        x_neighbors = self.combine(paddle.concat([x, x_neighbors], axis=-1))

        return x_neighbors


class EdgeEmbedding(nn.Layer):
    r"""The `EdgeEmbedding` module
    from the `"Enhancing geometric representations for molecules
    with equivariant vector-scalar interactive message passing"
    <https://arxiv.org/pdf/2210.16518.pdf>`_ paper.

    Args:
        num_rbf (int):
            The number of radial basis functions.
        hidden_channels (int):
            The number of hidden channels in the node embeddings.
    """

    def __init__(self, num_rbf: int, hidden_channels: int):
        super(EdgeEmbedding, self).__init__()
        self.edge_proj = nn.Linear(num_rbf, hidden_channels)

    def forward(self, edge_index: Tensor, edge_attr: Tensor, x: Tensor) -> Tensor:
        r"""Computes the edge embeddings of the graph.

        Args:
            edge_index (paddle.Tensor): The indices of the edges.
            edge_attr (paddle.Tensor): The edge features.
            x (paddle.Tensor): The node features.

        Returns:
            out_edge_attr (paddle.Tensor): The edge embeddings.
        """
        edges = edge_index.T

        graph = pgl.Graph(
            edges=edges,
            node_feat={
                "x": x,
            },
            edge_feat={
                "edge_attr": edge_attr,
            },
        )

        def _send_func(src_feat, dst_feat, edge_feat):
            edge_attr = edge_feat["edge_attr"]
            x_i, x_j = src_feat["x"], dst_feat["x"]
            edge_attr = (x_i + x_j) * self.edge_proj(edge_attr)
            return {"edge_attr": edge_attr}

        msg = graph.send(
            message_func=_send_func,
            node_feat={
                "x": x,
            },
            edge_feat={
                "edge_attr": edge_attr,
            },
        )

        return msg["edge_attr"]


class Atomref(nn.Layer):
    r"""Adds atom reference values to atomic energies.

    Args:
        atomref (paddle.Tensor, optional):  A tensor of atom reference values,
            or None if not provided.
        max_z (int): The maximum atomic numbers.
    """

    def __init__(self, atomref: Optional[Tensor] = None, max_z: int = 100):
        super(Atomref, self).__init__()
        if atomref is None:
            atomref = paddle.zeros((max_z, 1))
        else:
            atomref = paddle.to_tensor(atomref)

        if atomref.ndim == 1:
            atomref = atomref.reshape((-1, 1))
        self.register_buffer("initial_atomref", atomref)
        self.atomref = nn.Embedding(len(atomref), 1)
        paddle.assign(self.initial_atomref, self.atomref.weight)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        r"""Adds atom reference values to atomic energies.

        Args:
            x (paddle.Tensor): The atomic energies.
            z (paddle.Tensor): The atomic numbers.

        Returns:
            x (paddle.Tensor): The updated atomic energies.
        """
        return x + self.atomref(z)
