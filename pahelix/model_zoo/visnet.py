from typing import Optional, Tuple

import paddle
import pgl
from paddle import Tensor, nn
from pgl.message import Message
from pgl.utils import op

from pahelix.networks.visnet_output_modules import EquivariantScalar
from pahelix.networks.visnet_utils import (Atomref, CosineCutoff, Distance,
                                           EdgeEmbedding, ExpNormalSmearing,
                                           NeighborEmbedding, Sphere,
                                           VecLayerNorm)


class ViS_Graph(pgl.Graph):
    def recv(self, reduce_func, msg, recv_mode="dst"):
        r"""Receives messages and reduces them."""
        if not self._is_tensor:
            raise ValueError("You must call Graph.tensor()")

        if not isinstance(msg, dict):
            raise TypeError(
                "The input of msg should be a dict, but receives a %s" % (type(msg))
            )

        if not callable(reduce_func):
            raise TypeError("reduce_func should be callable")

        src, dst, eid = self.sorted_edges(sort_by=recv_mode)
        msg = op.RowReader(msg, eid)
        uniq_ind, segment_ids = self.get_segment_ids(src, dst, segment_by=recv_mode)
        bucketed_msg = Message(msg, segment_ids)
        output = reduce_func(bucketed_msg)
        x, vec = output

        x_output_dim = x.shape[-1]
        vec_output_dim1 = vec.shape[-1]
        vec_output_dim2 = vec.shape[-2]
        x_init_output = paddle.zeros(
            shape=[self._num_nodes, x_output_dim], dtype=x.dtype
        )
        x_final_output = paddle.scatter(x_init_output, uniq_ind, x)

        vec_init_output = paddle.zeros(
            shape=[self._num_nodes, vec_output_dim2, vec_output_dim1], dtype=vec.dtype
        )
        vec_final_output = paddle.scatter(vec_init_output, uniq_ind, vec)

        return x_final_output, vec_final_output


class ViS_MP(nn.Layer):
    r"""The message passing module without vertex geometric features
    of the equivariant vector-scalar interactive graph neural network (ViSNet)
    from the `"Enhancing geometric representations for molecules
    with equivariant vector-scalar interactive message passing"
    <https://arxiv.org/pdf/2210.16518.pdf>`_ paper.

    Args:
        num_heads (int): The number of attention heads.
        hidden_channels (int): The number of hidden channels
            in the node embeddings.
        cutoff (float): The cutoff distance.
        vecnorm_type (str): The type of normalization
            to apply to the vectors.
        trainable_vecnorm (bool): Whether the normalization weights
            are trainable.
        last_layer (bool): Whether this is the last layer
            in the model.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_channels: int,
        cutoff: float,
        vecnorm_type: str,
        trainable_vecnorm: bool,
        last_layer: bool = False,
    ):
        super(ViS_MP, self).__init__()
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.last_layer = last_layer

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.vec_layernorm = VecLayerNorm(
            hidden_channels, trainable=trainable_vecnorm, norm_type=vecnorm_type
        )

        self.act = nn.Silu()
        self.attn_activation = nn.Silu()

        self.cutoff = CosineCutoff(cutoff)

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias_attr=False)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dk_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dv_proj = nn.Linear(hidden_channels, hidden_channels)

        self.s_proj = nn.Linear(hidden_channels, hidden_channels * 2)
        if not self.last_layer:
            self.f_proj = nn.Linear(hidden_channels, hidden_channels)
            self.w_src_proj = nn.Linear(
                hidden_channels, hidden_channels, bias_attr=False
            )
            self.w_trg_proj = nn.Linear(
                hidden_channels, hidden_channels, bias_attr=False
            )

        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)

    @staticmethod
    def vector_rejection(vec: Tensor, d_ij: Tensor):
        r"""Computes the component of 'vec' orthogonal to 'd_ij'.

        Args:
            vec (paddle.Tensor): The input vector.
            d_ij (paddle.Tensor): The reference vector.

        Returns:
            vec_rej (paddle.Tensor): The component of 'vec'
                orthogonal to 'd_ij'.
        """
        vec_proj = (vec * d_ij.unsqueeze(2)).sum(axis=1, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(2)

    def forward(self, graph: pgl.Graph) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Computes the residual scalar and vector features
        of the nodes and scalar featues of the edges.

        Args:
            graph (pgl.Graph):
                - num_nodes,
                - edges <--> edge_index ,
                - node_feat <--> x, vec,
                - edge_feat <--> r_ij, f_ij, d_ij,

        Returns:
            dx (paddle.Tensor): The residual scalar features
                of the nodes.
            dvec (paddle.Tensor): The residual vector features
                of the nodes.
            df_ij (paddle.Tensor, optional): The residual scalar features
                of the edges, or None if this is the last layer.
        """
        x, vec, r_ij, f_ij, d_ij = (
            graph.node_feat["x"],
            graph.node_feat["vec"],
            graph.edge_feat["r_ij"],
            graph.edge_feat["f_ij"],
            graph.edge_feat["d_ij"],
        )
        x = self.layernorm(x)
        vec = self.vec_layernorm(vec)

        q = self.q_proj(x).reshape([-1, self.num_heads, self.head_dim])
        k = self.k_proj(x).reshape([-1, self.num_heads, self.head_dim])
        v = self.v_proj(x).reshape([-1, self.num_heads, self.head_dim])
        dk = self.act(self.dk_proj(f_ij)).reshape([-1, self.num_heads, self.head_dim])
        dv = self.act(self.dv_proj(f_ij)).reshape([-1, self.num_heads, self.head_dim])

        vec1, vec2, vec3 = paddle.split(self.vec_proj(vec), 3, axis=-1)
        vec_dot = (vec1 * vec2).sum(axis=1)

        def _send_func(src_feat, dst_feat, edge_feat):
            q_i = dst_feat["q"]
            k_j, v_j, vec_j = (src_feat["k"], src_feat["v"], src_feat["vec"])
            dk, dv, r_ij, d_ij = (
                edge_feat["dk"],
                edge_feat["dv"],
                edge_feat["r_ij"],
                edge_feat["d_ij"],
            )

            attn = (q_i * k_j * dk).sum(axis=-1)
            attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

            v_j = v_j * dv
            v_j = (v_j * attn.unsqueeze(2)).reshape([-1, self.hidden_channels])

            s1, s2 = paddle.split(self.act(self.s_proj(v_j)), 2, axis=1)
            vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)

            return {"x": v_j, "vec": vec_j}

        def _recv_func(msg: pgl.Message):
            x, vec = msg["x"], msg["vec"]
            return msg.reduce(x, pool_type="sum"), msg.reduce(vec, pool_type="sum")

        msg = graph.send(
            message_func=_send_func,
            node_feat={"q": q, "k": k, "v": v, "vec": vec},
            edge_feat={"dk": dk, "dv": dv, "r_ij": r_ij, "d_ij": d_ij},
        )

        x, vec_out = graph.recv(reduce_func=_recv_func, msg=msg)

        o1, o2, o3 = paddle.split(self.o_proj(x), 3, axis=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_out

        def _send_func_dihedral(src_feat, dst_feat, edge_feat):
            vec_i, vec_j = dst_feat["vec"], src_feat["vec"]
            d_ij, f_ij = edge_feat["d_ij"], edge_feat["f_ij"]

            w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
            w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
            w_dot = (w1 * w2).sum(axis=1)
            df_ij = self.act(self.f_proj(f_ij)) * w_dot

            return {"df_ij": df_ij}

        if not self.last_layer:
            edge_msg = graph.send(
                message_func=_send_func_dihedral,
                node_feat={
                    "vec": vec,
                },
                edge_feat={"f_ij": f_ij, "d_ij": d_ij},
            )

            df_ij = edge_msg["df_ij"]

            return dx, dvec, df_ij

        return dx, dvec, None


class ViSNetBlock(nn.Layer):
    r"""The representation module of the equivariant vector-scalar
    interactive graph neural network (ViSNet) from the
    `"Enhancing geometric representations for molecules
    with equivariant vector-scalar interactive message passing"
    <https://arxiv.org/pdf/2210.16518.pdf>`_ paper.

    Args:
        lmax (int): The maximum degree
            of the spherical harmonics.
        vecnorm_type (str): The type of normalization
            to apply to the vectors.
        trainable_vecnorm (bool):  Whether the normalization weights
            are trainable.
        num_heads (int): The number of attention heads.
        num_layers (int): The number of layers in the network.
        hidden_channels (int): The number of hidden channels
            in the node embeddings.
        num_rbf (int): The number of radial basis functions.
        trainable_rbf (bool):Whether the radial basis function
            parameters are trainable.
        max_z (int): The maximum atomic numbers.
        cutoff (float): The cutoff distance.
        max_num_neighbors (int):The maximum number of neighbors
            considered for each atom.
        vertex (bool): Whether to use vertex geometric features.
    """

    def __init__(
        self,
        lmax: int = 1,
        vecnorm_type: str = "none",
        trainable_vecnorm: bool = False,
        num_heads: int = 8,
        num_layers: int = 9,
        hidden_channels: int = 256,
        num_rbf: int = 32,
        trainable_rbf: bool = False,
        max_z: int = 100,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
    ):
        super().__init__()
        self.lmax = lmax
        self.vecnorm_type = vecnorm_type
        self.trainable_vecnorm = trainable_vecnorm
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_rbf = num_rbf
        self.trainable_rbf = trainable_rbf
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance = Distance(cutoff, max_num_neighbors, loop=True)
        self.sphere = Sphere(lmax=lmax)
        self.distance_expansion = ExpNormalSmearing(cutoff, num_rbf, trainable_rbf)
        self.neighbor_embedding = NeighborEmbedding(
            hidden_channels, num_rbf, cutoff, max_z
        )
        self.edge_embedding = EdgeEmbedding(num_rbf, hidden_channels)

        self.vis_mp_layers = nn.LayerList()
        vis_mp_kwargs = dict(
            num_heads=num_heads,
            hidden_channels=hidden_channels,
            cutoff=cutoff,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
        )
        vis_mp_class = ViS_MP
        for _ in range(num_layers - 1):
            layer = vis_mp_class(last_layer=False, **vis_mp_kwargs)
            self.vis_mp_layers.append(layer)
        self.vis_mp_layers.append(vis_mp_class(last_layer=True, **vis_mp_kwargs))

        self.out_norm = nn.LayerNorm(hidden_channels)
        self.vec_out_norm = VecLayerNorm(
            hidden_channels, trainable=trainable_vecnorm, norm_type=vecnorm_type
        )

    def forward(self, graph: pgl.Graph):
        r"""Computes the scalar and vector features of the nodes.

        Args:
            graph (pgl.Graph):
                - num_nodes,
                - node_feat <--> z, pos,

        Returns:
            x (paddle.Tensor): The scalar features of the nodes.
            vec (paddle.Tensor): The vector features of the nodes.
        """
        z, pos = graph.node_feat["z"], graph.node_feat["pos"]

        x = self.embedding(z)
        edge_index, edge_weight, edge_vec = self.distance(pos, graph.graph_node_id)
        edge_attr = self.distance_expansion(edge_weight)
        mask = edge_index[0] != edge_index[1]
        edge_vec[mask] = edge_vec[mask] / paddle.norm(edge_vec[mask], axis=1).unsqueeze(
            1
        )
        edge_vec = self.sphere(edge_vec)
        x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)
        vec = paddle.zeros((x.shape[0], ((self.lmax + 1) ** 2) - 1, x.shape[1]))
        edge_attr = self.edge_embedding(edge_index, edge_attr, x)

        vis_graph = ViS_Graph(
            num_nodes=x.shape[0],
            edges=edge_index.T,
            node_feat={"x": x, "vec": vec},
            edge_feat={"r_ij": edge_weight, "f_ij": edge_attr, "d_ij": edge_vec},
        )
        vis_graph._graph_node_index = graph._graph_node_index

        # ViS-MP Layers
        for attn in self.vis_mp_layers[:-1]:
            dx, dvec, dedge_attr = attn(vis_graph)
            x = x + dx
            vec = vec + dvec
            edge_attr = edge_attr + dedge_attr
            vis_graph.node_feat["x"] = x
            vis_graph.node_feat["vec"] = vec
            vis_graph.edge_feat["f_ij"] = edge_attr

        dx, dvec, _ = self.vis_mp_layers[-1](vis_graph)
        x = x + dx
        vec = vec + dvec

        x = self.out_norm(x)
        vec = self.vec_out_norm(vec)

        return x, vec


class ViSNet(nn.Layer):
    r"""A PyTorch module that implements
    the equivariant vector-scalar interactive graph neural network (ViSNet)
    from the `"Enhancing geometric representations for molecules
    with equivariant vector-scalar interactive message passing"
    <https://arxiv.org/pdf/2210.16518.pdf>`_ paper.

    Args:
        lmax (int): The maximum degree
            of the spherical harmonics.
        vecnorm_type (str): The type of normalization
            to apply to the vectors.
        trainable_vecnorm (bool): Whether the normalization weights
            are trainable.
        num_heads (int): The number of attention heads.
        num_layers (int): The number of layers in the network.
        hidden_channels (int): The number of hidden channels
            in the node embeddings.
        num_rbf (int): The number of radial basis functions.
        trainable_rbf (bool): Whether the radial basis function
            parameters are trainable.
        max_z (int): The maximum atomic numbers.
        cutoff (float): The cutoff distance.
        max_num_neighbors (int): The maximum number of neighbors
            considered for each atom.
        vertex (bool): Whether to use vertex geometric features.
        atomref (paddle.Tensor, optional): A tensor of atom reference values,
            or None if not provided.
        reduce_op (str): The type of reduction operation to apply
            ("sum", "mean").
        mean (float, optional): The mean of the output distribution,
            or 0 if not provided.
        std (float, optional): The standard deviation
            of the output distribution, or 1 if not provided.
        derivative (bool): Whether to compute the derivative of the output
            with respect to the positions.
    """

    def __init__(
        self,
        lmax: int = 1,
        vecnorm_type: str = "none",
        trainable_vecnorm: bool = False,
        num_heads: int = 8,
        num_layers: int = 6,
        hidden_channels: int = 128,
        num_rbf: int = 32,
        trainable_rbf: bool = False,
        max_z: int = 100,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        atomref: Optional[Tensor] = None,
        reduce_op: str = "sum",
        mean: Optional[float] = None,
        std: Optional[float] = None,
        derivative: bool = False,
    ):
        super(ViSNet, self).__init__()
        self.representation_model = ViSNetBlock(
            lmax=lmax,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            num_rbf=num_rbf,
            trainable_rbf=trainable_rbf,
            max_z=max_z,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
        )
        self.output_model = EquivariantScalar(hidden_channels=hidden_channels)
        self.prior_model = Atomref(atomref=atomref, max_z=max_z)
        self.reduce_op = reduce_op
        self.derivative = derivative

        mean = paddle.to_tensor(0) if mean is None else paddle.to_tensor(mean)
        self.register_buffer("mean", mean)
        std = paddle.to_tensor(1) if std is None else paddle.to_tensor(std)
        self.register_buffer("std", std)

    def forward(self, graph: pgl.Graph) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Computes the energies or properties (forces)
        for a batch of molecules.

        Args:
            graph (pgl.Graph):
                - num_nodes,
                - node_feat <--> z, pos,

        Returns:
            y (paddle.Tensor): The energies or properties for each molecule.
            dy (paddle.Tensor, optional): The negative derivative of energies.
        """
        if self.derivative:
            graph.node_feat["pos"].stop_gradient = False

        x, v = self.representation_model(graph)
        x = self.output_model.pre_reduce(x, v)
        x = x * self.std

        if self.prior_model is not None:
            x = self.prior_model(x, z=graph.node_feat["z"])

        y = pgl.math.segment_pool(x, graph.graph_node_id, pool_type=self.reduce_op)
        y = y + self.mean

        if self.derivative:
            try:
                dy = paddle.grad(
                    [y],
                    [graph.node_feat["pos"]],
                    grad_outputs=[paddle.ones_like(y)],
                    create_graph=True,
                    retain_graph=True,
                )[0]
                return y, -dy
            except RuntimeError:
                print(
                    "Since the Op segment_pool_grad doesn't have any gradop. " + \
                    "Can't compute the derivative of the energies with respect to the positions."
                )
                print(
                    "The derivative of the energies with respect to the positions is None."
                )
                return y, None
        return y, None

if __name__ == '__main__':
    
    graph = pgl.Graph(
        num_nodes=5,
        edges=paddle.to_tensor([]),
        node_feat={
            "z": paddle.randint(0, 100, (5,)),
            "pos": paddle.rand((5, 3)),
        },
    )
        
    model = ViSNet(
        lmax=1,
        vecnorm_type="none",
        trainable_vecnorm=False,
        num_heads=8,
        num_layers=6,
        hidden_channels=128,
        num_rbf=32,
        trainable_rbf=False,
        max_z=100,
        cutoff=5.0,
        max_num_neighbors=32,
        atomref=None,
        reduce_op="sum",
        mean=None,
        std=None,
        derivative=False,
    )

    out, _ = model(graph)
    
    assert out.shape == [1, 1]