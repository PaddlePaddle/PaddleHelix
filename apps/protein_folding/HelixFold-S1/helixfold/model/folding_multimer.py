#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

"""Modules and utilities for the multimer structure module."""


import numbers
import ml_collections
from typing import Union, Iterable, Tuple

import numpy as np
import paddle
import paddle.nn as nn

from helixfold.common import residue_constants
from helixfold.model import geometry
from helixfold.model import quat_affine


class PointProjection(nn.Layer):
    """Given input representation and frame produces points in global frame."""

    def __init__(self,
                 channel_num: int,
                 num_points: Union[Iterable[int], int],
                 global_config: ml_collections.ConfigDict,
                 return_local_points: bool = False):
        """Constructs Linear Module.

        Args:
            channel_num: number of channels.
            num_points: number of points to project. Can be tuple when
                outputting multiple dimensions.
            global_config: Global Config, passed through to underlying Linear.
            return_local_points: Whether to return points in local frame as well.
        """
        super(PointProjection, self).__init__()
        if isinstance(num_points, numbers.Integral):
            self.num_points = (num_points,)
        else:
            self.num_points = tuple(num_points)

        self.return_local_points = return_local_points
        self.global_config = global_config

        out_shape = self.num_points[:-1] + (3 * self.num_points[-1],)
        self.num_output_dims = len(out_shape)
        self.weight = paddle.create_parameter(
            [channel_num] + list(out_shape), 'float32',
            default_initializer=nn.initializer.XavierUniform())
        self.bias = paddle.create_parameter(
            list(out_shape), 'float32',
            default_initializer=nn.initializer.Constant(0.0))

    def forward(self, single_act, affine):
        out_letters = 'hijkl'[:self.num_output_dims]
        equation = f'...a,a{out_letters}->...{out_letters}'
        points_local = paddle.einsum(equation, single_act, self.weight)
        points_local += self.bias

        points_local = paddle.split(points_local, 3, axis=-1)

        nhead_nqk = np.prod(points_local[0].shape[-2:])
        bs_res_nhead_nqk = points_local[0].shape[:-2] + [nhead_nqk]

        points_local = [paddle.reshape(x, bs_res_nhead_nqk) for x in
                        points_local]
        points_local = geometry.Vec3Array(*points_local)
        points_global = affine[..., None].apply_to_point(points_local)

        if self.return_local_points:
            return points_global, points_local
        else:
            return points_global


class ScalarProjection(nn.Layer):
    def __init__(self,
                 channel_num: int,
                 num_points: Union[Iterable[int], int],
                 global_config: ml_collections.ConfigDict):
        super(ScalarProjection, self).__init__()
        if isinstance(num_points, numbers.Integral):
            self.num_points = (num_points,)
        else:
            self.num_points = tuple(num_points)

        self.global_config = global_config
        self.num_output_dims = len(self.num_points)
        self.weight = paddle.create_parameter(
            [channel_num] + list(self.num_points), 'float32',
            default_initializer=nn.initializer.XavierUniform())

    def forward(self, single_act):
        out_letters = 'hijkl'[:self.num_output_dims]
        equation = f'...a,a{out_letters}->...{out_letters}'
        act = paddle.einsum(equation, single_act, self.weight)
        return act


class InvariantPointAttention(nn.Layer):
    """Invariant Point attention module.

    The high-level idea is that this attention module works over a set of points
    and associated orientations in 3D space (e.g. protein residues).

    Each residue outputs a set of queries and keys as points in their local
    reference frame.  The attention is then defined as the euclidean distance
    between the queries and keys in the global frame.

    """
    def __init__(self, channel_num, config, global_config,
                 dist_epsilon=1e-8):
        super(InvariantPointAttention, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config
        self.dist_epsilon = dist_epsilon

        num_head = self.config.num_head
        num_point_qk = self.config.num_point_qk
        num_scalar_qk = self.config.num_scalar_qk
        num_point_v = self.config.num_point_v
        num_scalar_v = self.config.num_scalar_v
        num_output = self.config.num_channel

        tpw = np.log(np.exp(1.) - 1.)
        self.trainable_point_weights = paddle.create_parameter(
            [num_head], 'float32',
            default_initializer=nn.initializer.Constant(tpw))

        self.q_point_projection = PointProjection(
            channel_num['seq_channel'], [num_head, num_point_qk],
            self.global_config)

        self.k_point_projection = PointProjection(
            channel_num['seq_channel'], [num_head, num_point_qk],
            self.global_config)

        self.q_scalar_projection = ScalarProjection(
            channel_num['seq_channel'], [num_head, num_scalar_qk],
            self.global_config)

        self.k_scalar_projection = ScalarProjection(
            channel_num['seq_channel'], [num_head, num_scalar_qk],
            self.global_config)

        self.attention_2d = nn.Linear(channel_num['pair_channel'], num_head)

        self.v_scalar_projection = ScalarProjection(
            channel_num['seq_channel'], [num_head, num_scalar_v],
            self.global_config)

        self.v_point_projection = PointProjection(
            channel_num['seq_channel'], [num_head, num_point_v],
            self.global_config)

        if self.global_config.zero_init:
            init_w = nn.initializer.Constant(value=0.0)
        else:
            init_w = nn.initializer.XavierUniform()

        c = num_scalar_v + num_point_v * 4 + channel_num['pair_channel']
        self.output_projection = nn.Linear(
            num_head * c, num_output,
            weight_attr=paddle.ParamAttr(initializer=init_w))

    def forward(self, single_act: paddle.Tensor, pair_act: paddle.Tensor,
                mask: paddle.Tensor, affine: quat_affine.QuatAffine):
        # single_act: [B, N, C]
        # pair_act: [B, N, M, C']
        # mask: [B, N, 1]
        num_residues = single_act.shape[1]
        num_head = self.config.num_head
        num_scalar_qk = self.config.num_scalar_qk
        num_point_qk = self.config.num_point_qk
        num_scalar_v = self.config.num_scalar_v
        num_point_v = self.config.num_point_v
        num_output = self.config.num_channel

        attn_logits = 0.

        # We assume that all queries and keys come iid from N(0, 1) distribution
        # and compute the variances of the attention logits.
        # Each scalar pair (q, k) contributes Var q*k = 1
        scalar_variance = max(num_scalar_qk, 1) * 1.
        scalar_weights = np.sqrt(1.0 / scalar_variance)

        # Each point pair (q, k) contributes Var [0.5 ||q||^2 - <q, k>] = 9 / 2
        point_variance = max(num_point_qk, 1) * 9. / 2
        point_weights = np.sqrt(1.0 / point_variance)

        trainable_point_weights = nn.functional.softplus(
            self.trainable_point_weights)
        point_weights *= paddle.unsqueeze(trainable_point_weights,
                                          axis=1)

        q_point = self.q_point_projection(single_act, affine)
        k_point = self.k_point_projection(single_act, affine)

        q_point = [q_point.x, q_point.y, q_point.z]
        k_point = [k_point.x, k_point.y, k_point.z]

        q_point = [
            paddle.reshape(x, [-1, num_residues, num_head, num_point_qk])
            for x in q_point]
        k_point = [
            paddle.reshape(x, [-1, num_residues, num_head, num_point_qk])
            for x in k_point]

        # [B, R, H, C] => [B, H, R, C], put head dim first
        q_point = [paddle.transpose(x, [0, 2, 1, 3]) for x in q_point]
        k_point = [paddle.transpose(x, [0, 2, 1, 3]) for x in k_point]

        dist2 = [
            paddle.square(paddle.unsqueeze(qx, axis=-2) - \
                          paddle.unsqueeze(kx, axis=-3))
            for qx, kx in zip(q_point, k_point)]
        dist2 = sum(dist2)

        attn_qk_point = -0.5 * paddle.sum(
            paddle.unsqueeze(point_weights, axis=[1, 2]) * dist2, axis=-1)
        attn_logits += attn_qk_point

        q_scalar = self.q_scalar_projection(single_act)
        k_scalar = self.k_scalar_projection(single_act)
        q = paddle.transpose(scalar_weights * q_scalar, [0, 2, 1, 3])
        k = paddle.transpose(k_scalar, [0, 2, 1, 3])
        attn_qk_scalar = paddle.matmul(q, paddle.transpose(k, [0, 1, 3, 2]))
        attn_logits += attn_qk_scalar

        attention_2d = self.attention_2d(pair_act)
        attention_2d = paddle.transpose(attention_2d, [0, 3, 1, 2])
        attn_logits += attention_2d

        mask_2d = mask * paddle.transpose(mask, [0, 2, 1])
        attn_logits -= 1e5 * (1. - mask_2d.unsqueeze(1))
        attn_logits *= np.sqrt(1. / 3)

        # [batch_size, num_head, num_query_residues, num_target_residues]
        attn = nn.functional.softmax(attn_logits)

        v_scalar = self.v_scalar_projection(single_act)
        v = paddle.transpose(v_scalar, [0, 2, 1, 3])

        # o_i^h
        # [batch_size, num_query_residues, num_head, num_head * num_scalar_v]
        result_scalar = paddle.matmul(attn, v)
        result_scalar = paddle.transpose(result_scalar, [0, 2, 1, 3])

        v_point = self.v_point_projection(single_act, affine)
        v_point = [v_point.x, v_point.y, v_point.z]
        v_point = [
            paddle.reshape(x, [-1, num_residues, num_head, num_point_v])
            for x in v_point]
        v_point = [paddle.transpose(x, [0, 2, 1, 3]) for x in v_point]

        # o_i^{hp}
        # [batch_size, num_query_residues, num_head, num_head * num_point_v]
        result_point_global = [
            paddle.sum(paddle.unsqueeze(attn, -1) * paddle.unsqueeze(vx, -3),
                       axis=-2) for vx in v_point]
        result_point_global = [
            paddle.transpose(x, [0, 2, 1, 3]) for x in result_point_global]

        # \tilde{o}_i^h
        # [batch_size, num_residues, num_head, pair_channel]
        result_attention_over_2d = paddle.einsum(
            'nhij,nijc->nihc', attn, pair_act)

        # Reshape, global-to-local and save
        result_scalar = paddle.reshape(
            result_scalar, [-1, num_residues, num_head * num_scalar_v])
        result_point_global = [
            paddle.reshape(x, [-1, num_residues, num_head * num_point_v])
            for x in result_point_global]
        result_point_global = geometry.Vec3Array(*result_point_global)
        result_point_local = affine[..., None].apply_inverse_to_point(
            result_point_global)

        result_point_local_norm = result_point_local.norm(self.dist_epsilon)
        result_attention_over_2d = paddle.reshape(
            result_attention_over_2d,
            [-1, num_residues, num_head * self.channel_num['pair_channel']])

        output_features = [result_scalar]
        output_features.extend(
            [result_point_local.x, result_point_local.y, result_point_local.z])
        output_features.extend(
            [result_point_local_norm, result_attention_over_2d])

        final_act = paddle.concat(output_features, axis=-1)
        return self.output_projection(final_act)


def make_transform_from_reference(n_xyz, ca_xyz, c_xyz) -> geometry.Rigid3Array:
    """Returns rotation and translation matrices to convert
    from reference."""
    rotation = geometry.Rot3Array.from_two_vectors(
        c_xyz - ca_xyz, n_xyz - ca_xyz)
    return geometry.Rigid3Array(rotation, ca_xyz)


def make_backbone_affine(
        positions: paddle.Tensor,
        mask: paddle.Tensor) -> Tuple[geometry.Rigid3Array, paddle.Tensor]:
    """Make backbone Rigid3Array and mask."""
    n, ca, c = [residue_constants.atom_order[a] for a in ('N', 'CA', 'C')]

    rigid_mask = mask[..., n] * mask[..., ca] * mask[..., c]

    n_xyz = geometry.Vec3Array.from_array(positions[..., n, :])
    ca_xyz = geometry.Vec3Array.from_array(positions[..., ca, :])
    c_xyz = geometry.Vec3Array.from_array(positions[..., c, :])
    rigid = make_transform_from_reference(n_xyz, ca_xyz, c_xyz)

    return rigid, rigid_mask
