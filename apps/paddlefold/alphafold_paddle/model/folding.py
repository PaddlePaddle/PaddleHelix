#   Copyright (c) 2021 PaddlePaddle Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Modules and utilities for the structure module."""

import paddle
import paddle.nn as nn

import functools
from typing import Dict
from alphafold_paddle.common import residue_constants
from alphafold_paddle.model import all_atom
from alphafold_paddle.model import quat_affine
from alphafold_paddle.model import r3

import ml_collections
import numpy as np


class InvariantPointAttention(nn.Layer):
    """Invariant Point attention module.

    The high-level idea is that this attention module works over a set of points
    and associated orientations in 3D space (e.g. protein residues).

    Each residue outputs a set of queries and keys as points in their local
    reference frame.  The attention is then defined as the euclidean distance
    between the queries and keys in the global frame.

    Jumper et al. (2021) Suppl. Alg. 22 "InvariantPointAttention"
    """
    def __init__(self, channel_num, config, global_config,
                 dist_epsilon=1e-8):
        super(InvariantPointAttention, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config
        self.dist_epsilon = dist_epsilon

        num_head = self.config.num_head
        num_scalar_qk = self.config.num_scalar_qk
        num_point_qk = self.config.num_point_qk
        num_scalar_v = self.config.num_scalar_v
        num_point_v = self.config.num_point_v
        num_output = self.config.num_channel

        assert num_scalar_qk > 0
        assert num_point_qk > 0
        assert num_point_v > 0

        self.q_scalar = nn.Linear(
            channel_num['seq_channel'], num_head * num_scalar_qk)
        self.kv_scalar = nn.Linear(
            channel_num['seq_channel'],
            num_head * (num_scalar_v + num_scalar_qk))

        self.q_point_local = nn.Linear(
            channel_num['seq_channel'], num_head * 3 * num_point_qk)
        self.kv_point_local = nn.Linear(
            channel_num['seq_channel'],
            num_head * 3 * (num_point_qk + num_point_v))

        tpw = np.log(np.exp(1.) - 1.)
        self.trainable_point_weights = paddle.create_parameter(
            [num_head], 'float32',
            default_initializer=nn.initializer.Constant(tpw))

        self.attention_2d = nn.Linear(channel_num['pair_channel'], num_head)

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

        # Construct scalar queries of shape:
        # [batch_size, num_query_residues, num_head, num_points]
        q_scalar = self.q_scalar(single_act)
        q_scalar = paddle.reshape(
            q_scalar, [-1, num_residues, num_head, num_scalar_qk])

        # Construct scalar keys/values of shape:
        # [batch_size, num_target_residues, num_head, num_points]
        kv_scalar = self.kv_scalar(single_act)
        kv_scalar = paddle.reshape(
            kv_scalar,
            [-1, num_residues, num_head, num_scalar_v + num_scalar_qk])
        k_scalar, v_scalar = paddle.split(
            kv_scalar, [num_scalar_qk, -1], axis=-1)

        # Construct query points of shape:
        # [batch_size, num_residues, num_head, num_point_qk]
        q_point_local = self.q_point_local(single_act)
        q_point_local = paddle.split(q_point_local, 3, axis=-1)

        q_point_global = affine.apply_to_point(q_point_local, extra_dims=1)
        q_point = [
            paddle.reshape(x, [-1, num_residues, num_head, num_point_qk])
            for x in q_point_global]

        # Construct key and value points.
        # Key points shape [batch_size, num_residues, num_head, num_point_qk]
        # Value points shape [batch_size, num_residues, num_head, num_point_v]
        kv_point_local = self.kv_point_local(single_act)
        kv_point_local = paddle.split(kv_point_local, 3, axis=-1)

        kv_point_global = affine.apply_to_point(kv_point_local, extra_dims=1)
        kv_point_global = [
            paddle.reshape(x, [-1, num_residues, num_head, num_point_qk + num_point_v])
            for x in kv_point_global]

        k_point, v_point = list(
            zip(*[
                paddle.split(x, [num_point_qk, -1], axis=-1)
                for x in kv_point_global
            ]))

        # We assume that all queries and keys come iid from N(0, 1) distribution
        # and compute the variances of the attention logits.
        # Each scalar pair (q, k) contributes Var q*k = 1
        scalar_variance = max(num_scalar_qk, 1) * 1.
        # Each point pair (q, k) contributes Var [0.5 ||q||^2 - <q, k>] = 9 / 2
        point_variance = max(num_point_qk, 1) * 9. / 2

        # Allocate equal variance to scalar, point and attention 2d parts so that
        # the sum is 1.

        num_logit_terms = 3
        scalar_weights = np.sqrt(1.0 / (num_logit_terms * scalar_variance))
        point_weights = np.sqrt(1.0 / (num_logit_terms * point_variance))
        attention_2d_weights = np.sqrt(1.0 / (num_logit_terms))

        trainable_point_weights = nn.functional.softplus(
            self.trainable_point_weights)
        point_weights *= paddle.unsqueeze(
            trainable_point_weights, axis=1)

        # [B, R, H, C] => [B, H, R, C], put head dim first
        q_point = [paddle.transpose(x, [0, 2, 1, 3]) for x in q_point]
        k_point = [paddle.transpose(x, [0, 2, 1, 3]) for x in k_point]
        v_point = [paddle.transpose(x, [0, 2, 1, 3]) for x in v_point]

        dist2 = [
            paddle.square(paddle.unsqueeze(qx, axis=-2) - \
                          paddle.unsqueeze(kx, axis=-3))
            for qx, kx in zip(q_point, k_point)]
        dist2 = sum(dist2)

        attn_qk_point = -0.5 * paddle.sum(
            paddle.unsqueeze(point_weights, axis=[1, 2]) * dist2, axis=-1)

        q = paddle.transpose(scalar_weights * q_scalar, [0, 2, 1, 3])
        k = paddle.transpose(k_scalar, [0, 2, 1, 3])
        v = paddle.transpose(v_scalar, [0, 2, 1, 3])
        attn_qk_scalar = paddle.matmul(q, paddle.transpose(k, [0, 1, 3, 2]))
        attn_logits = attn_qk_scalar + attn_qk_point

        attention_2d = self.attention_2d(pair_act)
        attention_2d = paddle.transpose(attention_2d, [0, 3, 1, 2])
        attention_2d = attention_2d_weights * attention_2d
        attn_logits += attention_2d

        mask_2d = mask * paddle.transpose(mask, [0, 2, 1])
        attn_logits -= 1e5 * (1. - mask_2d)

        # [batch_size, num_head, num_query_residues, num_target_residues]
        attn = nn.functional.softmax(attn_logits)

        # o_i^h
        # [batch_size, num_query_residues, num_head, num_head * num_scalar_v]
        result_scalar = paddle.matmul(attn, v)
        result_scalar = paddle.transpose(result_scalar, [0, 2, 1, 3])

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
        result_point_local = affine.invert_point(
            result_point_global, extra_dims=1)
        result_attention_over_2d = paddle.reshape(
            result_attention_over_2d,
            [-1, num_residues, num_head * self.channel_num['pair_channel']])

        result_point_local_norm = paddle.sqrt(
            self.dist_epsilon + paddle.square(result_point_local[0]) + \
            paddle.square(result_point_local[1]) + \
            paddle.square(result_point_local[2]))

        output_features = [result_scalar]
        output_features.extend(result_point_local)
        output_features.extend(
            [result_point_local_norm, result_attention_over_2d])

        final_act = paddle.concat(output_features, axis=-1)
        return self.output_projection(final_act)


class FoldIteration(nn.Layer):
    """A single iteration of the main structure module loop.

    Jumper et al. (2021) Suppl. Alg. 20 "StructureModule" lines 6-21

    First, each residue attends to all residues using InvariantPointAttention.
    Then, we apply transition layers to update the hidden representations.
    Finally, we use the hidden representations to produce an update to the
    affine of each residue.
    """
    def __init__(self, channel_num, config, global_config):
        super(FoldIteration, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        self.invariant_point_attention = InvariantPointAttention(
            channel_num, config, global_config)
        self.attention_layer_norm = nn.LayerNorm(channel_num['seq_channel'])

        for i in range(self.config.num_layer_in_transition):
            if i < self.config.num_layer_in_transition - 1:
                init_w = nn.initializer.KaimingNormal()
            elif self.global_config.zero_init:
                init_w = nn.initializer.Constant(value=0.0)
            else:
                init_w = nn.initializer.XavierUniform()

            layer_name, c_in = 'transition', channel_num['seq_channel']
            if i > 0:
                layer_name, c_in = f'transition_{i}', self.config.num_channel

            setattr(self, layer_name, nn.Linear(
                c_in, self.config.num_channel,
                weight_attr=paddle.ParamAttr(initializer=init_w)))

        self.transition_dropout = nn.Dropout(p=self.config.dropout)
        self.transition_layer_norm = nn.LayerNorm(self.config.num_channel)

        if self.global_config.zero_init:
            last_init_w = nn.initializer.Constant(value=0.0)
        else:
            last_init_w = nn.initializer.XavierUniform()

        # Jumper et al. (2021) Alg. 23 "Backbone update"
        self.affine_update = nn.Linear(
            self.config.num_channel, 6,
            weight_attr=paddle.ParamAttr(initializer=last_init_w))

        self.rigid_sidechain = MultiRigidSidechain(
            channel_num, self.config.sidechain, self.global_config)

    def forward(self, activations, init_single_act, static_pair_act,
                seq_mask, aatype):
        affine = quat_affine.QuatAffine.from_tensor(activations['affine'])
        act = activations['act']

        attn = self.invariant_point_attention(
            act, static_pair_act, seq_mask, affine)
        act += attn
        act = self.attention_layer_norm(act)

        input_act = act
        for i in range(self.config.num_layer_in_transition):
            layer_name = 'transition'
            if i > 0:
                layer_name = f'transition_{i}'

            act = getattr(self, layer_name)(act)

            if i < self.config.num_layer_in_transition - 1:
                act = nn.functional.relu(act)

        act += input_act
        act = self.transition_dropout(act)
        act = self.transition_layer_norm(act)

        affine_update = self.affine_update(act)
        affine = affine.pre_compose(affine_update)

        sc = self.rigid_sidechain(
            affine.scale_translation(self.config.position_scale),
            act, init_single_act, aatype)
        outputs = {'affine': affine.to_tensor(), 'sc': sc}

        affine.stop_rot_gradient()
        new_activations = {
            'act': act,
            'affine': affine.to_tensor()
        }
        return new_activations, outputs


class StructureModule(nn.Layer):
    """StructureModule as a network head.

    Jumper et al. (2021) Suppl. Alg. 20 "StructureModule"
    """
    def __init__(self, channel_num, config, global_config):
        super(StructureModule, self).__init__()
        assert config.num_layer > 0

        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        self.single_layer_norm = nn.LayerNorm(channel_num['seq_channel'])
        self.initial_projection = nn.Linear(
            channel_num['seq_channel'], config.num_channel)
        self.pair_layer_norm = nn.LayerNorm(channel_num['pair_channel'])

        self.fold_iteration = FoldIteration(
            channel_num, config, global_config)

    def forward(self, representations, batch):
        output = self._generate_affines(representations, batch)

        ret = dict()
        ret['representations'] = {'structure_module': output['act']}

        # NOTE: pred unit is nanometer, *position_scale to scale back to
        # angstroms to match unit of PDB files.
        # (L, B, N, 7), L = FoldIteration layers
        scale = paddle.to_tensor(
            [1.] * 4 + [self.config.position_scale] * 3, 'float32')
        ret['traj'] = output['affine'] * paddle.unsqueeze(
            scale, axis=[0, 1, 2])

        ret['sidechains'] = output['sc']

        # (B, N, 14, 3)
        atom14_pred_positions = output['sc']['atom_pos'][-1]
        ret['final_atom14_positions'] = atom14_pred_positions

        # (B, N, 14)
        ret['final_atom14_mask'] = batch['atom14_atom_exists']

        # (B, N, 37, 3)
        atom37_pred_positions = all_atom.atom14_to_atom37(
            atom14_pred_positions, batch)
        atom37_pred_positions *= paddle.unsqueeze(
            batch['atom37_atom_exists'], axis=-1)
        ret['final_atom_positions'] = atom37_pred_positions

        # (B, N, 37)
        ret['final_atom_mask'] = batch['atom37_atom_exists']

        # (B, N, 7)
        ret['final_affines'] = ret['traj'][-1]

        if self.training:
            return ret
        else:
            no_loss_features = ['final_atom_positions', 'final_atom_mask',
                                'representations']
            no_loss_ret = {k: ret[k] for k in no_loss_features}
            return no_loss_ret

    def _generate_affines(self, representations, batch):
        """Generate predicted affines for a single chain.

        Jumper et al. (2021) Suppl. Alg. 20 "StructureModule"

        This is the main part of the structure module - it iteratively applies
        folding to produce a set of predicted residue positions.

        Args:
            representations: Representations dictionary.
            batch: Batch dictionary.

        Returns:
            A dictionary containing residue affines and sidechain positions.
        """
        seq_mask = paddle.unsqueeze(batch['seq_mask'], axis=-1)

        single_act = self.single_layer_norm(representations['single'])

        init_single_act = single_act
        single_act = self.initial_projection(single_act)

        pair_act = self.pair_layer_norm(representations['pair'])

        affine = generate_new_affine(seq_mask)

        outputs = []
        activations = {'act': single_act, 'affine': affine.to_tensor()}
        for _ in range(self.config.num_layer):
            activations, output = self.fold_iteration(
                activations, init_single_act, pair_act,
                seq_mask, batch['aatype'])
            outputs.append(output)

        output = dict()
        for k in outputs[0].keys():
            if k == 'sc':
                output[k] = dict()
                for l in outputs[0][k].keys():
                    output[k][l] = paddle.stack([o[k][l] for o in outputs])
            else:
                output[k] = paddle.stack([o[k] for o in outputs])

        output['act'] = activations['act']
        return output


def compute_renamed_ground_truth(
        batch: Dict[str, paddle.Tensor],
        atom14_pred_positions: paddle.Tensor) -> Dict[str, paddle.Tensor]:
    """Find optimal renaming of ground truth based on the predicted positions.

    Jumper et al. (2021) Suppl. Alg. 26 "renameSymmetricGroundTruthAtoms"

    This renamed ground truth is then used for all losses,
    such that each loss moves the atoms in the same direction.
    Shape (N).

    Args:
        batch: Dictionary containing:
            * atom14_gt_positions: Ground truth positions.
            * atom14_alt_gt_positions: Ground truth positions with renaming swaps.
            * atom14_atom_is_ambiguous: 1.0 for atoms that are affected by
                renaming swaps.
            * atom14_gt_exists: Mask for which atoms exist in ground truth.
            * atom14_alt_gt_exists: Mask for which atoms exist in ground truth
                after renaming.
            * atom14_atom_exists: Mask for whether each atom is part of the given
                amino acid type.
        atom14_pred_positions: Array of atom positions in global frame with shape
            (N, 14, 3).

    Returns:
        Dictionary containing:
            alt_naming_is_better: Array with 1.0 where alternative swap is better.
            renamed_atom14_gt_positions: Array of optimal ground truth positions
                after renaming swaps are performed.
            renamed_atom14_gt_exists: Mask after renaming swap is performed.
    """
    pass


def backbone_loss(ret, batch, value, config):
    """Backbone FAPE Loss.

  Jumper et al. (2021) Suppl. Alg. 20 "StructureModule" line 17

  Args:
    ret: Dictionary to write outputs into, needs to contain 'loss'.
    batch: Batch, needs to contain 'backbone_affine_tensor',
      'backbone_affine_mask'.
    value: Dictionary containing structure module output, needs to contain
      'traj', a trajectory of rigids.
    config: Configuration of loss, should contain 'fape.clamp_distance' and
      'fape.loss_unit_distance'.
  """
    pass


def sidechain_loss(batch, value, config):
    """All Atom FAPE Loss using renamed rigids."""
    # Rename Frames
    # Jumper et al. (2021) Suppl. Alg. 26 "renameSymmetricGroundTruthAtoms" line 7
    pass


def structural_violation_loss(ret, batch, value, config):
    """Computes loss for structural violations."""
    assert config.sidechain.weight_frac
    pass


def find_structural_violations(
        batch: Dict[str, paddle.Tensor],
        atom14_pred_positions: paddle.Tensor,  # (N, 14, 3)
        config: ml_collections.ConfigDict):
    """Computes several checks for structural violations."""
    pass


def compute_violation_metrics(
    batch: Dict[str, paddle.Tensor],
    atom14_pred_positions: paddle.Tensor,  # (N, 14, 3)
    violations: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
    """Compute several metrics to assess the structural violations."""
    pass


def supervised_chi_loss(ret, batch, value, config):
    """Computes loss for direct chi angle supervision.

    Jumper et al. (2021) Suppl. Alg. 27 "torsionAngleLoss"

    Args:
        ret: Dictionary to write outputs into, needs to contain 'loss'.
        batch: Batch, needs to contain 'seq_mask', 'chi_mask', 'chi_angles'.
        value: Dictionary containing structure module output, needs to contain
            value['sidechains']['angles_sin_cos'] for angles and
            value['sidechains']['unnormalized_angles_sin_cos'] for unnormalized
            angles.
        config: Configuration of loss, should contain 'chi_weight' and
            'angle_norm_weight', 'angle_norm_weight' scales angle norm term,
            'chi_weight' scales torsion term.
    """
    eps = 1e-6
    pass


def generate_new_affine(sequence_mask):
    t_shape = sequence_mask.shape[:-1] # (batch, N_res, 1)
    assert len(t_shape) == 2
    t_shape.append(3) # (batch, N_res, 3)
    q_shape = sequence_mask.shape[:-1] + [1] # (batch, N_res, 1)
    quaternion = paddle.tile(
                    paddle.reshape(
                        paddle.to_tensor([1.0, 0.0, 0.0, 0.0]), [1, 1, 4]),
                    repeat_times=q_shape)
    translation = paddle.zeros(t_shape)
    return quat_affine.QuatAffine(quaternion, translation)


def l2_normalize(x, axis=-1, epsilon=1e-12):
    return x / paddle.sqrt(
        paddle.maximum(
            paddle.sum(paddle.square(x), axis=axis, keepdim=True),
            paddle.to_tensor([epsilon], dtype='float32')))


class MultiRigidSidechain(nn.Layer):
    """Class to make side chain atoms."""
    def __init__(self, channel_num, config, global_config):
        super(MultiRigidSidechain, self).__init__()

        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        c = self.config.num_channel
        self.input_projection = nn.Linear(channel_num['seq_channel'], c)
        self.input_projection_1 = nn.Linear(channel_num['seq_channel'], c)

        for i in range(self.config.num_residual_block):
            l1, l2 = 'resblock1', 'resblock2'
            if i > 0:
                l1, l2 = f'resblock1_{i}', f'resblock2_{i}'

            init_w_1 = nn.initializer.KaimingNormal()
            if self.global_config.zero_init:
                init_w_2 = nn.initializer.Constant(value=0.)
            else:
                init_w_2 = nn.initializer.XavierUniform()

            setattr(self, l1, nn.Linear(
                c, c, weight_attr=paddle.ParamAttr(initializer=init_w_1)))
            setattr(self, l2, nn.Linear(
                c, c, weight_attr=paddle.ParamAttr(initializer=init_w_2)))

        self.unnormalized_angles = nn.Linear(c, 14)

    def forward(self, affine, single_act, init_single_act, aatype):
        single_act = self.input_projection(nn.functional.relu(single_act))
        init_single_act = self.input_projection_1(
            nn.functional.relu(init_single_act))
        act = single_act + init_single_act

        for i in range(self.config.num_residual_block):
            l1, l2 = 'resblock1', 'resblock2'
            if i > 0:
                l1, l2 = f'resblock1_{i}', f'resblock2_{i}'

            old_act = act
            act = getattr(self, l1)(nn.functional.relu(act))
            act = getattr(self, l2)(nn.functional.relu(act))
            act += old_act

        # Map activations to torsion angles. Shape: (num_res, 14).
        num_res = act.shape[1]
        unnormalized_angles = self.unnormalized_angles(
            nn.functional.relu(act))
        unnormalized_angles = paddle.reshape(
            unnormalized_angles, [-1, num_res, 7, 2])
        angles = l2_normalize(unnormalized_angles, axis=-1)

        outputs = {
            'angles_sin_cos': angles,  #  (B, N, 7, 2)
            'unnormalized_angles_sin_cos':
                unnormalized_angles,   #  (B, N, 7, 2)
        }

        # Jumper et al. (2021) Suppl. Alg. 24 "computeAllAtomCoordinates"
        backbone_to_global = r3.rigids_from_quataffine(affine)
        all_frames_to_global = all_atom.torsion_angles_to_frames(
            aatype, backbone_to_global, angles)
        pred_positions = all_atom.frames_and_literature_positions_to_atom14_pos(
            aatype, all_frames_to_global)

        outputs.update({
            'atom_pos': pred_positions.translation,  # (B, N, 14, 3)
            'frames_rot': all_frames_to_global.rot.rotation,  # (B, N, 8, 3, 3)
            'frames_trans': all_frames_to_global.trans.translation,  # (B, N, 8, 3)
        })

        return outputs
