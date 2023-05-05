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

"""Modules and utilities for the structure module."""

import gc
import ml_collections
import numpy as np
import paddle
import paddle.nn as nn
import functools
from typing import Dict
from alphafold_paddle.common import residue_constants
from alphafold_paddle.model import all_atom
from alphafold_paddle.model import quat_affine
from alphafold_paddle.model import r3
from alphafold_paddle.model import utils
from paddle.fluid.framework import _dygraph_tracer

def squared_difference(x, y):
    return paddle.square(x - y)

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
        attn_logits -= 1e5 * (1. - mask_2d.unsqueeze(1))

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

        self.ipa_dropout = nn.Dropout(p=self.config.dropout)
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
        act = self.ipa_dropout(act)
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

        affine = affine.stop_rot_gradient()
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
        """tbd."""

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

        return ret

    def loss(self, value, batch):
        ret = {'loss': 0.}

        ret['metrics'] = {}
        # If requested, compute in-graph metrics.
        if self.config.compute_in_graph_metrics:
            atom14_pred_positions = value['final_atom14_positions']
            # Compute renaming and violations.
            value.update(compute_renamed_ground_truth(batch, paddle.to_tensor(atom14_pred_positions)))
            value['violations'] = find_structural_violations(
                batch, atom14_pred_positions, self.config)

            # Several violation metrics:
            violation_metrics = compute_violation_metrics(
                batch=batch,
                atom14_pred_positions=atom14_pred_positions,
                violations=value['violations'])
            ret['metrics'].update(violation_metrics)
        
        backbone_loss(ret, batch, value, self.config)

        if 'renamed_atom14_gt_positions' not in value:
            tmp_atom14_positions = value['final_atom14_positions']
            value.update(compute_renamed_ground_truth(batch, paddle.to_tensor(tmp_atom14_positions)))

        sc_loss = sidechain_loss(batch, value, self.config)

        ret['loss'] = ((1 - self.config.sidechain.weight_frac) * ret['loss'] + self.config.sidechain.weight_frac * sc_loss['loss'])
        ret['sidechain_fape'] = sc_loss['fape']

        supervised_chi_loss(ret, batch, value, self.config)
        
        # Finetune loss
        if self.config.structural_violation_loss_weight:
            if 'violations' not in value:
                value['violations'] = find_structural_violations(batch, value['final_atom14_positions'], self.config)
            structural_violation_loss(ret, batch, value, self.config)
        return ret

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

        if not self.training and self.global_config.low_memory is True:
        # if not self.training:
            pair_act_cpu = pair_act.cpu()
            del pair_act
            gc.collect()
        affine = generate_new_affine(seq_mask)

        outputs = []
        activations = {'act': single_act, 'affine': affine.to_tensor()}
        for _ in range(self.config.num_layer):
            activations, output = self.fold_iteration(
                activations, init_single_act, pair_act_cpu if not self.training and self.global_config.low_memory is True else pair_act,
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
    Shape (B, N).

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
            (B, N, 14, 3).

    Returns:
        Dictionary containing:
            alt_naming_is_better: Array with 1.0 where alternative swap is better.
            renamed_atom14_gt_positions: Array of optimal ground truth positions
                after renaming swaps are performed.
            renamed_atom14_gt_exists: Mask after renaming swap is performed.
    """
    atom14_gt_positions_pd = paddle.to_tensor(batch['atom14_gt_positions'])
    atom14_alt_gt_positions_pd = paddle.to_tensor(batch['atom14_alt_gt_positions'])
    atom14_atom_is_ambiguous_pd = paddle.to_tensor(batch['atom14_atom_is_ambiguous'])
    atom14_gt_exists_pd = paddle.to_tensor(batch['atom14_gt_exists'])
    atom14_atom_exists_pd = paddle.to_tensor(batch['atom14_atom_exists'])
    # (B, N)
    alt_naming_is_better = all_atom.find_optimal_renaming(
        atom14_gt_positions=atom14_gt_positions_pd,
        atom14_alt_gt_positions=atom14_alt_gt_positions_pd,
        atom14_atom_is_ambiguous=atom14_atom_is_ambiguous_pd,
        atom14_gt_exists=atom14_gt_exists_pd,
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=atom14_atom_exists_pd)

    renamed_atom14_gt_positions = (
        (1. - alt_naming_is_better[:, :, None, None])
        * atom14_gt_positions_pd
        + alt_naming_is_better[:, :, None, None]
        * atom14_alt_gt_positions_pd
    )

    tmp_atom14_alt_gt_exists = paddle.to_tensor(batch['atom14_alt_gt_exists'])

    renamed_atom14_gt_mask = (
        (1. - alt_naming_is_better[:, :, None]) * atom14_gt_exists_pd
        + alt_naming_is_better[:, :, None] * tmp_atom14_alt_gt_exists)

    return {
        'alt_naming_is_better': alt_naming_is_better,  # (B, N)
        'renamed_atom14_gt_positions': renamed_atom14_gt_positions,  # (B, N, 14, 3)
        'renamed_atom14_gt_exists': renamed_atom14_gt_mask,  # (B, N, 14)
    }
    

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
    affine_trajectory = quat_affine.QuatAffine.from_tensor(value['traj'])
    rigid_trajectory = r3.rigids_from_quataffine(affine_trajectory)

    gt_rot = paddle.to_tensor(batch['backbone_affine_tensor_rot'], dtype='float32')
    gt_trans = paddle.to_tensor(batch['backbone_affine_tensor_trans'], dtype='float32')
    gt_affine = quat_affine.QuatAffine(
        quaternion=None,
        translation=gt_trans,
        rotation=gt_rot)
    gt_rigid = r3.rigids_from_quataffine(gt_affine)
    backbone_mask = batch['backbone_affine_mask']
    backbone_mask = paddle.to_tensor(backbone_mask) 

    fape_loss_fn = functools.partial(
        all_atom.frame_aligned_point_error,
        l1_clamp_distance=config.fape.clamp_distance,
        length_scale=config.fape.loss_unit_distance)
    
    fape_loss = []
    index = 0
    for rigid_trajectory_rot_item,rigid_trajectory_trans_item in zip(rigid_trajectory.rot,rigid_trajectory.trans):
        rigid_trajectory_item = r3.Rigids(rigid_trajectory_rot_item, rigid_trajectory_trans_item)
        index+=1
        middle_fape_loss = fape_loss_fn(rigid_trajectory_item, gt_rigid, backbone_mask,
                           rigid_trajectory_trans_item, gt_rigid.trans,
                           backbone_mask)
        fape_loss.append(middle_fape_loss)
    fape_loss = paddle.stack(fape_loss)

    if 'use_clamped_fape' in batch:
        # Jumper et al. (2021) Suppl. Sec. 1.11.5 "Loss clamping details"
        use_clamped_fape = batch['use_clamped_fape'][0, 0]

        unclamped_fape_loss_fn = functools.partial(
            all_atom.frame_aligned_point_error,
            l1_clamp_distance=None,
            length_scale=config.fape.loss_unit_distance)

        fape_loss_unclamped = []
        index_t = 0
        for rigid_trajectory_rot_item_t, rigid_trajectory_trans_item_t in zip(rigid_trajectory.rot, rigid_trajectory.trans):
            rigid_trajectory_item_t = r3.Rigids(rigid_trajectory_rot_item_t, rigid_trajectory_trans_item_t)
            index_t+=1
            middle_fape_loss_t = unclamped_fape_loss_fn(rigid_trajectory_item_t, gt_rigid, backbone_mask,
                            rigid_trajectory_trans_item_t, gt_rigid.trans,
                            backbone_mask)
            fape_loss_unclamped.append(middle_fape_loss_t)
        fape_loss_unclamped = paddle.stack(fape_loss_unclamped)

        fape_loss = (fape_loss * use_clamped_fape + fape_loss_unclamped * (1 - use_clamped_fape))
    
    ret['fape'] = fape_loss[-1]
    ret['backbone_fape'] = paddle.mean(fape_loss)
    ret['loss'] += paddle.mean(fape_loss)


def sidechain_loss(batch, value, config):
    """All Atom FAPE Loss using renamed rigids."""
    # Rename Frames
    # Jumper et al. (2021) Suppl. Alg. 26 "renameSymmetricGroundTruthAtoms" line 7
    alt_naming_is_better = value['alt_naming_is_better']

    renamed_gt_frames = (
        (1. - alt_naming_is_better[:, :, None, None])
        * batch['rigidgroups_gt_frames']
        + alt_naming_is_better[:, :, None, None]
        * batch['rigidgroups_alt_gt_frames'])

    batch_size = renamed_gt_frames.shape[0]
    flat_gt_frames = r3.rigids_from_tensor_flat12(
            paddle.reshape(renamed_gt_frames, [batch_size, -1, 12]))
    flat_frames_mask = paddle.reshape(batch['rigidgroups_gt_exists'], [batch_size, -1])

    flat_gt_positions = r3.vecs_from_tensor(
            paddle.reshape(value['renamed_atom14_gt_positions'], [batch_size, -1, 3]))
    flat_positions_mask = paddle.reshape(value['renamed_atom14_gt_exists'], [batch_size, -1])

    # Compute frame_aligned_point_error score for the final layer.
    pred_frames_rot = value['sidechains']['frames_rot']
    pred_frames_trans = value['sidechains']['frames_trans']
    tmp_rots = paddle.reshape(pred_frames_rot[-1], [batch_size, -1, 3, 3])
    tmp_vecs = paddle.reshape(pred_frames_trans[-1], [batch_size, -1, 3])
    tmp_rots = r3.rots_from_tensor3x3(tmp_rots)
    tmp_vecs = r3.vecs_from_tensor(tmp_vecs) 
    flat_pred_frames = r3.Rigids(rot=tmp_rots, trans=tmp_vecs)

    pred_positions = value['sidechains']['atom_pos']
    pred_positions = paddle.reshape(pred_positions[-1], [batch_size, -1, 3])
    flat_pred_positions = r3.vecs_from_tensor(pred_positions)

    # FAPE Loss on sidechains
    fape = all_atom.frame_aligned_point_error(
        pred_frames=flat_pred_frames,
        target_frames=flat_gt_frames,
        frames_mask=flat_frames_mask,
        pred_positions=flat_pred_positions,
        target_positions=flat_gt_positions,
        positions_mask=flat_positions_mask,
        l1_clamp_distance=config.sidechain.atom_clamp_distance,
        length_scale=config.sidechain.length_scale)

    return {
      'fape': fape,
      'loss': fape}


def structural_violation_loss(ret, batch, value, config):
    """Computes loss for structural violations."""
    assert config.sidechain.weight_frac
    
    # Put all violation losses together to one large loss.
    violations = value['violations']
    num_atoms = paddle.sum(batch['atom14_atom_exists'], dtype='float32')
    ret['loss'] += (config.structural_violation_loss_weight * (
        violations['between_residues']['bonds_c_n_loss_mean'] +
        violations['between_residues']['angles_ca_c_n_loss_mean'] +
        violations['between_residues']['angles_c_n_ca_loss_mean'] +
        paddle.sum(
            violations['between_residues']['clashes_per_atom_loss_sum'] +
            violations['within_residues']['per_atom_loss_sum']) /
        (1e-6 + num_atoms)))


def find_structural_violations(
        batch: Dict[str, paddle.Tensor],
        atom14_pred_positions: paddle.Tensor,  # (B, N, 14, 3)
        config: ml_collections.ConfigDict):
    """Computes several checks for structural violations."""

    # Compute between residue backbone violations of bonds and angles.
    connection_violations = all_atom.between_residue_bond_loss(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=batch['atom14_atom_exists'],
        residue_index=paddle.cast(batch['residue_index'], 'float32'),
        aatype=batch['aatype_index'],
        tolerance_factor_soft=config.violation_tolerance_factor,
        tolerance_factor_hard=config.violation_tolerance_factor)

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (B, N, 14).
    temp_atomtype_radius = np.array([
        residue_constants.van_der_waals_radius[name[0]]
        for name in residue_constants.atom_types
    ], dtype=np.float32)
    atomtype_radius = paddle.to_tensor(temp_atomtype_radius)
    atom14_atom_radius = batch['atom14_atom_exists'] * utils.batched_gather(
        atomtype_radius, batch['residx_atom14_to_atom37'])

    # Compute the between residue clash loss.
    between_residue_clashes = all_atom.between_residue_clash_loss(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=batch['atom14_atom_exists'],
        atom14_atom_radius=atom14_atom_radius,
        residue_index=paddle.cast(batch['residue_index'], 'float32'),
        overlap_tolerance_soft=config.clash_overlap_tolerance,
        overlap_tolerance_hard=config.clash_overlap_tolerance)

    # Compute all within-residue violations (clashes,
    # bond length and angle violations).
    restype_atom14_bounds = residue_constants.make_atom14_dists_bounds(
        overlap_tolerance=config.clash_overlap_tolerance,
        bond_length_tolerance_factor=config.violation_tolerance_factor)
    atom14_dists_lower_bound = utils.batched_gather(
        paddle.to_tensor(restype_atom14_bounds['lower_bound']), batch['aatype_index'])
    atom14_dists_upper_bound = utils.batched_gather(
        paddle.to_tensor(restype_atom14_bounds['upper_bound']), batch['aatype_index'])
    within_residue_violations = all_atom.within_residue_violations(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=batch['atom14_atom_exists'],
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0)
    
    # Combine them to a single per-residue violation mask (used later for LDDT).
    per_residue_violations_mask = paddle.max(paddle.stack([
        connection_violations['per_residue_violation_mask'],
        paddle.max(between_residue_clashes['per_atom_clash_mask'], axis=-1),
        paddle.max(within_residue_violations['per_atom_violations'], axis=-1)]), axis=0)

    return {
        'between_residues': {
            'bonds_c_n_loss_mean':
                connection_violations['c_n_loss_mean'],  # (B)
            'angles_ca_c_n_loss_mean':
                connection_violations['ca_c_n_loss_mean'],  # (B)
            'angles_c_n_ca_loss_mean':
                connection_violations['c_n_ca_loss_mean'],  # (B)
            'connections_per_residue_loss_sum':
                connection_violations['per_residue_loss_sum'],  # (B, N)
            'connections_per_residue_violation_mask':
                connection_violations['per_residue_violation_mask'],  # (B, N)
            'clashes_mean_loss':
                between_residue_clashes['mean_loss'],  # (B)
            'clashes_per_atom_loss_sum':
                between_residue_clashes['per_atom_loss_sum'],  # (B, N, 14)
            'clashes_per_atom_clash_mask':
                between_residue_clashes['per_atom_clash_mask'],  # (B, N, 14)
        },
        'within_residues': {
            'per_atom_loss_sum':
                within_residue_violations['per_atom_loss_sum'],  # (B, N, 14)
            'per_atom_violations':
                within_residue_violations['per_atom_violations'],  # (B, N, 14),
        },
        'total_per_residue_violations_mask':
            per_residue_violations_mask,  # (B, N)
    }


def compute_violation_metrics(
    batch: Dict[str, paddle.Tensor],
    atom14_pred_positions: paddle.Tensor,  # (B, N, 14, 3)
    violations: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
    """Compute several metrics to assess the structural violations."""
    batch_size = atom14_pred_positions.shape[0]
    ret = {}
    extreme_ca_ca_violations = all_atom.extreme_ca_ca_distance_violations(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=paddle.cast(batch['atom14_atom_exists'], 'float32'),
        residue_index=paddle.cast(batch['residue_index'], 'float32'))
    ret['violations_extreme_ca_ca_distance'] = extreme_ca_ca_violations

    violations_between_residue_bond_tmp = []
    for i in range(batch_size):
        violations_between_residue_bond_i = utils.mask_mean(mask=batch['seq_mask'][i], 
                    value=violations['between_residues']['connections_per_residue_violation_mask'][i])
        violations_between_residue_bond_tmp.append(violations_between_residue_bond_i)
    violations_between_residue_bond = paddle.to_tensor(violations_between_residue_bond_tmp, 
                    stop_gradient=False)
    violations_between_residue_bond = paddle.squeeze(violations_between_residue_bond, axis=-1)
    ret['violations_between_residue_bond'] = violations_between_residue_bond

    violations_between_residue_clash_tmp = []
    for i in range(batch_size):
        violations_between_residue_clash_i = utils.mask_mean(mask=batch['seq_mask'][i], 
                    value=paddle.max(violations['between_residues']['clashes_per_atom_clash_mask'],
                    axis=-1)[i])
        violations_between_residue_clash_tmp.append(violations_between_residue_clash_i)
    violations_between_residue_clash = paddle.to_tensor(violations_between_residue_clash_tmp, 
                    stop_gradient=False)
    violations_between_residue_clash = paddle.squeeze(violations_between_residue_clash, axis=-1)
    ret['violations_between_residue_clash'] = violations_between_residue_clash

    violations_within_residue_tmp = []
    for i in range(batch_size):
        violations_within_residue_i = utils.mask_mean(mask=batch['seq_mask'][i], 
                    value=paddle.max(violations['within_residues']['per_atom_violations'], axis=-1)[i])
        violations_within_residue_tmp.append(violations_within_residue_i)
    violations_within_residue = paddle.to_tensor(violations_within_residue_tmp, 
                    dtype='float32', stop_gradient=False)
    violations_within_residue = paddle.squeeze(violations_within_residue, axis=-1)
    ret['violations_within_residue'] = violations_within_residue

    violations_per_residue_tmp = []
    for i in range(batch_size):
        violations_per_residue_i = utils.mask_mean(mask=batch['seq_mask'][i], 
                    value=violations['total_per_residue_violations_mask'][i])
        violations_per_residue_tmp.append(violations_per_residue_i)
    violations_per_residue = paddle.to_tensor(violations_per_residue_tmp, 
                    dtype='float32', stop_gradient=False)
    violations_per_residue = paddle.squeeze(violations_per_residue, axis=-1)
    ret['violations_per_residue'] = violations_per_residue
    return ret


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
    
    sequence_mask = batch['seq_mask']
    num_res = sequence_mask.shape[1]
    batch_size = sequence_mask.shape[0]
    chi_mask = batch['chi_mask']
    pred_angles = paddle.reshape(value['sidechains']['angles_sin_cos'], [-1, batch_size, num_res, 7, 2])
    pred_angles = pred_angles.transpose([1, 0, 2, 3, 4])
    pred_angles = pred_angles[:, :, :, 3:]

    residue_type_one_hot = paddle.nn.functional.one_hot(batch['aatype_index'], 
                            num_classes=residue_constants.restype_num + 1)
    chi_pi_periodic = paddle.einsum('nijk,nkl->nijl', residue_type_one_hot[:, None, ...], 
                            paddle.to_tensor(residue_constants.chi_pi_periodic)[None])

    sin_cos_true_chi = batch['chi_angles_sin_cos'][:, None, ...]

    # This is -1 if chi is pi-periodic and +1 if it's 2pi-periodic
    shifted_mask = (1 - 2 * chi_pi_periodic)[..., None]
    sin_cos_true_chi_shifted = shifted_mask * sin_cos_true_chi

    sq_chi_error = paddle.sum(squared_difference(sin_cos_true_chi, pred_angles), axis=-1)
    sq_chi_error_shifted = paddle.sum(squared_difference(sin_cos_true_chi_shifted, pred_angles), axis=-1)
    sq_chi_error = paddle.minimum(sq_chi_error, sq_chi_error_shifted)

    sq_chi_loss_tmp = []
    for i in range(batch_size):
        sq_chi_loss_i = utils.mask_mean(mask=paddle.unsqueeze(chi_mask[i], axis=0), value=sq_chi_error[i])
        sq_chi_loss_tmp.append(sq_chi_loss_i)
    sq_chi_loss = paddle.to_tensor(sq_chi_loss_tmp, stop_gradient=False)
    sq_chi_loss = paddle.squeeze(sq_chi_loss, axis=-1)
    ret['chi_loss'] = sq_chi_loss
    ret['loss'] += config.chi_weight * sq_chi_loss

    unnormed_angles = paddle.reshape(value['sidechains']['unnormalized_angles_sin_cos'], [batch_size, -1, num_res, 7, 2])
    angle_norm = paddle.sqrt(paddle.sum(paddle.square(unnormed_angles), axis=-1) + eps)
    norm_error = paddle.abs(angle_norm - 1.)
    angle_norm_loss_tmp = []
    for i in range(batch_size):
        angle_norm_loss_i = utils.mask_mean(mask=paddle.unsqueeze(sequence_mask[i], axis=[0,2]), value=norm_error[i])
        angle_norm_loss_tmp.append(angle_norm_loss_i)
    angle_norm_loss = paddle.to_tensor(angle_norm_loss_tmp, stop_gradient=False)
    angle_norm_loss = paddle.squeeze(angle_norm_loss, axis=-1)
    ret['angle_norm_loss'] = angle_norm_loss
    ret['loss'] += config.angle_norm_weight * angle_norm_loss


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

        # Outputs1 (Rot + Trans)
        outputs.update({
            'atom_pos': pred_positions.translation,  # (B, N, 14, 3)
            'frames_rot': all_frames_to_global.rot.rotation,  # (B, N, 8, 3, 3)
            'frames_trans': all_frames_to_global.trans.translation,  # (B, N, 8, 3)
        })

        # ## Outputs2 (Rigids)
        # outputs.update({
        #     'atom_pos': pred_positions.translation,  # (B, N, 14, 3)
        #     'frames': all_frames_to_global,  # (B, N, 8, 3, 3)
        # })

        return outputs