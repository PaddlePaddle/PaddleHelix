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

"""Ops for all atom representations.

Generally we employ two different representations for all atom coordinates,
one is atom37 where each heavy atom corresponds to a given position in a 37
dimensional array, This mapping is non amino acid specific, but each slot
corresponds to an atom of a given name, for example slot 12 always corresponds
to 'C delta 1', positions that are not present for a given amino acid are
zeroed out and denoted by a mask.
The other representation we employ is called atom14, this is a more dense way
of representing atoms with 14 slots. Here a given slot will correspond to a
different kind of atom depending on amino acid type, for example slot 5
corresponds to 'N delta 2' for Aspargine, but to 'C delta 1' for Isoleucine.
14 is chosen because it is the maximum number of heavy atoms for any standard
amino acid.
The order of slots can be found in 'residue_constants.residue_atoms'.
Internally the model uses the atom14 representation because it is
computationally more efficient.
The internal atom14 representation is turned into the atom37 at the output of
the network to facilitate easier conversion to existing protein datastructures.
"""

import numpy as np
from typing import Dict, Optional
import paddle
import paddle.nn as nn
from tools import residue_constants, r3
import tools.model_utils as utils


def squared_difference(x, y):
    return paddle.square(x - y)


def get_chi_atom_indices():
    """Returns atom indices needed to compute chi angles for all residue types.

    Returns:
        A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
        in the order specified in residue_constants.restypes + unknown residue type
        at the end. For chi angles which are not defined on the residue, the
        positions indices are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in residue_constants.restypes:
        residue_name = residue_constants.restype_1to3[residue_name]
        residue_chi_angles = residue_constants.chi_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append(
                [residue_constants.atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append([0, 0, 0, 0])  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    return paddle.to_tensor(chi_atom_indices)


def atom14_to_atom37(atom14_data: paddle.Tensor,  # (B, N, 14, ...)
                     batch: Dict[str, paddle.Tensor]
                     ) -> paddle.Tensor:  # (B, N, 37, ...)
    """Convert atom14 to atom37 representation."""
    assert len(atom14_data.shape) in [3, 4]
    assert 'residx_atom37_to_atom14' in batch
    assert 'atom37_atom_exists' in batch

    atom37_data = utils.batched_gather(atom14_data,
                                    batch['residx_atom37_to_atom14'],
                                    batch_dims=2)
    if len(atom14_data.shape) == 3:
        atom37_data *= batch['atom37_atom_exists']
    elif len(atom14_data.shape) == 4:
        atom37_data *= batch['atom37_atom_exists'][:, :, :, 
                                            None].astype(atom37_data.dtype)
    return atom37_data


def atom37_to_atom14(
    atom37_data: paddle.Tensor,  # (B, N, 37, ...)
    batch: Dict[str, paddle.Tensor]) -> paddle.Tensor:  # (B, N, 14, ...)
    """Convert atom14 to atom37 representation."""
    assert len(atom37_data.shape) in [3, 4]
    assert 'residx_atom14_to_atom37' in batch
    assert 'atom14_atom_exists' in batch

    atom14_data = utils.batched_gather(atom37_data,
                                    batch['residx_atom14_to_atom37'],
                                    batch_dims=2)
    if len(atom37_data.shape) == 3:
        atom14_data *= batch['atom14_atom_exists'].astype(atom14_data.dtype)
    elif len(atom37_data.shape) == 4:
        atom14_data *= batch['atom14_atom_exists'][:, :, :,
                                               None].astype(atom14_data.dtype)
    return atom14_data


def atom37_to_frames(
    aatype: paddle.Tensor,  # (B, N)
    all_atom_positions: paddle.Tensor,  # (B, N, 37, 3)
    all_atom_mask: paddle.Tensor,  # (B, N, 37)
) -> Dict[str, paddle.Tensor]:
    """Computes the frames for the up to 8 rigid groups for each residue.

    The rigid groups are defined by the possible torsions in a given amino acid.
    We group the atoms according to their dependence on the torsion angles into
    "rigid groups".  E.g., the position of atoms in the chi2-group depend on
    chi1 and chi2, but do not depend on chi3 or chi4.
    Jumper et al. (2021) Suppl. Table 2 and corresponding text.

    Args:
        aatype: Amino acid type, given as Tensor with integers.
        all_atom_positions: atom37 representation of all atom coordinates.
        all_atom_mask: atom37 representation of mask on all atom coordinates.
    Returns:
        Dictionary containing:
        * 'rigidgroups_gt_frames': 8 Frames corresponding to 'all_atom_positions'
            represented as flat 12 dimensional array.
        * 'rigidgroups_gt_exists': Mask denoting whether the atom positions for
            the given frame are available in the ground truth, e.g. if they were
            resolved in the experiment.
        * 'rigidgroups_group_exists': Mask denoting whether given group is in
            principle present for given amino acid type.
        * 'rigidgroups_group_is_ambiguous': Mask denoting whether frame is
            affected by naming ambiguity.
        * 'rigidgroups_alt_gt_frames': 8 Frames with alternative atom renaming
            corresponding to 'all_atom_positions' represented as flat
            12 dimensional array.
    """
    # 0: 'backbone group',
    # 1: 'pre-omega-group', (empty)
    # 2: 'phi-group', (currently empty, because it defines only hydrogens)
    # 3: 'psi-group',
    # 4,5,6,7: 'chi1,2,3,4-group'
    aatype_in_shape = aatype.shape

    # If there is a batch axis, just flatten it away, and reshape everything
    # back at the end of the function.
    aatype = paddle.reshape(aatype, [-1])
    all_atom_positions = paddle.reshape(all_atom_positions, [-1, 37, 3])
    all_atom_mask = paddle.reshape(all_atom_mask, [-1, 37])

    # Create an array with the atom names.
    # shape (num_restypes, num_rigidgroups, 3_atoms): (21, 8, 3)
    restype_rigidgroup_base_atom_names = np.full([21, 8, 3], '', dtype=object)

    # 0: backbone frame
    restype_rigidgroup_base_atom_names[:, 0, :] = ['C', 'CA', 'N']

    # 3: 'psi-group'
    restype_rigidgroup_base_atom_names[:, 3, :] = ['CA', 'C', 'O']

    # 4,5,6,7: 'chi1,2,3,4-group'
    for restype, restype_letter in enumerate(residue_constants.restypes):
        resname = residue_constants.restype_1to3[restype_letter]
        for chi_idx in range(4):
            if residue_constants.chi_angles_mask[restype][chi_idx]:
                atom_names = residue_constants.chi_angles_atoms[resname][chi_idx]
                restype_rigidgroup_base_atom_names[
                    restype, chi_idx + 4, :] = atom_names[1:]

    # Create mask for existing rigid groups.
    restype_rigidgroup_mask = np.zeros([21, 8], dtype=np.float32)
    restype_rigidgroup_mask[:, 0] = 1
    restype_rigidgroup_mask[:, 3] = 1
    restype_rigidgroup_mask[:20, 4:] = residue_constants.chi_angles_mask

    # Translate atom names into atom37 indices.
    lookuptable = residue_constants.atom_order.copy()
    lookuptable[''] = 0
    restype_rigidgroup_base_atom37_idx = np.vectorize(lambda x: lookuptable[x])(
        restype_rigidgroup_base_atom_names)
    restype_rigidgroup_base_atom37_idx = paddle.to_tensor(restype_rigidgroup_base_atom37_idx)

    # Compute the gather indices for all residues in the chain.
    # shape (B, N, 8, 3)
    residx_rigidgroup_base_atom37_idx = utils.batched_gather(
        restype_rigidgroup_base_atom37_idx, aatype)

    # Gather the base atom positions for each rigid group.
    base_atom_pos = utils.batched_gather(
        all_atom_positions,
        residx_rigidgroup_base_atom37_idx,
        batch_dims=1)

    # Compute the Rigids.
    gt_frames = r3.rigids_from_3_points_vecs(
        point_on_neg_x_axis=r3.vecs_from_tensor(base_atom_pos[:, :, 0, :]),
        origin=r3.vecs_from_tensor(base_atom_pos[:, :, 1, :]),
        point_on_xy_plane=r3.vecs_from_tensor(base_atom_pos[:, :, 2, :])
    )

    # Compute a mask whether the group exists.
    # (B, N, 8)
    restype_rigidgroup_mask = paddle.to_tensor(restype_rigidgroup_mask)
    group_exists = utils.batched_gather(restype_rigidgroup_mask, aatype)

    # Compute a mask whether ground truth exists for the group
    gt_atoms_exist = utils.batched_gather(  # shape (B, N, 8, 3)
        all_atom_mask.astype('float32'),
        residx_rigidgroup_base_atom37_idx,
        batch_dims=1)
    gt_exists = paddle.min(gt_atoms_exist, axis=-1) * group_exists  # (B, N, 8)

    # Adapt backbone frame to old convention (mirror x-axis and z-axis).
    rots = np.tile(np.eye(3, dtype=np.float32), [8, 1, 1])
    rots[0, 0, 0] = -1
    rots[0, 2, 2] = -1
    rots = paddle.to_tensor(rots, dtype='float32')
    gt_frames = r3.rigids_mul_rots(gt_frames, r3.rots_from_tensor3x3(rots))

    # The frames for ambiguous rigid groups are just rotated by 180 degree around
    # the x-axis. The ambiguous group is always the last chi-group.
    restype_rigidgroup_is_ambiguous = np.zeros([21, 8], dtype=np.float32)
    restype_rigidgroup_rots = np.tile(np.eye(3, dtype=np.float32), [21, 8, 1, 1])

    for resname, _ in residue_constants.residue_atom_renaming_swaps.items():
        restype = residue_constants.restype_order[
            residue_constants.restype_3to1[resname]]
        chi_idx = int(sum(residue_constants.chi_angles_mask[restype]) - 1)
        restype_rigidgroup_is_ambiguous[restype, chi_idx + 4] = 1
        restype_rigidgroup_rots[restype, chi_idx + 4, 1, 1] = -1
        restype_rigidgroup_rots[restype, chi_idx + 4, 2, 2] = -1

    # Gather the ambiguity information for each residue.
    restype_rigidgroup_is_ambiguous = paddle.to_tensor(restype_rigidgroup_is_ambiguous, dtype='float32')
    restype_rigidgroup_rots = paddle.to_tensor(restype_rigidgroup_rots, dtype='float32')
    residx_rigidgroup_is_ambiguous = utils.batched_gather(
        restype_rigidgroup_is_ambiguous, aatype)
    residx_rigidgroup_ambiguity_rot = utils.batched_gather(
        restype_rigidgroup_rots, aatype)

    # Create the alternative ground truth frames.
    alt_gt_frames = r3.rigids_mul_rots(
        gt_frames, r3.rots_from_tensor3x3(residx_rigidgroup_ambiguity_rot))

    gt_frames_flat12 = r3.rigids_to_tensor_flat12(gt_frames)
    alt_gt_frames_flat12 = r3.rigids_to_tensor_flat12(alt_gt_frames)

    # reshape back to original residue layout
    gt_frames_flat12 = paddle.reshape(gt_frames_flat12, aatype_in_shape + [8, 12])
    gt_exists = paddle.reshape(gt_exists, aatype_in_shape + [8])
    group_exists = paddle.reshape(group_exists, aatype_in_shape + [8])
    residx_rigidgroup_is_ambiguous = paddle.reshape(
            residx_rigidgroup_is_ambiguous, aatype_in_shape + [8])
    alt_gt_frames_flat12 = paddle.reshape(
            alt_gt_frames_flat12, aatype_in_shape + [8, 12])

    return {
        'rigidgroups_gt_frames': gt_frames_flat12,  # (B, N, 8, 12)
        'rigidgroups_gt_exists': gt_exists,  # (B, N, 8)
        'rigidgroups_group_exists': group_exists,  # (B, N, 8)
        'rigidgroups_group_is_ambiguous': residx_rigidgroup_is_ambiguous,  # (B, N, 8)
        'rigidgroups_alt_gt_frames': alt_gt_frames_flat12,  # (B, N, 8, 12)
    }


def atom37_to_torsion_angles(
    aatype: paddle.Tensor,  # (B, T, N)
    all_atom_pos: paddle.Tensor,  # (B, T, N, 37, 3)
    all_atom_mask: paddle.Tensor,  # (B, T, N, 37)
    placeholder_for_undefined=False,
) -> Dict[str, paddle.Tensor]:
    """Computes the 7 torsion angles (in sin, cos encoding) for each residue.

    The 7 torsion angles are in the order
    '[pre_omega, phi, psi, chi_1, chi_2, chi_3, chi_4]',
    here pre_omega denotes the omega torsion angle between the given amino acid
    and the previous amino acid.

    Args:
        aatype: Amino acid type, given as array with integers.
        all_atom_pos: atom37 representation of all atom coordinates.
        all_atom_mask: atom37 representation of mask on all atom coordinates.
        placeholder_for_undefined: flag denoting whether to set masked torsion
        angles to zero.
    Returns:
        Dict containing:
        * 'torsion_angles_sin_cos': Array with shape (B, N, 7, 2) where the final
            2 dimensions denote sin and cos respectively
        * 'alt_torsion_angles_sin_cos': same as 'torsion_angles_sin_cos', but
            with the angle shifted by pi for all chi angles affected by the naming
            ambiguities.
        * 'torsion_angles_mask': Mask for which chi angles are present.
    """

    # Map aatype > 20 to 'Unknown' (20).
    aatype = paddle.minimum(aatype.astype('int'), paddle.to_tensor([20]).astype('int'))
    
    num_batch, num_temp, num_res = aatype.shape

    # Compute the backbone angles.
    pad = paddle.zeros([num_batch, num_temp, 1, 37, 3])
    prev_all_atom_pos = paddle.concat([pad, all_atom_pos[..., :-1, :, :]], axis=-3)

    pad = paddle.zeros([num_batch, num_temp, 1, 37])
    prev_all_atom_mask = paddle.concat([pad, all_atom_mask[..., :-1, :]], axis=-2)

    # For each torsion angle collect the 4 atom positions that define this angle.
    # shape (B, T, N, atoms=4, xyz=3)
    pre_omega_atom_pos = paddle.concat(
        [prev_all_atom_pos[..., 1:3, :],  # prev CA, C
        all_atom_pos[..., 0:2, :]  # this N, CA
        ], axis=-2)

    phi_atom_pos = paddle.concat(
        [prev_all_atom_pos[..., 2:3, :],  # prev C
        all_atom_pos[..., 0:3, :]  # this N, CA, C
        ], axis=-2)

    psi_atom_pos = paddle.concat(
        [all_atom_pos[..., 0:3, :],  # this N, CA, C
        all_atom_pos[..., 4:5, :]  # this O
        ], axis=-2)

    # Collect the masks from these atoms.
    # Shape [batch, n_temp, num_res]
    pre_omega_mask = (
        paddle.prod(prev_all_atom_mask[..., 1:3], axis=-1)  # prev CA, C
        * paddle.prod(all_atom_mask[..., 0:2], axis=-1))  # this N, CA
    phi_mask = (
        prev_all_atom_mask[..., 2]  # prev C
        * paddle.prod(all_atom_mask[..., 0:3], axis=-1))  # this N, CA, C
    psi_mask = (
        paddle.prod(all_atom_mask[..., 0:3], axis=-1) *  # this N, CA, C
        all_atom_mask[..., 4])  # this O

    # Collect the atoms for the chi-angles.
    # Compute the table of chi angle indices. Shape: [restypes, chis=4, atoms=4].
    chi_atom_indices = get_chi_atom_indices()

    # Select atoms to compute chis. Shape: [batch, num_temp, num_res, chis=4, atoms=4].
    atom_indices = utils.batched_gather(
        params=chi_atom_indices, indices=aatype, axis=0, batch_dims=0)

    # Gather atom positions. Shape: [batch, num_temp, num_res, chis=4, atoms=4, xyz=3].
    chis_atom_pos = utils.batched_gather(
        params=all_atom_pos, indices=atom_indices, axis=0,
        batch_dims=3)

    # Copy the chi angle mask, add the UNKNOWN residue. Shape: [restypes, 4].
    chi_angles_mask = list(residue_constants.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = paddle.to_tensor(chi_angles_mask)

    # Compute the chi angle mask. I.e. which chis angles exist according to the
    # aatype. Shape [batch, num_temp, num_res, chis=4].
    chis_mask = utils.batched_gather(params=chi_angles_mask, indices=aatype,
                                axis=0, batch_dims=0)
    # Constrain the chis_mask to those chis, where the ground truth coordinates of
    # all defining four atoms are available.
    # Gather the chi angle atoms mask. Shape: [batch, num_temp, num_res, chis=4, atoms=4].
    chi_angle_atoms_mask = utils.batched_gather(
        params=all_atom_mask, indices=atom_indices, axis=0,
        batch_dims=3)
    # Check if all 4 chi angle atoms were set. Shape: [batch, num_temp, num_res, chis=4].
    chi_angle_atoms_mask = paddle.prod(chi_angle_atoms_mask, axis=[-1])
    chis_mask = chis_mask * chi_angle_atoms_mask

    # Stack all torsion angle atom positions.
    # Shape (B, T, N, torsions=7, atoms=4, xyz=3)
    torsions_atom_pos = paddle.concat(
        [pre_omega_atom_pos[:, :, :, None, :, :],
        phi_atom_pos[:, :, :, None, :, :],
        psi_atom_pos[:, :, :, None, :, :],
        chis_atom_pos
        ], axis=3)

    # Stack up masks for all torsion angles.
    # shape (B, T, N, torsions=7)
    torsion_angles_mask = paddle.concat(
        [pre_omega_mask[..., None],
        phi_mask[..., None],
        psi_mask[..., None],
        chis_mask
        ], axis=-1)

    # Create a frame from the first three atoms:
    # First atom: point on x-y-plane
    # Second atom: point on negative x-axis
    # Third atom: origin
    # r3.Rigids (B, T, N, torsions=7)
    torsion_frames = r3.rigids_from_3_points_vecs(
        point_on_neg_x_axis=r3.Vecs(torsions_atom_pos[..., 1, :]),
        origin=r3.Vecs(torsions_atom_pos[..., 2, :]),
        point_on_xy_plane=r3.Vecs(torsions_atom_pos[..., 0, :]))

    # Compute the position of the forth atom in this frame (y and z coordinate
    # define the chi angle)
    # r3.Vecs (B, T, N, torsions=7)
    forth_atom_rel_pos = r3.rigids_mul_vecs(
        r3.invert_rigids(torsion_frames),
        r3.vecs_from_tensor(torsions_atom_pos[..., 3, :]))

    # Normalize to have the sin and cos of the torsion angle.
    # paddle.Tensor (B, T, N, torsions=7, sincos=2)
    torsion_angles_sin_cos = paddle.stack(
        [forth_atom_rel_pos.z, forth_atom_rel_pos.y], axis=-1)
    torsion_angles_sin_cos /= paddle.sqrt(
        paddle.sum(paddle.square(torsion_angles_sin_cos), axis=-1, keepdim=True)
        + 1e-8)

    # Mirror psi, because we computed it from the Oxygen-atom.
    torsion_angles_sin_cos *= paddle.to_tensor(
        [1., 1., -1., 1., 1., 1., 1.])[None, None, None, :, None]

    # Create alternative angles for ambiguous atom names.
    chi_is_ambiguous = utils.batched_gather(
        paddle.to_tensor(residue_constants.chi_pi_periodic), aatype)
    # chi_is_ambiguous (B, T, N, torsions=4)
    mirror_torsion_angles = paddle.concat(
        [paddle.ones([num_batch, num_temp, num_res, 3]),
        1.0 - 2.0 * chi_is_ambiguous], axis=-1)
    # mirror_torsion_angles (B, T, N, torsions=7)
    alt_torsion_angles_sin_cos = (
        torsion_angles_sin_cos * mirror_torsion_angles[:, :, :, :, None])

    if placeholder_for_undefined:
        # Add placeholder torsions in place of undefined torsion angles
        # (e.g. N-terminus pre-omega)
        placeholder_torsions = paddle.stack([
            paddle.ones(torsion_angles_sin_cos.shape[:-1]),
            paddle.zeros(torsion_angles_sin_cos.shape[:-1])
        ], axis=-1)
        torsion_angles_sin_cos = torsion_angles_sin_cos * torsion_angles_mask[
            ..., None] + placeholder_torsions * (1 - torsion_angles_mask[..., None])
        alt_torsion_angles_sin_cos = alt_torsion_angles_sin_cos * torsion_angles_mask[
            ..., None] + placeholder_torsions * (1 - torsion_angles_mask[..., None])

    return {
        'torsion_angles_sin_cos': torsion_angles_sin_cos,  # (B, T, N, 7, 2)
        'alt_torsion_angles_sin_cos': alt_torsion_angles_sin_cos,  # (B, T, N, 7, 2)
        'torsion_angles_mask': torsion_angles_mask  # (B, T, N, 7)
    }


def torsion_angles_to_frames(
    aatype: paddle.Tensor,  # (B, N)
    backb_to_global: r3.Rigids,  # (B, N)
    torsion_angles_sin_cos: paddle.Tensor  # (B, N, 7, 2)
) -> r3.Rigids:  # (B, N, 8)
    """Compute rigid group frames from torsion angles.

    Jumper et al. (2021) Suppl. Alg. 24 "computeAllAtomCoordinates" lines 2-10
    Jumper et al. (2021) Suppl. Alg. 25 "makeRotX"

    Args:
        aatype: aatype for each residue
        backb_to_global: Rigid transformations describing transformation from
        backbone frame to global frame.
        torsion_angles_sin_cos: sin and cosine of the 7 torsion angles
    Returns:
        Frames corresponding to all the Sidechain Rigid Transforms
    """
    assert len(aatype.shape) == 2
    assert len(backb_to_global.rot.xx.shape) == 2
    assert len(torsion_angles_sin_cos.shape) == 4
    assert torsion_angles_sin_cos.shape[2] == 7
    assert torsion_angles_sin_cos.shape[3] == 2

    # Gather the default frames for all rigid groups.(New)
    # # r3.Rigids with shape (B, N, 8)
    restype_rigid_group_default_frame = paddle.to_tensor(
        residue_constants.restype_rigid_group_default_frame)  # (21, 8, 4, 4)
    # (B, N, 8, 4, 4)
    m = utils.batched_gather(restype_rigid_group_default_frame, aatype)

    default_frames = r3.rigids_from_tensor4x4(m)

    # Create the rotation matrices according to the given angles (each frame is
    # defined such that its rotation is around the x-axis).
    sin_angles = torsion_angles_sin_cos[..., 0]
    cos_angles = torsion_angles_sin_cos[..., 1]

    # insert zero rotation for backbone group.
    num_batch, num_residues = aatype.shape
    sin_angles = paddle.concat([paddle.zeros([num_batch, num_residues, 1]), sin_angles],
                                axis=-1)
    cos_angles = paddle.concat([paddle.ones([num_batch, num_residues, 1]), cos_angles],
                                axis=-1)
    zeros = paddle.zeros_like(sin_angles)
    ones = paddle.ones_like(sin_angles)

    # all_rots are r3.Rots with shape (B, N, 8)
    all_rots = r3.Rots(ones, zeros, zeros,
                        zeros, cos_angles, -sin_angles,
                        zeros, sin_angles, cos_angles)

    # Apply rotations to the frames.
    all_frames = r3.rigids_mul_rots(default_frames, all_rots)

    # slice, concat and unsqueeze Rigids
    def slice_rigids(rigid, start, end):
        """slice along the last axis of rot.xx and trans.x"""
        assert len(rigid.rot.xx.shape) == 3
        rotation = rigid.rot.rotation[..., start:end, :, :]
        translation = rigid.trans.translation[..., start:end, :]
        return r3.Rigids(rot=r3.Rots(rotation), trans=r3.Vecs(translation))

    def concat_rigids(*arg):
        """concat along the last axis of rot.xx and trans.x"""
        assert len(arg) > 1
        assert len(arg[0].rot.xx.shape) == len(arg[1].rot.xx.shape)
        rotation = paddle.concat([r.rot.rotation for r in arg], axis=-3)
        translation = paddle.concat([r.trans.translation for r in arg], axis=-2)
        return r3.Rigids(rot=r3.Rots(rotation), trans=r3.Vecs(translation))

    def unsqueeze_rigids(rigid, axis=-1):
        """add an axis in the axis of rot.xx and trans.x"""
        if axis < 0:
            axis_t = axis - 1
            axis_r = axis - 2
        else:
            axis_t = axis
            axis_r = axis

        rotation = paddle.unsqueeze(rigid.rot.rotation, axis=axis_r)
        translation = paddle.unsqueeze(rigid.trans.translation, axis=axis_t)
        return r3.Rigids(rot=r3.Rots(rotation), trans=r3.Vecs(translation))

    # chi2, chi3, and chi4 frames do not transform to the backbone frame but to
    # the previous frame. So chain them up accordingly.

    chi2_frame_to_frame = slice_rigids(all_frames, 5, 6)
    chi3_frame_to_frame = slice_rigids(all_frames, 6, 7)
    chi4_frame_to_frame = slice_rigids(all_frames, 7, 8)

    chi1_frame_to_backb = slice_rigids(all_frames, 4, 5)
    chi2_frame_to_backb = r3.rigids_mul_rigids(chi1_frame_to_backb,
                                                chi2_frame_to_frame)
    chi3_frame_to_backb = r3.rigids_mul_rigids(chi2_frame_to_backb,
                                                chi3_frame_to_frame)
    chi4_frame_to_backb = r3.rigids_mul_rigids(chi3_frame_to_backb,
                                             chi4_frame_to_frame)

    all_frames_to_backb = concat_rigids(
        slice_rigids(all_frames, 0, 5),
        chi2_frame_to_backb,
        chi3_frame_to_backb,
        chi4_frame_to_backb)

    # Create the global frames.
    # shape (B, N, 8)
    all_frames_to_global = r3.rigids_mul_rigids(
                    unsqueeze_rigids(backb_to_global),
                    all_frames_to_backb)

    return all_frames_to_global


def frames_and_literature_positions_to_atom14_pos(
        aatype: paddle.Tensor,  # (B, N)
        all_frames_to_global: r3.Rigids  # (B, N, 8)
) -> r3.Vecs:  # (B, N, 14)
    """Put atom literature positions (atom14 encoding) in each rigid group.

    Jumper et al. (2021) Suppl. Alg. 24 "computeAllAtomCoordinates" line 11

    Args:
        aatype: aatype for each residue.
        all_frames_to_global: All per residue coordinate frames.
    Returns:
        Positions of all atom coordinates in global frame.
    """
    # Pick the appropriate transform for every atom.
    restype_atom14_to_rigid_group = paddle.to_tensor(
        residue_constants.restype_atom14_to_rigid_group)[None, ...]
    
    # [1, 21, 14] -> # [n_batch, 21, 14]
    n_batch = aatype.shape[0]
    if n_batch > 1:
        restype_atom14_to_rigid_group = paddle.tile(
            restype_atom14_to_rigid_group, repeat_times=[n_batch, 1, 1])
    
    residx_to_group_idx = utils.batched_gather(
        restype_atom14_to_rigid_group,
        aatype, batch_dims=1)

    # 8 rigid groups:
    # 0: 'backbone group',
    # 1: 'pre-omega-group', (empty)
    # 2: 'phi-group', (currently empty, because it defines only hydrogens)
    # 3: 'psi-group',
    # 4,5,6,7: 'chi1,2,3,4-group'
    # (B, N, 14, 8)
    group_mask = nn.functional.one_hot(
        residx_to_group_idx, num_classes=8)

    def _convert(x, y):
        return paddle.sum(paddle.unsqueeze(x, -2) * y, axis=-1)

    # r3.Rigids with shape (B, N, 14)
    map_atoms_to_global = r3.Rigids(
        rot=all_frames_to_global.rot.map(_convert, group_mask),
        trans=all_frames_to_global.trans.map(_convert, group_mask))

    # Gather the literature atom positions for each residue.
    # r3.Vecs with shape (B, N, 14)
    restype_atom14_rigid_group_positions = paddle.to_tensor(
        residue_constants.restype_atom14_rigid_group_positions)[None, ...]
    # [1, 21, 14, 3] -> [B, 21, 14, 3]
    if n_batch > 1:
        restype_atom14_rigid_group_positions = paddle.tile(
            restype_atom14_rigid_group_positions, repeat_times=[n_batch, 1, 1, 1])
    
    lit_positions = r3.vecs_from_tensor(
        utils.batched_gather(
            restype_atom14_rigid_group_positions,
            aatype, batch_dims=1))

    # Transform each atom from its local frame to the global frame.
    # r3.Vecs with shape (B, N, 14)
    pred_positions = r3.rigids_mul_vecs(map_atoms_to_global, lit_positions)

    # Mask out non-existing atoms.
    restype_atom14_mask = paddle.to_tensor(
        residue_constants.restype_atom14_mask)[None, ...]
    # [1, 21, 14] -> [B, 21, 14]
    if n_batch > 1:
        restype_atom14_mask = paddle.tile(
            restype_atom14_mask, repeat_times=[n_batch, 1, 1])

    mask = utils.batched_gather(
        restype_atom14_mask, aatype, batch_dims=1)
    pred_positions = pred_positions.map(lambda x, m: x * m, mask)

    return pred_positions


def extreme_ca_ca_distance_violations(
    pred_atom_positions: paddle.Tensor,  # (B, N, 37(14), 3)
    pred_atom_mask: paddle.Tensor,  # (B, N, 37(14))
    residue_index: paddle.Tensor,  # (B, N)
    max_angstrom_tolerance=1.5
    ) -> paddle.Tensor:
    """Counts residues whose Ca is a large distance from its neighbour.

    Measures the fraction of CA-CA pairs between consecutive amino acids that are
    more than 'max_angstrom_tolerance' apart.

    Args:
        pred_atom_positions: Atom positions in atom37/14 representation
        pred_atom_mask: Atom mask in atom37/14 representation
        residue_index: Residue index for given amino acid, this is assumed to be
        monotonically increasing.
        max_angstrom_tolerance: Maximum distance allowed to not count as violation.
    Returns:
        Fraction of consecutive CA-CA pairs with violation.
    """
    batch_size = pred_atom_positions.shape[0]
    this_ca_pos = pred_atom_positions[:, :-1, 1, :]  # (B, N - 1, 3)
    this_ca_mask = pred_atom_mask[:, :-1, 1]         # (B, N - 1)
    next_ca_pos = pred_atom_positions[:, 1:, 1, :]  # (B, N - 1, 3)
    next_ca_mask = pred_atom_mask[:, 1:, 1]  # (B, N - 1)
    has_no_gap_mask = ((residue_index[:, 1:] - residue_index[:, :-1]) == 1.0)

    ca_ca_distance = paddle.sqrt(1e-6 + paddle.sum(squared_difference(this_ca_pos, next_ca_pos), axis=-1))
    violations = (ca_ca_distance - residue_constants.ca_ca) > max_angstrom_tolerance
    mask = this_ca_mask * next_ca_mask * has_no_gap_mask
    ca_ca_violation_tmp = []
    for i in range(batch_size):
        ca_ca_violation_i = utils.mask_mean(mask=mask[i], value=violations[i])
        ca_ca_violation_tmp.append(ca_ca_violation_i)
    ca_ca_violation = paddle.to_tensor(ca_ca_violation_tmp, stop_gradient=False)
    ca_ca_violation = paddle.squeeze(ca_ca_violation, axis=-1)
    return ca_ca_violation


def between_residue_bond_loss(
    pred_atom_positions: paddle.Tensor,  # (B, N, 37(14), 3)
    pred_atom_mask: paddle.Tensor,  # (B, N, 37(14))
    residue_index: paddle.Tensor,  # (B, N)
    aatype: paddle.Tensor,  # (B, N)
    tolerance_factor_soft=12.0,
    tolerance_factor_hard=12.0
) -> Dict[str, paddle.Tensor]:
    """Flat-bottom loss to penalize structural violations between residues.

    This is a loss penalizing any violation of the geometry around the peptide
    bond between consecutive amino acids. This loss corresponds to
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 44, 45.

    Args:
        pred_atom_positions: Atom positions in atom37/14 representation
        pred_atom_mask: Atom mask in atom37/14 representation
        residue_index: Residue index for given amino acid, this is assumed to be
        monotonically increasing.
        aatype: Amino acid type of given residue
        tolerance_factor_soft: soft tolerance factor measured in standard deviations
        of pdb distributions
        tolerance_factor_hard: hard tolerance factor measured in standard deviations
        of pdb distributions

    Returns:
        Dict containing:
        * 'c_n_loss_mean': Loss for peptide bond length violations
        * 'ca_c_n_loss_mean': Loss for violations of bond angle around C spanned
            by CA, C, N
        * 'c_n_ca_loss_mean': Loss for violations of bond angle around N spanned
            by C, N, CA
        * 'per_residue_loss_sum': sum of all losses for each residue
        * 'per_residue_violation_mask': mask denoting all residues with violation
            present.
    """
    batch_size = aatype.shape[0]

    assert len(pred_atom_positions.shape) == 4
    assert len(pred_atom_mask.shape) == 3
    assert len(residue_index.shape) == 2
    assert len(aatype.shape) == 2

    # Get the positions of the relevant backbone atoms.
    this_ca_pos = pred_atom_positions[:, :-1, 1, :]  # (B, N - 1, 3)
    this_ca_mask = pred_atom_mask[:, :-1, 1]         # (B, N - 1)
    this_c_pos = pred_atom_positions[:, :-1, 2, :]   # (B, N - 1, 3)
    this_c_mask = pred_atom_mask[:, :-1, 2]          # (B, N - 1)
    next_n_pos = pred_atom_positions[:, 1:, 0, :]    # (B, N - 1, 3)
    next_n_mask = pred_atom_mask[:, 1:, 0]           # (B, N - 1)
    next_ca_pos = pred_atom_positions[:, 1:, 1, :]   # (B, N - 1, 3)
    next_ca_mask = pred_atom_mask[:, 1:, 1]          # (B, N - 1)
    has_no_gap_mask = ((residue_index[:, 1:] - residue_index[:, :-1]) == 1.0)


    # Compute loss for the C--N bond.
    c_n_bond_length = paddle.sqrt(1e-6 + paddle.sum(squared_difference(this_c_pos, next_n_pos), axis=-1))

    # The C-N bond to proline has slightly different length because of the ring.
    next_is_proline = paddle.cast((aatype[:, 1:] == residue_constants.resname_to_idx['PRO']), 'float32')
    gt_length = (
        (1. - next_is_proline) * residue_constants.between_res_bond_length_c_n[0]
        + next_is_proline * residue_constants.between_res_bond_length_c_n[1])
    gt_stddev = (
        (1. - next_is_proline) *
        residue_constants.between_res_bond_length_stddev_c_n[0] +
        next_is_proline * residue_constants.between_res_bond_length_stddev_c_n[1])
    c_n_bond_length_error = paddle.sqrt(1e-6 + paddle.square(c_n_bond_length - gt_length))
    c_n_loss_per_residue = nn.functional.relu(c_n_bond_length_error - tolerance_factor_soft * gt_stddev)
    mask = this_c_mask * next_n_mask * has_no_gap_mask
    c_n_loss = paddle.sum(mask * c_n_loss_per_residue, axis=-1) / (paddle.sum(mask, axis=-1) + 1e-6)
    c_n_violation_mask = mask * (c_n_bond_length_error > (tolerance_factor_hard * gt_stddev))

    # Compute loss for the angles.
    ca_c_bond_length = paddle.sqrt(1e-6 + paddle.sum(
        squared_difference(this_ca_pos, this_c_pos), axis=-1))
    n_ca_bond_length = paddle.sqrt(1e-6 + paddle.sum(
        squared_difference(next_n_pos, next_ca_pos), axis=-1))

    ca_c_bond_length = paddle.unsqueeze(ca_c_bond_length, axis=-1)
    c_n_bond_length = paddle.unsqueeze(c_n_bond_length, axis=-1)
    n_ca_bond_length = paddle.unsqueeze(n_ca_bond_length, axis=-1)
    c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length
    c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length
    n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length

    ca_c_n_cos_angle = paddle.sum(c_ca_unit_vec * c_n_unit_vec, axis=-1)
    gt_angle = residue_constants.between_res_cos_angles_ca_c_n[0]
    gt_stddev = residue_constants.between_res_bond_length_stddev_c_n[0]
    ca_c_n_cos_angle_error = paddle.sqrt(1e-6 + paddle.square(ca_c_n_cos_angle - gt_angle))
    ca_c_n_loss_per_residue = nn.functional.relu(ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
    ca_c_n_loss = paddle.sum(mask * ca_c_n_loss_per_residue, axis=-1) / (paddle.sum(mask, axis=-1) + 1e-6)
    ca_c_n_violation_mask = mask * (ca_c_n_cos_angle_error >
                                    (tolerance_factor_hard * gt_stddev))

    c_n_ca_cos_angle = paddle.sum((-c_n_unit_vec) * n_ca_unit_vec, axis=-1)
    gt_angle = residue_constants.between_res_cos_angles_c_n_ca[0]
    gt_stddev = residue_constants.between_res_cos_angles_c_n_ca[1]
    c_n_ca_cos_angle_error = paddle.sqrt(1e-6 + paddle.square(c_n_ca_cos_angle - gt_angle))
    c_n_ca_loss_per_residue = nn.functional.relu(c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
    c_n_ca_loss = paddle.sum(mask * c_n_ca_loss_per_residue, axis=-1) / (paddle.sum(mask, axis=-1) + 1e-6)
    c_n_ca_violation_mask = mask * (c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev))

    # Compute a per residue loss (equally distribute the loss to both
    # neighbouring residues).
    tmpsum = paddle.zeros(shape=[batch_size, 1])    
    per_residue_loss_sum = (c_n_loss_per_residue + ca_c_n_loss_per_residue + c_n_ca_loss_per_residue)
    tmp_per_residue_loss1 = paddle.concat(x=[per_residue_loss_sum, tmpsum], axis=-1)
    tmp_per_residue_loss2 = paddle.concat(x=[tmpsum, per_residue_loss_sum], axis=-1)
    per_residue_loss_sum = 0.5 * (tmp_per_residue_loss1 + tmp_per_residue_loss2)

    # Compute hard violations.
    violation_mask = paddle.max(
        paddle.stack([c_n_violation_mask,
                    ca_c_n_violation_mask,
                    c_n_ca_violation_mask]), axis=0)
    tmp_violation_mask1 = paddle.concat(x=[violation_mask, tmpsum], axis=-1)
    tmp_violation_mask2 = paddle.concat(x=[tmpsum, violation_mask], axis=-1)
    violation_mask = paddle.maximum(tmp_violation_mask1, tmp_violation_mask2)

    return {'c_n_loss_mean': c_n_loss,  # shape (B)
            'ca_c_n_loss_mean': ca_c_n_loss,  # shape (B)
            'c_n_ca_loss_mean': c_n_ca_loss,  # shape (B)
            'per_residue_loss_sum': per_residue_loss_sum,  # shape (B, N)
            'per_residue_violation_mask': violation_mask  # shape (B, N)
            }


def between_residue_clash_loss(
    atom14_pred_positions: paddle.Tensor,  # (B, N, 14, 3)
    atom14_atom_exists: paddle.Tensor,  # (B, N, 14)
    atom14_atom_radius: paddle.Tensor,  # (B, N, 14)
    residue_index: paddle.Tensor,  # (B, N)
    overlap_tolerance_soft=1.5,
    overlap_tolerance_hard=1.5
) -> Dict[str, paddle.Tensor]:
    """Loss to penalize steric clashes between residues.

    This is a loss penalizing any steric clashes due to non bonded atoms in
    different peptides coming too close. This loss corresponds to the part with
    different residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
        atom14_pred_positions: Predicted positions of atoms in
        global prediction frame
        atom14_atom_exists: Mask denoting whether atom at positions exists for given
        amino acid type
        atom14_atom_radius: Van der Waals radius for each atom.
        residue_index: Residue index for given amino acid.
        overlap_tolerance_soft: Soft tolerance factor.
        overlap_tolerance_hard: Hard tolerance factor.

    Returns:
        Dict containing:
        * 'mean_loss': average clash loss
        * 'per_atom_loss_sum': sum of all clash losses per atom, shape (B, N, 14)
        * 'per_atom_clash_mask': mask whether atom clashes with any other atom
            shape (B, N, 14)
    """
    assert len(atom14_pred_positions.shape) == 4
    assert len(atom14_atom_exists.shape) == 3
    assert len(atom14_atom_radius.shape) == 3
    assert len(residue_index.shape) == 2

    # Create the distance matrix.
    # (B, N, N, 14, 14)
    atom14_pred_positions1 = paddle.unsqueeze(atom14_pred_positions, axis=[2,4])
    atom14_pred_positions2 = paddle.unsqueeze(atom14_pred_positions, axis=[1,3])
    dists = paddle.sqrt(1e-10 + paddle.sum(squared_difference(atom14_pred_positions1, atom14_pred_positions2), axis=-1))
    
    # Create the mask for valid distances.
    # shape (B, N, N, 14, 14)
    atom14_atom_exists1 = paddle.unsqueeze(atom14_atom_exists, axis=[2,4])
    atom14_atom_exists2 = paddle.unsqueeze(atom14_atom_exists, axis=[1,3])
    dists_mask = (atom14_atom_exists1 * atom14_atom_exists2)

    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
    # are handled separately.
    residue_index1 = paddle.unsqueeze(residue_index, axis=[2,3,4])
    residue_index2 = paddle.unsqueeze(residue_index, axis=[1,3,4])
    dists_mask *= (residue_index1 < residue_index2)

    # Backbone C--N bond between subsequent residues is no clash.
    c_one_hot = nn.functional.one_hot(paddle.to_tensor([2]), num_classes=14)
    n_one_hot = nn.functional.one_hot(paddle.to_tensor([0]), num_classes=14)
    neighbour_mask = ((residue_index1 + 1) == residue_index2)
    tmp_c_one_hot = paddle.unsqueeze(c_one_hot, axis=[1,2,4])
    tmp_n_one_hot = paddle.unsqueeze(n_one_hot, axis=[1,2,3])
    c_n_bonds = neighbour_mask * tmp_c_one_hot * tmp_n_one_hot

    dists_mask *= (1. - c_n_bonds)

    # Disulfide bridge between two cysteines is no clash.
    cys_sg_idx = residue_constants.restype_name_to_atom14_names['CYS'].index('SG')
    cys_sg_one_hot = nn.functional.one_hot(paddle.to_tensor(cys_sg_idx), num_classes=14)
    cys_sg_one_hot1 = paddle.unsqueeze(cys_sg_one_hot, axis=[1,2,4])
    cys_sg_one_hot2 = paddle.unsqueeze(cys_sg_one_hot, axis=[1,2,3])
    disulfide_bonds = (cys_sg_one_hot1 * cys_sg_one_hot2)
    dists_mask *= (1. - disulfide_bonds)

    # Compute the lower bound for the allowed distances.
    # shape (B, N, N, 14, 14)
    atom14_atom_radius1 = paddle.unsqueeze(atom14_atom_radius, axis=[2,4])
    atom14_atom_radius2 = paddle.unsqueeze(atom14_atom_radius, axis=[1,3])
    dists_lower_bound = dists_mask * (atom14_atom_radius1 + atom14_atom_radius2)

    # Compute the error.
    # shape (B, N, N, 14, 14)
    dists_to_low_error = dists_mask * nn.functional.relu(dists_lower_bound - overlap_tolerance_soft - dists)

    # Compute the mean loss.
    # shape (B)
    mean_loss = (paddle.sum(dists_to_low_error, axis=[1,2,3,4]) / (1e-6 + paddle.sum(dists_mask, axis=[1,2,3,4])))

    # Compute the per atom loss sum.
    # shape (B, N, 14)
    per_atom_loss_sum = (paddle.sum(dists_to_low_error, axis=[1, 3]) +
                        paddle.sum(dists_to_low_error, axis=[2, 4]))

    # Compute the hard clash mask.
    # shape (B, N, N, 14, 14)
    clash_mask = dists_mask * (dists < (dists_lower_bound - overlap_tolerance_hard))

    # Compute the per atom clash.
    # shape (B, N, 14)
    per_atom_clash_mask = paddle.maximum(
        paddle.max(clash_mask, axis=[1, 3]),
        paddle.max(clash_mask, axis=[2, 4]))
        
    return {'mean_loss': mean_loss,  # shape (B)
            'per_atom_loss_sum': per_atom_loss_sum,  # shape (B, N, 14)
            'per_atom_clash_mask': per_atom_clash_mask  # shape (B, N, 14)
            }


def within_residue_violations(
    atom14_pred_positions: paddle.Tensor,  # (B, N, 14, 3)
    atom14_atom_exists: paddle.Tensor,  # (B, N, 14)
    atom14_dists_lower_bound: paddle.Tensor,  # (B, N, 14, 14)
    atom14_dists_upper_bound: paddle.Tensor,  # (B, N, 14, 14)
    tighten_bounds_for_loss=0.0,
) -> Dict[str, paddle.Tensor]:
    """Loss to penalize steric clashes within residues.

    This is a loss penalizing any steric violations or clashes of non-bonded atoms
    in a given peptide. This loss corresponds to the part with
    the same residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
        atom14_pred_positions: Predicted positions of atoms in
        global prediction frame
        atom14_atom_exists: Mask denoting whether atom at positions exists for given
        amino acid type
        atom14_dists_lower_bound: Lower bound on allowed distances.
        atom14_dists_upper_bound: Upper bound on allowed distances
        tighten_bounds_for_loss: Extra factor to tighten loss

    Returns:
        Dict containing:
        * 'per_atom_loss_sum': sum of all clash losses per atom, shape (B, N, 14)
        * 'per_atom_clash_mask': mask whether atom clashes with any other atom
            shape (B, N, 14)
    """
    assert len(atom14_pred_positions.shape) == 4
    assert len(atom14_atom_exists.shape) == 3
    assert len(atom14_dists_lower_bound.shape) == 4
    assert len(atom14_dists_upper_bound.shape) == 4

    # Compute the mask for each residue.
    # shape (B, N, 14, 14)
    dists_masks = (1. - paddle.unsqueeze(paddle.eye(14, 14), axis=[0, 1]))
    atom14_atom_exists1 = paddle.unsqueeze(atom14_atom_exists, axis=-1)
    atom14_atom_exists2 = paddle.unsqueeze(atom14_atom_exists, axis=-2)
    dists_masks *= (atom14_atom_exists1 * atom14_atom_exists2)

    # Distance matrix
    # shape (B, N, 14, 14)
    atom14_pred_positions1 = paddle.unsqueeze(atom14_pred_positions, axis=-2)
    atom14_pred_positions2 = paddle.unsqueeze(atom14_pred_positions, axis=-3)
    dists = paddle.sqrt(1e-10 + paddle.sum(
        squared_difference(atom14_pred_positions1, atom14_pred_positions2),
        axis=-1))

    # Compute the loss.
    # shape (B, N, 14, 14)
    dists_to_low_error = nn.functional.relu(
        atom14_dists_lower_bound + tighten_bounds_for_loss - dists)
    dists_to_high_error = nn.functional.relu(
        dists - (atom14_dists_upper_bound - tighten_bounds_for_loss))
    loss = dists_masks * (dists_to_low_error + dists_to_high_error)

    # Compute the per atom loss sum.
    # shape (B, N, 14)
    per_atom_loss_sum = (paddle.sum(loss, axis=2) + paddle.sum(loss, axis=3))

    # Compute the violations mask.
    # shape (B, N, 14, 14)
    violations = dists_masks * ((dists < atom14_dists_lower_bound) |
                                (dists > atom14_dists_upper_bound))

    # Compute the per atom violations.
    # shape (B, N, 14)
    per_atom_violations = paddle.maximum(paddle.max(violations, axis=2), paddle.max(violations, axis=3))

    return {'per_atom_loss_sum': per_atom_loss_sum,  # shape (B, N, 14)
            'per_atom_violations': per_atom_violations  # shape (B, N, 14)
            }


def find_optimal_renaming(
    atom14_gt_positions: paddle.Tensor,  # (B, N, 14, 3)
    atom14_alt_gt_positions: paddle.Tensor,  # (B, N, 14, 3)
    atom14_atom_is_ambiguous: paddle.Tensor,  # (B, N, 14)
    atom14_gt_exists: paddle.Tensor,  # (B, N, 14)
    atom14_pred_positions: paddle.Tensor,  # (B, N, 14, 3)
    atom14_atom_exists: paddle.Tensor,  # (B, N, 14)
) -> paddle.Tensor:  # (B, N):
    """Find optimal renaming for ground truth that maximizes LDDT.

    Jumper et al. (2021) Suppl. Alg. 26
    "renameSymmetricGroundTruthAtoms" lines 1-5

    Args:
        atom14_gt_positions: Ground truth positions in global frame of ground truth.
        atom14_alt_gt_positions: Alternate ground truth positions in global frame of
        ground truth with coordinates of ambiguous atoms swapped relative to
        'atom14_gt_positions'.
        atom14_atom_is_ambiguous: Mask denoting whether atom is among ambiguous
        atoms, see Jumper et al. (2021) Suppl. Table 3
        atom14_gt_exists: Mask denoting whether atom at positions exists in ground
        truth.
        atom14_pred_positions: Predicted positions of atoms in
        global prediction frame
        atom14_atom_exists: Mask denoting whether atom at positions exists for given
        amino acid type

    Returns:
        Float array of shape [N] with 1. where atom14_alt_gt_positions is closer to
        prediction and 0. otherwise
    """
    assert len(atom14_gt_positions.shape) == 4
    assert len(atom14_alt_gt_positions.shape) == 4
    assert len(atom14_atom_is_ambiguous.shape) == 3
    assert len(atom14_gt_exists.shape) == 3
    assert len(atom14_pred_positions.shape) == 4
    assert len(atom14_atom_exists.shape) == 3

    # Create the pred distance matrix.
    # shape (B, N, N, 14, 14)
    atom14_pred_positions1 = paddle.unsqueeze(atom14_pred_positions, axis=[2,4])
    atom14_pred_positions2 = paddle.unsqueeze(atom14_pred_positions, axis=[1,3])
    pred_dists = paddle.sqrt(1e-10 + paddle.sum(
        squared_difference(atom14_pred_positions1, atom14_pred_positions2),
        axis=-1))

    # Compute distances for ground truth with original and alternative names.
    # shape (B, N, N, 14, 14)
    atom14_gt_positions1 = paddle.unsqueeze(atom14_gt_positions, axis=[2,4])
    atom14_gt_positions2 = paddle.unsqueeze(atom14_gt_positions, axis=[1,3])
    gt_dists = paddle.sqrt(1e-10 + paddle.sum(
        squared_difference(atom14_gt_positions1, atom14_gt_positions2),
        axis=-1))

    atom14_alt_gt_positions1 = paddle.unsqueeze(atom14_alt_gt_positions, axis=[2,4])
    atom14_alt_gt_positions2 = paddle.unsqueeze(atom14_alt_gt_positions, axis=[1,3])
    alt_gt_dists = paddle.sqrt(1e-10 + paddle.sum(
        squared_difference(atom14_alt_gt_positions1, atom14_alt_gt_positions2),
        axis=-1))

    # Compute LDDT's.
    # shape (B, N, N, 14, 14)
    lddt = paddle.sqrt(1e-10 + squared_difference(pred_dists, gt_dists))
    alt_lddt = paddle.sqrt(1e-10 + squared_difference(pred_dists, alt_gt_dists))

    # Create a mask for ambiguous atoms in rows vs. non-ambiguous atoms
    # in cols.
    # shape (B, N, N, 14, 14)
    atom14_gt_exists1 = paddle.unsqueeze(atom14_gt_exists, axis=[2,4])
    atom14_gt_exists2 = paddle.unsqueeze(atom14_gt_exists, axis=[1,3])
    atom14_atom_is_ambiguous1 = paddle.unsqueeze(atom14_atom_is_ambiguous, axis=[2,4])
    atom14_atom_is_ambiguous2 = paddle.unsqueeze(atom14_atom_is_ambiguous, axis=[1,3])
    mask = (atom14_gt_exists1 *  # rows
            atom14_atom_is_ambiguous1 *  # rows
            atom14_gt_exists2 *  # cols
            (1. - atom14_atom_is_ambiguous2))  # cols

    # Aggregate distances for each residue to the non-amibuguous atoms.
    # shape (B, N)
    per_res_lddt = paddle.sum(mask * lddt, axis=[2, 3, 4])
    alt_per_res_lddt = paddle.sum(mask * alt_lddt, axis=[2, 3, 4])

    # Decide for each residue, whether alternative naming is better.
    # shape (B, N)
    alt_naming_is_better = paddle.cast((alt_per_res_lddt < per_res_lddt), 'float32')

    return alt_naming_is_better  # shape (B, N)


def frame_aligned_point_error(
    pred_frames: r3.Rigids, 
    target_frames: r3.Rigids,  
    frames_mask: paddle.Tensor,  
    pred_positions: r3.Vecs,  
    target_positions: r3.Vecs,  
    positions_mask: paddle.Tensor,  
    length_scale: float,
    l1_clamp_distance: Optional[float] = None,
    epsilon=1e-4) -> paddle.Tensor:
    """Measure point error under different alignments.

    Jumper et al. (2021) Suppl. Alg. 28 "computeFAPE"

    Computes error between two structures with B points under A alignments derived
    from the given pairs of frames.
    Args:
        pred_frames: num_frames reference frames for 'pred_positions'.
        target_frames: num_frames reference frames for 'target_positions'.
        frames_mask: Mask for frame pairs to use.
        pred_positions: num_positions predicted positions of the structure.
        target_positions: num_positions target positions of the structure.
        positions_mask: Mask on which positions to score.
        length_scale: length scale to divide loss by.
        l1_clamp_distance: Distance cutoff on error beyond which gradients will
        be zero.
        epsilon: small value used to regularize denominator for masked average.
    Returns:
        Masked Frame Aligned Point Error.
    """
    def unsqueeze_rigids(rigid, axis=-1):
        """add an axis in the axis of rot.xx and trans.x"""
        if axis < 0:
            axis_t = axis - 1
            axis_r = axis - 2
        else:
            axis_t = axis
            axis_r = axis

        rotation = paddle.unsqueeze(rigid.rot.rotation, axis=axis_r)
        translation = paddle.unsqueeze(rigid.trans.translation, axis=axis_t)
        return r3.Rigids(rot=r3.Rots(rotation), trans=r3.Vecs(translation))

    def unsqueeze_vecs(vecs, axis=-1):
        """add an axis in the axis of rot.xx and trans.x"""
        if axis < 0:
            axis_t = axis - 1
        else:
            axis_t = axis

        translation = paddle.unsqueeze(vecs.translation, axis=axis_t)
        return r3.Vecs(translation)
    
    # Compute array of predicted positions in the predicted frames.
    # r3.Vecs (num_frames, num_positions)
    local_pred_pos = r3.rigids_mul_vecs(
        unsqueeze_rigids(r3.invert_rigids(pred_frames)),
        unsqueeze_vecs(pred_positions, axis=1))

    # Compute array of target positions in the target frames.
    # r3.Vecs (num_frames, num_positions)
    local_target_pos = r3.rigids_mul_vecs(
        unsqueeze_rigids(r3.invert_rigids(target_frames)),
        unsqueeze_vecs(target_positions, axis=1))

    # Compute errors between the structures.
    # paddle.Tensor (num_frames, num_positions)
    error_dist = paddle.sqrt(r3.vecs_squared_distance(local_pred_pos, local_target_pos) + epsilon)        

    if l1_clamp_distance:
        error_dist = paddle.clip(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error *= paddle.unsqueeze(frames_mask, axis=-1)
    normed_error *= paddle.unsqueeze(positions_mask, axis=-2)

    normalization_factor = (
        paddle.sum(frames_mask, axis=-1) *
        paddle.sum(positions_mask, axis=-1))
    return (paddle.sum(normed_error, axis=[-2, -1]) /
            (epsilon + normalization_factor))
