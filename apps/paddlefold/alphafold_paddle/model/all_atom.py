#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from alphafold_paddle.common import residue_constants

from alphafold_paddle.model import r3
from alphafold_paddle.model import utils

import paddle
import paddle.nn as nn


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
        aatype: Amino acid type, given as array with integers.
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
    pass #TODO


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
    aatype = paddle.minimum(aatype, paddle.to_tensor([20], dtype='int32'))

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
    torsion_frames = r3.rigids_from_3_points(
        p_neg_x_axis=torsions_atom_pos[..., 1, :],
        origin=torsions_atom_pos[..., 2, :],
        p_xy_plane=torsions_atom_pos[..., 0, :])

    # Compute the position of the forth atom in this frame (y and z coordinate
    # define the chi angle)
    # r3.Vecs (B, T, N, torsions=7)
    forth_atom_rel_pos = r3.rigids_mul_vecs(
        r3.invert_rigids(torsion_frames),
        r3.vecs_from_tensor(torsions_atom_pos[..., 3, :]))

    # Normalize to have the sin and cos of the torsion angle.
    # jnp.ndarray (B, T, N, torsions=7, sincos=2)
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

    # Gather the default frames for all rigid groups.
    # # r3.Rigids with shape (B, N, 8)
    restype_rigid_group_default_frame = paddle.to_tensor(
        residue_constants.restype_rigid_group_default_frame)[None, ...]
    m = utils.batched_gather(restype_rigid_group_default_frame,
                           aatype, batch_dims=1)

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
        residue_constants.restype_atom14_to_rigid_group)
    residx_to_group_idx = utils.batched_gather(
        restype_atom14_to_rigid_group[None, ...],
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
        residue_constants.restype_atom14_rigid_group_positions)
    lit_positions = r3.vecs_from_tensor(
        utils.batched_gather(
            restype_atom14_rigid_group_positions[None, ...],
            aatype, batch_dims=1))

    # Transform each atom from its local frame to the global frame.
    # r3.Vecs with shape (B, N, 14)
    pred_positions = r3.rigids_mul_vecs(map_atoms_to_global, lit_positions)

    # Mask out non-existing atoms.
    restype_atom14_mask = paddle.to_tensor(
        residue_constants.restype_atom14_mask)
    mask = utils.batched_gather(
        restype_atom14_mask[None, ...], aatype, batch_dims=1)
    pred_positions = pred_positions.map(lambda x, m: x * m, mask)

    return pred_positions
