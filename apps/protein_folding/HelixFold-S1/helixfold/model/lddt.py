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

"""lDDT protein distance score."""

import paddle
import paddle.nn as nn
import numpy as np


def get_interface_mask(true_points, true_points_mask, 
                       protein_mask, nucleic_mask, cutoff=10.0):
    """
    Create a mask for interface atoms between protein and nucleic acid.

    Args:
    true_points: (batch, length, 3) array of true atom positions
    true_points_mask: (batch, length) binary mask for valid true atom positions
    protein_mask: (batch, length, 1) binary mask for protein atoms
    nucleic_mask: (batch, length, 1) binary mask for nucleic acid atoms
    cutoff: float, distance cutoff for interface definition (in Angstroms)

    Returns:
    interface_mask: (batch, length, 1) binary mask indicating interface atoms
    """

    # Compute distance matrix (batch, length, length) - pairwise distances
    dmat = paddle.sqrt(1e-10 + paddle.sum(
        (true_points[:, :, None] - true_points[:, None, :])**2, axis=-1))  # Shape: (batch, length, length)

    # Ensure protein_mask and nucleic_mask have correct dimensions
    protein_mask = protein_mask.squeeze(-1)  # Shape: (batch, length)
    nucleic_mask = nucleic_mask.squeeze(-1)  # Shape: (batch, length)

    # Create true_pos_mask for valid atoms (pairwise)
    true_pos_mask = true_points_mask.unsqueeze(1) * true_points_mask.unsqueeze(2)  # Shape: (batch, length, length)

    # Create pairwise protein and nucleic acid masks
    protein_mask_mat = protein_mask.unsqueeze(2) * nucleic_mask.unsqueeze(1)  # (batch, length, length)
    nucleic_mask_mat = nucleic_mask.unsqueeze(2) * protein_mask.unsqueeze(1)  # (batch, length, length)

    # Masked distance matrix: Focus on distances between protein and nucleic atoms
    dmat_masked = true_pos_mask.squeeze(-1) * (protein_mask_mat + nucleic_mask_mat)  # Shape: (batch, length, length)

    # Create interface mask based on cutoff distance
    dmat_interface_mask = (dmat <= cutoff).astype('float32') * dmat_masked # Interface where distances are below cutoff
    dmat_interface_mask = dmat_interface_mask.astype('bool')

    return dmat_interface_mask


def lddt(predicted_points,
         true_points,
         true_points_mask,
         cutoff=15.,
         pair_mask=None,
         per_residue=False):
    """Measure (approximate) lDDT for a batch of coordinates.

    lDDT reference:
    Mariani, V., Biasini, M., Barbato, A. & Schwede, T. lDDT: A local
    superposition-free score for comparing protein structures and models using
    distance difference tests. Bioinformatics 29, 2722–2728 (2013).

    lDDT is a measure of the difference between the true distance matrix and the
    distance matrix of the predicted points.  The difference is computed only on
    points closer than cutoff *in the true structure*.

    This function does not compute the exact lDDT value that the original paper
    describes because it does not include terms for physical feasibility
    (e.g. bond length violations). Therefore this is only an approximate
    lDDT score.

    Args:
    predicted_points: (batch, length, 3) array of predicted 3D points
    true_points: (batch, length, 3) array of true 3D points
    true_points_mask: (batch, length, 1) binary-valued float array.  This mask
        should be 1 for points that exist in the true points.
    cutoff: float or tensor of (batch, length). Maximum distance for a pair 
        of points to be included
    per_residue: If true, return score for each residue.  Note that the overall
        lDDT is not exactly the mean of the per_residue lDDT's because some
        residues have more contacts than others.

    Returns:
    An (approximate, see above) lDDT score in the range 0-1.
    """

    assert len(predicted_points.shape) == 3
    assert predicted_points.shape[-1] == 3
    assert true_points_mask.shape[-1] == 1
    assert len(true_points_mask.shape) == 3

    # Compute true and predicted distance matrices.
    dmat_true = paddle.sqrt(1e-10 + paddle.sum(
        (true_points[:, :, None] - true_points[:, None, :])**2, axis=-1))

    dmat_predicted = paddle.sqrt(1e-10 + paddle.sum(
        (predicted_points[:, :, None] -
        predicted_points[:, None, :])**2, axis=-1))

    cutoff = paddle.to_tensor(cutoff)
    if cutoff.ndim > 0:
        cutoff = cutoff[:, None]

    true_points_mask = paddle.cast(true_points_mask, 'float32')
    # Apply pair_mask if provided (for interface pLDDT calculation)
    dists_to_score = (
        paddle.cast((dmat_true < cutoff), 'float32') * true_points_mask *
        paddle.transpose(true_points_mask, [0, 2, 1]) *
        (1. - paddle.eye(dmat_true.shape[1]))  # Exclude self-interaction.
    )
    if not pair_mask is None:
        dists_to_score = dists_to_score * pair_mask

    # Shift unscored distances to be far away.
    dist_l1 = paddle.abs(dmat_true - dmat_predicted)

    # True lDDT uses a number of fixed bins.
    # We ignore the physical plausibility correction to lDDT, though.
    score = 0.25 * (paddle.cast((dist_l1 < 0.5), 'float32') +
                    paddle.cast((dist_l1 < 1.0), 'float32') +
                    paddle.cast((dist_l1 < 2.0), 'float32') +
                    paddle.cast((dist_l1 < 4.0), 'float32'))

    # Normalize over the appropriate axes.
    reduce_axes = (-1,) if per_residue else (-2, -1)
    norm = 1. / (1e-10 + paddle.sum(dists_to_score, axis=reduce_axes))
    score = norm * (1e-10 + paddle.sum(dists_to_score * score, axis=reduce_axes))

    return score


def express_coords_in_frame(x, frame):
    """
    Args:
        x: (B, N, 3)
        frame: [(B, N_frame, 3), (B, N_frame, 3), (B, N_frame, 3)], stands for
                a, b, c
    
    Returns:
        x_trans: (B, N_frame, N, 3)
    """
    def _norm_vector(v):
        return v / paddle.sqrt((v ** 2).sum(-1, keepdim=True) + 1e-8)
    a, b, c = frame
    w1 = _norm_vector(a - b)    # (B, N_frame, 3)
    w2 = _norm_vector(c - b)
    e1 = _norm_vector(w1 + w2)
    e2 = _norm_vector(w2 - w1)
    e3 = paddle.cross(e1, e2)
    d = x[:, None] - b[:, :, None]  # (B, N_frame, N, 3)
    x_trans = paddle.stack([
            (d * e1[:, :, None]).sum(-1), 
            (d * e2[:, :, None]).sum(-1), 
            (d * e3[:, :, None]).sum(-1)], -1)    # (B, N_frame, N, 3)
    return x_trans


def alignment_error(
        pred_atom_pos, gt_atom_pos,
        frame_ai_indice, frame_bi_indice, frame_ci_indice,
        frame_mask):
    """
    Args:
        pred_atom_pos: (B, N_atom, 3)
        gt_atom_pos: (B, N_atom, 3)
        frame_ai_indice: (B, N_frame)
        frame_bi_indice: (B, N_frame)
        frame_ci_indice: (B, N_frame)
        frame_mask: (B, N_frame)
    """
    def _batch_indexing(x, index):
        ret = paddle.stack([v[i] for v, i 
                in zip(x, index)])
        return ret

    pred_ai = _batch_indexing(pred_atom_pos, frame_ai_indice)    # (B, N_frame, 3)
    pred_bi = _batch_indexing(pred_atom_pos, frame_bi_indice)
    pred_ci = _batch_indexing(pred_atom_pos, frame_ci_indice)
    gt_ai = _batch_indexing(gt_atom_pos, frame_ai_indice)
    gt_bi = _batch_indexing(gt_atom_pos, frame_bi_indice)
    gt_ci = _batch_indexing(gt_atom_pos, frame_ci_indice)

    pred_xij = express_coords_in_frame(
            pred_bi, [pred_ai, pred_bi, pred_ci])   # (B, N_frame, N_frame, 3)
    gt_xij = express_coords_in_frame(
            gt_bi, [gt_ai, gt_bi, gt_ci])   # (B, N_frame, N_frame, 3)

    error = paddle.sqrt(((pred_xij - gt_xij) ** 2).sum(-1) + 1e-8)  # (B, N_frame, N_frame)
    return error
