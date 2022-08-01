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

"""Utils for data preparation."""

import os
import copy
import ml_collections
import numpy as np
import paddle
from typing import Mapping, Optional, Sequence
from absl import logging
from typing import List, Mapping, Tuple
from alphafold_paddle.common import residue_constants, protein
from alphafold_paddle.data.mmcif_parsing import MmcifObject
from alphafold_paddle.data.templates import _get_atom_positions as get_atom_positions
from alphafold_paddle.data import parsers
from alphafold_paddle.data.pipeline import make_sequence_features, make_msa_features
from alphafold_paddle.model import quat_affine, all_atom


FeatureDict = Mapping[str, np.ndarray]


def single_sequence_to_features(sequence, description):
    makeup_a3m_string = f"{description}\n{sequence}"
    return a3m_to_features([makeup_a3m_string])


def Msas_to_features(Msas):
    """MSA to af2 processed features."""
    uniref90_msa, bfd_msa, mgnify_msa = Msas['uniref90_msa'], Msas['bfd_msa'], Msas['mgnify_msa']
    template_features = Msas['template_features'] if 'template_features' in Msas else {}

    input_sequence = uniref90_msa.sequences[0]
    input_description = uniref90_msa.descriptions[0]
    num_res = len(input_sequence)
    sequence_features = make_sequence_features(
        sequence=input_sequence,
        description=input_description,
        num_res=num_res)

    msa_features = make_msa_features(
            msas=(
                uniref90_msa.sequences, 
                bfd_msa.sequences, 
                mgnify_msa.sequences),
            deletion_matrices=(
                uniref90_msa.deletion_matrix, 
                bfd_msa.deletion_matrix, 
                mgnify_msa.deletion_matrix))

    raw_features = {**sequence_features, **msa_features, **template_features}
    return raw_features


def a3m_to_features(a3m_string_list):
    """MSA to af2 processed features."""
    msas = []
    deletion_matrices = []
    for a3m_str in a3m_string_list:
        msa, deletion_matrix = parsers.parse_a3m(a3m_str)
        msas.append(msa)
        deletion_matrices.append(deletion_matrix)

    input_sequence = msas[0][0]
    input_description = "xxx"   # TODO: description missing
    num_res = len(input_sequence)
    sequence_features = make_sequence_features(
            sequence=input_sequence,
            description=input_description,
            num_res=num_res)

    msa_features = make_msa_features(
            msas=msas,
            deletion_matrices=deletion_matrices)

    raw_features = {**sequence_features, **msa_features}
    return raw_features


def aatype_to_sequence(aatype):
    return ''.join([
        residue_constants.restypes_with_x[aatype[i]] 
        for i in range(len(aatype))
    ])


def load_pdb_chain(distill_obj, confidence_threshold=0.5):
    """Load pdb label."""

    pdb_label = {}
    aatype = distill_obj.aatype
    sequence = aatype_to_sequence(aatype)
    order_map = residue_constants.restype_order_with_x
    aatype_idx = np.array([order_map.get(rn, order_map['X']) for rn in sequence])
    all_atom_positions = distill_obj.atom_positions
    all_atom_mask = distill_obj.atom_mask

    pdb_label["aatype_index"] = aatype_idx
    pdb_label["all_atom_positions"] = all_atom_positions.astype(np.float32)
    pdb_label["all_atom_mask"] = all_atom_mask.astype(np.float32)
    pdb_label["resolution"] = np.array([0.]).astype(np.float32)

    high_confidence = distill_obj.b_factors > confidence_threshold
    high_confidence = np.any(high_confidence, axis=-1)
    for i, confident in enumerate(high_confidence):
        if(not confident):
            pdb_label["all_atom_mask"][i] = 0

    return pdb_label


def load_chain(mmcif_obj, chain_id='A'):
    """Load chain info."""
    all_atom_positions, all_atom_mask = get_atom_positions(mmcif_obj, chain_id, max_ca_ca_distance=float('inf'))
    # Directly parses sequence from fasta, should be consistent to 'aatype' in input features (from .fasta or .pkl)
    sequence = mmcif_obj.chain_to_seqres[chain_id]           
    order_map = residue_constants.restype_order_with_x
    aatype_idx = np.array([order_map.get(rn, order_map['X']) for rn in sequence], dtype=np.int32)
    resolution = np.array([mmcif_obj.header['resolution']], dtype=np.float32)
    return {
        'aatype_index':       aatype_idx,           # [NR,]
        'all_atom_positions': all_atom_positions,   # [NR, 37, 3]
        'all_atom_mask':      all_atom_mask,        # [NR, 37]
        'resolution':         resolution            # [,]
    }


def shape_list(x):
    """Return list of dimensions of an array."""
    x = np.array(x)

    if x.ndim is None:
        return x.shape
    static = x.shape
    ret = []
    for _, dim in enumerate(static):
        ret.append(dim)
    return ret


def one_hot(depth, indices):
    """tbd."""
    res = np.eye(depth)[indices.reshape(-1)]
    return res.reshape(list(indices.shape) + [depth])


def shaped_categorical(probs):
    """tbd."""
    ds = shape_list(probs)
    num_classes = ds[-1]
    probs = np.reshape(probs, (-1, num_classes))
    nums = list(range(num_classes))
    counts = []
    for prob in probs:
        counts.append(np.random.choice(nums, p=prob))

    return np.reshape(np.array(counts, np.int32), ds[:-1])


def generate_seq_mask(protein):
    """Generate seq mask."""
    protein['seq_mask'] = np.ones(shape_list(protein['aatype_index']),
                                dtype=np.float32)
    return protein


def generate_template_mask(protein):
    """Generate template mask."""
    protein['template_mask'] = np.ones(shape_list(protein['template_domain_names']),
                                dtype=np.float32)
    return protein


def generate_hhblits_profile(protein):
    """Compute the HHblits MSA profile if not already present."""
    if 'hhblits_profile' in protein:
        return protein
    # Compute the profile for every residue (over all MSA sequences).
    protein['hhblits_profile'] = np.mean(one_hot(22, protein['msa']), axis=0)
    return protein


def sample_raw_msa(raw_features, data_config):
    """
    related key in raw_features:
        msa: (msa_depth, seq_len)
        deletion_matrix_int: (msa_depth, seq_len)
    """
    D = len(raw_features['msa'])
    new_depth = np.random.randint(
            data_config.msa_sample.min_depth, 
            data_config.msa_sample.max_depth + 1)
    new_depth = min(new_depth, D)
    if new_depth == 1:
        indices = np.array([0])
    else:
        indices = np.random.choice(D - 1, size=[new_depth - 1], replace=False)
        indices = np.append(0, indices + 1).astype('int')

    for k in ['msa', 'deletion_matrix_int']:
        raw_features[k] = raw_features[k][indices]
    raw_features['num_alignments'][:] = new_depth
    return raw_features


def generate_backbone_affine(prot):
    """Generate backbone affine label."""
    n, ca, c = [residue_constants.atom_order[a] for a in ('N', 'CA', 'C')]
    rot, trans = quat_affine.make_transform_from_reference_np(
            n_xyz=prot['all_atom_positions'][:, n],
            ca_xyz=prot['all_atom_positions'][:, ca],
            c_xyz=prot['all_atom_positions'][:, c])

    backbone_affine_mask = (
            prot['all_atom_mask'][..., n] *
            prot['all_atom_mask'][..., ca] *
            prot['all_atom_mask'][..., c])
    prot['backbone_affine_tensor_rot'] = rot
    prot['backbone_affine_tensor_trans'] = trans
    prot['backbone_affine_mask'] = backbone_affine_mask
    return prot


def generate_rigidgroups(prot):
    """Generate rigid groups label."""
    rigidgroups = all_atom.atom37_to_frames(
        paddle.to_tensor(prot['aatype_index'][None]),          # (B, N)
        paddle.to_tensor(prot['all_atom_positions'][None]),    # (B, N, 37, 3)
        paddle.to_tensor(prot['all_atom_mask'][None]))         # (B, N, 37)
    rigidgroups = {k: v.squeeze(0).numpy() for k, v in rigidgroups.items()}
    prot.update(rigidgroups)
    return prot


def generate_torsion_angles(prot):
    """Generate torsion angles label."""

    torsion_angles_dict = all_atom.atom37_to_torsion_angles(
        aatype=paddle.to_tensor(prot['aatype_index'][None, None]),
        all_atom_pos=paddle.to_tensor(prot['all_atom_positions'][None, None], 'float32'),
        all_atom_mask=paddle.to_tensor(prot['all_atom_mask'][None, None], 'float32'),
        placeholder_for_undefined=True)
    torsion_angles_dict = {k: v.squeeze([0, 1]).numpy() for k, v in torsion_angles_dict.items()}
    #prot.update(torsion_angles_dict)
    chi_angles_sin_cos = torsion_angles_dict['torsion_angles_sin_cos'][:, 3:, :]  # [N, 7, 2] -> [N, 4, 2]
    chi_mask = torsion_angles_dict['torsion_angles_mask'][:, 3:]                  # [N, 7]    -> [N, 4]
    prot['chi_angles_sin_cos'] = chi_angles_sin_cos
    prot['chi_mask'] = chi_mask
    return prot


def generate_atom14_positions(prot):
    """Change key 'aatype' to 'aatype_index'."""
    restype_atom14_to_atom37 = []  # mapping (restype, atom14) --> atom37
    restype_atom37_to_atom14 = []  # mapping (restype, atom37) --> atom14
    restype_atom14_mask = []

    for rt in residue_constants.restypes:
        atom_names = residue_constants.restype_name_to_atom14_names[
            residue_constants.restype_1to3[rt]]

        restype_atom14_to_atom37.append([
            (residue_constants.atom_order[name] if name else 0)
            for name in atom_names
        ])

        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append([
            (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
            for name in residue_constants.atom_types
        ])

        restype_atom14_mask.append([(1. if name else 0.) for name in atom_names])

    # Add dummy mapping for restype 'UNK'.
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.] * 14)

    restype_atom14_to_atom37 = np.array(restype_atom14_to_atom37, dtype=np.int32)
    restype_atom37_to_atom14 = np.array(restype_atom37_to_atom14, dtype=np.int32)
    restype_atom14_mask = np.array(restype_atom14_mask, dtype=np.float32)

    # Create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein.
    residx_atom14_to_atom37 = restype_atom14_to_atom37[prot['aatype_index']]
    residx_atom14_mask = restype_atom14_mask[prot['aatype_index']]

    # Create a mask for known ground truth positions.
    residx_atom14_gt_mask = residx_atom14_mask * np.take_along_axis(
        prot["all_atom_mask"], residx_atom14_to_atom37, axis=1).astype(np.float32)

    # Gather the ground truth positions.
    residx_atom14_gt_positions = residx_atom14_gt_mask[:, :, None] * (
        np.take_along_axis(prot["all_atom_positions"],
                            residx_atom14_to_atom37[..., None],
                            axis=1))

    prot["atom14_atom_exists"] = residx_atom14_mask
    prot["atom14_gt_exists"] = residx_atom14_gt_mask
    prot["atom14_gt_positions"] = residx_atom14_gt_positions

    prot["residx_atom14_to_atom37"] = residx_atom14_to_atom37

    # Create the gather indices for mapping back.
    residx_atom37_to_atom14 = restype_atom37_to_atom14[prot['aatype_index']]
    prot["residx_atom37_to_atom14"] = residx_atom37_to_atom14

    # Create the corresponding mask.
    restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
    for restype, restype_letter in enumerate(residue_constants.restypes):
        restype_name = residue_constants.restype_1to3[restype_letter]
        atom_names = residue_constants.residue_atoms[restype_name]
        for atom_name in atom_names:
          atom_type = residue_constants.atom_order[atom_name]
          restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[prot['aatype_index']]
    prot["atom37_atom_exists"] = residx_atom37_mask

    # As the atom naming is ambiguous for 7 of the 20 amino acids, provide
    # alternative ground truth coordinates where the naming is swapped
    restype_3 = [
        residue_constants.restype_1to3[res] for res in residue_constants.restypes
    ]
    restype_3 += ["UNK"]

    # Matrices for renaming ambiguous atoms.
    all_matrices = {res: np.eye(14, dtype=np.float32) for res in restype_3}
    for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
        correspondences = np.arange(14)
        for source_atom_swap, target_atom_swap in swap.items():
          source_index = residue_constants.restype_name_to_atom14_names[
            resname].index(source_atom_swap)
          target_index = residue_constants.restype_name_to_atom14_names[
            resname].index(target_atom_swap)
          correspondences[source_index] = target_index
          correspondences[target_index] = source_index
          renaming_matrix = np.zeros((14, 14), dtype=np.float32)
          for index, correspondence in enumerate(correspondences):
            renaming_matrix[index, correspondence] = 1.
        all_matrices[resname] = renaming_matrix.astype(np.float32)
    renaming_matrices = np.stack([all_matrices[restype] for restype in restype_3])

    # Pick the transformation matrices for the given residue sequence
    # shape (num_res, 14, 14).
    renaming_transform = renaming_matrices[prot['aatype_index']]

    # Apply it to the ground truth positions. shape (num_res, 14, 3).
    alternative_gt_positions = np.einsum("rac,rab->rbc",
                                        residx_atom14_gt_positions,
                                        renaming_transform)
    prot["atom14_alt_gt_positions"] = alternative_gt_positions

    # Create the mask for the alternative ground truth (differs from the
    # ground truth mask, if only one of the atoms in an ambiguous pair has a
    # ground truth position).
    alternative_gt_mask = np.einsum("ra,rab->rb",
                                    residx_atom14_gt_mask,
                                    renaming_transform)

    prot["atom14_alt_gt_exists"] = alternative_gt_mask

    # Create an ambiguous atoms mask.  shape: (21, 14).
    restype_atom14_is_ambiguous = np.zeros((21, 14), dtype=np.float32)
    for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
        for atom_name1, atom_name2 in swap.items():
          restype = residue_constants.restype_order[
            residue_constants.restype_3to1[resname]]
          atom_idx1 = residue_constants.restype_name_to_atom14_names[resname].index(
            atom_name1)
          atom_idx2 = residue_constants.restype_name_to_atom14_names[resname].index(
            atom_name2)
          restype_atom14_is_ambiguous[restype, atom_idx1] = 1
          restype_atom14_is_ambiguous[restype, atom_idx2] = 1

    # From this create an ambiguous_mask for the given sequence.
    prot["atom14_atom_is_ambiguous"] = (
        restype_atom14_is_ambiguous[prot['aatype_index']])

    return prot


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    """Create pseudo beta features."""
    is_gly = np.equal(aatype, residue_constants.restype_order['G'])
    ca_idx = residue_constants.atom_order['CA']
    cb_idx = residue_constants.atom_order['CB']
    pseudo_beta = np.where(
        np.tile(is_gly[..., None].astype("int32"),
                [1,] * len(is_gly.shape) + [3,]).astype("bool"),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :])

    if all_atom_masks is not None:
        pseudo_beta_mask = np.where(is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
        pseudo_beta_mask = pseudo_beta_mask.astype(np.float32)
        return pseudo_beta, pseudo_beta_mask

    return pseudo_beta


def generate_masked_msa(protein, replace_fraction=0.15):
    """Create data for BERT on raw MSA."""
    # Add a random amino acid uniformly.
    random_aa = np.array([0.05] * 20 + [0., 0.], dtype=np.float32)

    categorical_probs = 0.1 * random_aa + \
        0.1 * protein['hhblits_profile'] + \
        0.1 * one_hot(22, protein['msa'])
    # Put all remaining probability on [MASK] which is a new column.
    pad_shapes = [[0, 0] for _ in range(len(categorical_probs.shape))]
    pad_shapes[-1][1] = 1
    mask_prob = 0.7
    assert mask_prob >= 0.

    categorical_probs = np.pad(categorical_probs, pad_shapes,
                               constant_values=(mask_prob,))

    mask_position = np.random.uniform(size=shape_list(protein['msa']),
                                      low=0, high=1) < 0.15

    bert_msa = shaped_categorical(categorical_probs)
    bert_msa = np.where(mask_position, bert_msa, protein['msa'])

    protein['bert_mask'] = mask_position.astype(np.int32)
    protein['true_msa'] = protein['msa']
    protein['msa'] = bert_msa

    return protein


def curry1(f):
  """Supply all arguments but the first."""
  def fc(*args, **kwargs):
    return lambda x: f(x, *args, **kwargs)

  return fc


def generate_pseudo_beta(protein):
    """Create pseudo-beta (alpha for glycine) position and mask."""
    pseudo_beta, pseudo_beta_mask = pseudo_beta_fn(
        protein['aatype_index'],
        protein['all_atom_positions'],
        protein['all_atom_mask'])

    protein['pseudo_beta'] = pseudo_beta
    protein['pseudo_beta_mask'] = pseudo_beta_mask
    return protein


@curry1
def compose(x, fs):
  """tbd."""
  for f in fs:    
      x = f(x)
  return x


def generate_label(protein_chain):
  """Generate label from chain atom37 positions."""
  label_process = [
    generate_seq_mask,
    generate_atom14_positions,
    generate_backbone_affine,
    generate_rigidgroups,
    generate_torsion_angles,
    generate_pseudo_beta,
    # generate_template_mask,
    # generate_hhblits_profile,
    # generate_masked_msa,
  ]

  protein_label = compose(label_process)(protein_chain)
  return protein_label
