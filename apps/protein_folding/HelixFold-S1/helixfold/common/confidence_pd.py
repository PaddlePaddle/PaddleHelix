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

"""Functions for processing confidence metrics."""

from typing import Dict, Optional, Tuple
import numpy as np
import paddle


def compute_plddt(logits: paddle.Tensor) -> paddle.Tensor:
  """Computes per-residue pLDDT from logits.

  Args:
    logits: [num_res, num_bins] output from the PredictedLDDTHead.

  Returns:
    plddt: [num_res] per-residue pLDDT.
  """
  num_bins = logits.shape[-1]
  bin_width = 1.0 / num_bins
  bin_centers = paddle.arange(0.5 * bin_width, 1.0, bin_width)
  probs = paddle.nn.functional.softmax(logits, axis=-1)
  predicted_lddt_ca = paddle.sum(probs * bin_centers[None, :], axis=-1)
  return predicted_lddt_ca * 100


def compute_chain_plddt(atom_plddt, perm_asym_id):
  """
  Args:
    atom_plddt: [num_atoms] atom-level pLDDT values.
    perm_asym_id: [num_atoms] atom-level chain IDs.
  Returns:
    chain_plddt: [num_chains] per-chain pLDDT.
  """
  uniq_asym_ids = paddle.unique(perm_asym_id)
  chain_plddt = []
  for idx, asym_id in enumerate(uniq_asym_ids):
    chain_mask = (perm_asym_id == asym_id)
    chain_plddt.append(paddle.mean(atom_plddt[chain_mask]))
  return uniq_asym_ids, paddle.stack(chain_plddt)


def predicted_prob_to_value(logits, breaks):
  """Convert predicted probability into value
  Args:
    logits: (*, num_bins)
    breaks: [num_bins - 1]
  Returns:
    value: (*,)
  """
  prob = paddle.nn.functional.softmax(logits, axis=-1)
  bin_centers = _calculate_bin_centers(breaks)
  value = paddle.sum(prob * bin_centers, axis=-1)
  return value


def compute_chain_inter_pde(logits, breaks, asym_id):
  """
  Args:
    logits: (num_res, num_res, num_bins)
    breaks: [num_bins - 1]
    asym_id: [num_res]
  Returns:
    chain_inter_pde: [num_chains]
  """
  pde_value = predicted_prob_to_value(logits, breaks)
  uniq_asym_ids = paddle.unique(asym_id)
  chain_inter_pde = []
  if uniq_asym_ids.shape[0] < 2: # return nan for single-chain samples where inter_pde is not applicable.
    return uniq_asym_ids, paddle.stack([paddle.to_tensor(float('nan'))])
  for aid in uniq_asym_ids:
    pair_mask = (asym_id == aid)[:, None] * (asym_id != aid)[None]
    pair_mask = paddle.logical_or(pair_mask, pair_mask.T)
    chain_inter_pde.append(paddle.mean(pde_value[pair_mask]))
  return uniq_asym_ids, paddle.stack(chain_inter_pde)


def _calculate_bin_centers(breaks: np.ndarray):
  """Gets the bin centers from the bin edges.

  Args:
    breaks: [num_bins - 1] the error bin edges.

  Returns:
    bin_centers: [num_bins] the error bin centers.
  """
  step = float((breaks[1] - breaks[0]).astype('float32'))

  # Add half-step to get the center
  bin_centers = breaks + step / 2
  # Add a catch-all bin at the end.
  bin_centers = paddle.concat([bin_centers, # [num_bins -1] 
        bin_centers[-1] + step if len(bin_centers[-1].shape) > 0 # for paddle2.*
        else bin_centers[-1].unsqueeze(0) + step  # for paddle3
        ], axis=0)
  return bin_centers


def _calculate_expected_aligned_error(
    alignment_confidence_breaks: paddle.Tensor,
    aligned_distance_error_probs: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
  """Calculates expected aligned distance errors for every pair of residues.

  Args:
    alignment_confidence_breaks: [num_bins - 1] the error bin edges.
    aligned_distance_error_probs: [num_res, num_res, num_bins] the predicted
      probs for each error bin, for each pair of residues.

  Returns:
    predicted_aligned_error: [num_res, num_res] the expected aligned distance
      error for each pair of residues.
    max_predicted_aligned_error: The maximum predicted error possible.
  """
  bin_centers = _calculate_bin_centers(alignment_confidence_breaks)

  # Tuple of expected aligned distance error and max possible error.
  return (paddle.sum(aligned_distance_error_probs * bin_centers, axis=-1),
          paddle.to_tensor(bin_centers[-1]))


def compute_predicted_aligned_error(
    logits: paddle.Tensor,
    breaks: paddle.Tensor) -> Dict[str, paddle.Tensor]:
  """Computes aligned confidence metrics from logits.

  Args:
    logits: [num_res, num_res, num_bins] the logits output from
      PredictedAlignedErrorHead.
    breaks: [num_bins - 1] the error bin edges.

  Returns:
    aligned_confidence_probs: [num_res, num_res, num_bins] the predicted
      aligned error probabilities over bins for each residue pair.
    predicted_aligned_error: [num_res, num_res] the expected aligned distance
      error for each pair of residues.
    max_predicted_aligned_error: The maximum predicted error possible.
  """
  aligned_confidence_probs = paddle.nn.functional.softmax(
      logits,
      axis=-1)
  predicted_aligned_error, max_predicted_aligned_error = (
      _calculate_expected_aligned_error(
          alignment_confidence_breaks=breaks,
          aligned_distance_error_probs=aligned_confidence_probs))
  return {
      'aligned_confidence_probs': aligned_confidence_probs,
      'predicted_aligned_error': predicted_aligned_error,
      'max_predicted_aligned_error': max_predicted_aligned_error,
  }


def predicted_tm_score(
    logits: paddle.Tensor,
    breaks: paddle.Tensor,
    residue_weights: Optional[paddle.Tensor] = None,
    asym_id: Optional[paddle.Tensor] = None,
    interface: bool = False) -> paddle.Tensor:
  """Computes predicted TM alignment or predicted interface TM alignment score.

  Args:
    logits: [num_res, num_res, num_bins] the logits output from
      PredictedAlignedErrorHead.
    breaks: [num_bins] the error bins.
    residue_weights: [num_res] the per residue weights to use for the
      expectation.
    asym_id: [num_res] the asymmetric unit ID - the chain ID. Only needed for
      ipTM calculation, i.e. when interface=True.
    interface: If True, interface predicted TM score is computed.

  Returns:
    ptm_score: The predicted TM alignment or the predicted iTM score.
  """

  # residue_weights has to be in [0, 1], but can be floating-point, i.e. the
  # exp. resolved head's probability.
  if residue_weights is None:
    residue_weights = paddle.ones(logits.shape[0])

  bin_centers = _calculate_bin_centers(breaks)

  num_res = int(paddle.sum(residue_weights))
  # Clip num_res to avoid negative/undefined d0.
  clipped_num_res = max(num_res, 19)

  # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
  # "Scoring function for automated assessment of protein structure template
  # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
  d0 = 1.24 * (clipped_num_res - 15) ** (1./3) - 1.8

  # Convert logits to probs.
  probs = paddle.nn.functional.softmax(logits, axis=-1)
  # TM-Score term for every bin.
  tm_per_bin = 1. / (1 + paddle.square(bin_centers) / np.square(d0))
  # E_distances tm(distance).
  predicted_tm_term = paddle.sum(probs * tm_per_bin, axis=-1)

  pair_mask = paddle.ones_like(predicted_tm_term, dtype=bool)
  if interface:
    pair_mask *= asym_id[:, None] != asym_id[None, :]

  predicted_tm_term *= pair_mask.astype(predicted_tm_term.dtype)

  pair_residue_weights = (pair_mask.astype(residue_weights.dtype) * (
      residue_weights[None, :] * residue_weights[:, None])).astype(predicted_tm_term.dtype)
  normed_residue_mask = pair_residue_weights / (1e-8 + paddle.sum(
      pair_residue_weights, axis=-1, keepdim=True))
  per_alignment = paddle.sum(predicted_tm_term * normed_residue_mask, axis=-1)

  valid_per_alignment = per_alignment * residue_weights.astype(per_alignment.dtype)
  return {
      "score": per_alignment[valid_per_alignment.argmax()], 
      "token_score": valid_per_alignment,
  } 


def predicted_chain_pair_iptm(
    logits: paddle.Tensor,
    breaks: paddle.Tensor,
    residue_weights: Optional[paddle.Tensor] = None,
    asym_id: Optional[paddle.Tensor] = None) -> paddle.Tensor:
  """compute chain pair ipTM score
  Args:
    logits: [num_res, num_res, num_bins] the logits output from
      PredictedAlignedErrorHead.
    breaks: [num_bins] the error bins.
    residue_weights: [num_res] the per residue weights to use for the
      expectation.
    asym_id: [num_res] the asymmetric unit ID - the chain ID. 

  Returns:
    chain_pair_asym_ids: [N_chain]
    chain_pair_iptm: [N_chain, N_chain]
  """
  uniq_asym_ids = paddle.unique(asym_id)
  n_chains = len(uniq_asym_ids)
  chain_pair_iptm = paddle.zeros((n_chains, n_chains), 'float32')
  chain_pair_mask = paddle.ones((n_chains, n_chains), 'float32')
  for i, asym_a in enumerate(uniq_asym_ids):
    for j, asym_b in enumerate(uniq_asym_ids):
      flag = paddle.logical_or(asym_id == asym_a, asym_id == asym_b)
      cur_logits = logits[flag]
      cur_logits = cur_logits.transpose([1, 0, 2])[flag]
      cur_logits = cur_logits.transpose([1, 0, 2])
      cur_weight = residue_weights[flag]
      cur_asym = asym_id[flag]
      if paddle.sum(cur_weight) == 0:
        chain_pair_mask[i, j] = 0
        continue
      if asym_a == asym_b:
        score = predicted_tm_score(
            cur_logits, breaks, cur_weight, cur_asym, 
            interface=False)['score']
      else:
        score = predicted_tm_score(
            cur_logits, breaks, cur_weight, cur_asym,
            interface=True)['score']
      chain_pair_iptm[i, j] = score
  return {
    'chain_pair_asym_ids': uniq_asym_ids,
    'chain_pair_iptm': chain_pair_iptm,
    'chain_pair_mask': chain_pair_mask,
  }


def get_has_clash(atom_pos, atom_mask, asym_id, is_polymer_chain):
    """
    A structure is marked as having a clash (has_clash) if for any two
    polymer chains A,B in the prediction clashes(A,B) > 100 or 
    clashes(A,B) / min(NA,NB) > 0.5 where NA is the number of atoms in 
    chain A.
    Args:
        atom_pos: [N_atom, 3]
        atom_mask: [N_atom]
        asym_id: [N_atom]
        is_polymer_chain: [N_atom]
    """
    flag = paddle.logical_and(atom_mask == 1, is_polymer_chain == 1)
    atom_pos = atom_pos[flag]
    asym_id = asym_id[flag]
    uniq_asym_ids = paddle.unique(asym_id)
    n = len(uniq_asym_ids)
    if n == 1:
        return 0
    for idx1, aid1 in enumerate(uniq_asym_ids[:-1]):
        for idx2, aid2 in enumerate(uniq_asym_ids[idx1 + 1:]):
            pos1 = atom_pos[asym_id == aid1]
            pos2 = atom_pos[asym_id == aid2]
            dist = paddle.sqrt(paddle.sum((pos1[None] - pos2[:, None]) ** 2, -1))
            n_clash = paddle.sum(dist < 1.1).astype('float32')
            if n_clash > 100 or n_clash / min(len(pos1), len(pos2)) > 0.5:
                return 1
    return 0
  

def get_has_clash_token_level(atom_pos, atom_mask, asym_id, token2atom, is_polymer_chain):
    """
    A structure is marked as having a clash (has_clash) if for any two
    polymer chains A,B in the prediction clashes(A,B) > 100 or 
    clashes(A,B) / min(NA,NB) > 0.5 where NA is the number of atoms in 
    chain A.
    Args:
        atom_pos: [N_atom, 3]
        atom_mask: [N_atom]
        token2atom: [N_atom], mapping from token to atom
        asym_id: [N_atom]
        is_polymer_chain: [N_atom]
    Returns:
        token_clash: [N_token]
        chain_clash: [N_chain, N_chain]
    """

    def _map_to_continuous_indices(arr):
        """ 
          Map index array to continuous indices.
        """
        if arr.shape[0] <= 1:
            return arr
        if not isinstance(arr, paddle.Tensor):
            arr = paddle.to_tensor(arr)
        diffs = arr[1:] - arr[:-1]
        assert paddle.all(diffs >= 0), \
            f"Not an ascending array: position where diff < 0 found."
        _, inverse_indices = paddle.unique(arr, return_inverse=True)
        
        return inverse_indices

    atom_pos = atom_pos[atom_mask == 1]
    asym_id = asym_id[atom_mask == 1]
    is_polymer_chain = is_polymer_chain[atom_mask == 1]
    token2atom = _map_to_continuous_indices(token2atom[atom_mask == 1])

    uniq_asym_ids = paddle.unique(asym_id)
    n_chain = len(uniq_asym_ids)
    n_token = int(paddle.max(token2atom).numpy()) + 1
    chain_clash = paddle.zeros((n_chain, n_chain), 'int32')
    token_clash = paddle.zeros(n_token, 'int32')
    atom_clash = paddle.zeros(len(atom_pos), 'int32')
    if n_chain == 1:
        return {'token_has_clash': token_clash, 
                'chain_has_clash': chain_clash}

    for idx1, aid1 in enumerate(uniq_asym_ids[:-1]):
        for idx2, aid2 in enumerate(uniq_asym_ids[idx1 + 1:]):
            is_poly_1 = is_polymer_chain[asym_id == aid1]
            is_poly_2 = is_polymer_chain[asym_id == aid2]
            if not paddle.any(is_poly_1) or not paddle.any(is_poly_2):
                continue
            pos1 = atom_pos[asym_id == aid1]
            pos2 = atom_pos[asym_id == aid2]
            dist = paddle.sqrt(paddle.sum((pos2[None] - pos1[:, None]) ** 2, -1))
            clash_mask = dist < 1.1
            n_clash = paddle.sum(clash_mask).astype('float32')
            if n_clash > 100 or n_clash / min(len(pos1), len(pos2)) > 0.5:
                chain_clash[idx1, idx1 + 1 + idx2] = 1
                chain_clash[idx1 + 1 + idx2, idx1] = 1

            # Mark atoms involved in clashes
            indices1 = paddle.arange(len(atom_pos))[asym_id == aid1]
            indices2 = paddle.arange(len(atom_pos))[asym_id == aid2]
            clash_indices1, clash_indices2 = paddle.where(clash_mask)
            if len(clash_indices1) > 0:
                atom_clash[indices1[clash_indices1]] = 1
                atom_clash[indices2[clash_indices2]] = 1 

    with paddle.amp.auto_cast(enable=False):
      # FIXME: paddle.geometric.segment_sum is not support bfloat16 now (3.0.0dev20241231)
      ac = atom_clash.astype('float32')
      token_clash = paddle.geometric.segment_sum(ac, token2atom)
      token_clash = token_clash.astype('bfloat16')
    
    token_clash = paddle.clip(token_clash, 0, 1)
    return {
      'token_has_clash': token_clash, 
      'chain_has_clash': chain_clash}