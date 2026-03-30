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
import scipy.special


def compute_plddt(logits: np.ndarray) -> np.ndarray:
  """Computes per-residue pLDDT from logits.

  Args:
    logits: [num_res, num_bins] output from the PredictedLDDTHead.

  Returns:
    plddt: [num_res] per-residue pLDDT.
  """
  num_bins = logits.shape[-1]
  bin_width = 1.0 / num_bins
  bin_centers = np.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)
  probs = scipy.special.softmax(logits, axis=-1)
  predicted_lddt_ca = np.sum(probs * bin_centers[None, :], axis=-1)
  return predicted_lddt_ca * 100


def compute_chain_plddt(atom_plddt, perm_asym_id):
  """
  Args:
    atom_plddt: [num_atoms] atom-level pLDDT values.
    perm_asym_id: [num_atoms] atom-level chain IDs.
  Returns:
    chain_plddt: [num_chains] per-chain pLDDT.
  """
  uniq_asym_ids = np.unique(perm_asym_id)
  chain_plddt = []
  for idx, asym_id in enumerate(uniq_asym_ids):
    chain_mask = (perm_asym_id == asym_id)
    chain_plddt.append(np.mean(atom_plddt[chain_mask]))
  return uniq_asym_ids, np.array(chain_plddt)


def predicted_prob_to_value(logits, breaks):
  """Convert predicted probability into value
  Args:
    logits: (*, num_bins)
    breaks: [num_bins - 1]
  Returns:
    value: (*,)
  """
  prob = scipy.special.softmax(logits, axis=-1)
  bin_centers = _calculate_bin_centers(breaks)
  value = np.sum(prob * bin_centers, axis=-1)
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
  uniq_asym_ids = np.unique(asym_id)
  chain_inter_pde = []
  for aid in uniq_asym_ids:
    pair_mask = (asym_id == aid)[:, None] * (asym_id != aid)[None]
    pair_mask = np.logical_or(pair_mask, pair_mask.T)
    chain_inter_pde.append(np.mean(pde_value[pair_mask]))
  return uniq_asym_ids, np.array(chain_inter_pde)


def _calculate_bin_centers(breaks: np.ndarray):
  """Gets the bin centers from the bin edges.

  Args:
    breaks: [num_bins - 1] the error bin edges.

  Returns:
    bin_centers: [num_bins] the error bin centers.
  """
  step = (breaks[1] - breaks[0])

  # Add half-step to get the center
  bin_centers = breaks + step / 2
  # Add a catch-all bin at the end.
  bin_centers = np.concatenate([bin_centers, [bin_centers[-1] + step]],
                               axis=0)
  return bin_centers


def _calculate_expected_aligned_error(
    alignment_confidence_breaks: np.ndarray,
    aligned_distance_error_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
  return (np.sum(aligned_distance_error_probs * bin_centers, axis=-1),
          np.asarray(bin_centers[-1]))


def compute_predicted_aligned_error(
    logits: np.ndarray,
    breaks: np.ndarray) -> Dict[str, np.ndarray]:
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
  aligned_confidence_probs = scipy.special.softmax(
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
    logits: np.ndarray,
    breaks: np.ndarray,
    residue_weights: Optional[np.ndarray] = None,
    asym_id: Optional[np.ndarray] = None,
    interface: bool = False,
    pairwise_residue_weights: Optional[np.ndarray] = None,
    aggregation_type: str = 'max') -> np.ndarray:
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
    pairwise_residue_weights: [num_res, num_res] the per residue weights to use for the
      expectation.
    aggregation_type: The aggregation method for aggregating among alignments. Choices
      include 'max', 'mean'. Default is 'max'.

  Returns:
    ptm_score: The predicted TM alignment or the predicted iTM score.
  """

  # residue_weights has to be in [0, 1], but can be floating-point, i.e. the
  # exp. resolved head's probability.
  if residue_weights is None:
    residue_weights = np.ones(logits.shape[0])
  if pairwise_residue_weights is None:
    pairwise_residue_weights = np.ones((logits.shape[0], logits.shape[0]))

  bin_centers = _calculate_bin_centers(breaks)

  num_res = int(np.sum(residue_weights))
  # Clip num_res to avoid negative/undefined d0.
  clipped_num_res = max(num_res, 19)

  # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
  # "Scoring function for automated assessment of protein structure template
  # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
  d0 = 1.24 * (clipped_num_res - 15) ** (1./3) - 1.8

  # Convert logits to probs.
  probs = scipy.special.softmax(logits, axis=-1)

  # TM-Score term for every bin.
  tm_per_bin = 1. / (1 + np.square(bin_centers) / np.square(d0))
  # E_distances tm(distance).
  predicted_tm_term = np.sum(probs * tm_per_bin, axis=-1)

  pair_mask = np.ones_like(predicted_tm_term, dtype=bool)
  if interface:
    pair_mask *= asym_id[:, None] != asym_id[None, :]

  predicted_tm_term *= pair_mask

  pair_residue_weights = pair_mask * (
      residue_weights[None, :] * residue_weights[:, None]) * pairwise_residue_weights
  normed_residue_mask = pair_residue_weights / (1e-8 + np.sum(
      pair_residue_weights, axis=-1, keepdims=True))
  
  if aggregation_type == 'max':
    per_alignment = np.sum(predicted_tm_term * normed_residue_mask, axis=-1)
    return np.asarray(per_alignment[(per_alignment * residue_weights).argmax()])
  elif aggregation_type == 'mean':
    return np.sum(predicted_tm_term * normed_residue_mask) / (1e-8 + np.sum(
        pair_residue_weights))
  else:
    raise ValueError('Unknown aggregation type {}'.format(aggregation_type))


def predicted_chain_pair_iptm(
    logits: np.ndarray,
    breaks: np.ndarray,
    residue_weights: Optional[np.ndarray] = None,
    asym_id: Optional[np.ndarray] = None) -> np.ndarray:
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
  uniq_asym_ids = np.unique(asym_id)
  n_chains = len(uniq_asym_ids)
  chain_pair_iptm = np.zeros((n_chains, n_chains), 'float32')
  chain_pair_mask = np.ones((n_chains, n_chains), 'float32')
  for i, asym_a in enumerate(uniq_asym_ids):
    for j, asym_b in enumerate(uniq_asym_ids):
      flag = np.logical_or(asym_id == asym_a, asym_id == asym_b)
      cur_logits = logits[flag][:, flag]
      cur_weight = residue_weights[flag]
      cur_asym = asym_id[flag]
      if np.sum(cur_weight) == 0:
        chain_pair_mask[i, j] = 0
        continue
      if asym_a == asym_b:
        score = predicted_tm_score(
            cur_logits, breaks, cur_weight, cur_asym, 
            interface=False)
      else:
        score = predicted_tm_score(
            cur_logits, breaks, cur_weight, cur_asym,
            interface=True)
      chain_pair_iptm[i, j] = score
  return {
    'chain_pair_asym_ids': uniq_asym_ids,
    'chain_pair_iptm': chain_pair_iptm,
    'chain_pair_mask': chain_pair_mask,
  }


def get_actifPTM(result, use_contact_map_prob=True, contact_dist=8.0, interface_dist=5.0) -> dict:
    """
    This function return the actifpTM (actual interface pTM) score for whole complex.

    Args:
        results: The result from HelixFold. should contain:
          - asym_id: unique chain ids for whole complex. (N_token, )
          - predicted_aligned_error: predicted aligned error logits and breaks.
          - distogram: distogram logits
          
        use_contact_map_prob: If True, calculate actifpTM based on contact probabilities. 
                Otherwise, calculate actifpTM based on interface mask. Default is True.
        contact_dist: The distance threshold for contact map probability. Default is 8.0.
        interface_dist: The distance threshold for interface mask. Default is 5.0.
    Returns:
        actifpTM score.
    """

    def _get_dgram_bins(result):
        """calculate bin boundaries of distogram"""
        dgram = result["distogram"]["logits"]
        if dgram.shape[-1] == 64:
            dgram_bins = np.append(0,np.linspace(2.3125,21.6875,63))
        if dgram.shape[-1] == 39:
            dgram_bins = np.linspace(3.25,50.75,39) + 1.25
        return dgram_bins

    def _get_chain_indices(asym_id):
        """Returns a list of tuples indicating the start and end indices for each chain."""

        chain_starts_ends = []
        unique_chains = np.unique(asym_id) # chains are numbered 0, 1, 2, ...

        for chain in unique_chains:
            positions = np.where(asym_id == chain)[0]
            chain_starts_ends.append((chain, positions[0], positions[-1]))

        return chain_starts_ends

    def _np_softmax(x, axis=-1):
        # 为了数值稳定性，减去每行的最大值
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def _get_contact_map(result, dist=8.0):
        """get contact map from distogram, probability of contact"""
        dist_logits = result["distogram"]["logits"]
        dist_bins =  np.append(0, result["distogram"]['bin_edges'])
        return np.sum(_np_softmax(dist_logits) * (dist_bins < dist), axis=-1)
  
    def _get_interface_mask(result, dist=5.0):
        """get binary mask for residue in interface between two chains"""
        asym_id = result['asym_id'] # (N_token, )
        ref_token2atom_idx = result['ref_token2atom_idx'].astype(int) # (N_atom, )
        atom_pos = result['final_atom_positions'] # (N_atom, 3)
        atom_mask = result['final_atom_mask'].astype(bool) # (N_atom, )

        pair_dist = np.linalg.norm(atom_pos[:, None] - atom_pos[None, :], axis=-1)
        interface_mask = np.where(pair_dist <= dist, 1, 0).astype(bool) # (N_atom, N_atom)
        interface_mask *= (atom_mask[:, None] * atom_mask[None, :]) # remove the invalide atom
        interface_mask *= (1 - np.eye(len(atom_pos))).astype(bool) # remove self-interaction (N_atom, N_atom)
        interface_mask *= (asym_id[ref_token2atom_idx][:, None] != asym_id[ref_token2atom_idx][None, :]) # (N_atom, N_atom)
        
        ## mapping interface mask to token level
        interface_indices = np.where(interface_mask)
        n_tokens = np.max(ref_token2atom_idx) + 1
        interface_mask_token = np.zeros((n_tokens, n_tokens))
        for i, (atom_idx, atom_jdx) in enumerate(zip(*interface_indices)):
            interface_mask_token[ref_token2atom_idx[atom_idx], ref_token2atom_idx[atom_jdx]] = 1

        return interface_mask_token


    logits_pae = result['logits_pae']
    breaks_pae = result['breaks_pae']
    residue_weights = result['frame_mask']
    asym_id = result['asym_id']

    return {
        'actifpTM': predicted_tm_score(
            logits=logits_pae,
            breaks=breaks_pae,
            asym_id=asym_id,
            interface=True,
            residue_weights=residue_weights,
            pairwise_residue_weights=_get_contact_map(result, contact_dist) \
                    if use_contact_map_prob else _get_interface_mask(result, interface_dist)
        )
    }
