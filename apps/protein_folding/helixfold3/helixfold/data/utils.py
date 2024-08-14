#   Copyright (c) 2024 PaddleHelix Authors. All Rights Reserved.
#
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# International License (the "License");  you may not use this file  except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for data."""

from typing import *
from absl import logging
import numpy as np
import pickle
import os
import time
import json
import gzip
from helixfold.data.pipeline import FeatureDict, DataPipeline

# macros for retrying getting queue items.
CROPPING_DOWN_SAMPLING_SIZE = 5_000


ignored_keys = [
    'domain_name',
    'sequence',
    'is_distillation',
    'template_domain_names',
    'template_e_value',
    'template_neff',
    'template_prob_true',
    'template_release_date',
    'template_score',
    'template_similarity',
    'template_sequence',
    'template_sum_probs',
    'seq_length',
    'msa_row_mask',
    'random_crop_to_size_seed',
    'extra_msa_row_mask',
    'resolution',
    'template_mask',
]

batched_keys = [
    'deletion_matrix_int',
    'msa',
    'msa_mask',
    'template_aatype',
    'template_all_atom_masks',
    'template_all_atom_positions',
    'template_confidence_scores',
    'extra_msa',
    'extra_msa_mask',
    'bert_mask',
    'true_msa',
    'extra_has_deletion',
    'extra_deletion_value',
    'msa_feat',
    'template_pseudo_beta',
    'template_pseudo_beta_mask',
    # extended for hf3
    'has_deletion', 'deletion_value', 
    'template_restype', 'template_pseudo_beta_mask',
    'template_backbone_frame_mask', 
    # extedned for hf3 online
    'template_distogram', 'template_unit_vector'
]


atom_level_keys = [
    'perm_entity_id', 'perm_asym_id', 'all_chain_ids', 'all_ccd_ids', 'all_atom_ids', 'perm_atom_index', 

    'ref_pos', 'ref_mask', 'ref_element', 'ref_charge', 'ref_atom_name_chars',
    'ref_space_uid', 'ref_token2atom_idx', 

    'label_ccd_ids', 'label_atom_ids', 'all_atom_pos',
    'all_atom_pos_mask',
]



def crop_spatial_all_atom(feat, label, list_n_k, crop_size, for_recycle, 
                          targeted_asym_ids=None, inf=3e4):
    """ 
    Crop spatial. 
    Randomly select target rediue from targeted chains
    """
    ca_coords = label['all_atom_pos'][label['all_centra_token_indice']]  # [N_token, 3]
    ca_mask = label['all_centra_token_indice_mask'].astype('bool')  # [N_token]
    mask = np.logical_and(np.isin(feat['asym_id'], targeted_asym_ids), ca_mask)
    if np.sum(mask) == 0:
        return crop_contiguous(list_n_k, crop_size), 'contiguous'
    target_coords = ca_coords[mask]
    center_coords = target_coords[np.random.randint(len(target_coords))]
    return crop_spatial_all_atom_by_center(
            feat, label, center_coords, crop_size)


def crop_spatial_inter_all_atom(feat, label, list_n_k, crop_size, for_recycle, 
                                targeted_asym_ids=None, ca_ca_threshold=15.0, inf=3e4):
    """ 
    Crop spatial interface.
    Select interface rediue from targeted chains
    """
    ca_coords = label['all_atom_pos'][label['all_centra_token_indice']]  # [N_token, 3]
    ca_mask = label['all_centra_token_indice_mask'].astype('bool')  # [N_token]
    asym_id = feat['asym_id']

    # down sample token
    num_token = label['all_centra_token_indice'].shape[0]
    if num_token > CROPPING_DOWN_SAMPLING_SIZE:
        down_sampling_indices = np.sort(np.random.choice(
            np.arange(num_token), CROPPING_DOWN_SAMPLING_SIZE, replace=False))
        ca_coords = ca_coords[down_sampling_indices]
        ca_mask = ca_mask[down_sampling_indices]
        asym_id = asym_id[down_sampling_indices]

    # if there are not enough atoms to construct interface, use contiguous crop
    if (ca_mask.sum(axis=-1) <= 1).all():
        # return crop_contiguous(list_n_k, crop_size)
        return crop_spatial_all_atom(feat, label, list_n_k, crop_size, for_recycle, inf=3e4)

    pair_mask = ca_mask[..., None] * ca_mask[..., None, :]
    # get_pairwise_distances
    coord_diff = np.expand_dims(ca_coords, -2) - np.expand_dims(ca_coords, -3)
    ca_distances = np.sqrt(np.sum(coord_diff**2, axis=-1))
    # get_interface_candidates
    in_same_asym = asym_id[..., None] == asym_id[..., None, :]
    ca_distances = ca_distances * (1.0 - in_same_asym.astype('float')) * pair_mask
    cnt_interfaces = np.sum((ca_distances > 0) & (ca_distances < ca_ca_threshold), axis=-1)
    if for_recycle: # [num_recycle, num_res]
        cnt_interfaces = cnt_interfaces[0]
    # idx of residue whose to-other-entitiy distance < ca_ca_threshold
    interface_candidates = cnt_interfaces.nonzero()[0]
    if not targeted_asym_ids is None:
        # keep interface_candidates from targeted(sampled) chains 
        interface_candidates = interface_candidates[
            np.isin(asym_id[interface_candidates], targeted_asym_ids)
        ]

    if np.any(interface_candidates):
        target_res = int(np.random.choice(interface_candidates))
    else:
        # return crop_contiguous(list_n_k, crop_size)
        return crop_spatial_all_atom(feat, label, list_n_k, crop_size, for_recycle, inf=3e4)

    # map down sampled target token back to full token
    if num_token > CROPPING_DOWN_SAMPLING_SIZE:
        target_res = down_sampling_indices[target_res]

    center_coords = label['all_atom_pos'][label['all_centra_token_indice']][target_res]
    ret, _ = crop_spatial_all_atom_by_center(
            feat, label, center_coords, crop_size)
    return ret, "crop_spatial_inter"


def crop_spatial_all_atom_by_center(feat, label, center_coords, crop_size):
    """ tbd. """
    ca_coords = label['all_atom_pos'][label['all_centra_token_indice']]  # [N_token, 3]
    ca_mask = label['all_centra_token_indice_mask'].astype('bool')  # [N_token]
    dists = np.sqrt(((center_coords[None] - ca_coords) ** 2).sum(-1))  # (N_token,)
    dists[~ca_mask] = float('Inf')
    indices = np.argsort(dists)[:crop_size]
    indices.sort()
    return indices, "crop_spatial"


def crop_contiguous(list_n_k, N_res):
    n_added = 0
    n_remain = np.sum(list_n_k)
    list_m_k = [np.zeros([n_k]) for n_k in list_n_k]
    chain_orders = np.random.permutation(len(list_n_k))
    for chain_i in chain_orders:
        n_k = list_n_k[chain_i]
        n_remain -= n_k
        # get crop range
        crop_size_max = min(N_res - n_added, n_k)
        crop_size_min = min(n_k, max(0, N_res - (n_added + n_remain)))
        crop_size = np.random.randint(crop_size_min, crop_size_max + 1)
        crop_start = np.random.randint(0, n_k - crop_size + 1)
        # update mask
        list_m_k[chain_i][crop_start: crop_start + crop_size] = 1
        n_added += crop_size
    crop_contiguous_idx = np.where(np.concatenate(list_m_k))[0]
    return crop_contiguous_idx



def map_to_continuous_indices(arr):
    """ 
    map index array to continous indices
    input: [3, 3, 3, 3, 74, 74, 74, ... , n-2, n-1, n-1, n, n, n]
    output: [0, 0, 0, 0, 1, 1, 1, ....., m-2, m-1, m-1, m, m, m]
    """
    if arr.shape[0] == 0: return 
    index_map = {arr[0]:0}
    counter_idx = 0
    for i in range(1, len(arr)):
        assert arr[i] >= arr[i-1], \
            f"not an ascending array at pos {i} i: {arr[i]} i-1: {arr[i-1]}"
        if not arr[i] == arr[i-1]:
            counter_idx += 1
            index_map[arr[i]] = counter_idx
    for i in range(len(arr)):
        arr[i] = index_map[arr[i]]
    return arr

def get_crop_mask_all_atom(
    crop_features: FeatureDict,
    seq_infos: Mapping,
    crop_size: int = 256,                      
    spatial_crop_ratio=0.4, 
    spatial_inter_crop_ratio=0.4, 
    targeted_asym_ids=None,
    max_atom_num=None) -> List:
    """ get cropping idx. """

    def _crop_idx_to_mask(crop_idx, chain_len_dict):
        seq_lens = list(chain_len_dict.values())
        crop_mask = np.zeros(sum(seq_lens), dtype=bool)
        crop_mask[crop_idx] = True
        chain_crop_dict = {}
        offset = 0
        for chain_id, seq_len in chain_len_dict.items():
            chain_crop_dict[chain_id] = crop_mask[offset: offset + seq_len]
            offset += seq_len
        return chain_crop_dict

    # token_size = crop_features["seq_token"]["restype"].shape[0]
    # atom_size = crop_features["seq_token"]["ref_pos"].shape[0]

    seq_lens = seq_infos['seq_lens']
    rand_drop = np.random.random()

    # map asym_id to chain_id
    asym_to_chain = {}
    for token_id, chain_id in zip(crop_features["conf_bond"]["ref_token2atom_idx"], 
                                  crop_features["seq_token"]['all_chain_ids']):
        asym_to_chain[crop_features["seq_token"]["asym_id"][token_id]] = chain_id

    raw_features = {
        'asym_id': crop_features["seq_token"]["asym_id"],
        'sym_id': crop_features["seq_token"]["sym_id"],
        'entity_id': crop_features["seq_token"]["entity_id"],
        'residue_index': crop_features["seq_token"]["residue_index"],
        'is_protein': crop_features["seq_token"]["is_protein"],
        'is_dna': crop_features["seq_token"]["is_dna"],
        'is_rna': crop_features["seq_token"]["is_rna"],
        'is_ligand': crop_features["seq_token"]["is_ligand"],
        'ref_token2atom_idx': crop_features["conf_bond"]["ref_token2atom_idx"],
    }
    raw_labels = crop_features['labels']

    if rand_drop < (1 - spatial_crop_ratio - spatial_inter_crop_ratio):
        crop_token_idx, crop_method = crop_contiguous(seq_lens, crop_size), 'contiguous'
    elif rand_drop > (1 - spatial_crop_ratio):
        crop_token_idx, crop_method = crop_spatial_all_atom(feat=raw_features, 
                                label=raw_labels, list_n_k=seq_lens, \
                                crop_size=crop_size, for_recycle=False, 
                                targeted_asym_ids=targeted_asym_ids)
    else:
        crop_token_idx, crop_method = crop_spatial_inter_all_atom(feat=raw_features, 
                        label=raw_labels, list_n_k=seq_lens, \
                        crop_size=crop_size, for_recycle=False, 
                        targeted_asym_ids=targeted_asym_ids)
    
    crop_token_idx = crop_token_idx.astype(int)

    # Redo a smaller cropping if it contains too many atoms
    estimated_atom_num = np.sum(raw_features['is_protein'][crop_token_idx] * 14 
            + raw_features['is_dna'][crop_token_idx] * 22 
            + raw_features['is_rna'][crop_token_idx] * 22 
            + raw_features['is_ligand'][crop_token_idx] * 1)
    if estimated_atom_num >= max_atom_num:
        # print('[Crop] redo cropping', estimated_atom_num)
        return get_crop_mask_all_atom(crop_features, 
                seq_infos, 
                int(crop_size * 0.8), 
                spatial_crop_ratio, 
                spatial_inter_crop_ratio, 
                targeted_asym_ids,
                max_atom_num)

    crop_idx_mask = _crop_idx_to_mask(crop_token_idx, 
        chain_len_dict=dict(zip(np.unique(raw_features["asym_id"]), seq_lens)))

    # crop_idx_mask = filter_invalid_chains(crop_idx_mask, raw_features)

    # map asym_id to chain
    return {asym_to_chain[k]: v for k, v in crop_idx_mask.items()}, crop_method


def generate_pkl_features_from_fasta(
        fasta_path: str,
        name: str,
        output_dir: str,
        data_pipeline: DataPipeline,
        timings: Optional[Dict[str, float]] = None,
        use_gzip: bool = False):
    """Generate features.pkl from FASTA sequence."""
    if timings is None:
        timings = {}

    # Check output dir.
    output_dir = os.path.join(output_dir, name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    msa_output_dir = os.path.join(output_dir, 'msas')
    if not os.path.exists(msa_output_dir):
        os.makedirs(msa_output_dir)

    # Get features.
    pt = time.time()
    logging.info(f"processing file {fasta_path}...")
    features = data_pipeline.process(
        input_fasta_path=fasta_path,
        msa_output_dir=msa_output_dir)
    timings['data_pipeline'] = time.time() - pt

    # Write out features as a pickled dictionary.
    if use_gzip:
        features_output_path = os.path.join(output_dir, 'features.pkl.gz')
        with gzip.open(features_output_path, 'wb') as f:
            pickle.dump(features, f, protocol=4)

    else:
        features_output_path = os.path.join(output_dir, 'features.pkl')
        with open(features_output_path, 'wb') as f:
            pickle.dump(features, f, protocol=4)

    logging.info(f"process file {fasta_path} done.")

    # Save timings.
    timings_output_path = os.path.join(output_dir, 'timings.json')
    with open(timings_output_path, 'w') as fp:
        json.dump(timings, fp, indent=4)

    return features



ignored_keys_multimer = ignored_keys + [
    'num_templates',
    'num_alignments',
    'assembly_num_chains',
    'cluster_bias_mask',
    'release_date',
    'resolution',
    'protein_num_templates',
    'protein_num_alignments',
    'protein_cluster_bias_mask',
    'ligand_num_templates',
    'ligand_num_alignments',
    'ligand_cluster_bias_mask',
    'rna_num_templates',
    'rna_num_alignments',
    'rna_cluster_bias_mask',
    'dna_num_templates',
    'dna_num_alignments',
    'dna_cluster_bias_mask',
]
batched_keys_multimer = batched_keys + [
    'deletion_matrix'
]

# keys that should be ignored when conducting crop & pad
def is_ignored_key_multimer(k):
    return k in ignored_keys_multimer

# keys that have batch dim, e.g. msa features which have shape [N_msa, N_res, ...]
def is_batched_key_multimer(k):
    return k in batched_keys_multimer

