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

"""Utils for training data."""
import math
from typing import *
from absl import logging
import numpy as np
import pickle
import pathlib
import pandas as pd
import os
import sys
import time
import json
import gzip
import traceback
from copy import deepcopy
from multiprocessing import Queue
from functools import reduce
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist, squareform

import paddle
from helixfold.common import residue_constants
from helixfold.common.residue_constants import restype_order_with_x
from helixfold.data.mmcif_parsing import MmcifObject
from helixfold.data.mmcif_parsing import parse as parse_mmcif_string
from helixfold.data.pipeline import FeatureDict, DataPipeline
from helixfold.data.templates import _get_atom_positions as get_atom_positions
from Bio.PDB import protein_letters_3to1
from helixfold.data import label_utils
from utils.interface_utils import InterfaceInfo

INT_MAX = 0x7fffffff

# macros for retrying getting queue items.
MAX_TIMEOUT = 60
MAX_FAILED = 5
CROPPING_DOWN_SAMPLING_SIZE = 5_000
LABEL_NEED_OFFSET_KEYS = label_utils.NEED_OFFSET_KEYS

def cif_to_fasta(mmcif_object: MmcifObject, chain_id: str) -> str:
    """mmcif to fasta."""
    residues = mmcif_object.seqres_to_structure[chain_id]
    residue_names = [residues[t].name for t in range(len(residues))]
    residue_letters = [protein_letters_3to1.get(n, 'X') for n in residue_names]
    filter_out_triple_letters = lambda x: x if len(x) == 1 else 'X'
    fasta_string = ''.join([filter_out_triple_letters(n) for n in residue_letters])
    return fasta_string

def load_features(path: str) -> FeatureDict:
    """Load features."""
    assert path.endswith('.pkl'), f"only pickle features supported, {path} provided."
    return pickle.load(open(path, 'rb'))

def load_labels(cif_path: str, pdb_id: str, chain_id: str = 'A') -> FeatureDict:
    """Load labels."""
    if cif_path.endswith('.gz'):
        with gzip.open(cif_path, 'rb') as f:
            cif_string = f.read().decode('utf-8')
    else:
        cif_string = open(cif_path, 'r').read()

    # parse cif string
    mmcif_obj = parse_mmcif_string(
        file_id=pdb_id, mmcif_string=cif_string).mmcif_object
    # fetch useful labels
    if mmcif_obj is not None:
        all_atom_positions, all_atom_mask = get_atom_positions(mmcif_obj, chain_id, max_ca_ca_distance=float('inf'))
        # directly parses sequence from fasta, should be consistent to 'aatype' in input features (from .fasta or .pkl)
        sequence = cif_to_fasta(mmcif_obj, chain_id)           
        aatype_idx = np.array([restype_order_with_x[rn] for rn in sequence])
        resolution = np.array([mmcif_obj.header['resolution']])

    return {
        'aatype_index':       aatype_idx,           # [NR,]
        'all_atom_positions': all_atom_positions,   # [NR, 37, 3]
        'all_atom_mask':      all_atom_mask,        # [NR, 37]
        'resolution':         resolution            # [,]
    }


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
]

batched_pair_keys = [
    'template_distogram', 'template_unit_vector'
]
pair_keys = [
    'token_bonds', 'covalent_bonds',
    # extedned for hf3 service
    'template_distogram', 'template_unit_vector',
    'interface_mask',
    'interface_info_sample', 'interface_info_full', 'interface_info_mix',
    'interface_info',
]

FRAME_PAE_REQUIRED_KEYS = ['frame_ai_indice', 'frame_bi_indice', 'frame_ci_indice']  # N_token

atom_level_keys = [
    'perm_entity_id', 'perm_asym_id', 'all_chain_ids', 'all_ccd_ids', 'all_atom_ids', 'perm_atom_index', 

    'ref_pos', 'ref_mask', 'ref_element', 'ref_charge', 'ref_atom_name_chars',
    'ref_space_uid', 'ref_token2atom_idx', 

    'label_ccd_ids', 'label_atom_ids', 'all_atom_pos',
    'all_atom_pos_mask',
]

label_keys = ['frame_mask', 'all_centra_token_indice', 'all_centra_token_indice_mask']

# keys that should be ignored when conducting crop & pad
def is_ignored_key(k):
    return k in ignored_keys

# keys that have batch dim, e.g. msa features which have shape [N_msa, N_res, ...]
def is_batched_key(k):
    return k in batched_keys


def align_feat(feat, size, is_hf3=False, n_query=32, align_atom=False):
    """ pad seq to ensure seq_len is divisable for size(dap_degree) """
    # get num res from aatype
    if is_hf3:
        assert 'restype' in feat.keys(), \
            "'restype' missing from batch, which is not expected."
        num_res = feat['restype'].shape[1]
    else:
        assert 'aatype' in feat.keys(), \
            "'aatype' missing from batch, which is not expected."
        num_res = feat['aatype'].shape[2]

    if num_res % size != 0:
        align_size = (num_res // size + 1) * size

        # pad short seq (0 padding and (automatically) create masks)
        def pad(key, array, start_axis, align_size, num_res):
            if is_ignored_key(key):
                return array
            if is_hf3 and key in atom_level_keys:
                return array
            d_seq = start_axis      # choose the dim to crop / pad
            if is_batched_key(key) or is_batched_key_multimer(key) \
                    or key in batched_pair_keys:
                d_seq += 1

            pad_shape = list(array.shape)
            pad_shape[d_seq] = align_size - num_res
            pad_array = paddle.zeros(pad_shape, dtype=array.dtype)
            array = paddle.concat([array, pad_array], axis=d_seq)

            if is_hf3 and key in pair_keys:
                d_seq += 1
                pad_shape = list(array.shape)
                pad_shape[d_seq] = align_size - num_res
                pad_array = paddle.zeros(pad_shape, dtype=array.dtype)
                array = paddle.concat([array, pad_array], axis=d_seq)
            return array

        s_axis = 2 if not is_hf3 else 1
        feat = {k: pad(k, v, s_axis, align_size, num_res) for k, v in feat.items()}
        if not is_hf3:
            feat['seq_length'] = (align_size * paddle.ones_like(feat['seq_length']))

    if is_hf3 and 'msa' in feat:
        array = feat['msa']
        pad_shape = list(array.shape)
        msa_depth = pad_shape[1]
        if msa_depth % size != 0:
            msa_align_size = (msa_depth // size + 1) * size
            pad_shape[1] = msa_align_size - msa_depth
            pad_array = paddle.zeros(pad_shape, dtype=array.dtype)
            feat['msa'] = paddle.concat([array, pad_array], axis=1)

        if align_atom:
          num_atom = feat['ref_pos'].shape[1]
          num_center = math.ceil(num_atom / n_query)
          if num_center % size != 0 or num_atom % size != 0:
            align_center_size = (num_center // size + 1) * size
            _align_atom_size = (align_center_size - 1) * n_query + 1
            align_atom_size = (_align_atom_size // size + 1) * size
            print('num_atom', num_atom, 'align_atom_size', align_atom_size)
            def pad_atom(key, array, start_axis, align_size, num_atom):
                if not key in atom_level_keys: return array
                if not num_atom in np.shape(array): return array  # filter out feat with no atom dim
                d_seq = start_axis      # choose the dim to crop / pad
                pad_shape = list(array.shape)
                assert pad_shape[d_seq] == num_atom, f"feat {key} 's atom_dim != 1: {pad_shape}"
                pad_shape[d_seq] = align_size - num_atom

                pad_array = paddle.zeros(pad_shape, dtype=array.dtype)
                if key == 'ref_token2atom_idx':  # pad with val of the last element
                    pad_array += array[-1][-1]
                array = paddle.concat([array, pad_array], axis=d_seq)    
                return array
            feat = {k: pad_atom(k, v, 1, align_atom_size, num_atom) for k, v in feat.items()}
    return feat


def align_label(label, size, n_query=32, align_atom=False):
    """Align label."""
    if align_atom:
      if 'all_atom_pos_mask' in label:
        num_atom = label['all_atom_pos_mask'].shape[1]
        num_center = math.ceil(num_atom / n_query)
        s_axis = 1
        if num_center % size != 0 or num_atom % size != 0:
            align_center_size = (num_center // size + 1) * size
            _align_atom_size = (align_center_size - 1) * n_query + 1
            align_atom_size = (_align_atom_size // size + 1) * size
            print('num_atom', num_atom, 'align_atom_size', align_atom_size)
            for k in label:
                array = label[k]
                if not k in atom_level_keys: continue
                if not num_atom in np.shape(array): continue # filter out feat with no atom dim
                pad_shape = list(array.shape)
                assert pad_shape[s_axis] == num_atom, f"feat {k} 's atom_dim != 1: {pad_shape}"
                pad_shape[s_axis] = align_atom_size - num_atom
                pad_array = paddle.zeros(pad_shape, dtype=array.dtype)
                array = paddle.concat([array, pad_array], axis=s_axis)
                label[k] = array

    num_res = label['asym_id'].shape[1]

    if num_res % size == 0:
        return label
    
    align_size = (num_res // size + 1) * size
    s_axis = 1
    for k in label_keys:
        array = label[k]
        pad_shape = list(array.shape)
        pad_shape[s_axis] = align_size - num_res
        pad_array = paddle.zeros(pad_shape, dtype=array.dtype)
        array = paddle.concat([array, pad_array], axis=s_axis)
        label[k] = array

    return label


def unpad_prediction(feat, pred, is_hf3=False):
    """ remove dap paddings"""
    unpad_pred = deepcopy(pred)
    if is_hf3:
        n = feat['restype'].shape[0]
    else:
        n = feat['aatype'].shape[0]

    if is_hf3:
        k0 = 'confidence_head'
        for k1 in ['logits_pae', 'logits_pde', 'pae']:
            unpad_pred[k0][k1] = pred[k0][k1][:, :, :n, :n]
    else:
        k1 = 'logits'

        k0 = 'distogram'
        unpad_pred[k0][k1] = pred[k0][k1][:, :n, :n]

        k0 = 'experimentally_resolved'
        unpad_pred[k0][k1] = pred[k0][k1][:, :n]

        k0 = 'masked_msa'
        unpad_pred[k0][k1] = pred[k0][k1][:, :, :n]

        k0 = 'predicted_lddt'
        unpad_pred[k0][k1] = pred[k0][k1][:, :n]

        k0 = 'structure_module'
        for k1 in pred[k0].keys():
            if k1.startswith('final_'):
                unpad_pred[k0][k1] = pred[k0][k1][:, :n]

            elif k1 == 'sidechains':
                for k2 in pred[k0][k1].keys():
                    unpad_pred[k0][k1][k2] = pred[k0][k1][k2][:, :, :n]

            elif k1 == 'traj':
                unpad_pred[k0][k1] = pred[k0][k1][:, :, :n]

        k0 = 'representations'
        if k0 in pred.keys():
            for k1 in pred[k0].keys():
                if k1 == 'pair':
                    unpad_pred[k0][k1] = pred[k0][k1][:, :n, :n]

                elif k1 == 'msa':
                    unpad_pred[k0][k1] = pred[k0][k1][:, :, :n]

                else:
                    unpad_pred[k0][k1] = pred[k0][k1][:, :n]

    return unpad_pred


def crop_and_pad(
    raw_features: FeatureDict,
    raw_labels: FeatureDict,
    crop_size: int = 256,
    pad_for_shorter_seq: bool = True,
    return_crop_range: bool = False) -> FeatureDict:
    """Cropping and padding."""

    # get num res from aatype
    assert 'aatype' in raw_features.keys(), \
        "'aatype' missing from batch, which is not expected."
    num_res = raw_features['aatype'].shape[1]

    crop_start, crop_end = 0, num_res
    if num_res < crop_size and pad_for_shorter_seq:
        # pad short seq (0 padding and (automatically) create masks)
        def pad(key: str, array: np.ndarray, start_axis: int):
            if is_ignored_key(key):
                return array
            d_seq = start_axis      # choose the dim to crop / pad
            if is_batched_key(key):
                d_seq += 1
            pad_shape = list(array.shape)
            pad_shape[d_seq] = crop_size - num_res
            pad_array = np.zeros(pad_shape)
            pad_array = pad_array.astype(array.dtype)
            array = np.concatenate([array, pad_array], axis=d_seq)
            return array
        raw_features = {k: pad(k, v, 1) for k, v in raw_features.items()}
        raw_labels = {k: pad(k, v, 0) for k, v in raw_labels.items()}
    elif num_res > crop_size:
        # crop long seq.
        crop_start = np.random.randint(num_res - crop_size)
        crop_end = crop_start + crop_size
        def crop(key: str, array: np.ndarray, start_axis: int):
            if is_ignored_key(key):
                return array
            d_seq = start_axis      # choose the dim to crop / pad
            if is_batched_key(key):
                d_seq += 1
            slices = [slice(None)] * len(array.shape)
            slices[d_seq] = slice(crop_start, crop_end)
            return array[tuple(slices)]
        raw_features = {k: crop(k, v, 1) for k, v in raw_features.items()}
        raw_labels = {k: crop(k, v, 0) for k, v in raw_labels.items()}
    else:
        # seq len == crop size
        pass

    # fix for input seq length
    raw_features['seq_length'] = (crop_size * np.ones_like(raw_features['seq_length'])).astype(np.int32)
    if return_crop_range:
        return raw_features, raw_labels, (crop_start, crop_end)
    return raw_features, raw_labels

def get_heterodimer_chains(feat, label, ca_ca_threshold=10.0, inf=3e4, prot_name=None, disable_random=False):
    """ select heterodimer chains"""
    for_recycle = len(feat['asym_id'].shape) == 2  #[for_recycle, num_res]
    ca_idx = residue_constants.atom_order['CA']
    ca_coords = label['all_atom_positions'][..., ca_idx, :]
    ca_mask = label['all_atom_mask'][..., ca_idx].astype('bool')

    pair_mask = ca_mask[..., None] * ca_mask[..., None, :]
    # get_pairwise_distances
    coord_diff = np.expand_dims(ca_coords, -2) - np.expand_dims(ca_coords, -3)
    ca_distances = np.sqrt(np.sum(coord_diff**2, axis=-1))
    # get_interface_candidates
    in_same_entity = feat['entity_id'][..., None] == feat['entity_id'][..., None, :]
    ca_distances = ca_distances * (1.0 - in_same_entity.astype('float')) * pair_mask
    cnt_interfaces = np.sum((ca_distances > 0) & (ca_distances < ca_ca_threshold), axis=-1)
    if for_recycle: # [num_recycle, num_res]
        cnt_interfaces = cnt_interfaces[0]
    # idx of residue whose to-other-entitiy distance < ca_ca_threshold
    interface_candidates = cnt_interfaces.nonzero()[0]

    if np.any(interface_candidates):
        if not disable_random:
            choice_aa = int(np.random.choice(interface_candidates))
        else:
            # choice_aa = int(interface_candidates[0]) # the first interface aa 
            choice_aa = int(cnt_interfaces.argmax(-1)) # skip_homo_fixed_v2: aa with most number of connected aa
        to_target_distances = ca_distances[..., choice_aa]  # distance to selected interface candidate(choice_aa)
        to_target_distances = np.where(to_target_distances == 0, inf, to_target_distances)
        match_aa = np.argsort(to_target_distances)[0]  # [num_res]
        feat_asym_id = feat['asym_id']
        if for_recycle: # [num_recycle, num_res]
            feat_asym_id = feat_asym_id[0]
        choice_asym_id, match_asym_id = feat_asym_id[choice_aa], feat_asym_id[match_aa]  # [1] [num_res]
        nearest_asym_id = match_asym_id[0]
        choice_aa_idx = np.where(feat_asym_id == choice_asym_id)[0]  # select res with asym_id of the selected chain
        match_aa_idx = np.where(feat_asym_id == nearest_asym_id)[0] # select res of the nearest chain
        all_idx = np.concatenate([choice_aa_idx, match_aa_idx]) # concat idx of the selected chain and its nearest chain
        all_idx.sort()

        def select(key: str, array: np.ndarray, start_axis: int):
            if is_ignored_key_multimer(key):
                return array
            d_seq = start_axis         # choose the dim to crop / pad
            if is_batched_key_multimer(key):
                d_seq += 1
            new_array = np.take(array, all_idx, axis=d_seq)
            return new_array

        start_axis = 1 if for_recycle else 0
        feat = {k: select(k, v, start_axis) for k, v in feat.items()}
        label = {k: select(k, v, 0) for k, v in label.items()}

        if for_recycle:
            # items in for_recycle are repeatedly the same. 
            # num_recycle == 1 when data.common.num_recycle == 0 and model.num_recycle == 0
            recycle_idx = 0  
            print(f"chains {set(feat['asym_id'][recycle_idx])} selected as heterodimer in {prot_name}")
            feat['seq_length'][:] = feat['aatype'].shape[1] # [num_recycle, num_res]
            # assign 1 to the first asym_id/entity_id and 2 to the rest
            feat['asym_id'][:] = np.where(feat['asym_id'][recycle_idx] == \
                                          feat['asym_id'][recycle_idx][0], 1, 2) # [num_recycle, num_res] 
            feat['entity_id'][:] = np.where(feat['entity_id'][recycle_idx] == feat['entity_id'][recycle_idx][0], 1, 2) 
        else:
            print(f"chains {set(feat['asym_id'])} selected as heterodimer")
            feat['seq_length'] = feat['aatype'].shape[0]
            feat['asym_id'] = np.where(feat['asym_id'] == feat['asym_id'][0], 1, 2)
            feat['entity_id'] = np.where(feat['entity_id'] == feat['entity_id'][0], 1, 2)
        label['asym_id'] = np.where(label['asym_id'] == label['asym_id'][0], 1, 2)
        label['entity_id'] = np.where(label['entity_id'] == label['entity_id'][0], 1, 2)
        feat['sym_id'].fill(0)
        label['sym_id'].fill(0)

        return feat, label

    return None, None


def add_G_linker(feat, label, seq_infos=None):
    def _insert(array, offset, length, fill_value, axis):
        shape = list(array.shape)
        shape[axis] = shape[axis] + length
        new_array = np.full(shape, fill_value, array.dtype)
        if axis == 0:
            new_array[:offset] = array[:offset]
            new_array[offset + length:] = array[offset:]
        elif axis == 1:
            new_array[:, :offset] = array[:, :offset]
            new_array[:, offset + length:] = array[:, offset:]
        else:
            raise ValueError(axis)
        return new_array

    asym_id = feat['asym_id']
    assert len(np.unique(asym_id)) == 2, feat['asym_id']
    G_offset = np.where(asym_id == 2)[0][0]
    G_length = 30
    raw_seq_len = len(asym_id)

    new_feat = {}
    ## insert 0
    for k, v in feat.items():
        if isinstance(v, np.ndarray) and raw_seq_len in v.shape:
            new_feat[k] = _insert(v, G_offset, G_length, 0, axis=v.shape.index(raw_seq_len))
        else:
            new_feat[k] = v
    ## special keys
    new_feat['aatype'][G_offset: G_offset + G_length] = residue_constants.restype_order_with_x['G']
    new_feat['residue_index'] = np.arange(raw_seq_len + G_length)
    new_feat['seq_length'] = raw_seq_len + G_length
    new_feat['asym_id'][:] = 1
    new_feat['sym_id'][:] = 1
    new_feat['entity_id'][:] = 1

    new_label = {}
    ## insert 0
    for k, v in label.items():
        if isinstance(v, np.ndarray) and raw_seq_len in v.shape:
            new_label[k] = _insert(v, G_offset, G_length, 0, axis=v.shape.index(raw_seq_len))
        else:
            new_label[k] = v
    ## special keys
    for k in ['aatype', 'seq_length', 'asym_id', 'sym_id', 'entity_id']:
        if k in label:
            new_label[k] = deepcopy(new_feat[k])

    ## new_seq_infos
    new_seq_infos = {}
    if not seq_infos is None:
        new_seq_infos['seq_lens'] = [raw_seq_len + G_length]
        new_seq_infos['chain_ids'] = seq_infos['chain_ids'][0:1]

    ## G_mask
    G_mask = np.ones([raw_seq_len])
    G_mask = _insert(G_mask, G_offset, G_length, 0, axis=0)
    return new_feat, new_label, new_seq_infos, G_mask


def get_G_linker_fasta_info(fasta_info, chain_ids, G_len, G_type):
    assert len(chain_ids) == 2, "G_linker only support two chains so far"
    seq_A = fasta_info[chain_ids[0]]['seq']
    seq_B = fasta_info[chain_ids[1]]['seq']
    desc_A = fasta_info[chain_ids[0]]['desc']
    desc_B = fasta_info[chain_ids[1]]['desc']
    new_fasta_info = {
        chain_ids[0]: {
            'seq': seq_A + G_type * G_len + seq_B,
            'desc': desc_A + '>>G-linker<<' + desc_B
        }
    }

    ## for evaluation
    len_A, len_B = len(seq_A), len(seq_B)
    eval_info = {
        'chain_ids': chain_ids,
        'mask': np.array([1] * len_A + [0] * G_len + [1] * len_B, 'int64'),
        'asym_id': np.array([1] * len_A + [2] * len_B, 'int64'),
        'residue_index': np.append(np.arange(len_A), np.arange(len_B)),
    }
    return new_fasta_info, eval_info


def get_G_linker_chain_dict(G_len, G_type):
    order_map = residue_constants.restype_order_with_x
    aatype_idx = np.array([order_map[G_type]] * G_len, dtype=np.int32)
    all_atom_positions = np.zeros([G_len, 37, 3], 'float32')
    all_atom_mask = np.zeros([G_len, 37], 'int64')
    resolution = np.array([0], 'float32')
    return {
        'aatype_index':       aatype_idx,           # [NR,]
        'all_atom_positions': all_atom_positions,   # [NR, 37, 3]
        'all_atom_mask':      all_atom_mask,        # [NR, 37]
        'resolution':         resolution            # [,]
    }


def crop_spatial(feat, label, list_n_k, crop_size, for_recycle, ca_ca_threshold=10.0, inf=3e4):
    """ tbd. """
    ca_idx = residue_constants.atom_order['CA']
    ca_coords = label['all_atom_positions'][..., ca_idx, :]
    ca_mask = label['all_atom_mask'][..., ca_idx].astype('bool')

    # if there are not enough atoms to construct interface, use contiguous crop
    if (ca_mask.sum(axis=-1) <= 1).all():
        return crop_contiguous(list_n_k, crop_size)

    pair_mask = ca_mask[..., None] * ca_mask[..., None, :]
    # get_pairwise_distances
    coord_diff = np.expand_dims(ca_coords, -2) - np.expand_dims(ca_coords, -3)
    ca_distances = np.sqrt(np.sum(coord_diff**2, axis=-1))
    # get_interface_candidates
    in_same_asym = feat['asym_id'][..., None] == feat['asym_id'][..., None, :]
    ca_distances = ca_distances * (1.0 - in_same_asym.astype('float')) * pair_mask
    cnt_interfaces = np.sum((ca_distances > 0) & (ca_distances < ca_ca_threshold), axis=-1)
    if for_recycle: # [num_recycle, num_res]
        cnt_interfaces = cnt_interfaces[0]
    # idx of residue whose to-other-entitiy distance < ca_ca_threshold
    interface_candidates = cnt_interfaces.nonzero()[0]

    print("num interface_candidates:", interface_candidates.shape, "has interface_candidates:", \
         np.any(interface_candidates))
    if np.any(interface_candidates):
        target_res = int(np.random.choice(interface_candidates))
    else:
        return crop_contiguous(list_n_k, crop_size)

    if for_recycle: # [num_recycle, num_res, num_res]
        ca_distances = ca_distances[0]
    to_target_distances = ca_distances[target_res]
    to_target_distances[~ca_mask] = inf
    break_tie = (np.arange(0, to_target_distances.shape[-1]).astype('float') * 1e-3)
    to_target_distances += break_tie
    ret = np.argsort(to_target_distances)[:crop_size]
    ret.sort()
    assert len(ret.shape) == 1, f"crop_idx.shape {ret.shape} is invalid"
    return ret


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


def crop_spatial_ranking(feat, label, list_n_k, crop_size, for_recycle, ca_ca_threshold=10.0, inf=3e4):
    if not for_recycle:
        chain_asym_ids = np.unique(feat['asym_id'])
        chains_indices = {chain_asym_id: np.where(feat['asym_id']==chain_asym_id)[0] for chain_asym_id in chain_asym_ids}
    else:
        chain_asym_ids = np.unique(feat['asym_id'][0])
        chains_indices = {chain_asym_id: np.where(feat['asym_id'][0]==chain_asym_id)[0] for chain_asym_id in chain_asym_ids}

    protein_asym_id = 1
    pos_peptide_asym_id = 2
    neg_peptide_asym_id = 3

    if pos_peptide_asym_id not in chain_asym_ids:
        return np.concatenate((crop_contiguous(list_n_k, crop_size-len(chains_indices[neg_peptide_asym_id])), chains_indices[neg_peptide_asym_id]), axis=0)

    crop_idx = np.concatenate((chains_indices[protein_asym_id], chains_indices[pos_peptide_asym_id]))

    def crop(key: str, array: np.ndarray, indices: np.ndarray, start_axis: int):
        if len(array.shape) <= start_axis:
            return array
        if is_ignored_key_multimer(key):
            return array
        # choose the dim to crop / pad
        d_seq = start_axis
        if is_batched_key_multimer(key):
            d_seq += 1
        new_array = np.take(array, indices, axis=d_seq)
        return new_array

    start_axis = 1 if for_recycle else 0
    cropped_feat = {k: crop(k, v, crop_idx, start_axis) for k, v in feat.items()}
    cropped_label = {k: crop(k, v, crop_idx, 0) for k, v in label.items()}

    # cropped_feat['seq_length'] = cropped_feat['aatype'][0].shape[0] if for_recycle else cropped_feat['aatype'].shape[0]
    ca_idx = residue_constants.atom_order['CA']
    ca_coords = cropped_label['all_atom_positions'][..., ca_idx, :]
    ca_mask = cropped_label['all_atom_mask'][..., ca_idx].astype('bool')

    # if there are not enough atoms to construct interface, use contiguous crop
    if (ca_mask.sum(axis=-1) <= 1).all():
        return crop_contiguous(list_n_k, crop_size)

    pair_mask = ca_mask[..., None] * ca_mask[..., None, :]
    # get_pairwise_distances
    coord_diff = np.expand_dims(ca_coords, -2) - np.expand_dims(ca_coords, -3)
    ca_distances = np.sqrt(np.sum(coord_diff**2, axis=-1))
    # get_interface_candidates
    in_same_asym = cropped_feat['asym_id'][0][..., None] == cropped_feat['asym_id'][0][..., None, :]
    ca_distances = ca_distances * (1.0 - in_same_asym.astype('float')) * pair_mask
    cnt_interfaces = np.sum((ca_distances > 0) & (ca_distances < ca_ca_threshold), axis=-1)
    # indices for all interface residue
    interface_candidates = cnt_interfaces.nonzero()[0]

    if np.any(interface_candidates):
        target_res = int(np.random.choice(interface_candidates))
    else:
        return crop_contiguous(list_n_k, crop_size)

    to_target_distances = ca_distances[target_res]
    to_target_distances[~ca_mask] = inf
    break_tie = (np.arange(0, to_target_distances.shape[-1]).astype('float') * 1e-3)
    to_target_distances += break_tie

    # under ranking model, we only crop the protein sequence,
    # which is the longest over all the sequences,
    # assuming all other sequences are peptides instead of long proteins of length [1,40]

    pos_peptide_len = len(chains_indices[pos_peptide_asym_id]) if pos_peptide_asym_id in chain_asym_ids else 0
    neg_peptide_len = len(chains_indices[neg_peptide_asym_id]) if neg_peptide_asym_id in chain_asym_ids else 0

    pos_peptide_indices = chains_indices[pos_peptide_asym_id] if pos_peptide_asym_id in chain_asym_ids else []
    neg_peptide_indices = chains_indices[neg_peptide_asym_id] if neg_peptide_asym_id in chain_asym_ids else []

    ret = np.argsort(to_target_distances[chains_indices[protein_asym_id]])[:crop_size-pos_peptide_len-neg_peptide_len]
    ret.sort()

    return np.concatenate((ret, pos_peptide_indices, neg_peptide_indices))


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

def remove_idx_from_chain_w_min_res(crop_idx, asym_id, min_res=1):
    """
    remove crop_idx for subchain whose num_res <= min_res.
    """
    asym_id_shape = asym_id.shape
    crop_idx_shape = crop_idx.shape
    assert len(crop_idx_shape) == 1, f"crop_idx.shape {crop_idx_shape} not valid, suppose to be equals to [num_res]"

    if len(asym_id_shape) == 2:  # [batch, num_res]
        cropped_asym_id = np.take(asym_id, crop_idx, 1)[0] # [cropped_size]
    elif len(asym_id_shape) == 1:  # [num_res]
        cropped_asym_id = np.take(asym_id, crop_idx, 0) # [cropped_size]
    else: assert asym_id_shape in [1, 2], f"asym_id.shape == {asym_id_shape} not supported"
    
    # find index of cropped chains with num_res <= min_res 
    num_elements = []
    drop_idx = []
    for pos, chain_id in enumerate(cropped_asym_id):
        if pos == 0: num_elements.append(1)
        else:
            if chain_id == cropped_asym_id[pos - 1]:
                # res from same chain as previous
                num_elements.append(num_elements[pos - 1 ] + 1)
            else:
                # res from new chain
                num_elements.append(1)
                # num res in previous chain < min_res
                if num_elements[pos - 1] <= min_res: drop_idx.append(pos - 1)
    
    return np.delete(crop_idx, drop_idx)

def select_from_samped_chain_center(all_chain_info_dict, sampled_chain_ids,
            max_cropping_token, assembly_mmcif_object, ccd_preprocessed_dict):
    '''
    select chains neighbouring to the center of sampled chains by center distance
    '''
    # 1. load chain atom pos
    chain_pos = {}
    for chain_id in all_chain_info_dict:
        try:
            chain_label = label_utils.load_chain(
                    assembly_mmcif_object,
                    ccd_preprocessed_dict,
                    chain_id)
        except Exception as exception:
            print(f"[Error] get position label failed! {exception}")
            traceback.print_exc()
            continue
        if chain_label is None: 
            continue
        chain_pos[chain_id] = chain_label['all_atom_pos']

    # 2. compute center atom pos of sampled chains
    sampled_chain_pos = np.concatenate(
        [chain_pos[c] for c in sampled_chain_ids if c in chain_pos], axis=0
    )
    sampled_chain_center = np.mean(sampled_chain_pos, axis=0)

    # 3. get chain center distance from sampled_chain_center
    chain_center_dist = {}
    def get_distance(a, b):
        coord_diff = a - b
        pair_distances = np.sqrt(np.sum(coord_diff ** 2, axis=-1))
        return pair_distances
    for chain_id in all_chain_info_dict:
        if chain_id in sampled_chain_ids: continue
        if not chain_id in chain_pos: continue
        chain_center = np.mean(chain_pos[chain_id], 0)
        chain_center_dist[chain_id] = get_distance(
            chain_center, sampled_chain_center
        )

    # 4. sort chain_ids by center distance
    chain_ids_ranked_by_dist =  sorted(chain_center_dist, key=chain_center_dist.get)

    # 5. select nearest chains within length
    final_chain_ids = []
    cum_num = 0
    for chain_id in sampled_chain_ids + chain_ids_ranked_by_dist:
        n_token = all_chain_info_dict[chain_id]['n_token']
        if cum_num + n_token < max_cropping_token:
            final_chain_ids.append(chain_id)
            cum_num += n_token
    return final_chain_ids


def remove_incomplete_ligands(crop_idx, asym_id, is_protein, ligand_id=4):
    '''
    remove incomplete ligand chains
    '''
    # get complete length of each ligand chain
    non_polymer_asym_id = asym_id[is_protein == ligand_id]
    non_polymer_complete_len = {}
    for asym in np.unique(non_polymer_asym_id):
        non_polymer_complete_len[asym] = len(non_polymer_asym_id[non_polymer_asym_id == asym])
    
    keep_idx = np.ones_like(crop_idx)
    # check cropped length for each ligand chain
    asym_id_cropped = asym_id[crop_idx]  # [crop_size,]
    for asym in non_polymer_complete_len:
        complete_len = non_polymer_complete_len[asym]
        chain_asym_id_cropped = asym_id_cropped[asym_id_cropped == asym]
        # print(chain_asym_id_cropped, len(chain_asym_id_cropped), complete_len)
        if not len(chain_asym_id_cropped) == complete_len:
            # remove non_polymer if cropped_len != complete_len
            # set indices related to incomplete ligand to False
            keep_idx[asym_id_cropped == asym] = 0

    return crop_idx[keep_idx.astype(bool)] # extract polymers and complete non-polymers


def _frame_feats_valid_check(feats):
    atom_nums = feats['all_atom_pos'].shape[0] # N_atom
    for key in FRAME_PAE_REQUIRED_KEYS:
        assert np.max(feats[key]) < atom_nums
    
    token_length = [len(feats['frame_mask'])]
    for key in FRAME_PAE_REQUIRED_KEYS:
        token_length.append(len(feats[key]))
    assert len(set(token_length)) == 1


def map_frame_absolute_indice(absolute_indice, frame_mask, keep_indices):
    """
        absolute_indice: N_token, `cropped`, absolute idx, to indexes the N_atom features.
        frame_mask: N_token, `cropped`, mask, 1 if the token is the valid frame, 0 otherwise.
        keep_indices: N_atom, `cropped`, keep indexes;
        Return: 
            new_indices, N_token, `cropped` absolute idx, to indexes the `cropped` N_atom features.
            new_frame_mask, N_token, `cropped` mask, 1 if the token is the valid frame, 0 otherwise.
    """
    assert absolute_indice.shape[0] == frame_mask.shape[0]
    index_map = {old_index: new_index for new_index, old_index in enumerate(keep_indices)}
    new_frame_mask = []
    new_indices = []
    for i, idx in enumerate(absolute_indice):
        if (frame_mask[i] == 0) or (not idx in index_map):
            new_indices.append(0)
            new_frame_mask.append(0)
            continue
        new_indices.append(index_map[idx])
        new_frame_mask.append(1)

    return np.array(new_indices), np.array(new_frame_mask)


def map_frame_indice_and_mask(crop_absolute_indices: dict, 
                                crop_frame_mask, crop_keep_indices, crop_size):
    ## crop_absolute_indices: # N_token, all is `cropped`.        
    ## crop_size：for padding;       
    new_frame_mapping = {k:[] for k in FRAME_PAE_REQUIRED_KEYS}
    new_frame_mask_mapping = {k:[] for k in FRAME_PAE_REQUIRED_KEYS}
    for key, val in crop_absolute_indices.items():
        _indices, _mask = map_frame_absolute_indice(val, crop_frame_mask, crop_keep_indices)
        new_frame_mapping[key] = _indices
        new_frame_mask_mapping[key] = _mask

    new_mask = None
    for k in FRAME_PAE_REQUIRED_KEYS:
        _nmask = new_frame_mask_mapping[k]
        if new_mask is None:
            new_mask = _nmask
        else:
            new_mask = new_mask & _nmask
    
    ## padding
    if len(new_mask) < crop_size:
        _pad_zeros = np.zeros([crop_size - len(new_mask)])
        new_mask = np.concatenate([new_mask, _pad_zeros])
        for k in FRAME_PAE_REQUIRED_KEYS:
            _pad_zeros = np.zeros([crop_size - len(new_mask)])
            new_frame_mapping[k] = np.concatenate([new_frame_mapping[k], _pad_zeros])
        
    return {**new_frame_mapping, 'frame_mask': new_mask}

def get_relative_token_centra(ref_token2atom_idx, centra_token_indice):
    """
    convert absolute centra token indices to token-relative centra indices
    """
    token_centra = {}
    for centra_idx in centra_token_indice:
        token_idx = ref_token2atom_idx[centra_idx]
        token_centra[token_idx] = np.count_nonzero(
            ref_token2atom_idx[:centra_idx] == token_idx
        )
    return token_centra

def map_to_absolute_token_centra(ref_token2atom_idx, token_centra, crop_size):
    """
    map relative token_centra to absolute indices
    """
    abs_token_centra = [0]
    token_set = set()
    for pos, token_idx in enumerate(ref_token2atom_idx):
        if not token_idx in token_set:
            token_set.add(token_idx)
            abs_token_centra.append(
                pos + token_centra[token_idx]
            )
    abs_token_centra = abs_token_centra[1:]
    if len(abs_token_centra) < crop_size:  # padding
        abs_token_centra += [0] * (crop_size - len(abs_token_centra))

    return np.array(abs_token_centra)

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

def filter_invalid_chains(crop_mask, feat):
    """ remove chains with invalid number of tokens. """
    offset = 0
    for key in crop_mask:
        chain_len = crop_mask[key].shape[0]
        is_protein = np.unique(feat['is_protein'][offset: offset + chain_len])
        is_dna = np.unique(feat['is_dna'][offset: offset + chain_len])
        is_rna = np.unique(feat['is_rna'][offset: offset + chain_len])
        is_ligand = np.unique(feat['is_ligand'][offset: offset + chain_len])

        offset += chain_len
        assert len(is_protein) == 1, f"invalid is_protein in chain {key} {is_protein}"
        assert len(is_dna) == 1, f"invalid is_dna in chain {key} {is_dna}"
        assert len(is_rna) == 1, f"invalid is_rna in chain {key} {is_rna}"
        assert len(is_ligand) == 1, f"invalid is_ligand in chain {key} {is_ligand}"

        is_protein = is_protein[0]
        is_dna = is_dna[0]
        is_rna = is_rna[0]
        is_ligand = is_ligand[0]
        if is_protein:  # remove chains with less than 4 res
            if sum(crop_mask[key]) < 4:
                crop_mask[key][crop_mask[key] == True] = False
        
        # TODO: ligand may be incomplete in label
        # if is_ligand:  # remove incomplete ligand
        #     if sum(crop_mask[key]) != chain_len:
        #         crop_mask[key][crop_mask[key] == True] = False
    return crop_mask


def get_crop_mask_all_atom(
    raw_features: FeatureDict,
    raw_labels: FeatureDict,
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

    seq_lens = seq_infos['seq_lens']
    rand_drop = np.random.random()

    # map asym_id to chain_id
    asym_to_chain = {}
    for token_id, chain_id in zip(raw_features["ref_token2atom_idx"], 
                                  raw_features['all_chain_ids']):
        asym_to_chain[raw_features["asym_id"][token_id]] = chain_id

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
        return get_crop_mask_all_atom(
                raw_features, 
                raw_labels,
                seq_infos, 
                int(crop_size * 0.8), 
                spatial_crop_ratio, 
                spatial_inter_crop_ratio, 
                targeted_asym_ids,
                max_atom_num)

    crop_idx_mask = _crop_idx_to_mask(crop_token_idx, 
        chain_len_dict=dict(zip(np.unique(raw_features["asym_id"]), seq_lens)))
    # map asym_id to chain
    return {asym_to_chain[k]: v for k, v in crop_idx_mask.items()}, crop_method


def get_crop_mask_all_atom_by_center(
    center_coords: np.ndarray,
    crop_features: FeatureDict,
    seq_infos: Mapping,
    crop_size: int = 256):                 
    """get_crop_mask_all_atom_by_center"""

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

    seq_lens = seq_infos['seq_lens']
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

    crop_token_idx, _ = crop_spatial_all_atom_by_center(
            raw_features, raw_labels, center_coords, crop_size)
    crop_token_idx = crop_token_idx.astype(int)

    crop_idx_mask = _crop_idx_to_mask(crop_token_idx, 
        chain_len_dict=dict(zip(np.unique(raw_features["asym_id"]), seq_lens)))

    # crop_idx_mask = filter_invalid_chains(crop_idx_mask, raw_features)

    # map asym_id to chain
    return {asym_to_chain[k]:v for k, v in crop_idx_mask.items()}


def apply_crop_mask_and_pad(
    raw_features: FeatureDict,
    raw_labels: FeatureDict,
    feat_crop_idx: np.array,
    label_crop_idx: np.array,
    crop_size: int,
    pad_for_shorter_seq: bool = True,
    for_recycle: bool = False) -> FeatureDict:
    """Cropping and padding."""
    # init size before cropping
    token_size = raw_features["restype"].shape[0]
    atom_size = raw_features["ref_pos"].shape[0]

    start_axis = 1 if for_recycle else 0
    
    def crop(key: str, array: np.ndarray, start_axis: int, crop_idx: np.array, crop_idx_atom: np.array):
        if is_ignored_key_multimer(key):
            return array
        if len(array.shape) <= start_axis:
            return array
        d_seq = start_axis         # choose the dim to crop / pad
        if is_batched_key_multimer(key):
            d_seq += 1
        if key in batched_pair_keys:
            d_seq += 1
        d_seq_size = array.shape[d_seq]
        if not key in atom_level_keys:
            new_array = np.take(array, crop_idx, axis=d_seq)
            if key in batched_pair_keys + pair_keys: # further cropping for paired axis
                new_array = np.take(new_array, crop_idx, axis=d_seq + 1)
        else:
            #  atom level cropping
            new_array = np.take(array, crop_idx_atom, axis=d_seq)
        assert d_seq_size in [token_size, atom_size], \
                f"d_seq size {d_seq_size} of {key} {array.shape}" \
                + f"not in [{token_size}, {atom_size}]"
        return new_array

    ref_atom2token_idx = raw_features['ref_token2atom_idx']  # [all_atom]
    relative_token_centra = get_relative_token_centra(ref_atom2token_idx, 
                                centra_token_indice=raw_labels['all_centra_token_indice'])
    feat_crop_idx_atom = np.arange(ref_atom2token_idx.shape[0])[
        np.isin(ref_atom2token_idx, feat_crop_idx)]
    label_crop_idx_atom = np.arange(ref_atom2token_idx.shape[0])[
        np.isin(ref_atom2token_idx, label_crop_idx)]
    
    new_features = {k: crop(k, v, start_axis, 
            feat_crop_idx, feat_crop_idx_atom) 
            for k, v in raw_features.items()}
    new_labels = {k: crop(k, v, 0, 
            label_crop_idx, label_crop_idx_atom) 
            for k, v in raw_labels.items()}
    num_res = new_features['restype'].shape[start_axis]

    # update size after cropping
    token_size = new_features["restype"].shape[0]
    atom_size = new_features["ref_pos"].shape[0]

    # TODO(zhukunrui): add atom padding if batch size > 1
    if num_res < crop_size and pad_for_shorter_seq: 
        # pad short seq (0 padding and (automatically) create masks)
        def pad(key: str, array: np.ndarray, start_axis: int):
            if is_ignored_key_multimer(key):
                return array
            if len(array.shape) <= start_axis:
                return array
            d_seq = start_axis      # choose the dim to crop / pad
            if is_batched_key_multimer(key):
                d_seq += 1
            if key in batched_pair_keys: 
                d_seq += 1
            d_seq_size = array.shape[d_seq]
            if not key in atom_level_keys:
                pad_shape = list(array.shape)
                pad_shape[d_seq] = crop_size - num_res
                pad_array = np.zeros(pad_shape)
                pad_array = pad_array.astype(array.dtype)
                array = np.concatenate([array, pad_array], axis=d_seq)
                if key in batched_pair_keys + pair_keys: # further padding for paired axis
                    pad_shape = list(array.shape)
                    pad_shape[d_seq + 1] = crop_size - num_res
                    pad_array = np.zeros(pad_shape)
                    pad_array = pad_array.astype(array.dtype)
                    array = np.concatenate([array, pad_array], axis=d_seq + 1)
            else:
                # temporally skip atom level padding, TODO(zhukunrui) 
                new_array = array
            assert d_seq_size in [token_size, atom_size], \
                f"d_seq size {d_seq_size} of {key} {array.shape}" \
                + f"not in [{token_size}, {atom_size}]"
            return array

        new_features = {k: pad(k, v, start_axis) for k, v in new_features.items()}
        new_labels = {k: pad(k, v, 0) for k, v in new_labels.items()}

    # new_features['crop_idx'] = feat_crop_idx
    new_labels['all_centra_token_indice'] = map_to_absolute_token_centra(new_features['ref_token2atom_idx'], 
                                                token_centra=relative_token_centra, crop_size=crop_size)
    new_features['ref_token2atom_idx'] = map_to_continuous_indices(new_features['ref_token2atom_idx'])


    ## map frame pae indices to absolute indices
    frame_crop_indices = {}
    crop_frame_mask = new_labels.pop('frame_mask')
    for key in FRAME_PAE_REQUIRED_KEYS:
        frame_crop_indices[key] = new_labels.pop(key)
    new_frame_feats = map_frame_indice_and_mask(crop_absolute_indices=frame_crop_indices, 
                                                crop_frame_mask=crop_frame_mask, 
                                                crop_keep_indices=label_crop_idx_atom, 
                                                crop_size=crop_size)
    new_labels.update(new_frame_feats)
    _frame_feats_valid_check(new_labels)

    # fix for input seq length
    # new_features['seq_length'] = (crop_size * np.ones_like(new_features['seq_length'])).astype(np.int32)
    return new_features, new_labels

def crop_and_pad_multimer(
    raw_features: FeatureDict,
    raw_labels: FeatureDict,
    seq_infos: Mapping,
    crop_size: int = 256,
    pad_for_shorter_seq: bool = True,
    for_recycle: bool = False,
    model_preset: str = '', spatial_crop_ratio=0.5) -> FeatureDict:
    """Cropping and padding."""

    # get num res from aatype
    assert 'aatype' in raw_features.keys(), \
        "'aatype' missing from batch, which is not expected."

    seq_lens = seq_infos['seq_lens']
    rand_drop = np.random.random()
    if model_preset == 'multimer_ranking':
        crop_idx = crop_spatial_ranking(raw_features, raw_labels, seq_lens, crop_size, for_recycle)
    elif rand_drop < (1 - spatial_crop_ratio):
        crop_idx = crop_contiguous(seq_lens, crop_size)
    else:
        crop_idx = crop_spatial(feat=raw_features, label=raw_labels, list_n_k=seq_lens, \
                                crop_size=crop_size, for_recycle=for_recycle)
    crop_idx = crop_idx.astype(int)
    # drop idx in cropped subchain whose num_res == 1
    crop_idx = remove_idx_from_chain_w_min_res(crop_idx, raw_features['asym_id'], min_res=1)

    start_axis = 1 if for_recycle else 0

    def crop(key: str, array: np.ndarray, start_axis: int):
        if len(array.shape) <= start_axis:
            return array
        if is_ignored_key_multimer(key):
            return array
        d_seq = start_axis         # choose the dim to crop / pad
        if is_batched_key_multimer(key):
            d_seq += 1
        new_array = np.take(array, crop_idx, axis=d_seq)
        return new_array

    raw_features = {k: crop(k, v, start_axis) for k, v in raw_features.items()}
    raw_labels = {k: crop(k, v, 0) for k, v in raw_labels.items()}
    num_res = raw_features['aatype'].shape[start_axis]

    ## copy assembly features
    for k in ['asym_id', 'sym_id', 'entity_id', 'residue_index']:
        if for_recycle:
            raw_labels[k] = np.copy(raw_features[k][0])
        else:
            raw_labels[k] = np.copy(raw_features[k])

    if num_res < crop_size and pad_for_shorter_seq:
        # pad short seq (0 padding and (automatically) create masks)
        def pad(key: str, array: np.ndarray, start_axis: int):
            if len(array.shape) <= start_axis:
                return array
            if is_ignored_key_multimer(key):
                return array
            d_seq = start_axis      # choose the dim to crop / pad
            if is_batched_key_multimer(key):
                d_seq += 1
            pad_shape = list(array.shape)
            pad_shape[d_seq] = crop_size - num_res
            pad_array = np.zeros(pad_shape)
            pad_array = pad_array.astype(array.dtype)
            array = np.concatenate([array, pad_array], axis=d_seq)
            return array

        raw_features = {k: pad(k, v, start_axis) for k, v in raw_features.items()}
        raw_labels = {k: pad(k, v, 0) for k, v in raw_labels.items()}

    raw_features['crop_idx'] = crop_idx
    # fix for input seq length
    raw_features['seq_length'] = (crop_size * np.ones_like(raw_features['seq_length'])).astype(np.int32)
    return raw_features, raw_labels


def crop_and_pad_split_chains(
    raw_features: FeatureDict,
    raw_labels: FeatureDict,
    crop_size: int = 256,
    pad_for_shorter_seq: bool = True) -> FeatureDict:
    """Cropping and padding for spliting chains"""

    # get num res from aatype
    assert 'aatype' in raw_features.keys(), \
        "'aatype' missing from batch, which is not expected."

    # 1. crop_pad for sequence features

    chain_lens = np.bincount(raw_features['asym_id'].astype('int64'))
    chain_aatypes, chain_seq_lengths = [], []
    start_idx = 0
    crop_idx = []
    for chain_len in chain_lens:
        if chain_len == 0:
            continue
        if chain_len < crop_size and pad_for_shorter_seq:
            crop_idx += [*range(start_idx, start_idx + chain_len)]
            pad_len = crop_size - chain_len
            chain_aatype = raw_features['aatype'][start_idx : start_idx + chain_len]
            chain_aatype = np.pad(chain_aatype, (0, pad_len), 'constant', constant_values=0)
            crop_len = chain_len
        elif chain_len > crop_size:
            crop_start = np.random.randint(chain_len - crop_size) + start_idx
            crop_end = crop_start + crop_size
            crop_idx += [*range(crop_start, crop_end)]
            chain_aatype = raw_features['aatype'][crop_start : crop_end]
            crop_len = crop_size
        else:
            crop_idx += [*range(start_idx, start_idx + chain_len)]
            chain_aatype = raw_features['aatype'][start_idx : start_idx + chain_len]
            crop_len = crop_size
        chain_aatypes.append(np.expand_dims(chain_aatype, axis=0))
        chain_seq_lengths.append(crop_len)
        start_idx += chain_len
    chain_aatypes = np.concatenate(chain_aatypes, axis=0)

    def crop(key: str, array: np.ndarray, start_axis: int):
        if len(array.shape) <= start_axis:
            return array
        if is_ignored_key_multimer(key):
            return array
        d_seq = start_axis         # choose the dim to crop / pad
        if is_batched_key_multimer(key):
            d_seq += 1
        new_array = np.take(array, crop_idx, axis=d_seq)
        return new_array

    # 2. crop for other features
    # for k, v in raw_features.items():
    #     print(k, v.shape, is_ignored_key_multimer(k), is_batched_key_multimer(k))
    raw_features = {k: crop(k, v, 0) for k, v in raw_features.items()}
    raw_labels = {k: crop(k, v, 0) for k, v in raw_labels.items()}
    raw_features['seq_length'] = raw_features['aatype'].shape[0]

    return raw_features, raw_labels, chain_aatypes, chain_seq_lengths

def make_pair_pocket_mask_by_receptor_residue(
        all_atom_positions, 
        all_atom_mask, 
        asym_ids,
        receptor_asym_id=None,
        seq_mask=None,
        threshold_min=10.0, 
        threshold_max=30.0,
        max_inter_residue_num=5):
    """
    Get pair_pocket_mask by firstly select several receptor residues 
    in the interface, then find the receptor pocket residues within a 
    distance threshold, finally make a mask between the receptor pocket 
    and the full ligand.
    Args:
        all_atom_positions: [N, 37, 3]
        all_atom_mask: [N, 37]
        asym_ids: [N,]
        receptor_asym_id: int, if not given, use the longest chain as receptor
        seq_mask: [N,]
    
    Return:
        pair_pocket_mask: [N, N]
    """
    def _get_dist(coord1, coord2):
        """
        coor1: [n1, 3]
        coor2: [n2, 3]
        return [n1, n2]
        """
        coord_diff = coord1[:, None] - coord2[None]
        dist = np.sqrt(np.sum(coord_diff ** 2, axis=-1))    # (n1, n2)
        return dist

    ca_idx = residue_constants.atom_order['CA']
    ca_coords = all_atom_positions[..., ca_idx, :]
    ca_mask = all_atom_mask[..., ca_idx].astype('bool')
    N = len(all_atom_positions)

    if receptor_asym_id is None:
        uniq_asym_ids, uniq_counts = np.unique(asym_ids, return_counts=True)
        receptor_asym_id = uniq_asym_ids[np.argmax(uniq_counts)]

    ## randomize threshold and inter residue num
    threshold = np.random.uniform(threshold_min, threshold_max)
    inter_residue_num = np.random.randint(1, max_inter_residue_num + 1)

    ## create pair_pocket_mask
    pair_pocket_mask = np.zeros([N, N], 'float32')
    in_same_asym = asym_ids[..., None] == asym_ids[..., None, :]
    pair_pocket_mask[in_same_asym] = 1.0

    ## go over all ligand chains
    uniq_asym_ids = np.unique(asym_ids)
    all_indices = np.arange(N, dtype='int64')
    for ligand_asym_id in uniq_asym_ids:
        if ligand_asym_id == receptor_asym_id:
            continue
        l_coords = ca_coords[asym_ids == ligand_asym_id]    # (n1, 3)
        r_coords = ca_coords[asym_ids == receptor_asym_id]  # (n2, 3)
        l_indices = all_indices[asym_ids == ligand_asym_id]   # (n1,)
        r_indices = all_indices[asym_ids == receptor_asym_id]   # (n2,)

        # get receptor residue in the interface as anchor point
        lr_dist = _get_dist(l_coords, r_coords)    # (n1, n2)
        inter_r_indices = r_indices[np.sum(lr_dist <= 10, axis=0) > 0]    # (m,)
        if len(inter_r_indices) == 0:
            print('no receptor residue in the interface')
            continue
        np.random.shuffle(inter_r_indices)
        inter_r_indices = inter_r_indices[:inter_residue_num]
        
        # get receptor pocket close to the anchor point
        anchor_coords = ca_coords[inter_r_indices]    # (m, 3)
        anchor_r_dist = _get_dist(anchor_coords, r_coords)    # (m, n2)
        pocket_r_indices = r_indices[np.sum(anchor_r_dist <= threshold, axis=0) > 0]    # (k,)
        
        # make mask between the receptor pocket and the full ligand
        mask1 = np.full([N], False)
        mask1[l_indices] = True
        mask2 = np.full([N], False)
        mask2[pocket_r_indices] = True
        pair_pocket_mask += mask1[:, None] * mask2[None] + mask1[None] * mask2[:, None]
    assert np.max(pair_pocket_mask) < 2

    ## apply seq_mask
    if not seq_mask is None:
        pair_pocket_mask *= seq_mask[..., None] * seq_mask[None]
    return pair_pocket_mask

def make_pair_pocket_mask_by_receptor_res_idx(
        receptor_res_idx,
        asym_ids,
        receptor_asym_id,
        seq_mask=None):
    """
    Get pair_pocket_mask by masking all receptor residues except for 
    indicies specified by receptor_res_idx, and unmask residues in the
    rest of the chains.
    Args:
        pocket_res_idx: [k] list of pocket indices at the receptor chain
        asym_ids: [N,]
        seq_mask: [N,]
    
    Return:
        pair_pocket_mask: [N, N]
    """
    N = len(asym_ids)
    ## create pair_pocket_mask
    pair_pocket_mask = np.zeros([N, N], 'float32')
    in_same_asym = asym_ids[..., None] == asym_ids[..., None, :]
    pair_pocket_mask[in_same_asym] = 1.0
    ## go over all ligand chains
    uniq_asym_ids = np.unique(asym_ids)
    all_indices = np.arange(N, dtype='int64')
    for ligand_asym_id in uniq_asym_ids:
        if ligand_asym_id == receptor_asym_id:
            continue
        l_indices = all_indices[asym_ids == ligand_asym_id]   # (n1,)
        r_indices = all_indices[asym_ids == receptor_asym_id]   # (n2,)
        pocket_r_indices = r_indices[receptor_res_idx]    # (k,)
        # make mask between the receptor pocket and the full ligand
        mask1 = np.full([N], False)
        mask1[l_indices] = True
        mask2 = np.full([N], False)
        mask2[pocket_r_indices] = True
        pair_pocket_mask += mask1[:, None] * mask2[None] + mask1[None] * mask2[:, None]
    assert np.max(pair_pocket_mask) < 2
    ## apply seq_mask
    if not seq_mask is None:
        pair_pocket_mask *= seq_mask[..., None] * seq_mask[None]
    return pair_pocket_mask


def remove_masked_residues(raw_labels: FeatureDict):
    """Remove masked residues."""
    mask = raw_labels['all_atom_mask'][:,0].astype(bool)
    return {k: v[mask] for k, v in raw_labels.items()}


def get_queue_item(q: Queue):
    """tbd."""
    # waiting time upperbound = MAX_FAILED * MAX_TIMEOUT
    for t in range(MAX_FAILED):
        try:
            item = q.get(block=True, timeout=MAX_TIMEOUT)
            logging.debug(f"get queue item succeeded. current qsize = {q.qsize()}.")
            return item
        except:
            logging.warning(f"get queue item timeout after {MAX_TIMEOUT}s "
                            f"({t + 1}/{MAX_FAILED}).")
    # exit subprogram:
    logging.error("get queue item failed for too many times. subprogram quit.")
    return None


def load_params_from_npz(npz_path):
    """tbd."""
    params = np.load(npz_path, allow_pickle=True)
    return params['arr_0'].flat[0]


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


def generate_pkl_features_from_fasta_a3m(
        fasta_path: str,
        a3m_msa_path: str,
        output_dir: str,
        m8_path: str,
        data_pipeline: DataPipeline,
        timings: Optional[Dict[str, float]] = None,
        use_gzip: bool = False):
    """Generate features.pkl from FASTA and colabfold_search results."""
    if timings is None:
        timings = {}

    name = pathlib.Path(fasta_path).stem
    numseq = pathlib.Path(a3m_msa_path).stem

    feat_list = [
        'query_seq', 'target_seq', 'identity', 'align_lenth', 'mismatch_num',
        'gap_num', 'query_start', 'query_end', 'target_start', 'target_end',
        'E_value', 'bit_score'
    ]
    template_df = pd.read_csv(m8_path, sep='\t', header=None, names=feat_list, index_col=False)

    # Check output dir.
    output_dir = os.path.join(output_dir, name)
    os.makedirs(output_dir, exist_ok=True)

    msa_output_dir = os.path.join(output_dir, 'msas')
    os.makedirs(msa_output_dir, exist_ok=True)

    # Get features.
    pt = time.time()
    logging.info(f"processing file {fasta_path}...")
    features = data_pipeline.process(input_fasta_path=fasta_path,
                                     a3m_msa_path=a3m_msa_path,
                                     template_df=template_df,
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
    'interface_source',
    'interface_gen_size',
    'interface_gen_seed',
    'interface_gen_temperature',
    'interface_gen_repeats',
    'interface_sampling_method',
    'interface_sampling_beam'
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



def group_min_distance(a, b):
    """
    # example1
    a1 = np.array([
        [0, 1, 2, 3],
        [1, 0, 4, 5],
        [2, 4, 0, 6],
        [3, 5, 6, 0]
    ])
    b1 = np.array([0, 0, 1, 1])

    # example2
    a2 = np.array([
        [0, 5, 4, 3, 2],
        [5, 0, 6, 5, 4],
        [4, 6, 0, 7, 8],
        [3, 5, 7, 0, 3],
        [2, 4, 8, 3, 0]
    ])
    b2 = np.array([0, 0, 1, 1, 1])

    print("Example 1 group distances:", group_min_distance(a1, b1))
    print("Example 2 group distances:", group_min_distance(a2, b2))
    """
    unique_groups = np.unique(b)
    group_distances = {}

    # check each pair of groups
    for g1 in unique_groups:
        for g2 in unique_groups:
            if g1 < g2:  # only consider different group
                group1_indices = np.where(b == g1)[0]
                group2_indices = np.where(b == g2)[0]

                # min distance between g1 and g2
                min_distance = float('inf')
                for i in group1_indices:
                    for j in group2_indices:
                        min_distance = min(min_distance, a[i, j])
                group_distances[(g1, g2)] = min_distance

    return group_distances

def dist_dict_2_mat(dist_dict, n_token):
    """
    convert distance dic to mat
    """
    distance_matrix = np.zeros((n_token, n_token))
    # add dist to matrix
    for (i, j), dist in dist_dict.items():
        distance_matrix[i][j] = dist
        distance_matrix[j][i] = dist
    return distance_matrix

def get_min_token_pair_distance(atom_pos, atom_mask, atom_to_token_mapping, n_token_padded):
    """
    atom_pos: [M, 3]
    atom_to_token_mapping: grouping infor like [0,0,0,1,2,2,2,3,3,4,5,6,6,6,7,8]
    """
    # atom_dist = pairwise_distances(atom_pos, atom_pos) # [M, M]

    # filter out masked atoms
    atom_mask = atom_mask.astype('bool')
    atom_pos = atom_pos[atom_mask]
    atom_to_token_mapping = atom_to_token_mapping[atom_mask]

    atom_dist = pdist(atom_pos) # [M * (M-1)/2]
    atom_dist = squareform(atom_dist) # [M, M]
    min_token_dist_dict = group_min_distance(atom_dist, b=atom_to_token_mapping)
    return dist_dict_2_mat(min_token_dist_dict, n_token_padded)


def build_pocket_info(asym_id, is_ligand, token_pos, token_pos_mask):
    """build_pocket_info
    Args:
        asym_id: [n_token,]
        is_ligand:  [n_token,]
        token_pos:  [n_token, 3]
        token_pos_mask: [n_token]
    Returns:
        pocket_info: [N, N]
    """
    N = len(asym_id)
    pocket_info = np.zeros([N, N], 'float32')
    dist = pairwise_distances(token_pos, token_pos)
    uniq_lig_asym_ids = np.unique(asym_id[is_ligand == 1])
    for aid in uniq_lig_asym_ids:
        target_mask = np.logical_and(is_ligand != 1, token_pos_mask)
        lig_mask = np.logical_and(asym_id == aid, token_pos_mask)
        inter_mask = target_mask[None] * lig_mask[:, None]
        inter_dist = dist + (1 - inter_mask) * 1e9
        pocket_mask = np.logical_and(np.sum(inter_dist < 10, 0, keepdims=True), inter_mask).astype('float32')
        pocket_info += pocket_mask
        pocket_info += pocket_mask.T
    return pocket_info



def token_completion(token_id, asym_id, is_ligand, restype, residue_index):
    '''
    find all related tokens for the input token if its entity type is ligand or unk-residue
    token_id: 1
    asym_id: [n_token]
    is_ligand: [n_token]
    restype: [n_token]
    residue_index: [n_token]
    '''
    assert len(asym_id.shape) == 1, f"asym_id shape {asym_id.shape} != 1"
    tok_id_completed = []
    if not is_ligand[token_id] > 0:
        tok_id_completed = np.array([token_id])
    else:
        if restype[token_id] == restype_order_with_x["X"]: # restype == unk
            tok_asym_id = asym_id[token_id]
            tok_res_id = residue_index[token_id]
            tok_chain_ids = np.where(asym_id == tok_asym_id)[0]  # unk chain
            tok_chain_res_ids = np.where(residue_index[tok_chain_ids] == tok_res_id)[0] # same residue_index in chain
            tok_id_completed = tok_chain_ids[tok_chain_res_ids]
        else: 
            tok_asym_id = asym_id[token_id]
            tok_id_completed = np.where(asym_id == tok_asym_id)[0]
    
    return tok_id_completed


def dist_to_bins(arr, buckets):
    """
    map distances in arr to the nearest buckets
    arr = np.array([20, 19.9, 5, 5.2, 50, 21])
    buckets = np.arange(1,21)
    dist_to_bins(arr, buckets)
    >>> array([19, 19,  4, 5, 20, 20])
    """
    bucket_arr = []
    for n in arr:
        limit_found = False
        for bid, limit in enumerate(buckets):
            if n <= limit and n > buckets[bid - 1]:
                bucket_arr.append(bid)
                limit_found = True
                break
        if not limit_found:
            bucket_arr.append(len(buckets))
    return np.array(bucket_arr)


def get_bin_masks(bins, bucket_size):
    """
    convert bins to mask
    """
    return (np.arange(bucket_size) >= bins[:, None]).astype('float')


def build_pocket_info_with_random_residues(asym_id, is_ligand, 
        token_pos, token_pos_mask, top_n, m):
    """build_pocket_info by finding the `top_n` nearest and centra_dist<10A residues
        from the ligand and then randomly selecting `m` residues from the top_n
    Args:
        asym_id: [n_token,]
        is_ligand:  [n_token,]
        token_pos:  [n_token, 3]
        token_pos_mask: [n_token]
        top_n: int
        m: int
    Returns:
        pocket_info: [N, N]
    """
    N = len(asym_id)
    pocket_info = np.zeros([N, N], 'float32')
    dist = pairwise_distances(token_pos, token_pos)
    uniq_lig_asym_ids = np.unique(asym_id[is_ligand == 1])
    for aid in uniq_lig_asym_ids:
        target_mask = np.logical_and(is_ligand != 1, token_pos_mask)
        lig_mask = np.logical_and(asym_id == aid, token_pos_mask)
        inter_mask = target_mask[None] * lig_mask[:, None]
        inter_dist = dist + (1 - inter_mask) * 1e9
        # for each residue, find its nearest dist to ligand atoms 
        prot_min_dist = np.min(inter_dist, 0)
        top_n_indices = np.argsort(prot_min_dist)[:top_n]
        top_n_indices = top_n_indices[prot_min_dist[top_n_indices] < 10]
        if len(top_n_indices) == 0:
            continue
        m = min(len(top_n_indices), m)
        rand_m_indices = np.random.choice(top_n_indices, m, replace=False)
        # create mask
        pocket_mask = np.zeros([N, N], 'bool')
        pocket_mask[:, rand_m_indices] = True
        pocket_mask = np.logical_and(pocket_mask, inter_mask).astype('float32')
        # add mask to pocket_info
        pocket_info += pocket_mask
        pocket_info += pocket_mask.T
    return pocket_info


def build_pocket_info_with_random_residues(asym_id, binder_mask, 
        token_pos, token_pos_mask, top_n, m, target_mask=None, seed=None):
    """build_pocket_info by finding the `top_n` nearest and centra_dist<10A residues
        from the binder and then randomly selecting `m` residues from the top_n,
        If `top_n` is negative, randomly select residues within 10A.
        - binder: ligand or antibody
        - target: protein or antigen
        
    Args:
        asym_id: [n_token,]
        binder_mask:  [n_token,]
        token_pos:  [n_token, 3]
        token_pos_mask: [n_token]
        top_n: int
        m: int
        target_mask: [n_token]
        seed: int, consistently choose residues if set
    Returns:
        pocket_info: [N, N]
    """
    N = len(asym_id)
    pocket_info = np.zeros([N, N], 'float32')
    dist = pairwise_distances(token_pos, token_pos)
    uniq_bid_asym_ids = np.unique(asym_id[binder_mask == 1])
    if target_mask is None:
        target_mask = np.logical_and(binder_mask != 1, token_pos_mask)
    for aid in uniq_bid_asym_ids:
        lig_mask = np.logical_and(asym_id == aid, token_pos_mask)
        inter_mask = target_mask[None] * lig_mask[:, None]
        inter_dist = dist + (1 - inter_mask) * 1e9
        if top_n > 0:
            # for each residue, find its nearest dist to binder atoms
            prot_min_dist = np.min(inter_dist, 0)
            # pick up top_n residues
            top_n_indices = np.argsort(prot_min_dist)[:top_n]
            top_n_indices = top_n_indices[prot_min_dist[top_n_indices] < 10]
        else:
            # randomly select 5 residues within 10A
            in_pocket = np.any(inter_dist <= 10.0, axis=0)
            top_n_indices = np.where(in_pocket)[0]
        if len(top_n_indices) == 0:
            continue
        m = min(len(top_n_indices), m)
        if seed is None:
            rand_m_indices = np.random.choice(top_n_indices, m, replace=False)
        else:
            # Fix choice for testing reproduction
            valid_len = token_pos_mask.sum()
            rand_m_indices = np.random.default_rng(seed + valid_len).choice(
                top_n_indices, m, replace=False)
        # create mask
        pocket_mask = np.zeros([N, N], 'bool')
        pocket_mask[:, rand_m_indices] = True
        pocket_mask = np.logical_and(pocket_mask, inter_mask).astype('float32')
        # add mask to pocket_info
        pocket_info += pocket_mask
        pocket_info += pocket_mask.T
    return pocket_info


def build_interface_info_with_random_residues_module1(asym_id, entity_id,
        atom_pos, atom_mask, atom_to_token_mapping,
        token_pos, token_pos_mask, 
        top_n, m, 
        interface_mask=None, binder_mask=None, target_mask=None, 
        dist_thres=5, dist_type='heavy_atom', seed=None):
    """build_interface_info by find centra_dist<dist_thres residue pairs across any two chains 
        in the complex. 

        If interface_mask is specified, only token pairs with positive value (1) in the interface_mask are
        considered in the interface sampling.

        If interface_mask is not specified, it will be derived from the asym_id and binder/target masks:
        If the binder (binder_mask==1) and target (target_mask==1) are specified (not None), 
        only keep residue pairs across the binder-target interface. If only one of the binder_mask
        and target_mask are specified, the other is derived from the sepcified one by a NOT operation.

        If `top_n` >= 0, only keep the `top_n` nearest residue pairs from the found set.
        If m >= 0, randomly select `m` residue pairs from the top_n. 
        - binder: one or more chains of any type (protein, dna, rna, ligand)
        - target: one or more chains of any type (protein, dna, rna, ligand)
        
    Args:
        asym_id: [n_token,]
        entity_id: [n_token,]
        token_pos:  [n_token, 3]
        token_pos_mask: [n_token]
        top_n: int
        m: int
        interface_mask: [n_token, n_token]
        binder_mask:  [n_token,]
        target_mask: [n_token,]
        dist_thres: int
        seed: int, consistently choose residues if set
    Returns:
        interface_info: [N, N]
    """

    N = len(asym_id)
    interface_info_sample = np.zeros([N, N], 'float32')
    if dist_type == 'central_atom':
        dist = pairwise_distances(token_pos, token_pos)
    elif dist_type == 'heavy_atom':
        dist = get_min_token_pair_distance(atom_pos, atom_mask, atom_to_token_mapping, n_token_padded=N)
    else:
        raise ValueError(f"Unsupported dist_type: {dist_type}")

    if interface_mask is None:
        # get the interface mask only containing inter-chain token pairs
        interface_mask = build_possible_interface_mask(asym_id, token_pos_mask, 
                            binder_mask, target_mask)

    # Mask out no-interface token-pairs with large distance
    inter_dist = dist + (1 - interface_mask) * 1e9
    # Mask the upper triangle of inter_dist
    inter_dist += np.triu(np.ones_like(inter_dist), k=0) * 1e9
    # Find all residue pairs within dist_thres distance value
    in_interface = (inter_dist <= dist_thres)
    # compute the full interface info
    interface_info_full = (in_interface + in_interface.T).astype('float32')
    # mix different interfaces from exchanged homo chains
    # interface_info_mix = mix_homo_interface_info(interface_info_full, asym_id, entity_id)

    ## Sample part of the interface
    # get the indices of token pairs in in_interface
    indices_1st, indices_2nd = np.where(in_interface)
    # get the distance of token pairs in in_interface 
    interface_dist = inter_dist[indices_1st, indices_2nd]

    # if top_n is setted, only keep top_n indices
    if top_n >= 0:
        # joint sort indices_1st and indices_2nd according to values in interface_dist
        sorted_indices = np.argsort(interface_dist)
        indices_1st = indices_1st[sorted_indices][:top_n]  
        indices_2nd = indices_2nd[sorted_indices][:top_n] 

    # if m is setted, randomly choose m indices from top_n
    if m >= 0 and m < len(indices_1st):
        if seed is None:
            rand_m_indices = np.random.choice(range(len(indices_1st)), m, replace=False)
        else:
            # Fix choice for testing reproduction
            valid_len = token_pos_mask.sum()
            rand_m_indices = np.random.default_rng(seed + valid_len).choice(
                range(len(indices_1st)), m, replace=False)

        indices_1st = indices_1st[rand_m_indices]
        indices_2nd = indices_2nd[rand_m_indices]
    
    # update interface_info_sample with the found residue pairs (indicated by indices_1st and indices_2nd)
    for i, j in zip(indices_1st, indices_2nd):
        interface_info_sample[i, j] = 1
        interface_info_sample[j, i] = 1

    ret = {
        'interface_info_sample': interface_info_sample,
        'interface_info_full': interface_info_full,
        # 'interface_info_mix': interface_info_mix
    }

    return ret


def build_interface_info_with_random_residues_module2(asym_id, 
        atom_pos, atom_mask, atom_to_token_mapping,
        token_pos, token_pos_mask, top_n, m, binder_mask=None, target_mask=None, 
        dist_thres=5, dist_type='heavy_atom', seed=None, sample_times=1):
    """build_interface_info by find centra_dist<dist_thres residue pairs across any two chains 
        in the complex. 
        If the binder (binder_mask==1) and target (target_mask==1) are specified (not None), 
        only keep residue pairs across the binder-target interface. If only one of the binder_mask
        and target_mask are specified, the other is derived from the sepcified one by a NOT operation.
        If `top_n` >= 0, only keep the `top_n` nearest residue pairs from the found set.
        If m >= 0, randomly select `m` residue pairs from the top_n. 
        - binder: one or more chains of any type (protein, dna, rna, ligand)
        - target: one or more chains of any type (protein, dna, rna, ligand)
        
    Args:
        asym_id: [n_token,]
        binder_mask:  [n_token,]
        target_mask: [n_token,]
        token_pos:  [n_token, 3]
        token_pos_mask: [n_token]
        top_n: int
        m: int
        dist_thres：int
        seed: int, consistently choose residues if set
    Returns:
        interface_info: [N, N]
    """

    N = len(asym_id)
    token_pos_mask = token_pos_mask.astype('bool')
    if dist_type == 'central_atom':
        dist = pairwise_distances(token_pos, token_pos)
    elif dist_type == 'heavy_atom':
        dist = get_min_token_pair_distance(atom_pos, atom_mask, atom_to_token_mapping, n_token_padded=N)
    else:
        raise ValueError(f"Unsupported dist_type: {dist_type}")

    # Construct inter_dist from dist. 
    if target_mask is not None or binder_mask is not None:
        # construct inter_dist between target and binder
        if target_mask is None:
            target_mask = np.logical_and(binder_mask != 1, token_pos_mask)
        elif binder_mask is None:
            binder_mask = np.logical_and(target_mask != 1, token_pos_mask)
        # mask no-target-binder distances with large distance value
        target_binder_mask = np.outer(binder_mask, target_mask)
        target_binder_mask += target_binder_mask.T
        inter_dist = dist + (1 - target_binder_mask) * 1e9
    else:
        # construct inter_dist between any two chains
        inter_dist = deepcopy(dist)
        uniq_asym_ids = np.unique(asym_id)
        for aid in uniq_asym_ids:
            chain_mask = (asym_id == aid)
            # mask intra-chain distances with large distance
            inter_dist += np.outer(chain_mask, chain_mask) * 1e9
        # mask invalid token positions with large distance value
        inter_dist[:, ~token_pos_mask] += 1e9
        inter_dist[~token_pos_mask, :] += 1e9

    # Mask the upper triangle of inter_dist
    inter_dist += np.triu(np.ones_like(inter_dist), k=0) * 1e9
    # Find all residue pairs within dist_thres distance value
    in_interface = (inter_dist <= dist_thres)
    # compute the full interface info
    interface_info_full = (in_interface + in_interface.T).astype('float32')
    # get the indices of token pairs in in_interface
    indices_1st, indices_2nd = np.where(in_interface)
    # get the distance of token pairs in in_interface 
    interface_dist = inter_dist[indices_1st, indices_2nd]

    # if top_n is setted, only keep top_n indices
    if top_n >= 0:
        # joint sort indices_1st and indices_2nd according to values in interface_dist
        sorted_indices = np.argsort(interface_dist)
        indices_1st = indices_1st[sorted_indices][:top_n]  
        indices_2nd = indices_2nd[sorted_indices][:top_n] 

    sampled_interface_infos = []
    for _ in range(sample_times):
        # if m is setted, randomly choose m indices from top_n
        if m >= 0 and m < len(indices_1st):
            if seed is None:
                rand_m_indices = np.random.choice(range(len(indices_1st)), m, replace=False)
            else:
                # Fix choice for testing reproduction
                valid_len = token_pos_mask.sum()
                rand_m_indices = np.random.default_rng(seed + valid_len).choice(
                    range(len(indices_1st)), m, replace=False)

            indices_1st = indices_1st[rand_m_indices]
            indices_2nd = indices_2nd[rand_m_indices]
        
        # update interface_info with the found residue pairs (indicated by indices_1st and indices_2nd)
        interface_info = InterfaceInfo(N, probability=1.0)
        for i, j in zip(indices_1st, indices_2nd):
            interface_info.add_node([i, j], 1)

        sampled_interface_infos.append(interface_info)

    return sampled_interface_infos


def build_possible_interface_mask(asym_id, token_pos_mask, 
        binder_mask=None, target_mask=None) -> np.ndarray:
    """ Bulid a N*N mask array containing all possible inter-chain token pairs.

        If the binder (binder_mask==1) and target (target_mask==1) are specified (not None), 
        only keep token pairs across the binder and target. If only one of the binder_mask
        and target_mask are specified, the other is derived from the sepcified one by a NOT operation.
    """

    N = len(asym_id)
    token_pos_mask = token_pos_mask.astype('bool')

    # Construct inter_dist from dist. 
    if target_mask is not None or binder_mask is not None:
        # construct inter_dist between target and binder
        if target_mask is None:
            target_mask = np.logical_and(binder_mask != 1, token_pos_mask)
        elif binder_mask is None:
            binder_mask = np.logical_and(target_mask != 1, token_pos_mask)
        # create a mask indicating all possible in-interface token pairs between the target and binder 
        interface_mask = np.outer(binder_mask, target_mask).astype('float32')
        interface_mask += interface_mask.T
    else:
        # construct interface_mask contains all possible token pairs across the interface of any two chains
        interface_mask = np.ones([N, N], 'float32')
        
        uniq_asym_ids = np.unique(asym_id)
        for aid in uniq_asym_ids:
            chain_mask = (asym_id == aid)
            # mask out intra-chain token pairs
            interface_mask -= np.outer(chain_mask, chain_mask).astype('float32')
        # mask invalid token positions 
        interface_mask[:, ~token_pos_mask] = 0.
        interface_mask[~token_pos_mask, :] = 0.

    return interface_mask


def mix_homo_interface_info(interface_info, asym_id, entity_id):
    """
    Process the interface information array by considering chain pairs with the same entity ID.

    Parameters:
    - interface_info (numpy.ndarray): A 2D array of shape (N, N) representing the interface information.
    - asym_id (numpy.ndarray): A 1D array of shape (N,) indicating the asymmetric ID for each residue.
    - entity_id (numpy.ndarray): A 1D array of shape (N,) indicating the entity ID for each residue.

    Returns:
    - numpy.ndarray: A processed 2D array of shape (N, N) where the interface information is updated
                     by considering all chain pairs with the same entity ID and combining their interface
                     information using logical OR.

    The function follows these steps:
    1. Initialize a new array `interface_info_mix` with the same data as `interface_info`.
    2. Identify all unique pairs of `asym_id` values (asym_id_1, asym_id_2) that belong to the same `entity_id`
       and satisfy the condition asym_id_1 < asym_id_2.
    3. For each identified pair, create a modified copy of `interface_info_mix` called `interface_info_mix_aug`
       where the positions of the chains corresponding to asym_id_1 and asym_id_2 are swapped.
    4. Update `interface_info_mix` by applying logical OR between it and `interface_info_mix_aug`.
    5. Return the final processed `interface_info_mix`.
    """
    N = interface_info.shape[0]
    
    # Step 1: Initialize interface_info_mix with interface_info
    interface_info_mix = np.copy(interface_info) # N * N
    
    # Step 2: Find all unique (asym_id_1, asym_id_2) pairs with the same entity_id and asym_id_1 < asym_id_2
    unique_entities = np.unique(entity_id)
    pairs = []
    
    for entity in unique_entities:
        asym_ids_in_entity = list(set(asym_id[entity_id == entity]))
        if len(asym_ids_in_entity) > 1:
            for i in range(len(asym_ids_in_entity)):
                for j in range(i + 1, len(asym_ids_in_entity)):
                    pairs.append((asym_ids_in_entity[i], asym_ids_in_entity[j]))
    
    # Step 3: Process each pair
    for asym_id_1, asym_id_2 in pairs:
        # Copy the current interface_info_mix to interface_info_mix_aug
        interface_info_mix_aug = np.copy(interface_info_mix)
        
        # Swap the positions of asym_id_1 and asym_id_2
        temp = interface_info_mix_aug[:, asym_id == asym_id_1]
        interface_info_mix_aug[:, asym_id == asym_id_1] = interface_info_mix_aug[:, asym_id == asym_id_2]
        interface_info_mix_aug[:, asym_id == asym_id_2] = temp

        temp = interface_info_mix_aug[asym_id == asym_id_1, :]
        interface_info_mix_aug[asym_id == asym_id_1, :] = interface_info_mix_aug[asym_id == asym_id_2, :]
        interface_info_mix_aug[asym_id == asym_id_2, :] = temp
        
        # Update interface_info_mix with the logical OR
        interface_info_mix = np.logical_or(interface_info_mix, interface_info_mix_aug).astype('float32')
    
    # Step 5: Return the processed interface_info_mix
    return interface_info_mix


if __name__ == '__main__':
    import paddle
    import numpy as np
    from scipy.spatial.transform import Rotation
    from utils.frame_diff.data import diffuse_utils
    from helixfold.data.utils import make_pair_pocket_mask_by_receptor_residue
    from helixfold.data.utils import build_interface_info_with_random_residues
    from helixfold.common import protein as protein_utils

    def res_idx_to_asym_id(res_idx): 
        """ convert res_idx to asym_id """
        asym_id = []
        chain_id = 0
        for i, idx in enumerate(res_idx):
            previous_idx = 0 if i == 0 else res_idx[i - 1]
            if idx < previous_idx: chain_id += 1
            asym_id.append(chain_id)
        return np.array(asym_id)

    pdb_file = "exp-7kp8_A_B_C_E.pdb"
    pdb_file = "exp-4e1h_A_C_K_B_D.pdb"
    pdb_file = "exp-7uym_H_P.pdb"
    pdb_file = "log/debug-model_1_multimer_v3-chainaffine_v2-pae0-pair_pocket-homo_pocket/" \
                + "node_/test_pdbs-7kp8_EA/1/exp-7eic_C_B.pdb"
    prot_obj = diffuse_utils.read_pdb(pdb_file)

    atom_pos = prot_obj.atom_positions
    seq_mask = np.ones(len(atom_pos), dtype='bool')
    seq_mask[[0, 1, 2, 3, 5, 6, 7, 8, 20, 21, 22, 23]] = False
    diffuse_utils.update_atom_pos(atom_pos, prot_obj, pdb_file=f"{pdb_file}_masked.pdb", seq_mask=seq_mask)


    raw_labels = {
        'all_atom_positions': prot_obj.atom_positions,
        'all_atom_mask': prot_obj.atom_mask,
        'asym_id': res_idx_to_asym_id(prot_obj.residue_index)
    }
    receptor_id = 0
    pocket_size = 20
    pair_pocket_mask = make_pair_pocket_mask_by_receptor_residue(
        raw_labels['all_atom_positions'], 
        raw_labels['all_atom_mask'], 
        raw_labels['asym_id'],
        receptor_asym_id=receptor_id,
        seq_mask=None,
        threshold_min=pocket_size, 
        threshold_max=pocket_size
    )
    is_interface = raw_labels['asym_id'][:, None] != raw_labels['asym_id'][None]
    pair_mask = pair_pocket_mask * is_interface
    seq_mask = np.sum(pair_mask, 0) > 0  # bool [num_res]
    diffuse_utils.update_atom_pos(atom_pos, prot_obj, 
                                    pdb_file=f"{pdb_file}-receptor{receptor_id}-pocket.pdb", 
                                    seq_mask=seq_mask)

    pair_pocket_mask = make_pair_pocket_mask_by_receptor_res_idx(
        receptor_res_idx=[95, 96, 97, 98, 99, 100],
        asym_ids=raw_labels['asym_id'],
        receptor_asym_id=1
    )


    


