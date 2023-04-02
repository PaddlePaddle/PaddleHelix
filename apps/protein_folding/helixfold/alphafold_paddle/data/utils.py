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

from typing import *
from absl import logging
import numpy as np
import paddle
import pickle
import os
import sys
from copy import deepcopy
from multiprocessing import Queue

from alphafold_paddle.common.residue_constants import restype_order_with_x
from alphafold_paddle.data.mmcif_parsing import MmcifObject
from alphafold_paddle.data.mmcif_parsing import parse as parse_mmcif_string
from alphafold_paddle.data.pipeline import FeatureDict, DataPipeline
from alphafold_paddle.data.templates import _get_atom_positions as get_atom_positions
from Bio.PDB import protein_letters_3to1

INT_MAX = 0x7fffffff

# macros for retrying getting queue items.
MAX_TIMEOUT = 60
MAX_FAILED = 5

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
    # get cif string
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
]

# keys that should be ignored when conducting crop & pad
def is_ignored_key(k):
    """tbd."""
    return k in ignored_keys

# keys that have batch dim, e.g. msa features which have shape [N_msa, N_res, ...]
def is_batched_key(k):
    """tbd."""
    return k in batched_keys

def align_feat(feat, size):
    """Align feature."""
    # get num res from aatype
    assert 'aatype' in feat.keys(), \
        "'aatype' missing from batch, which is not expected."
    num_res = feat['aatype'].shape[2]

    if num_res % size != 0:
        align_size = (num_res // size + 1) * size

        # pad short seq (0 padding and (automatically) create masks)
        def pad(key, array, start_axis, align_size, num_res):
            if is_ignored_key(key):
                return array
            d_seq = start_axis      # choose the dim to crop / pad
            if is_batched_key(key):
                d_seq += 1
            pad_shape = list(array.shape)
            pad_shape[d_seq] = align_size - num_res
            pad_array = paddle.zeros(pad_shape, dtype=array.dtype)
            array = paddle.concat([array, pad_array], axis=d_seq)
            return array

        feat = {k: pad(k, v, 2, align_size, num_res) for k, v in feat.items()}
        feat['seq_length'] = (align_size * paddle.ones_like(feat['seq_length']))

    return feat


def align_label(label, size):
    """Align label."""
    num_res = label['all_atom_mask'].shape[1]

    if num_res % size != 0:
        align_size = (num_res // size + 1) * size

        def pad(key, array, start_axis, align_size, num_res):
            if is_ignored_key(key):
                return array
            d_seq = start_axis      # choose the dim to crop / pad
            if is_batched_key(key):
                d_seq += 1
            pad_shape = list(array.shape)
            pad_shape[d_seq] = align_size - num_res
            pad_array = paddle.zeros(pad_shape, dtype=array.dtype)
            array = paddle.concat([array, pad_array], axis=d_seq)
            return array

        label = {k: pad(k, v, 1, align_size, num_res) for k, v in label.items()}

    return label


def unpad_prediction(feat, pred):
    """Unpad prediction."""
    unpad_pred = deepcopy(pred)
    n = feat['aatype'].shape[0]

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
    pad_for_shorter_seq: bool = True) -> FeatureDict:
    """Cropping and padding."""

    # get num res from aatype
    assert 'aatype' in raw_features.keys(), \
        "'aatype' missing from batch, which is not expected."
    num_res = raw_features['aatype'].shape[1]

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
    return raw_features, raw_labels


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
    timings: Optional[Dict[str, float]] = None):
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
    features_output_path = os.path.join(output_dir, 'features.pkl')
    with open(features_output_path, 'wb') as f:
        pickle.dump(features, f, protocol=4)
    logging.info(f"process file {fasta_path} done.")
    
    # Save timings.
    timings_output_path = os.path.join(output_dir, 'timings.json')
    with open(timings_output_path, 'w') as fp:
        json.dump(timings, fp, indent=4)

    return features
