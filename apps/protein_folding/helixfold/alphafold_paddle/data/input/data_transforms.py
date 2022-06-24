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

"""Data transforms."""

import numpy as np
from alphafold_paddle.common import residue_constants


## Helper function
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


class SeedMaker(object):
    """Return unique seeds."""

    def __init__(self, initial_seed=0):
        self.next_seed = initial_seed

    def __call__(self):
        i = self.next_seed
        self.next_seed += 1
        return i

NUM_RES = 'num residues placeholder'
NUM_MSA_SEQ = 'msa placeholder'
NUM_EXTRA_SEQ = 'extra msa placeholder'
NUM_TEMPLATES = 'num templates placeholder'

MS_MIN32 = -2147483648
MS_MAX32 = 2147483647

Seed_maker = SeedMaker()

def make_random_seed(size, low=MS_MIN32, high=MS_MAX32):
    global Seed_maker
    np.random.seed(Seed_maker())
    return np.random.uniform(size=size, low=low, high=high)


def cast_64bit_ints(protein):
    """Cast 64bit ints."""
    for k, v in protein.items():
        if v.dtype == np.int64:
            protein[k] = v.astype(np.int32)
    return protein


_MSA_FEATURE_NAMES = [
    'msa', 'deletion_matrix', 'msa_mask', 'msa_row_mask', 'bert_mask',
    'true_msa'
]


def make_seq_mask(protein):
    """Make seq mask."""
    protein['seq_mask'] = np.ones(shape_list(protein['aatype']),
                                dtype=np.float32)
    return protein


def make_template_mask(protein):
    """Make template mask."""
    protein['template_mask'] = np.ones(shape_list(protein['template_domain_names']),
                                dtype=np.float32)
    return protein


def curry1(f):
    """Supply all arguments except the first."""
    def fc(*args, **kwargs):
        return lambda x: f(x, *args, **kwargs)

    return fc


@curry1
def add_distillation_flag(protein, distillation):
    """Add distillation flag."""
    protein['is_distillation'] = np.array(
        float(distillation), dtype=np.float32)
    return protein


def make_all_atom_aatype(protein):
  protein['all_atom_aatype'] = protein['aatype']
  return protein


def fix_templates_aatype(protein):
    """Fixes aatype encoding of templates."""
    # Map one-hot to indices.
    protein['template_aatype'] = np.argmax(protein['template_aatype'],
                                           axis=-1).astype(np.int32)
    # Map hhsearch-aatype to our aatype.
    new_order_list = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    new_order = np.array(new_order_list, np.int32)
    protein['template_aatype'] = new_order[protein['template_aatype']]
    return protein


def correct_msa_restypes(protein):
    """Correct MSA restype to have the same order as residue_constants."""
    new_order_list = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    new_order = np.array(new_order_list, dtype=protein['msa'].dtype)
    protein['msa'] = new_order[protein['msa']]

    perm_matrix = np.zeros((22, 22), dtype=np.float32)
    perm_matrix[range(len(new_order_list)), new_order_list] = 1.

    for k in protein:
        if 'profile' in k:  # Include both hhblits and psiblast profiles
            num_dim = protein[k].shape.as_list()[-1]
            assert num_dim in [20, 21, 22], (
                'num_dim for %s out of expected range: %s' % (k, num_dim))
            protein[k] = np.tensordot(protein[k], perm_matrix[:num_dim, :num_dim], axes=1)
    return protein


def squeeze_features(protein):
    """Remove singleton and repeated dimensions in protein features."""
    protein['aatype'] = np.argmax(protein['aatype'], axis=-1)
    for k in ['msa', 'num_alignments', 'seq_length', 'sequence',
              'superfamily', 'deletion_matrix', 'resolution',
              'between_segment_residues', 'residue_index',
              'template_all_atom_masks']:
        if k in protein:
            final_dim = shape_list(protein[k])[-1]
            if isinstance(final_dim, int) and final_dim == 1:
                protein[k] = np.squeeze(protein[k], axis=-1)

    for k in ['seq_length', 'num_alignments']:
        if k in protein:
            # Remove fake sequence dimension
            protein[k] = protein[k][0]
    return protein


def make_random_crop_to_size_seed(protein):
    """Random seed for cropping residues and templates."""
    protein['random_crop_to_size_seed'] = np.array(
        make_random_seed([2]), np.int32)
    return protein


@curry1
def randomly_replace_msa_with_unknown(protein, replace_proportion):
    """Replace a proportion of the MSA with 'X'."""
    msa_mask = np.random.uniform(size=shape_list(protein['msa']),
                                 low=0, high=1) < replace_proportion
    x_idx, gap_idx = 20, 21
    msa_mask = np.logical_and(msa_mask, protein['msa'] != gap_idx)
    protein['msa'] = np.where(
        msa_mask, np.ones_like(protein['msa']) * x_idx, protein['msa'])
    aatype_mask = np.random.uniform(size=shape_list(protein['aatype']),
                                    low=0, high=1) < replace_proportion
    protein['aatype'] = np.where(
        aatype_mask, np.ones_like(protein['aatype']) * x_idx,
        protein['aatype'])
    return protein


@curry1
def sample_msa(protein, max_seq, keep_extra):
    """Sample MSA randomly, remaining sequences are stored as `extra_*`."""
    num_seq = protein['msa'].shape[0]

    shuffled = list(range(1, num_seq))
    np.random.shuffle(shuffled)
    shuffled.insert(0, 0)
    index_order = np.array(shuffled, np.int32)
    num_sel = min(max_seq, num_seq)

    sel_seq = index_order[:num_sel]
    not_sel_seq = index_order[num_sel:]
    is_sel = num_seq - num_sel

    for k in _MSA_FEATURE_NAMES:
        if k in protein:
            if keep_extra and not is_sel:
                new_shape = list(protein[k].shape)
                new_shape[0] = 1
                protein['extra_' + k] = np.zeros(new_shape)
            elif keep_extra and is_sel:
                protein['extra_' + k] = protein[k][not_sel_seq]
            if k == 'msa':
                protein['extra_msa'] = protein['extra_msa'].astype(np.int32)
            protein[k] = protein[k][sel_seq]
    return protein


@curry1
def crop_extra_msa(protein, max_extra_msa):
    """MSA features are cropped so only `max_extra_msa` sequences are kept."""
    if protein['extra_msa'].any():
        num_seq = protein['extra_msa'].shape[0]
        num_sel = np.minimum(max_extra_msa, num_seq)
        shuffled = list(range(num_seq))
        np.random.shuffle(shuffled)
        select_indices = shuffled[:num_sel]
        for k in _MSA_FEATURE_NAMES:
            if 'extra_' + k in protein:
                protein['extra_' + k] = protein['extra_' + k][
                    select_indices]
    return protein


def delete_extra_msa(protein):
    """Delete extra msa."""
    for k in _MSA_FEATURE_NAMES:
        if 'extra_' + k in protein:
            del protein['extra_' + k]
    return protein


@curry1
def nearest_neighbor_clusters(protein, gap_agreement_weight=0.):
    """Assign each extra MSA sequence to its nearest neighbor in sampled MSA."""
    weights = np.concatenate([
        np.ones(21), gap_agreement_weight * np.ones(1), np.zeros(1)], 0)

    sample_one_hot = protein['msa_mask'][:, :, None] * \
        one_hot(23, protein['msa'])
    num_seq, num_res, _ = shape_list(sample_one_hot)

    array_extra_msa_mask = protein['extra_msa_mask']
    if array_extra_msa_mask.any():
        extra_one_hot = protein['extra_msa_mask'][:, :, None] * \
            one_hot(23, protein['extra_msa'])
        extra_num_seq, _, _ = shape_list(extra_one_hot)

        agreement = np.matmul(
            np.reshape(extra_one_hot, [extra_num_seq, num_res * 23]),
            np.reshape(sample_one_hot * weights, [num_seq, num_res * 23]).T)
        protein['extra_cluster_assignment'] = np.argmax(agreement, axis=1)
    else:
        protein['extra_cluster_assignment'] = np.array([])

    return protein


@curry1
def summarize_clusters(protein):
    """Produce profile and deletion_matrix_mean within each cluster."""
    num_seq = shape_list(protein['msa'])[0]

    def _csum(x):
        result = []
        for i in range(num_seq):
            result.append(np.sum(x[np.where(
                protein['extra_cluster_assignment'] == i)], axis=0))

        return np.array(result)

    mask = protein['extra_msa_mask']
    mask_counts = 1e-6 + protein['msa_mask'] + _csum(mask)  # Include center

    msa_sum = _csum(mask[:, :, None] *
                    np.zeros(mask.shape + (23,), np.float32))
    msa_sum += one_hot(23, protein['msa'])  # Original sequence
    protein['cluster_profile'] = msa_sum / mask_counts[:, :, None]

    del msa_sum

    del_sum = _csum(mask * protein['extra_deletion_matrix'])
    del_sum += protein['deletion_matrix']  # Original sequence
    protein['cluster_deletion_mean'] = del_sum / mask_counts
    del del_sum

    return protein


def make_msa_mask(protein):
    """Mask features are all ones, but will later be zero-padded."""
    protein['msa_mask'] = np.ones(shape_list(protein['msa']),
                                  dtype=np.float32)
    protein['msa_row_mask'] = np.ones(shape_list(protein['msa'])[0],
                                      dtype=np.float32)
    return protein


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


@curry1
def make_pseudo_beta(protein, prefix=''):
    """Create pseudo-beta (alpha for glycine) position and mask."""
    assert prefix in ['', 'template_']
    pseudo_beta, pseudo_beta_mask = pseudo_beta_fn(
        protein['template_aatype' if prefix else 'all_atom_aatype'],
        protein[prefix + 'all_atom_positions'],
        protein['template_all_atom_masks' if prefix else 'all_atom_mask'])

    protein[prefix + 'pseudo_beta'] = pseudo_beta
    protein[prefix + 'pseudo_beta_mask'] = pseudo_beta_mask
    return protein


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


def make_hhblits_profile(protein):
    """Compute the HHblits MSA profile if not already present."""
    if 'hhblits_profile' in protein:
        return protein
    # Compute the profile for every residue (over all MSA sequences).
    protein['hhblits_profile'] = np.mean(one_hot(22, protein['msa']),
                                         axis=0)
    return protein


@curry1
def make_masked_msa(protein, config, replace_fraction):
    """Create data for BERT on raw MSA."""
    # Add a random amino acid uniformly.
    random_aa = np.array([0.05] * 20 + [0., 0.], dtype=np.float32)

    categorical_probs = config.uniform_prob * random_aa + \
        config.profile_prob * protein['hhblits_profile'] + \
        config.same_prob * one_hot(22, protein['msa'])
    # Put all remaining probability on [MASK] which is a new column.
    pad_shapes = [[0, 0] for _ in range(len(categorical_probs.shape))]
    pad_shapes[-1][1] = 1
    mask_prob = 1. - config.profile_prob - config.same_prob - \
        config.uniform_prob
    assert mask_prob >= 0.

    categorical_probs = np.pad(categorical_probs, pad_shapes,
                               constant_values=(mask_prob,))

    mask_position = np.random.uniform(size=shape_list(protein['msa']),
                                      low=0, high=1) < replace_fraction

    bert_msa = shaped_categorical(categorical_probs)
    bert_msa = np.where(mask_position, bert_msa, protein['msa'])

    protein['bert_mask'] = mask_position.astype(np.int32)
    protein['true_msa'] = protein['msa']
    protein['msa'] = bert_msa

    return protein


@curry1
def make_fixed_size(protein, shape_schema, msa_cluster_size, extra_msa_size,
                    num_res, num_templates=0):
    """Guess at the MSA and sequence dimensions to make fixed size."""

    pad_size_map = {
        NUM_RES: num_res,
        NUM_MSA_SEQ: msa_cluster_size,
        NUM_EXTRA_SEQ: extra_msa_size,
        NUM_TEMPLATES: num_templates,
    }

    for k, v in protein.items():
        if k == 'extra_cluster_assignment':
            continue
        shape = list(v.shape)
        schema = shape_schema[k]
        assert len(shape) == len(schema), f'Rank mismatch between ' + \
            f'shape and shape schema for {k}: {shape} vs {schema}'

        pad_size = [pad_size_map.get(s2, None) or s1
                    for (s1, s2) in zip(shape, schema)]
        padding = [(0, p - v.shape[i]) for i, p in enumerate(pad_size)]
        if padding:
            protein[k] = np.pad(v, padding)
            protein[k].reshape(pad_size)

    return protein


@curry1
def make_msa_feat(protein):
    """Create and concatenate MSA features."""
    has_break = np.clip(protein['between_segment_residues'].astype(
        np.float32), np.array(0), np.array(1))
    aatype_1hot = one_hot(21, protein['aatype'])

    target_feat = [np.expand_dims(has_break, axis=-1), aatype_1hot]

    msa_1hot = one_hot(23, protein['msa'])
    has_deletion = np.clip(protein['deletion_matrix'], np.array(0),
                           np.array(1))

    c = 2. / np.pi
    deletion_value = np.arctan(protein['deletion_matrix'] / 3.) * c

    msa_feat = [msa_1hot, np.expand_dims(has_deletion, axis=-1),
                np.expand_dims(deletion_value, axis=-1)]

    if 'cluster_profile' in protein:
        deletion_mean_value = (
            np.arctan(protein['cluster_deletion_mean'] / 3.) * c)
        msa_feat.extend([protein['cluster_profile'],
                         np.expand_dims(deletion_mean_value, axis=-1)])

    if 'extra_deletion_matrix' in protein:
        protein['extra_has_deletion'] = np.clip(
            protein['extra_deletion_matrix'], np.array(0), np.array(1))
        protein['extra_deletion_value'] = np.arctan(
            protein['extra_deletion_matrix'] / 3.) * c

    protein['msa_feat'] = np.concatenate(msa_feat, axis=-1)
    protein['target_feat'] = np.concatenate(target_feat, axis=-1)
    return protein


@curry1
def select_feat(protein, feature_list):
    return {k: v for k, v in protein.items() if k in feature_list}


@curry1
def crop_templates(protein, max_templates):
    for k, v in protein.items():
        if k.startswith('template_'):
            protein[k] = v[:max_templates]
    return protein


@curry1
def random_crop_to_size(protein, crop_size, max_templates, shape_schema,
                        subsample_templates=False):
    """Crop randomly to `crop_size`, or keep as is if shorter than that."""
    seq_length = protein['seq_length']
    seq_length_int = int(seq_length)
    if 'template_mask' in protein:
        num_templates = np.array(shape_list(
            protein['template_mask'])[0], np.int32)
    else:
        num_templates = np.array(0, np.int32)

    num_res_crop_size = np.minimum(seq_length, crop_size)
    num_res_crop_size_int = int(num_res_crop_size)

    if subsample_templates:
        templates_crop_start = make_random_seed(
            size=(), low=0, high=num_templates + 1)
    else:
        templates_crop_start = 0

    num_templates_crop_size = np.minimum(
        num_templates - templates_crop_start, max_templates)
    num_templates_crop_size_int = int(num_templates_crop_size)

    num_res_crop_start = int(make_random_seed(
        size=(), low=0, high=seq_length_int - num_res_crop_size_int + 1))

    templates_select_indices = np.argsort(
        make_random_seed(size=[num_templates]))

    for k, v in protein.items():
        if k not in shape_schema or \
           ('template' not in k and NUM_RES not in shape_schema[k]):
            continue

        if k.startswith('template') and subsample_templates:
            v = v[templates_select_indices]

        crop_sizes = []
        crop_starts = []
        for i, (dim_size, dim) in enumerate(
                zip(shape_schema[k], shape_list(v))):
            is_num_res = (dim_size == NUM_RES)
            if i == 0 and k.startswith('template'):
                crop_size = num_templates_crop_size_int
                crop_start = templates_crop_start
            else:
                crop_start = num_res_crop_start if is_num_res else 0
                crop_size = (num_res_crop_size_int if is_num_res
                             else (-1 if dim is None else dim))
            crop_sizes.append(crop_size)
            crop_starts.append(crop_start)
        if len(v.shape) == 1:
            protein[k] = v[crop_starts[0]:crop_starts[0] + crop_sizes[0]]
        elif len(v.shape) == 2:
            protein[k] = v[crop_starts[0]:crop_starts[0] + crop_sizes[0],
                           crop_starts[1]:crop_starts[1] + crop_sizes[1]]
        elif len(v.shape) == 3:
            protein[k] = v[crop_starts[0]:crop_starts[0] + crop_sizes[0],
                           crop_starts[1]:crop_starts[1] + crop_sizes[1],
                           crop_starts[2]:crop_starts[2] + crop_sizes[2]]
        else:
            protein[k] = v[crop_starts[0]:crop_starts[0] + crop_sizes[0],
                           crop_starts[1]:crop_starts[1] + crop_sizes[1],
                           crop_starts[2]:crop_starts[2] + crop_sizes[2],
                           crop_starts[3]:crop_starts[3] + crop_sizes[3]]

    protein['seq_length'] = num_res_crop_size
    return protein


def make_atom14_masks(protein):
    """Construct denser atom positions (14 dimensions instead of 37)."""
    restype_atom14_to_atom37 = []
    restype_atom37_to_atom14 = []
    restype_atom14_mask = []

    for rt in residue_constants.restypes:
        atom_names = residue_constants.restype_name_to_atom14_names[
            residue_constants.restype_1to3[rt]]

        restype_atom14_to_atom37.append([
            (residue_constants.atom_order[name] if name else 0)
            for name in atom_names])

        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append([
            (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
            for name in residue_constants.atom_types])

        restype_atom14_mask.append([
            (1. if name else 0.) for name in atom_names])

    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.] * 14)

    restype_atom14_to_atom37 = np.array(restype_atom14_to_atom37, np.int32)
    restype_atom37_to_atom14 = np.array(restype_atom37_to_atom14, np.int32)
    restype_atom14_mask = np.array(restype_atom14_mask, np.float32)

    residx_atom14_to_atom37 = restype_atom14_to_atom37[protein['aatype']]
    residx_atom14_mask = restype_atom14_mask[protein['aatype']]

    protein['atom14_atom_exists'] = residx_atom14_mask
    protein['residx_atom14_to_atom37'] = residx_atom14_to_atom37

    residx_atom37_to_atom14 = restype_atom37_to_atom14[protein['aatype']]
    protein['residx_atom37_to_atom14'] = residx_atom37_to_atom14

    restype_atom37_mask = np.zeros([21, 37], np.float32)
    for restype, restype_letter in enumerate(residue_constants.restypes):
        restype_name = residue_constants.restype_1to3[restype_letter]
        atom_names = residue_constants.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = residue_constants.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[protein['aatype']]
    protein['atom37_atom_exists'] = residx_atom37_mask

    return protein
