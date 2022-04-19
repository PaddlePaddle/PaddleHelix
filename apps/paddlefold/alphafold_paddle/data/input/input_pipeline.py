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

"""Feature pre-processing input pipeline for AlphaFold."""

import time
import numpy as np
import paddle
from alphafold_paddle.data.data_utils import compose
from alphafold_paddle.common import residue_constants
from alphafold_paddle.data.input import data_transforms
from alphafold_paddle.model import quat_affine, all_atom


NUM_RES = 'num residues placeholder'
NUM_SEQ = 'length msa placeholder'
NUM_TEMPLATES = 'num templates placeholder'

FEATURES = {
    #### Static features of a protein sequence ####
    'aatype': (np.float32, [NUM_RES, 21]),
    'between_segment_residues': (np.int64, [NUM_RES, 1]),
    'deletion_matrix': (np.float32, [NUM_SEQ, NUM_RES, 1]),
    'domain_name': (str, [1]),
    'msa': (np.int64, [NUM_SEQ, NUM_RES, 1]),
    'num_alignments': (np.int64, [NUM_RES, 1]),
    'residue_index': (np.int64, [NUM_RES, 1]),
    'seq_length': (np.int64, [NUM_RES, 1]),
    'sequence': (str, [1]),
    'all_atom_positions': (np.float32,
                           [NUM_RES, residue_constants.atom_type_num, 3]),
    'all_atom_mask': (np.int64, [NUM_RES, residue_constants.atom_type_num]),
    'resolution': (np.float32, [1]),
    'template_domain_names': (str, [NUM_TEMPLATES]),
    'template_sum_probs': (np.float32, [NUM_TEMPLATES, 1]),
    'template_aatype': (np.float32, [NUM_TEMPLATES, NUM_RES, 22]),
    'template_all_atom_positions': (np.float32, [
        NUM_TEMPLATES, NUM_RES, residue_constants.atom_type_num, 3
    ]),
    'template_all_atom_masks': (np.float32, [
        NUM_TEMPLATES, NUM_RES, residue_constants.atom_type_num, 1
    ]),
}


def nonensembled_map_fns(data_config):
    """Input pipeline functions which are not ensembled."""
    common_cfg = data_config.common

    map_fns = [
        data_transforms.correct_msa_restypes,
        data_transforms.add_distillation_flag(False),
        data_transforms.cast_64bit_ints,
        data_transforms.squeeze_features,
        # Keep to not disrupt RNG.
        data_transforms.randomly_replace_msa_with_unknown(0.0),
        data_transforms.make_seq_mask,
        data_transforms.make_msa_mask,
        # Compute the HHblits profile if it's not set. This has to be run before
        # sampling the MSA.
        data_transforms.make_hhblits_profile,
        data_transforms.make_random_crop_to_size_seed,
    ]
    if common_cfg.use_templates:
        map_fns.extend([
            data_transforms.fix_templates_aatype,
            data_transforms.make_pseudo_beta('template_')
        ])
    map_fns.extend([data_transforms.make_atom14_masks])
    return map_fns


def ensembled_map_fns(data_config):
    """Input pipeline functions that can be ensembled and averaged."""
    common_cfg = data_config.common
    eval_cfg = data_config.eval

    map_fns = []

    if common_cfg.reduce_msa_clusters_by_max_templates:
        pad_msa_clusters = eval_cfg.max_msa_clusters - \
            eval_cfg.max_templates
    else:
        pad_msa_clusters = eval_cfg.max_msa_clusters

    max_msa_clusters = pad_msa_clusters
    max_extra_msa = common_cfg.max_extra_msa

    map_fns.append(data_transforms.sample_msa(
        max_msa_clusters, keep_extra=True))

    if 'masked_msa' in common_cfg:
        map_fns.append(data_transforms.make_masked_msa(
            common_cfg.masked_msa, eval_cfg.masked_msa_replace_fraction))

    if common_cfg.msa_cluster_features:
        map_fns.append(data_transforms.nearest_neighbor_clusters())
        map_fns.append(data_transforms.summarize_clusters())

    if max_extra_msa:
        map_fns.append(data_transforms.crop_extra_msa(max_extra_msa))
    else:
        map_fns.append(data_transforms.delete_extra_msa)

    map_fns.append(data_transforms.make_msa_feat())

    crop_feats = dict(eval_cfg.feat)

    if eval_cfg.fixed_size:
        map_fns.append(data_transforms.select_feat(list(crop_feats)))
        map_fns.append(data_transforms.random_crop_to_size(
            eval_cfg.crop_size,
            eval_cfg.max_templates,
            crop_feats,
            eval_cfg.subsample_templates))
        map_fns.append(data_transforms.make_fixed_size(
            crop_feats,
            pad_msa_clusters,
            common_cfg.max_extra_msa,
            eval_cfg.crop_size,
            eval_cfg.max_templates))
    else:
        map_fns.append(data_transforms.crop_templates(
            eval_cfg.max_templates))

    return map_fns


def _make_features_metadata(feature_names):
    """Makes a feature name to type and shape mapping from a list of names."""
    required_features = ['sequence', 'domain_name', 'template_domain_names']
    feature_names = list(set(feature_names) - set(required_features))

    features_metadata = {name: FEATURES[name] for name in feature_names}
    return features_metadata


def feature_shape(feature_name,
                  num_residues,
                  msa_length,
                  num_templates,
                  features=None):
    """Get the shape for the given feature name."""
    features = features or FEATURES
    if feature_name.endswith("_unnormalized"):
        feature_name = feature_name[:-13]

    unused_dtype, raw_sizes = features[feature_name]
    replacements = {NUM_RES: num_residues, NUM_SEQ: msa_length}

    if num_templates is not None:
        replacements[NUM_TEMPLATES] = num_templates

    sizes = [replacements.get(dimension, dimension)
             for dimension in raw_sizes]
    for dimension in sizes:
        if isinstance(dimension, str):
            raise ValueError('Could not parse %s (shape: %s) with values: %s' % (
                feature_name, raw_sizes, replacements))
    size_r = [int(x) for x in sizes]
    return size_r


def parse_reshape_logic(parsed_features, features, num_template, key=None):
    """Transforms parsed serial features to the correct shape."""
    num_residues = np.reshape(
        parsed_features['seq_length'].astype(np.int32), (-1,))[0]

    if "num_alignments" in parsed_features:
        num_msa = np.reshape(
            parsed_features["num_alignments"].astype(np.int32), (-1,))[0]
    else:
        num_msa = 0

    if key is not None and "key" in features:
        parsed_features["key"] = [key]  # Expand dims from () to (1,).

    for k, v in parsed_features.items():
        new_shape = feature_shape(
            feature_name=k,
            num_residues=num_residues,
            msa_length=num_msa,
            num_templates=num_template,
            features=features)

        new_shape_size = 1
        for dim in new_shape:
            new_shape_size *= dim

        if np.size(v) != new_shape_size:
            raise ValueError('the size of feature {} ({}) could not '
                             'be reshaped into {}'.format(
                                 k, np.size(v), new_shape))

        if 'template' not in k:
            if np.size(v) <= 0:
                raise ValueError('The feature {} is not empty.'.format(k))

        parsed_features[k] = np.reshape(v, new_shape)

    return parsed_features


def np_to_array_dict(np_example, features, use_templates=False):
    """Creates dict of arrays."""
    features_metadata = _make_features_metadata(features)
    array_dict = {k: v for k, v in np_example.items()
                  if k in features_metadata}

    if 'template_domain_names' in np_example:
        num_template = len(np_example['template_domain_names'])
    else:
        num_template = 0

    array_dict = parse_reshape_logic(array_dict, features_metadata, num_template)
    if use_templates:
        array_dict['template_mask'] = np.ones([num_template], np.float32)

    return array_dict


def process_arrays_from_config(arrays, data_config):
    """Apply filters and maps to an existing dataset, based on the config."""
    def _wrap_ensemble_fn(data, i):
        """Function to be mapped over the ensemble dimension."""
        d = data.copy()
        fns = ensembled_map_fns(data_config)
        fn = compose(fns)
        d['ensemble_index'] = i
        return fn(d)

    eval_cfg = data_config.eval
    num_ensemble = eval_cfg.num_ensemble
    if data_config.common.resample_msa_in_recycling:
        num_ensemble *= data_config.common.num_recycle + 1

    eval_cfg = data_config.eval
    arrays = compose(
        nonensembled_map_fns(data_config))(arrays)
    arrays_0 = _wrap_ensemble_fn(arrays, np.array(0, np.int32))
    result_array = {x: () for x in arrays_0.keys()}

    if num_ensemble > 1:
        for i in range(num_ensemble):
            arrays_t = _wrap_ensemble_fn(arrays, np.array(i, np.int32))
            for key in arrays_0.keys():
                result_array[key] += (arrays_t[key][None],)

        for key in arrays_0.keys():
            result_array[key] = np.concatenate(result_array[key], axis=0)
    else:
        result_array = {key: arrays_0[key][None] for key in arrays_0.keys()}
    return result_array
