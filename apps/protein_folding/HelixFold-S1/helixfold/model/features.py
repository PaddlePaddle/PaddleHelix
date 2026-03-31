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
"""Code to generate processed features."""

import copy
import ml_collections
import numpy as np
from typing import List, Mapping, Tuple

# try:
#   import tensorflow.compat.v1 as tf
#   from helixfold.data.tf_input import input_pipeline
#   from helixfold.data.tf_input import proteins_dataset

#   USE_TF = True
# except Exception:
#   from helixfold.data.input import input_pipeline

#   USE_TF = False

from helixfold.data.input.data_transforms import Seed_maker
from helixfold.data.input import input_pipeline

FeatureDict = Mapping[str, np.ndarray]


def make_data_config(
    config: ml_collections.ConfigDict,
    num_res: int,
) -> Tuple[ml_collections.ConfigDict, List[str]]:
    """Makes a data config for the input pipeline."""
    cfg = copy.deepcopy(config.data)

    feature_names = cfg.common.unsupervised_features
    if cfg.common.use_templates:
        feature_names += cfg.common.template_features

    with cfg.unlocked():
        cfg.eval.crop_size = num_res

    return cfg, feature_names


def np_example_to_features(np_example: FeatureDict,
                           config: ml_collections.ConfigDict,
                           random_seed: int = None) -> FeatureDict:
    """Preprocesses NumPy feature dict."""
    np_example = dict(np_example)

    if len(np_example['seq_length'].shape) == 0:
        num_res = int(np_example['seq_length'])
    else:
        num_res = int(np_example['seq_length'][0])

    cfg, feature_names = make_data_config(config, num_res=num_res)

    if 'deletion_matrix_int' in np_example:
        np_example['deletion_matrix'] = (
            np_example.pop('deletion_matrix_int').astype(np.float32))

    # if USE_TF:
    if False:
        import tensorflow.compat.v1 as tf
        from helixfold.data.tf_input import input_pipeline
        from helixfold.data.tf_input import proteins_dataset
        tf_graph = tf.Graph()
        with tf_graph.as_default(), tf.device('/device:CPU:0'):
            tf.compat.v1.set_random_seed(random_seed)
            tensor_dict = proteins_dataset.np_to_tensor_dict(
                np_example=np_example, features=feature_names)

            processed_batch = input_pipeline.process_tensors_from_config(
                tensor_dict, cfg)

        tf_graph.finalize()

        with tf.Session(graph=tf_graph) as sess:
            features = sess.run(processed_batch)

    else:
        if random_seed is None: random_seed = np.random.randint(10000000)
        Seed_maker.reset_init_seed(random_seed)
        from helixfold.data.input import input_pipeline
        array_dict = input_pipeline.np_to_array_dict(
            np_example=np_example,
            features=feature_names,
            use_templates=cfg.common.use_templates,
            multimer_mode=config.model.global_config.get(
                'multimer_mode', False))
        features = input_pipeline.process_arrays_from_config(array_dict, cfg)
        features = {k: v for k, v in features.items() if v.dtype != 'O'}

        extra_msa_length = cfg.common.max_extra_msa
        for k in [
                'extra_msa', 'extra_has_deletion', 'extra_deletion_value',
                'extra_msa_mask'
        ]:
            features[k] = features[k][:, :extra_msa_length]

        for k in features.keys():
            if features[k].dtype == np.int64:
                features[k] = features[k].astype(np.int32)
            elif features[k].dtype == np.float64:
                features[k] = features[k].astype(np.float32)

        if config.model.global_config.get('multimer_mode', False):
            # Monomer mode: target_feat: has_break + 21 one-hot
            # Multimer mode: target_feat: 21 one-hot aa
            features['target_feat'] = features['target_feat'][..., 1:]

        ## IMPORTANT: disable the random seed set by `np_example_to_features`,
        ##      otherwise, every data item will have the same random seed.
        np.random.seed(None)
    return features
