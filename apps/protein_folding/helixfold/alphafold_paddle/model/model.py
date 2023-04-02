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

"""Model."""

import os
import io
import time
import pickle
import logging
import pathlib
import numpy as np
import ml_collections
from copy import deepcopy
from typing import Dict, Optional

import paddle
from alphafold_paddle.model import utils
from alphafold_paddle.relax import relax
from alphafold_paddle.model import modules
from alphafold_paddle.common import protein
from alphafold_paddle.common import residue_constants
from utils.param_fuse import get_fused_param_groups

try:
  import tensorflow.compat.v1 as tf
  from alphafold_paddle.data.tf_input import input_pipeline
  from alphafold_paddle.data.tf_input import proteins_dataset

  USE_TF = True
except Exception:
  from alphafold_paddle.data.input import input_pipeline

  USE_TF = False

logger = logging.getLogger(__name__)

TARGET_FEAT_DIM = 22
MSA_FEAT_DIM = 49


def print_shape(d, level=0):
    tabs = '\t' * level
    for k, v in d.items():
        if type(v) is dict:
            print(tabs + k)
            print_shape(v, level=level+1)
        else:
            print(tabs + f'{k}: {v.shape} {v.dtype}')


def tensor_to_numpy(pred_dict):
    for k in pred_dict.keys():
        if isinstance(pred_dict[k], paddle.Tensor):
            pred_dict[k] = pred_dict[k].numpy()

        elif type(pred_dict[k]) is dict:
            tensor_to_numpy(pred_dict[k])


def slice_pred_dict(pred_dict, slice_idx, ignores=['breaks', 'traj', 'sidechains']):
    for k in pred_dict.keys():
        if k in ignores:
            continue

        if type(pred_dict[k]) is dict:
            pred_dict[k] = slice_pred_dict(pred_dict[k], slice_idx,
                                           ignores=ignores)

        else:
            pred_dict[k] = pred_dict[k][slice_idx]

    return pred_dict


class RunModel(object):
    """Wrapper for paddle model."""

    def __init__(self,
                 name: str,
                 config: ml_collections.ConfigDict,
                 params_path: str,
                 dynamic_subbatch_size: bool = True):
        self.name = name
        self.config = config
        self.dynamic_subbatch_size = dynamic_subbatch_size

        channel_num = {
            'target_feat': TARGET_FEAT_DIM,
            'msa_feat': MSA_FEAT_DIM,
        }
        self.alphafold = modules.AlphaFold(channel_num, config.model)
        # just only fuse param for inference
        fused_parameters = get_fused_param_groups(self.alphafold, config.model.global_config.get('dist_model', False))
        self.init_params(str(params_path))
        self.alphafold.eval()
        
        register_hooks_recursively(self.alphafold)
        

    def init_params(self, params_path: str):
        if params_path.endswith('.npz'):
            with open(params_path, 'rb') as f:
                params = np.load(io.BytesIO(f.read()), allow_pickle=False)
                params = dict(params)

            pd_params = utils.jax_params_to_paddle(params)
            pd_params = {k[len('alphafold.'):]: v for k, v in pd_params.items()}
            
            from collections import defaultdict
            qkv_dicts = defaultdict(dict)
            
            if self.config.model.global_config.fuse_attention:
                for key in pd_params:
                    if 'msa_column_global_attention' not in key and 'attention' in key and ('query_w' in key or 'key_w' in key or 'value_w' in key):
                        prefix = key[:key.rfind('.')]
                        if 'extra_msa_stack' in key:
                            qkv_dicts[prefix][key] = pd_params[key]
                            #print(key)
                        elif 'evoformer_iteration' in key:
                            qkv_dicts[prefix][key] = pd_params[key]
                            #print(key)
                        elif 'template_pair_stack' in key:
                            qkv_dicts[prefix][key] = pd_params[key]
                            #print(key)

                for prefix in qkv_dicts:
                    query_w = qkv_dicts[prefix][prefix + '.query_w']
                    key_w = qkv_dicts[prefix][prefix + '.key_w']
                    value_w = qkv_dicts[prefix][prefix + '.value_w']
                    if query_w.shape[0] == key_w.shape[0] and key_w.shape[0] == value_w.shape[0]:
                        # 1. merge to [3, num_head, key_dim, q_dim]
                        qkv_w = np.stack([query_w, key_w, value_w], axis=0).transpose((0, 2, 3, 1))
                        
                        # 2. remove seperated param
                        del pd_params[prefix + '.query_w']
                        del pd_params[prefix + '.key_w']
                        del pd_params[prefix + '.value_w']
                        
                        # 3. add merged param to pd_params
                        pd_params[prefix + '.qkv_w'] = qkv_w

        elif params_path.endswith('.pdparams'):
            logger.info('Load as Paddle model')
            pd_params = paddle.load(params_path)

        else:
            raise ValueError('Unsupported params file type')

        self.alphafold.set_state_dict(pd_params)

    def preprocess(self,
                   raw_features: Dict[str, np.ndarray],
                   random_seed: int,
                   pkl: pathlib.Path = None) -> Dict[str, paddle.Tensor]:
        """Convert raw input features to model input features"""
        if pkl is not None and pkl.exists():
            logger.info(f'Use cached {pkl}')
            with open(pkl, 'rb') as f:
                features = pickle.load(f)

            print('########## feature shape ##########')
            print_shape(features)
            return utils.map_to_tensor(features, add_batch=True)

        print('Processing input features')
        data_config = deepcopy(self.config.data)
        feature_names = data_config.common.unsupervised_features
        if data_config.common.use_templates:
            feature_names += data_config.common.template_features

        num_residues = int(raw_features['seq_length'][0])
        data_config.eval.crop_size = num_residues

        if 'deletion_matrix_int' in raw_features:
            raw_features['deletion_matrix'] = (raw_features.pop(
                'deletion_matrix_int').astype(np.float32))

        if USE_TF:
            data_config.eval.delete_msa_block = False

            tf_graph = tf.Graph()
            with tf_graph.as_default(), tf.device('/device:CPU:0'):
                tf.compat.v1.set_random_seed(random_seed)
                tensor_dict = proteins_dataset.np_to_tensor_dict(
                    np_example=raw_features, features=feature_names)

                processed_batch = input_pipeline.process_tensors_from_config(
                    tensor_dict, data_config)

            tf_graph.finalize()

            with tf.Session(graph=tf_graph) as sess:
                features = sess.run(processed_batch)

        else:

            array_dict = input_pipeline.np_to_array_dict(
                np_example=raw_features, features=feature_names,
                use_templates=data_config.common.use_templates)
            features = input_pipeline.process_arrays_from_config(
                array_dict, data_config)
            features = {k: v for k, v in features.items() if v.dtype != 'O'}

            extra_msa_length = data_config.common.max_extra_msa
            for k in ['extra_msa', 'extra_has_deletion', 'extra_deletion_value',
                      'extra_msa_mask']:
                features[k] = features[k][:, :extra_msa_length]

            for k in features.keys():
                if features[k].dtype == np.int64:
                    features[k] = features[k].astype(np.int32)

                elif features[k].dtype == np.float64:
                    features[k] = features[k].astype(np.float32)

        if pkl is not None:
            with open(pkl, 'wb') as f:
                pickle.dump(features, f, protocol=4)

        print('Preprocessesing finished')
        print('########## feature shape ##########')
        print_shape(features)
        return utils.map_to_tensor(features, add_batch=True)

    def predict(self,
                feat: Dict[str, paddle.Tensor],
                ensemble_representations: bool = True,
                return_representations: bool = True):
        """Predict protein structure and encoding representation"""
        if self.dynamic_subbatch_size:
            seq_len = feat['aatype'].shape[-1]
            extra_msa_num = feat['extra_msa'].shape[-2]
            self.update_subbatch_size(seq_len, extra_msa_num)

        with paddle.no_grad():
            ret = self.alphafold(
                feat, {},
                ensemble_representations=ensemble_representations,
                return_representations=return_representations,
                compute_loss=False)

        print('Prediction finished')
        tensor_to_numpy(ret)
        return ret

    def postprocess(self,
                    aatype: np.ndarray,
                    residue_index: np.ndarray,
                    relaxer: relax.AmberRelaxation,
                    prediction: Dict[str, np.ndarray],
                    output_dir: pathlib.Path,
                    slice_idx: int = 0,
                    timings: Optional[Dict[str, float]] = None):
        """Compute pLDDT, save unrelaxed pdb and execute relaxation"""
        print('Running postprocessing, it may takes long time')
        single_pred = slice_pred_dict(prediction, slice_idx)
        prediction.update(utils.get_confidence_metrics(single_pred))
        plddt = prediction['plddt']
        logger.info(f'{self.name} average pLDDT: {np.mean(plddt)}')

        if 'max_predicted_aligned_error' in prediction:
            err = prediction['max_predicted_aligned_error']
            logger.info(f'{self.name} max predicted aligned error: {err}')

        with open(output_dir.joinpath(f'result_{self.name}.pkl'), 'wb') as f:
            pickle.dump(prediction, f, protocol=4)

        plddt_b_factors = np.repeat(
            plddt[:, None], residue_constants.atom_type_num, axis=-1)
        prot = utils.generate_unrelaxed_pdb(
            aatype, residue_index, single_pred,
            output_dir.joinpath(f'unrelaxed_{self.name}.pdb'),
            b_factors=plddt_b_factors)

        if relaxer is not None:
            t0 = time.time()
            relaxed_pdb_str = relaxer.process(prot=prot)[0]
            if timings is not None:
                timings[f'relax_{self.name}'] = time.time() - t0

            pdb = f'relaxed_{self.name}.pdb'
            with open(output_dir.joinpath(pdb), 'w') as f:
                f.write(relaxed_pdb_str)

            print('Saved {}'.format(output_dir.joinpath(pdb)))

        else:
            relaxed_pdb_str = protein.to_pdb(prot)

        print('Postprocessing finished')
        return relaxed_pdb_str

    def update_subbatch_size(self, seq_len, extra_msa_num):
        if extra_msa_num == 5120:
            if seq_len < 200:
                # disable subbatch
                self.alphafold.global_config.subbatch_size = 5120

        elif extra_msa_num == 1024:
            if seq_len < 600:
                # disable subbatch
                self.alphafold.global_config.subbatch_size = 1024

        else:
            raise ValueError('Unknown subbatch strategy')
