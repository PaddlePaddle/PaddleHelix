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

"""Full HelixFold3 structure prediction script."""

import os
import sys
import time
import json
import pickle
import random
import logging
import pathlib
import argparse
import shutil
import copy
import glob
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Mapping, Union

import paddle
from paddle import distributed as dist
from utils.model import RunModel
from utils.misc import set_logging_level
from utils.utils import get_custom_amp_list, find_top_k_upper_triangle_indices
from utils.interface_utils import (
    InterfaceInfo, 
    INFERENCE_INTERFACE_CONFIG, 
    get_interface_prob_from_module1)

from helixfold.model import config, utils
from helixfold.data.utils import (
    atom_level_keys, 
    map_to_continuous_indices,
    build_interface_info_with_random_residues_module2 as build_interface_info_with_random_residues)
from helixfold.common import all_atom_pdb_save
from ppfleetx.distributed.protein_folding.scg import scg

from infer_scripts.feature_processing_aa import load_ccd_dict
from infer_scripts.tools.post_calculate import (
    calculate_chain_pair_pae_matrix,
    calculate_token_plddts)
from infer_scripts.tools.utils import (
    read_json, 
    filter_job_name, 
    write_format_json, 
    convert_to_json_compatible,
    get_extra_infos_for_mmcif)
from infer_scripts.tools import mmcif_writer, draw_results

## results key need to be save.
DISPLAY_DIM = frozenset(["SINGLE", "NUM_CHAIN", "NUM_ATOM", "NUM_TOKEN", 
        "NUM_CHAIN, NUM_CHAIN", "NUM_ATOM, NUM_ATOM", "NUM_TOKEN, NUM_TOKEN"])
DISPLAY_RESULTS_KEYS = {
    'atom_chain_ids': "NUM_ATOM",
    'atom_plddt': "NUM_ATOM",
    'token_plddt': "NUM_TOKEN",
    'token_has_clash': "NUM_TOKEN",
    'token_chain_ids': "NUM_TOKEN",
    'token_res_ids': "NUM_TOKEN",
    'token_ptm': "NUM_TOKEN",
    'token_pair_pae': "NUM_TOKEN, NUM_TOKEN",
    'global_pae': "SINGLE",
    'global_pae_min': "SINGLE",
    'chain_has_clash': "NUM_CHAIN, NUM_CHAIN",
    'chain_plddt': "NUM_CHAIN", 
    'chain_pair_iptm': "NUM_CHAIN, NUM_CHAIN", 
    'chain_ptm': "NUM_CHAIN", 
    'chain_pair_iptm_min': "NUM_CHAIN",
    'chain_pair_pae_min': "NUM_CHAIN, NUM_CHAIN",
    'chain_pair_pae': "NUM_CHAIN, NUM_CHAIN",
    'global_iptm': "SINGLE",
    'global_iptm_min': "SINGLE",
    'global_ptm': "SINGLE",
    'global_has_clash': "SINGLE", 
    'global_plddt': "SINGLE",
    'ranking_confidence': "SINGLE",
}

## WARNING: 3.1.4 version, will be deprecated in the future.
OLD_DISPLAY_RESULTS_KEYS = [
    'atom_chain_ids',
    'atom_plddts',
    'pae',
    'token_chain_ids',
    'token_res_ids',
    'chain_plddt', ## chain level plddt
    'chain_pair_iptm', ## chain level iptm,
    'chain_ptm', ## chain level ptm,
    'iptm',
    'ptm',
    'ranking_confidence',
    'has_clash', 
    'mean_plddt',
]
OLD_DISPLAY_RESULTS_KEYS_MAPPING = {
    "token_pair_pae": "pae",
    "atom_plddt": "atom_plddts",
    "global_ptm": "ptm",
    "global_iptm": "iptm",
    "global_has_clash": "has_clash",
    "global_plddt": "mean_plddt",
}
## WARNING: 3.1.4 version, will be deprecated in the future.


NO_FUSE_ATTEN = os.getenv("NO_FUSE_ATTEN", "0") == "1"
DEBUG = os.getenv("DEBUG", "0") == "1"
DO_EVAL = os.getenv("DO_EVAL", "0") == "1"
SAVE_PDB = os.getenv("SAVE_PDB", "0") == "1"
print(f">>> DEBUG: {DEBUG}")
print(f">>> DO_EVAL: {DO_EVAL}")
print(f">>> SAVE_PDB: {SAVE_PDB}")
print(f">>> NO_FUSE_ATTEN: {NO_FUSE_ATTEN}")
logger = logging.getLogger(__file__)


def update_previous_direct_sampled_map(current_sampled_dict,
                                       previous_direct_sampled_map, 
                                       previous_direct_sampled_history, 
                                       previous_direct_sampled_info_pkl,
                                       only_update_history=False):
    """
        Update the previous direct sampled interfaces map and save to file.

        Args:
            current_sampled_dict: dict, containing 'score', 'prob', 'confidence', 'previous_indirect_samples', 'previous_direct_samples', 'top_indices'
            previous_direct_sampled_map: (n_token, n_token), sample history map from previous `direct` sampling (interface sample follow by top1 method)
            previous_direct_sampled_history: list, each item is a dict, containing 'score', 'prob', 'confidence', 'previous_indirect_samples', 'previous_direct_samples', 'top_indices'
            previous_direct_sampled_info_pkl: str, the path to save the updated previous direct sampled interfaces
            only_update_history: bool, if True, only update history, not update previous_direct_sampled_map
        
        Returns:
            None
    """
    previous_direct_sampled_history.append(current_sampled_dict)
    top_indices = current_sampled_dict['top_indices'] ## list of list [(1, 2)]
    assert len(top_indices) == 1
    
    # update previous direct sampled interfaces
    if not only_update_history:
        for index in top_indices:
            previous_direct_sampled_map[index[0], index[1]] += 1
            previous_direct_sampled_map[index[1], index[0]] += 1
    
    # save update previous direct sampled interfaces to file
    with open(previous_direct_sampled_info_pkl, 'wb') as f:
        previous_direct_sampled_info = {
            'sampled_map': previous_direct_sampled_map,
            'sampled_history': previous_direct_sampled_history
        }
        pickle.dump(previous_direct_sampled_info, f)

    if not only_update_history:
        print(f'[DEBUG-interface]: updated previous_direct_sampled_map sum is {previous_direct_sampled_map.sum()}')
    else:
        print(f'[DEBUG-interface]: only update history, history size is {len(previous_direct_sampled_history)}')


def sample_interface_by_top1(reweighted_interface_prob, interface_prob_batch_i, n_token, confidence_map, 
                             indirect_samples_map, previous_direct_sampled_map):
    """
        Sample interface point from direct prob map by top1 method.
        Args:
            reweighted_interface_prob: (n_token, n_token), the reweighted interface probability
            interface_prob_batch_i: (n_token, n_token), the raw interface probability from module1
            n_token: int, the number of tokens
            confidence_map: (n_token, n_token)
            indirect_samples_map: (n_token, n_token), sample history map from previous `indirect` sampling (interface cluster follow by diffusion results)
            previous_direct_sampled_map: (n_token, n_token), sample history map from previous `direct` sampling (interface sample follow by top1 method)
        Returns:
            selected_interface_info: (n_token, n_token), the selected interface info in this sample step. To inject into the feature later.
            current_sampled_dict: dict
    """
    top_indices, top_scores = find_top_k_upper_triangle_indices(reweighted_interface_prob, 1)
    print(f"[DEBUG-interface]: top_indices {top_indices}")

    sampled_prob = interface_prob_batch_i[top_indices[0][0], top_indices[0][1]]
    sampled_confidence = confidence_map[top_indices[0][0], top_indices[0][1]]
    sampled_previous_indirect_samples = indirect_samples_map[top_indices[0][0], top_indices[0][1]]
    sampled_previous_direct_samples = previous_direct_sampled_map[top_indices[0][0], top_indices[0][1]]

    current_sampled_dict = {
        'score': float(top_scores[0]),
        'prob': float(sampled_prob),
        'confidence': float(sampled_confidence),
        'previous_indirect_samples': int(sampled_previous_indirect_samples),
        'previous_direct_samples': int(sampled_previous_direct_samples),
        'top_indices': top_indices.tolist()
    }
    print(f'[DEBUG-interface]: sampled info is {current_sampled_dict}')
    
    selected_interface_info = InterfaceInfo(n_token, indexes=top_indices)
    selected_interface_map = selected_interface_info.interface_map()
    selected_interface_info = paddle.to_tensor(selected_interface_map, dtype='float32')

    return selected_interface_info, current_sampled_dict


def sample_interface_from_prob(batch, previous_sampled_interface_dir, repeat_i, sampling_method, 
                                    conf_metric_name, conf_metric_accumulate_method, weight_b, weight_c, 
                                    max_prob_decay_times=None, min_prob=None, min_prob_rank=None,
                                    enable_replacement_sampling=False, topn=None, total_repeat_nums=None):
    # get the interface probability
    interface_prob_batch = batch['feat']['interface_prob'].numpy() ## (n_sample=1, n_token, n_token)
    interface_sample_constraint_mask_batch = batch['feat']['interface_sample_constraint_mask'].numpy() ## (n_sample=1, n_token, n_token)
    interface_info_batch = []

    for batch_i in range(interface_prob_batch.shape[0]):
        n_token = batch['feat']['asym_id'].shape[1]
        interface_prob_batch_i = interface_prob_batch[batch_i]
        interface_sample_constraint_mask_batch_i = interface_sample_constraint_mask_batch[batch_i]
        interface_prob_batch_i *= interface_sample_constraint_mask_batch_i

        save_pkl = f"direct_sample_history.pkl"
        # get the previous direct sampled interfaces of the case
        previous_direct_sampled_info_pkl = os.path.join(previous_sampled_interface_dir, save_pkl)
        if os.path.exists(previous_direct_sampled_info_pkl):
            # load from file
            with open(previous_direct_sampled_info_pkl, 'rb') as pkl:
                previous_direct_sampled_info = pickle.load(pkl)
                previous_direct_sampled_map = previous_direct_sampled_info['sampled_map']
                previous_direct_sampled_history = previous_direct_sampled_info['sampled_history']
        else:
            logger.info(f'No previous direct sampled interfaces found in repeat {repeat_i},'\
                                f' initialize previous_direct_sampled_map to 0.')
            previous_direct_sampled_map = np.zeros((n_token, n_token), dtype=np.int32)
            previous_direct_sampled_history = []
            previous_direct_sampled_info = {
                'sampled_map': previous_direct_sampled_map,
                'sampled_history': previous_direct_sampled_history
            }

        print(f'[DEBUG-interface]: loaded previous_direct_sampled_map sum is {previous_direct_sampled_map.sum()}' \
                    f'| file is {save_pkl} | repeat_i is {repeat_i}')

        indirect_samples_map, confidence_map, previous_interface_maps_allnode = get_indirect_samples_and_confidence_maps(
                                                batch, previous_sampled_interface_dir, 
                                                repeat_i, 'indirect_interface', 
                                                conf_metric_name, conf_metric_accumulate_method, 
                                                max_history_nums=topn if topn and sampling_method.startswith('topn_cluster') \
                                                                        else total_repeat_nums)
        print(f'[DEBUG-interface]: indirect_samples_map sum is {indirect_samples_map.sum()}')

        print(f'[DEBUG-interface]: weight_b is {weight_b}')
        print(f'[DEBUG-interface]: weight_c is {weight_c}')
        print(f'[DEBUG-interface]: repeat_i is {repeat_i}')

        if min_prob is not None:
            if min_prob < interface_prob_batch_i.max():
                min_valid_prob = min_prob
            else:
                min_valid_prob = interface_prob_batch_i.max()/2
            print(f'[DEBUG-interface]: min_valid_prob is {min_valid_prob}')
        elif max_prob_decay_times is not None:
            min_valid_prob = interface_prob_batch_i.max() / max_prob_decay_times
            print(f'[DEBUG-interface]: max_prob_decay_times={max_prob_decay_times}, max_prob={interface_prob_batch_i.max()}, min_valid_prob={min_valid_prob}')
        elif min_prob_rank is not None:
            result_indices, top_k_values = find_top_k_upper_triangle_indices(interface_prob_batch_i, min_prob_rank)
            if min_prob_rank > len(top_k_values):
                min_valid_prob = top_k_values[-1]
            else:
                min_valid_prob = top_k_values[min_prob_rank-1]
        else:
            min_valid_prob = 0


        new_valid_interface_mask = (interface_prob_batch_i > min_valid_prob).astype(np.float32) # (n_token, n_token)
        if not enable_replacement_sampling:
            # maskout previous sampled positions
            new_valid_interface_mask *= (previous_direct_sampled_map < 1).astype(np.float32)
        
        if sampling_method.startswith('topn_cluster'):
            assert topn is not None, f'sampling_method is {sampling_method}, but topn is None, please check the input arguments.'
            print(f'[WARNING-interface]: {sampling_method} is {topn}, skip the previous indirect_interface_map and confidence_map; weight_b and weight_c are ignored.')
            if repeat_i < topn:
                ## token-pair within interface cluster are be sampling without replacement.
                _indirect_mask = (indirect_samples_map == 0).astype(np.float32)
                if _indirect_mask.sum() > 0:
                    new_valid_interface_mask *= _indirect_mask
                else:
                    print(f'==> WARNING: history of indirect_samples_map is all 1, set topn to -1 and sample from history now.')
                    topn = -1 # skip the top1 sampling and update previous direct sampled interfaces

            reweighted_interface_prob = interface_prob_batch_i * new_valid_interface_mask
        else:
            reweighted_interface_prob = (interface_prob_batch_i + weight_b * confidence_map \
                                        + weight_c * (np.log(repeat_i+1) / (indirect_samples_map+previous_direct_sampled_map+1))) * new_valid_interface_mask


        if reweighted_interface_prob.sum() == 0:
            print(f'==> [WARNING ZERO PROBS] reweighted_interface_prob is all 0 in repeat {repeat_i}, random sample token-pair.')
            ## NOTE: if reweighted_interface_prob is all 0, random sample token-pair by the sample constraint mask.
            reweighted_interface_prob += interface_sample_constraint_mask_batch_i

        ## sample a new interface from the reweighted probability
        ## NOTE: top1 sample is always performed.
        selected_interface_info, current_sampled_dict = sample_interface_by_top1(
                                        reweighted_interface_prob, interface_prob_batch_i, 
                                        n_token, confidence_map, 
                                        indirect_samples_map, previous_direct_sampled_map)

        if sampling_method == 'top1':
            interface_info_batch.append(selected_interface_info)
            update_previous_direct_sampled_map(current_sampled_dict, 
                                            previous_direct_sampled_map,
                                            previous_direct_sampled_history, 
                                            previous_direct_sampled_info_pkl)
        elif sampling_method in ['topn_random', 'topn_cluster', 'topn_cluster_Allnode']:
            assert topn is not None
            if repeat_i >= topn:
                previous_direct_sampled_map = previous_direct_sampled_info['sampled_map']
                previous_direct_sampled_history = previous_direct_sampled_info['sampled_history']
                print(f'[DEBUG-interface]: method={sampling_method} | repeat_i={repeat_i} >= topn={topn}, skip sampling, ' \
                                        f'history size of previous_direct_sampled_history is {len(previous_direct_sampled_history)}')

                if sampling_method == 'topn_cluster_Allnode':
                    history_size = len(previous_interface_maps_allnode) if topn != -1 else previous_direct_sampled_map.sum() // 2
                    print(f'[DEBUG-interface]: method={sampling_method} | history size of previous_interface_maps_allnode is {history_size}')
                    # selected_key = np.random.choice(list(previous_interface_maps_allnode.keys()))  # random select one key
                    circular_index = (repeat_i - history_size) % history_size ## circularly sampling
                    if (history_size > 0 and circular_index in previous_interface_maps_allnode 
                                        and len(previous_interface_maps_allnode[circular_index]) > 0):
                        _node_idx = np.random.choice(len(previous_interface_maps_allnode[circular_index]))
                        top_indices = np.array(previous_interface_maps_allnode[circular_index][_node_idx]).reshape(1, 2)
                        print(f'[DEBUG-interface]: method={sampling_method} | repeat_i={repeat_i} | selected_repeat_key is {circular_index}')
                    else:
                        print(f'[WARNING-interface]: method={sampling_method} | history of {circular_index} is empty, random sample from history of previous_direct_sampled_map')
                        indices_tuple = np.nonzero(previous_direct_sampled_map)
                        selected_index = np.random.choice(len(indices_tuple[0]))
                        top_indices = np.array([indices_tuple[0][selected_index], indices_tuple[1][selected_index]]).reshape(1, 2)
                else:
                    print(f'[DEBUG-interface]: method={sampling_method} | random sample from history of previous_direct_sampled_map')
                    # random select one interface from previous_direct_sampled_map
                    indices_tuple = np.nonzero(previous_direct_sampled_map)
                    selected_index = np.random.choice(len(indices_tuple[0]))
                    top_indices = np.array([indices_tuple[0][selected_index], indices_tuple[1][selected_index]]).reshape(1, 2)
                            
                print(f'[DEBUG-interface]: Selected top_indices from history is {top_indices}')
                selected_interface_info = InterfaceInfo(n_token, indexes=top_indices)
                selected_interface_map = selected_interface_info.interface_map()
                selected_interface_info = paddle.to_tensor(selected_interface_map, dtype='float32')
                
                ## NOTE: only update history, not update previous_direct_sampled_map when circularly sampling.
                history_repeat_sampled_dict = {
                    'score': float(reweighted_interface_prob[top_indices[0][0], top_indices[0][1]]),
                    'prob': float(interface_prob_batch_i[top_indices[0][0], top_indices[0][1]]),
                    'confidence': 'circularly sampling, no confidence',
                    'previous_indirect_samples': '-1',
                    'previous_direct_samples': '-1',
                    'top_indices': top_indices.tolist()
                }
                update_previous_direct_sampled_map(history_repeat_sampled_dict, 
                                                previous_direct_sampled_map, 
                                                previous_direct_sampled_history, 
                                                previous_direct_sampled_info_pkl,
                                                only_update_history=True)
            else:
                update_previous_direct_sampled_map(current_sampled_dict, 
                                                previous_direct_sampled_map, 
                                                previous_direct_sampled_history, 
                                                previous_direct_sampled_info_pkl)
            interface_info_batch.append(selected_interface_info) 
        else:
            raise ValueError(f'Unsupported interface sampling method: {sampling_method}')        

    # Add the sampled interface to the batch
    interface_info_batch = paddle.stack(interface_info_batch)
    batch['feat']['interface_info'] = interface_info_batch

    return batch


def get_indirect_samples_and_confidence_maps(batch, previous_sampled_interface_dir, repeat_i, case_name, conf_metric_name,  
                                conf_metric_accumulate_method, max_history_nums=None):
    assert max_history_nums is not None, f'max_history_nums in indirect_samples is None, please check the input arguments.'

    def median_without_zero(values):
        values_without_zero = values[values != 0.]
        if len(values_without_zero) == 0:
            return 0
        else:
            return np.median(values_without_zero)

    def mean_without_zero(values):
        non_zero_elements = values[values != 0.]
        return non_zero_elements.mean() if len(non_zero_elements) > 0 else 0

    previous_interface_maps = []
    previous_interface_confs = []
    previous_interface_maps_allnode = {} ## dict {repeat_i: sparse all_cluster_node}

    if repeat_i >= max_history_nums:
        print(f'[WARNING-interface]: get_indirect_samples_and_confidence_maps set repeat_i={repeat_i} to max_history_nums={max_history_nums}')
        repeat_i = max_history_nums

    for j in range(repeat_i):
        interface_pkl_path = os.path.join(previous_sampled_interface_dir, case_name,
                                                            f"structure-derived-interface-repeat{j}.pkl")
        if os.path.exists(interface_pkl_path):
            # load from file
            with open(interface_pkl_path, 'rb') as pkl:
                repeat_j_interface = pickle.load(pkl)

            interface_map = repeat_j_interface['interface_map']
            confidence_dict = repeat_j_interface.get('confidences', {})
            if conf_metric_name is None:
                conf_value = 0
            elif conf_metric_name in confidence_dict:
                conf_value = confidence_dict[conf_metric_name]
            else:
                raise ValueError(f"conf_metric_name {conf_metric_name} not in the confidence dict {confidence_dict} of {interface_pkl_path}")
            
            previous_interface_maps.append(interface_map)
            previous_interface_confs.append(interface_map * conf_value)
            previous_interface_maps_allnode[j] = list(zip(*np.nonzero(interface_map)))

    if len(previous_interface_maps) > 0:
        print(f'[DEBUG-interface]: Found {len(previous_interface_maps)} previous interface maps.')
        previous_interface_maps = np.stack(previous_interface_maps, axis=-1)
        previous_interface_confs = np.stack(previous_interface_confs, axis=-1)

        indirect_samples_map = np.sum(previous_interface_maps, axis=-1)
        if conf_metric_accumulate_method == 'max':
            confidence_map = np.amax(previous_interface_confs, axis=-1)
        elif conf_metric_accumulate_method == 'mean':
            confidence_map = np.apply_along_axis(mean_without_zero, -1, previous_interface_confs)
        elif conf_metric_accumulate_method == 'median':
            confidence_map = np.apply_along_axis(median_without_zero, -1, previous_interface_confs)
        else:
            raise ValueError(f"Unsupported conf_metric_accumulate_method: {conf_metric_accumulate_method}")
    else:
        print(f'[DEBUG-interface]: no previous interface found.')
        n_token = batch['feat']['asym_id'].shape[1]
        indirect_samples_map = np.zeros((n_token, n_token), dtype=np.int32)
        confidence_map = np.zeros((n_token, n_token), dtype=np.float32)

    return indirect_samples_map, confidence_map, previous_interface_maps_allnode


def update_previous_indirect_sampled_map(batch, results, previous_sampled_interface_dir, repeat_i, dist_thres):
    
    batch_size, n_token = batch['feat']['asym_id'].shape[:2]
    sample_constraint_mask_batch = batch['feat']['interface_sample_constraint_mask'].numpy() ## (n_sample=1, n_token, n_token)
    for batch_i in range(batch_size):
        sample_constraint_mask = sample_constraint_mask_batch[batch_i]
       
        # get new predicted interface
        diff_results = results['diffusion_module']
        pred_pos = diff_results['final_atom_positions'][batch_i].numpy()
        all_centra_token_indice = batch['label']['all_centra_token_indice'][batch_i].numpy()
        pred_token_pos = pred_pos[all_centra_token_indice]

        asym_id = batch['feat']['asym_id'][batch_i].numpy()
        entity_id = batch['feat']['entity_id'][batch_i].numpy()
        centra_token_mask = batch['label']['all_centra_token_indice_mask'][batch_i].numpy().astype('bool')
        # compute the predicted interface matrix from the predicted token positions
        pred_interface_list = build_interface_info_with_random_residues(asym_id,
            atom_pos=pred_pos, 
            atom_mask=batch['label']['all_atom_pos_mask'][batch_i].numpy(),
            atom_to_token_mapping=batch['feat']['ref_token2atom_idx'][batch_i].numpy(),
            token_pos=pred_token_pos,
            token_pos_mask=centra_token_mask, top_n=-1, m=-1, dist_thres=dist_thres,  seed=None)
        pred_interface = pred_interface_list[0].interface_map().astype('float32')
        pred_interface *= sample_constraint_mask

        # update previous_indirect_sampled_map
        # previous_indirect_sampled_map += pred_interface

        conf_results = results['confidence_head']
        confidence_score_names = [
                'ptm', 'iptm', 'has_clash',
                'ranking_confidence']
        confidence_score_dict = {k: float(conf_results[k][batch_i]) 
                for k in confidence_score_names}
        print(f'[DEBUG-interface]: confidence score dict is {confidence_score_dict}')
        repeat_i_interface = {
            'interface_map': pred_interface,
            'confidences': confidence_score_dict
        }

        case_sampled_interface_dir = os.path.join(previous_sampled_interface_dir, "indirect_interface")
        if not os.path.exists(case_sampled_interface_dir):
            os.makedirs(case_sampled_interface_dir)

        repeat_i_interface_pkl = os.path.join(case_sampled_interface_dir, f"structure-derived-interface-repeat{repeat_i}.pkl")
        # save updated previous_indirect_sampled_map
        with open(repeat_i_interface_pkl, 'wb') as f:
            pickle.dump(repeat_i_interface, f)
        
    return None


def update_model_config_quickly_debug(model_config, args):
    """
        Update the model config quickly for debug. Read variables from os.environ.
        Args:
            model_config: the model config.
            args: the args from the command line.
    """
    ####### IMPORTANT DEBUG #######
    if NO_FUSE_ATTEN:
        ### NOTE: A800 support fuse_attention, but is not support in a100 due to environment limitations.
        logger.warning('Mute Fuse_attention!!')
        model_config.model.global_config.fuse_attention = False
    if DEBUG:
        logger.warning('QUICKLY DEBUG MODE!! set test_diff_batch_size to 1, step_num to 10, num_recycle to 1')
        model_config.model.heads.diffusion_module.test_diff_batch_size = 1
        model_config.model.heads.diffusion_module.step_num = 10
        model_config.model.num_recycle = 0
    ####### IMPORTANT DEBUG #######


def fuse_attention_key_update(pd_params):
    """ update keys in pd_params"""
    qkv_dicts = defaultdict(dict)
    for key in pd_params:
        if  'attention' in key \
            and ('query_w' in key or 'key_w' in key or 'value_w' in key):
            prefix = key[:key.rfind('.')]
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
    return pd_params


def init_seed(seed):
    """ set seed for reproduct results"""
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def init_distributed_env(args):
    """ init ddp and dap distributed environment"""
    dp_rank = 0 # ID for current device in distributed data parallel collective communication group
    dp_nranks = 1 # The number of devices in distributed data parallel collective communication group
    if args.distributed:
        # init bp, dap, dp hybrid distributed environment
        assert args.bp_degree > 1 or args.dap_degree > 1, f"distributed inference required args.bp_degree > 1 or args.dap_degree > 1, \
            but got args.bp_degree: {args.bp_degree}, args.dap_degree: {args.dap_degree}"
        scg.init_group(bp_degree=args.bp_degree, dap_degree=args.dap_degree, dap_comm_sync=args.dap_comm_sync)

        dp_nranks = scg.get_dp_world_size()
        dp_rank = scg.get_dp_rank_in_group() if dp_nranks > 1 else 0

        if args.bp_degree > 1 or args.dap_degree > 1:
            assert args.seed is not None, "BP and DAP should be set seed!"

    return dp_rank, dp_nranks


def batch_convert(np_array, add_batch=True):
    """
        Func of convert numpy to paddle tensor, also add batch dim.
    """
    np_type = {}
    other_type = {}
    # 
    for key, value in np_array.items():
        if type(value) == np.ndarray:
            np_type.update(utils.map_to_tensor({key: value}, add_batch=add_batch))
        else:
            ## TODO: @zhukunrui, need to check this code?? why raise error??
            # raise ValueError(f"{key} dtype: {type(value)} not supported")
            other_type[key] = [value]

    return {**np_type, **other_type}


def tensor_to_numpy(common_feat):
    """
        Func of convert paddle(tensor) to numpy.
    """
    for feat_key in common_feat:
        if isinstance(common_feat[feat_key], paddle.Tensor):
            if common_feat[feat_key].dtype == paddle.bfloat16:
                common_feat[feat_key] = paddle.cast(common_feat[feat_key], 'float32').numpy()
            else:
                common_feat[feat_key] = common_feat[feat_key].numpy()
        if feat_key in ['residue_index', 'asym_id']:
            common_feat[feat_key] = common_feat[feat_key].astype(np.int32)
    
    return common_feat


def dump_cif(name, batch, results, output_dir, timings, mmcif_extra_infos):
    """
        HF3 inference to *.cif file.
            - batch. input data
            - results. model output
            - output_dir. to save output
            - timings. timings of each step, dict
            - mmcif_extra_infos. extra infos for cif wirter, dict
    """

    t0 = time.time()
    diff_results = results['diffusion_module'] 
    confidence_results = results['confidence_head']

    required_keys = copy.deepcopy(mmcif_writer.required_keys_for_saving)
    required_keys += ['token_bonds_type', 'ref_element', 'is_ligand', 'atom_plddts']

    # 1 feat extraction
    common_feat = {k: batch['feat'][k][0]
            for k in required_keys if k in batch['feat']}
    common_feat.update(
        {k: batch['label'][k][0]
            for k in required_keys if k in batch['label']}
    )
    common_feat.update(
        {'atom_plddts': confidence_results['atom_plddts'][0]})

    common_feat['all_chain_ids'] = str(common_feat['all_chain_ids']).split()
    common_feat['all_ccd_ids'] = str(common_feat['all_ccd_ids']).split()
    common_feat['all_atom_ids'] = str(common_feat['all_atom_ids']).split()
    ## asym_id start with 1
    common_feat['asym_id'] -= 1
    ## resid start with 1
    common_feat['residue_index'] += 1

    pred_dict = {
        "pos": diff_results['final_atom_positions'].numpy(),
        "mask": diff_results['final_atom_mask'].numpy(),
    }
    exp_dict = {
        "mask": batch['label']['all_atom_pos_mask'].numpy(),
    }

    atom_mask = np.logical_and(pred_dict["mask"] > 0, exp_dict["mask"] > 0)[0]  # [N_atom]
    token_mask = batch['label']['all_centra_token_indice_mask'][0].numpy().astype('bool')
    common_feat = tensor_to_numpy(common_feat)

    def apply_mask(key, val):
        """ apply mask to val """
        val = np.array(val)
        if key in atom_level_keys or key in ['atom_plddts']:
            if key in ['ref_token2atom_idx']:
                return map_to_continuous_indices(val[atom_mask])
            return val[atom_mask]
        else:
            if key in ['token_bonds_type']:
                return val[token_mask, :][:, token_mask] 
            return val[token_mask]
    common_feat_masked = {k: apply_mask(k, v) for k, v in common_feat.items()}

    ## save prediction masked 
    pred_cif_path = f'{output_dir}/predicted_structure.cif'
    mmcif_writer.prediction_to_mmcif(   # 注意这个函数会写两个 cif，两种链名模式
        entry_name=name, 
        atom_positions=pred_dict["pos"][0][atom_mask], 
        feats_dict=common_feat_masked, 
        mmcif_path=pred_cif_path,
        extra_infos=mmcif_extra_infos)
    
    if DO_EVAL or SAVE_PDB:
        ## save prediction pdb 
        pred_pdb_path = f'{output_dir}/predicted_structure.pdb'
        common_feat_masked_pdb = copy.deepcopy(common_feat_masked)
        common_feat_masked_pdb['asym_id'] += 1
        all_atom_pdb_save.prediction_to_pdb(
            pred_atom_pos=pred_dict["pos"][0][atom_mask], 
            FeatsDict=common_feat_masked_pdb, 
            pdb_file_path=pred_pdb_path)
        
        pred_atom_pos = f'{output_dir}/predicted_pos_no_mask.npy'
        np.save(pred_atom_pos, pred_dict["pos"][0])

    assert os.path.exists(pred_cif_path),\
              (f"pred: {pred_cif_path} not exists! please check it")
    timings[f'postprocess'] = time.time() - t0


def dump_metric_results(batch, results, output_dir, timings):
    """
        HF3 inference to all_results.json file.
    """
    def _reorder_results(results, ori_chain_ids, ori_chain_ids_token):
        """
            Reorder the inference results by the original chain ids.
            Args:
                results: dict, inference results.
                ori_chain_ids: np.ndarray, original chain ids for atom.
                ori_chain_ids_token: np.ndarray, original chain ids for token.
            Returns:
                reordered_results: dict, reordered inference results.
        """
        seen = set()
        # Maintain the order of unique chain IDs
        ori_chain_level = [x for x in ori_chain_ids_token if not (x in seen or seen.add(x))]
        ent_weights_chain, sym_weights_chain = mmcif_writer.user_asymid_to_weight(ori_chain_level)
        ent_weights_token, sym_weights_token = mmcif_writer.user_asymid_to_weight(ori_chain_ids_token)
        ent_weights_atom, sym_weights_atom = mmcif_writer.user_asymid_to_weight(ori_chain_ids)
        
        sorted_indices_chain = np.lexsort((sym_weights_chain, ent_weights_chain))
        sorted_indices_token = np.lexsort((sym_weights_token, ent_weights_token))
        sorted_indices_atom = np.lexsort((sym_weights_atom, ent_weights_atom))
        
        reordered_results = {}
        for key in list(results.keys()):
            assert DISPLAY_RESULTS_KEYS[key] in DISPLAY_DIM, \
                f"key {key} not in DISPLAY_DIM, Got {DISPLAY_RESULTS_KEYS[key]}, " \
                f"Expected keys are: {DISPLAY_DIM}"
            _res = results.pop(key)
            if DISPLAY_RESULTS_KEYS[key] == 'SINGLE':
                reordered_results[key] = _res
            elif DISPLAY_RESULTS_KEYS[key] == 'NUM_CHAIN':
                reordered_results[key] = np.take(_res, sorted_indices_chain, axis=0) 
            elif DISPLAY_RESULTS_KEYS[key] == 'NUM_TOKEN':
                reordered_results[key] = np.take(_res, sorted_indices_token, axis=0) 
            elif DISPLAY_RESULTS_KEYS[key] == 'NUM_ATOM':
                reordered_results[key] = np.take(_res, sorted_indices_atom, axis=0) 
            elif DISPLAY_RESULTS_KEYS[key] == 'NUM_CHAIN, NUM_CHAIN':
                reordered_results[key] = np.take(_res, sorted_indices_chain, axis=0) 
                reordered_results[key] = np.take(reordered_results[key], sorted_indices_chain, axis=1) 
            elif DISPLAY_RESULTS_KEYS[key] == 'NUM_TOKEN, NUM_TOKEN':
                reordered_results[key] = np.take(_res, sorted_indices_token, axis=0) 
                reordered_results[key] = np.take(reordered_results[key], sorted_indices_token, axis=1)
            else:
                raise ValueError(f"key {key} not supported in _reorder_results yet.")

        return reordered_results
    

    t0 = time.time()
    already_metric = ['chain_plddt', 'chain_pair_iptm', 
                        'chain_pair_mask', 'chain_has_clash', 'token_has_clash']
    confi_float_names = ['ptm', 'iptm', 'has_clash', 'ranking_confidence']
    confi_other_names = ['atom_plddts', 'pae', 'token_iptm', 'token_ptm']
    required_atom_level_keys = atom_level_keys + ['atom_plddts']
    display_required_keys = ['all_ccd_ids', 'all_atom_ids', 'all_chain_ids',
                             'chain_ids', 'asym_id',
                            'ref_token2atom_idx', 'restype', 
                            'residue_index',
                            'all_atom_pos_mask',]

    diff_results = results['diffusion_module']
    # 1 feat extraction
    common_feat = {k: batch['feat'][k][0]
            for k in display_required_keys if k in batch['feat']}
    common_feat.update(
        {k: batch['label'][k][0]
            for k in  display_required_keys if k in batch['label']})
    common_feat.update({k: results['confidence_head'][k][0] \
            for k in confi_other_names + already_metric})
    for k in ['all_chain_ids', 'all_ccd_ids', 'all_atom_ids', 'chain_ids']:
        common_feat[k] = str(common_feat[k]).split()
    common_feat['asym_id'] -= 1
    common_feat['residue_index'] += 1
    pred_dict = {
        "pos": diff_results['final_atom_positions'].numpy(),
        "mask": diff_results['final_atom_mask'].numpy()}
    exp_dict = {
        "mask": batch['label']['all_atom_pos_mask'].numpy()}

    atom_mask = np.logical_and(pred_dict["mask"] > 0, exp_dict["mask"] > 0)[0]  # [N_atom] get valid atom
    token_mask = batch['label']['all_centra_token_indice_mask'][0].numpy().astype('bool') # get valid token
    common_feat = tensor_to_numpy(common_feat)

    def apply_mask(key, val):
        """ apply mask to val """
        val = np.array(val)
        if key in already_metric:
            return val
        elif key in required_atom_level_keys:
            if key in ['ref_token2atom_idx']:
                return map_to_continuous_indices(val[atom_mask])
            return val[atom_mask]
        else:
            if key in ['token_bonds_type', 'pae']:
                return val[token_mask, :][:, token_mask] 
            return val[token_mask]
    common_feat_masked = {k: apply_mask(k, v) for k, v in common_feat.items()}

    ## NOTE: save display results.
    all_results = {}
    ref_token2atom_idx = common_feat_masked['ref_token2atom_idx']
    ori_chain_ids = common_feat_masked['all_chain_ids'] # N_atom
    ori_chain_ids_token = common_feat_masked['chain_ids'] # N_token

    ## 1. SINGLE
    all_results['global_ptm'] = float(results['confidence_head']['ptm'])
    all_results['global_iptm'] = float(results['confidence_head']['iptm']) \
                            if np.unique(ori_chain_ids_token).size > 1 else np.nan
    all_results['global_has_clash'] = float(results['confidence_head']['has_clash'])
    all_results['global_plddt'] = common_feat_masked['atom_plddts'].mean()
    all_results['ranking_confidence'] = float(results['confidence_head']['ranking_confidence']) \
        if np.unique(ori_chain_ids_token).size > 1 else all_results['global_ptm'] - all_results['global_has_clash']

    ## 2. token-level
    all_results['token_ptm'] = common_feat_masked['token_ptm']
    all_results['token_has_clash'] = common_feat_masked['token_has_clash']
    all_results['token_pair_pae'] = common_feat_masked['pae']
    all_results['global_pae'] = np.mean(common_feat_masked['pae'])
    all_results['global_pae_min'] = np.min(common_feat_masked['pae'])
    all_results['token_chain_ids'] = ori_chain_ids_token.tolist()
    all_results['token_res_ids'] = common_feat_masked['residue_index'].tolist()
    all_results['token_plddt'] = calculate_token_plddts(common_feat_masked['atom_plddts'], 
                                                            ref_token2atom_idx)

    ## 3. atom-level
    all_results['atom_plddt'] = common_feat_masked['atom_plddts']
    all_results['atom_chain_ids'] = ori_chain_ids.tolist()

    ## 4. chain-level
    all_results['chain_plddt'] = common_feat_masked['chain_plddt']
    all_results['chain_has_clash'] = common_feat_masked['chain_has_clash']
    all_results['chain_pair_iptm'] = np.where(common_feat_masked['chain_pair_mask'] == 1, 
                                                common_feat_masked['chain_pair_iptm'], np.nan)
    _chain_ptm = np.diagonal(common_feat_masked['chain_pair_iptm'])
    all_results['chain_ptm'] = np.where(np.diagonal(common_feat_masked['chain_pair_mask']) == 1, 
                                        _chain_ptm, np.nan)
    all_results['chain_iptm'] = np.full_like(all_results['chain_ptm'], np.nan, dtype=np.float64)
    all_results['chain_pair_iptm_min'] = np.full_like(all_results['chain_ptm'], np.nan, dtype=np.float64)
    _chain_iptm_mask = common_feat_masked['chain_pair_mask'] \
                * (1 - np.eye(common_feat_masked['chain_pair_mask'].shape[0]))
    for i in range(_chain_iptm_mask.shape[0]):
        mask = _chain_iptm_mask[i] == 1
        if mask.sum() > 0:
            all_results['chain_iptm'][i] = np.mean(common_feat_masked['chain_pair_iptm'][i][mask]) 
            all_results['chain_pair_iptm_min'][i] = np.min(common_feat_masked['chain_pair_iptm'][i][mask])
    _chain_iptm = all_results.pop('chain_iptm')
    all_results['global_iptm_min'] = np.min(_chain_iptm) if np.unique(ori_chain_ids_token).size > 1 else np.nan
    np.fill_diagonal(all_results['chain_pair_iptm'], np.nan)

    _chain_pae_mask = np.diagonal(common_feat_masked['chain_pair_mask']).astype('bool')
    all_results['chain_pair_pae_min'] = calculate_chain_pair_pae_matrix(
                                            all_results['token_pair_pae'], all_results['token_chain_ids'], 
                                            mask_chain=_chain_pae_mask, mask_value=np.nan, 
                                            metric_type='min')
    all_results['chain_pair_pae'] = calculate_chain_pair_pae_matrix(
                                            all_results['token_pair_pae'], all_results['token_chain_ids'], 
                                            mask_chain=_chain_pae_mask, mask_value=np.nan, 
                                            metric_type='mean')                            

    t1 = time.time()
    all_results = _reorder_results(all_results, ori_chain_ids, ori_chain_ids_token)
    t2 = time.time()

    ## final results and save to json file.
    final_results = {}
    for k in DISPLAY_RESULTS_KEYS:
        if k in all_results:
            final_results[k] = convert_to_json_compatible(all_results[k])
            if k in OLD_DISPLAY_RESULTS_KEYS_MAPPING:
                final_results[OLD_DISPLAY_RESULTS_KEYS_MAPPING[k]] = final_results[k]
        else:
            raise ValueError(f'Key {k} not found in result; Required keys are: {DISPLAY_RESULTS_KEYS}.')

    if os.environ.get('DEBUG', False):
        with open(output_dir.joinpath('all_results.pkl'), 'wb') as f:
            pickle.dump(final_results, f)
    write_format_json(final_results, file_path=output_dir.joinpath('all_results.json'), nan_to_none=True)

    timings[f'get_display_results'] = t1 - t0
    timings[f'reorder_results'] = t2 - t1


def dump_interface_and_summary_info(batch_feats: dict, results: dict, history_dir: str, out_dir: str):
    """Dump the interface info to a files, including interface_info, sample_history, and so on.
        Save overall interface info to a json file/csv file and token-pair interface probability to a heatmap png.
            <out_dir>/
                sample_infos.csv
                    - model_id, sampling_left_entity, sampling_right_entity, sampling_probability
                predicted_interface.json 
                    - token_chain_ids: [chain_id1, chain_id2, ...] (N_token)
                    - token_pair_interface_probs: [[prob1, prob2, ...], [prob1, prob2, ...], ...] (N_token, N_token)
                predicted_interface.png
        NOTE: Becareful, this function will reorder the interface info to the original order of the input data.
        
        Args:
            batch_feats: dict, the features from the input data.
            results: dict, the results from the inference.
            history_dir: str, the directory contains sample_history.pkl
                such as: **/TEMP_OUTPUT/previous_sampled_interface/direct_sample_history.pkl
    """
    def _get_chain_indices(chain_ids: np.ndarray) -> dict:
        """Returns a list of tuples indicating the start and end indices for each chain.

            Args:
                chain_ids: np.ndarray (N_token), the chain ids for the token. such as: [1-1, 1-1, 1-2, 2-1, 2-2, ...]
            Returns:
                chain_starts_ends: dict, the start and end indices for each chain.
                    such as: {1-1: (0, 1), 1-2: (2, 3), 2-1: (4, 5), 2-2: (6, 7), ...}
        """

        chain_starts_ends = {}
        unique_chains = np.unique(chain_ids) # chains are numbered 1-1, 1-2, 2-1, 2-2, ...

        for chain in unique_chains:
            positions = np.where(chain_ids == chain)[0]
            chain_starts_ends[chain] = (positions[0], positions[-1])

        return chain_starts_ends

    def _get_token_pair_offset_for_entity(chain_ids: np.ndarray, row: int, col: int, 
                                          chain_idx_map: dict,
                                          all_chain_ids: np.ndarray,
                                          is_ligand_feat: np.ndarray, 
                                          all_atom_ids: np.ndarray) -> str:
        """Get the token pair offset for each entity.

            Args:
                chain_ids: np.ndarray (N_token), the chain ids for the token. such as: [1-1, 1-1, 1-2, 2-1, 2-2, ...]
                row: int, the row index of the token. 
                col: int, the column index of the token.
                chain_idx_map: dict, the start and end indices for each chain id.
                is_ligand_feat: np.ndarray (N_token), the feature for the token, which is ligand or not.
                all_atom_ids: np.ndarray (N_atom), the atom_ids for the predicted
                all_chain_ids: np.ndarray (N_atom), the chain_ids for the predicted
            Returns:
                token_pair_offset: str, the token pair offset for the entity.
                    Format: '<entity_id_1>-<sym_id_1>-<token/res_offset_1>-<atom_offset_1>, <entity_id_2>-<sym_id_2>-<token/res_offset_2>-<atom_offset_2>'
                    such as: '1-1-59-2', '2-1-40-3' , offset is the token index in the chain. start from 1.
        """
        entity_id_1 = chain_ids[row] # entity_id: <entity_id>-<sym_id>
        entity_id_2 = chain_ids[col]
        start_idx_1 = chain_idx_map[entity_id_1][0]
        start_idx_2 = chain_idx_map[entity_id_2][0]
        token_offset_1 = row - start_idx_1 + 1
        token_offset_2 = col - start_idx_2 + 1

        if is_ligand_feat[row] == 1:
            ## ligand
            token_offset_1, atom_offset_1 = 1, token_offset_1
            atom_offset_1 = '-' + str(atom_offset_1)
        else:
            atom_offset_1 = ''

        if is_ligand_feat[col] == 1:
            ## ligand
            token_offset_2, atom_offset_2 = 1, token_offset_2
            atom_offset_2 = '-' + str(atom_offset_2)
        else:
            atom_offset_2 = ''

        return {'sampling_left_entity': f'{entity_id_1}-{token_offset_1}{atom_offset_1}', 
                'sampling_right_entity': f'{entity_id_2}-{token_offset_2}{atom_offset_2}'}


    os.makedirs(out_dir, exist_ok=True)
    json_info = {'token_chain_ids': [], 'token_pair_interface_probs': []}
    sample_infos = pd.DataFrame(columns=['model_id', 'sampling_left_entity', 
                                         'sampling_right_entity', 'sampling_probability'])

    sample_history_path = os.path.join(history_dir, 'previous_sampled_interface/direct_sample_history.pkl')
    if not os.path.exists(sample_history_path):
        raise ValueError(f"sample_history_path: {sample_history_path} not exists")
    with open(sample_history_path, 'rb') as f:
        sample_history_obj = pickle.load(f)
    sample_history = sample_history_obj['sampled_history']

    token_mask = batch_feats['label']['all_centra_token_indice_mask'][0].numpy().astype('bool') # (N_token)
    ori_chain_ids_token = np.array(str(batch_feats['feat']['chain_ids'][0]).split()) # (N_token)
    ori_chain_ids = np.array(str(batch_feats['feat']['all_chain_ids'][0]).split()) # (N_atom)
    ori_atom_ids = np.array(str(batch_feats['feat']['all_atom_ids'][0]).split()) # (N_atom)
    atom_mask = batch_feats['label']['all_atom_pos_mask'][0].numpy().astype('bool') # (N_atom)
    is_ligand_feat = batch_feats['feat']['is_ligand'][0].numpy() # (N_token)
    interface_prob = batch_feats['feat']['interface_prob'][0].numpy() # (N_token, N_token)

    ## NOTE: set temporary variable for recording the interface sample order.
    _interface_sample_order = np.full_like(interface_prob, dtype=object, fill_value='') # (N_token, N_token)
    for sample_i, sample_i_info in enumerate(sample_history, start=1):
        _idx_i, _idx_j = sample_i_info['top_indices'][0]
        _interface_sample_order[_idx_i, _idx_j] += f'{sample_i},'
        if DEBUG:
            print('Before:', sample_i, interface_prob[_idx_i, _idx_j])

    ## First, Reorder the interface sample order and interface prob.
    ent_weights_token, sym_weights_token = mmcif_writer.user_asymid_to_weight(ori_chain_ids_token)
    ent_weights_atom, sym_weights_atom = mmcif_writer.user_asymid_to_weight(ori_chain_ids)
    sorted_indices_token = np.lexsort((sym_weights_token, ent_weights_token))
    sorted_indices_atom = np.lexsort((sym_weights_atom, ent_weights_atom))
    _interface_sample_order = np.take(_interface_sample_order, sorted_indices_token, axis=0)
    _interface_sample_order = np.take(_interface_sample_order, sorted_indices_token, axis=1)
    _token_pair_interface_probs = np.take(interface_prob, sorted_indices_token, axis=0)
    
    token_pair_interface_probs = np.take(_token_pair_interface_probs, sorted_indices_token, axis=1)
    token_chain_ids = ori_chain_ids_token[sorted_indices_token]
    token_mask = token_mask[sorted_indices_token]
    is_ligand_feat = is_ligand_feat[sorted_indices_token]
    atom_chain_ids = ori_chain_ids[sorted_indices_atom]
    atom_ids = ori_atom_ids[sorted_indices_atom]
    atom_mask = atom_mask[sorted_indices_atom]

    ## Second, mask the interface info by valid token. NOTE: Only save the valid token.
    token_pair_interface_probs = token_pair_interface_probs[token_mask, :][:, token_mask]
    _interface_sample_order = _interface_sample_order[token_mask, :][:, token_mask]
    token_chain_ids = token_chain_ids[token_mask]
    is_ligand_feat = is_ligand_feat[token_mask]
    atom_chain_ids = atom_chain_ids[atom_mask]
    atom_ids = atom_ids[atom_mask]

    ## 1. save the interface info to json file.
    json_info['token_chain_ids'] = token_chain_ids.tolist()
    json_info['token_pair_interface_probs'] = token_pair_interface_probs.tolist()
    write_format_json(json_info, file_path=os.path.join(out_dir, 'predicted_interface.json'), nan_to_none=True)

    ## 2. save the interface sample order to csv file.
    chain_idx_map = _get_chain_indices(token_chain_ids)
    non_empty_indices = np.nonzero(_interface_sample_order)
    non_empty_values = _interface_sample_order[non_empty_indices]
    for i, (row, col) in enumerate(zip(*non_empty_indices)):
        values = non_empty_values[i].split(',')
        for value in values:
            if value == '': continue
            if DEBUG:
                print('After:', value, token_pair_interface_probs[row, col])
            
            _sample_pair = _get_token_pair_offset_for_entity(token_chain_ids, row, col, chain_idx_map, 
                                            all_chain_ids=atom_chain_ids, all_atom_ids=atom_ids, is_ligand_feat=is_ligand_feat)
            sample_infos = sample_infos.append({
                'model_id': int(value),
                **_sample_pair,
                'sampling_probability': token_pair_interface_probs[row, col],
            }, ignore_index=True)
    sample_infos.sort_values(by=['model_id'], inplace=True)
    sample_infos.to_csv(os.path.join(out_dir, 'sample_infos.csv'), index=False)

    ## 3. save the token pair interface prob to a heatmap png file.
    interface_heatmap_path = os.path.join(out_dir, 'predicted_interface.png')
    draw_results.draw_interface_heatmap(token_pair_interface_probs, 
                                        chain_idx_map=chain_idx_map,
                                        path=interface_heatmap_path, 
                                        title='Predicted Interface Probability')

    logger.info(f'save all interface infos results to {out_dir}')


def save_result(json_name, batch_feats, 
                prediction, output_dir, timings, extra_infos):
    """
        Save the prediction results.
        Args:
            json_name: str, the name of the json file.
            batch_feats: dict, the batch features from the input data.
            prediction: dict, the prediction from the inference.
            output_dir: str, the directory to save the prediction results.
            timings: dict, the timings for the prediction.
            extra_infos: dict, the extra information for the prediction.
                It has keys: mmcif, model, job_name
    """

    base_timings = copy.deepcopy(timings)    
    
    ## 1. save pdb or mmcif. 
    dump_cif(name=json_name,
             batch=batch_feats, 
             results=prediction,
             output_dir=output_dir,
             timings=base_timings, 
             mmcif_extra_infos=extra_infos['mmcif'])

    ## 2. save the plddts, pae, and so on;
    dump_metric_results(batch=batch_feats,
                        results=prediction,
                        output_dir=output_dir,
                        timings=base_timings)

    ## 3. save the timings
    logger.info('Final timings for %s: %s', json_name, base_timings)
    with open(output_dir.joinpath('timings.json'), 'w') as f:
        f.write(json.dumps(base_timings, indent=4))


def split_prediction(pred: dict, rank: int, extra_infos: dict) -> list:
    """
        Split the prediction into multiple sub-predictions.
        Args:
            pred: dict, the prediction from the inference.
            rank: int, the number of sub-predictions, that is the diffusion batch size.
            extra_infos: dict, the extra information for the prediction.
                It has keys: mmcif, model, job_name
        Returns:
            prediction: list, the list of sub-predictions.
    """

    RETURN_KEYS = ['diffusion_module', 'confidence_head']
    
    prediction = []
    feat_key_list = [pred[rk].keys() for rk in RETURN_KEYS]
    feat_key_table = dict(zip(RETURN_KEYS, feat_key_list))
    
    for i in range(rank):
        sub_pred = {}
        for rk in RETURN_KEYS:
            feat_keys = feat_key_table[rk]
            ## Shape is [batch_size, diffusion_size, *]
            sub_feat = dict(zip(feat_keys, [pred[rk][fk][:, i] for fk in feat_keys]))
            sub_pred[rk] = sub_feat
        
        prediction.append(sub_pred)
    
    return prediction


def load_features_from_pkl(args, extra_infos: dict,
                          dp_rank=0, dp_nranks=1) -> Mapping[str, Union[paddle.Tensor, list]]:
    """
        Load the features from the pkl file.
        Args:
            args: argparse.Namespace, the arguments from the command line.
            extra_infos: dict, the extra information for the prediction.
                        It has keys: mmcif, model, job_name
            dp_rank: int, the rank of the distributed process.
            dp_nranks: int, the number of distributed processes.
        Returns:
            batch_feats: dict, the features from the pkl file.
    """
    output_dir_base = pathlib.Path(args.output_dir)

    batch_feats = None
    features_pkl = output_dir_base.joinpath('final_features.pkl')
    logger.info(f"loading pkl from: {features_pkl}")

    if features_pkl.exists():
        logger.info(f'[RANK-{dp_rank}] Use cached features.pkl')
        _t0 = time.time()
        with open(features_pkl, 'rb') as f:
            batch_feats = pickle.load(f)
                
        if dp_rank == 0:
            logger.info('Load features.pkl cost time: %.4fs', time.time() - _t0)

        batch_feats['feat'] = batch_convert(batch_feats['feat'], add_batch=True)
        batch_feats['label'] = batch_convert(batch_feats['label'], add_batch=True)

        if DEBUG:
            for feat_key in batch_feats['feat'] :
                feat_val = batch_feats['feat'][feat_key]
                logger.info(f'feat {feat_key} {type(feat_val)} {np.shape(feat_val)}')

            for feat_key in batch_feats['label'] :
                feat_val = batch_feats['label'][feat_key]
                logger.info(f'input label {feat_key} {type(feat_val)} {np.shape(feat_val)}')

        if dp_rank == 0:
            logger.info('Features cost time: %.4fs', time.time() - _t0)

        if args.distributed and args.dap_degree > 1:
            raise NotImplementedError(
                'Distributed inference is not supported for AF3 yet.')
            # ## FIXME: this function is not ready for AF3. Only support AF2
            # batch_feats = align_feat(batch_feats, args.dap_degree)      
    else:
        if dp_rank == 0:
            logger.error(f'{features_pkl} is not existed. Please check the input data !!')
        raise ValueError('No cached features.pkl')

    if extra_infos['model']['model_type'] == "HelixFold-S1":
        interface_prob = get_interface_prob_from_module1(os.path.join(args.output_dir, 'module1'))
        _feats = {'interface_prob': interface_prob}
        batch_feats['feat'].update(batch_convert(_feats, add_batch=True))

    if args.distributed:
        dist.barrier()
    
    return batch_feats


def predict_structure(
        dp_rank: int,
        dp_nranks: int,
        sub_dir_name: str,
        output_dir_base: str,
        diff_batch_size: int,
        batch_feats: Mapping[str, Union[paddle.Tensor, list]],
        model_runner: RunModel,
        extra_infos: dict) -> list:
    """
        Predict the structure of the input data.
        Args:
            dp_rank: int, the rank of the distributed process.
            dp_nranks: int, the number of distributed processes.
            sub_dir_name: str, the name of the sub-directory for the prediction.
            output_dir_base: str, the base directory for the prediction.
            diff_batch_size: int, the batch size for the diffusion module.
            batch_feats: Mapping[str, Union[paddle.Tensor, list]], the features from the pkl file.
            model_runner: RunModel, the model runner for the prediction.
            extra_infos: dict, the extra information for the prediction.
                        It has keys: mmcif, model, job_name
        Returns:
            all_pred_paths: list, the list of paths to the sub-predictions.
    """

    timings = dict()
    output_dir_base = pathlib.Path(output_dir_base)

    all_preds = []
    t0 = time.time()
    with paddle.no_grad():
        black_list, white_list = get_custom_amp_list()
        with paddle.amp.auto_cast(enable=True,
                                    custom_white_list=white_list, 
                                    custom_black_list=black_list, 
                                    level="O1", 
                                    dtype='bfloat16'):
            model_runner.eval()
            prediction = model_runner(batch_feats, compute_loss=False)

    if dp_rank == 0: 
        logger.info('########## prediction done ##########')
        logger.info('prediction cost time: %.4fs', time.time() - t0)

    if DEBUG:
        for head_name in prediction:
            for pred_key in prediction[head_name].keys():
                pred_val = prediction[head_name][pred_key]
                print(head_name, pred_key, type(pred_val), np.shape(pred_val))
        timings['per_module'] = prediction["duration"]

    if args.distributed and dp_rank == 0 and args.dap_degree > 1:
            # ## WARNING: this function is not ready for AF3. Only support AF2
            # prediction = unpad_prediction(feature_dict, prediction)
            raise NotImplementedError(
                'Distributed inference is not supported for AF3 yet.')

    if dp_rank == 0:
        timings[f'predict'] = time.time() - t0
        prediction = split_prediction(prediction, diff_batch_size, extra_infos)
        for rank_id in range(diff_batch_size):
            sub_dir_name_extra = sub_dir_name + f'-{str(rank_id + 1)}'
            output_dir = output_dir_base.joinpath(sub_dir_name_extra)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            save_result(json_name=sub_dir_name_extra, batch_feats=batch_feats,
                        prediction=prediction[rank_id], 
                        output_dir=output_dir, 
                        timings=timings,
                        extra_infos=extra_infos)
    
            all_preds.append({
                'path': output_dir,
                'results_wo_diffsize': prediction[rank_id]
            })
            if diff_batch_size > 1 and extra_infos['model']['model_type'] == "HelixFold-S1":
                logger.warning(f'[S1 model] only save the first when diff_batch_size={diff_batch_size}')
                break


    return all_preds


def update_subbatch_size_by_seqLen(batch_feats: dict, model_runner: RunModel):
    """
        Update the subbatch size by the inputsequence length.
        Args:
            batch_feats: dict, the features needed for the prediction.
            model_runner: RunModel, the model runner for the prediction.
    """
    subbatch_size = model_runner.model_config.model.global_config.subbatch_size
    feat_seq_len = batch_feats['feat']['restype'].shape[1] # [batch, seq_len]
    if feat_seq_len <= 1024:
        subbatch_size = 1024
    elif feat_seq_len <= 2500: # rna-2500?
        subbatch_size = 384
    elif feat_seq_len <= 2900:
        subbatch_size = 256
    elif feat_seq_len <= 3036:
        subbatch_size = 128
    model_runner.model_config.model.global_config.subbatch_size = subbatch_size
    logger.info(f"setting subbatch_size to {subbatch_size} for sample with seq_len: "\
                    f"{feat_seq_len} restype: {batch_feats['feat']['restype'].shape}")


def parse_infos_from_json(args, json_path: str) -> dict:
    """
        Parse the extra information from the json file or args, including the model type, job name, and other information.
        The extra information will be used to save the prediction results or update model config.

        Args:
            args: argparse.Namespace, the arguments from the command line.
            json_path: str, the path to the json file.
        Returns:
            extra_infos: dict, the extra information for the prediction. It has keys: mmcif, model, job_name
    """
    extra_infos = {}
    user_input_json = read_json(json_path)

    ## 1. get job name and extra infos for mmcif writing.
    job_name = user_input_json['job_name']
    format_job_name = 'job-' + filter_job_name(job_name)
    extra_infos['job_name'] = format_job_name
    
    mmcif_extra_infos = get_extra_infos_for_mmcif(args, length=10)
    extra_infos['mmcif'] = mmcif_extra_infos

    ## 2. get model type and recycle times for model config update.
    _model_infos = {}
    model_type = user_input_json.get('model_type', "HelixFold3")
    logger.info(f'model_type: {model_type}')
    _model_infos['model_type'] = model_type
    
    if 'recycle' in user_input_json and user_input_json['recycle']:
        recycle = int(user_input_json['recycle'])
        logger.info(f'"recycle" found in input json; use value {recycle}.')
    else:
        recycle = 10
        logger.info(f'No "recycle" found in input json; use default value {recycle}.')
    _model_infos['recycle'] = recycle

    _model_infos['ensemble_times'] = int(user_input_json.get('ensemble', 1))
    logger.info(f'ensemble_times: {_model_infos["ensemble_times"]}')

    ## 3. get s1 sample constraint:
    _model_infos['s1_sample_constraint'] = user_input_json.get('s1_sample_constraint', [])
    logger.info(f's1_sample_constraint: {_model_infos["s1_sample_constraint"]}')
    

    extra_infos['model'] = _model_infos

    return extra_infos


def main(args):

    set_logging_level("INFO")
    logger.info(f'[ARG] {args}')

    extra_infos = parse_infos_from_json(args, args.json_path)
    if extra_infos['model']['model_type'] == "HelixFold-S1" and \
        len(glob.glob(os.path.join(args.output_dir, 'module1', 'job-*', '*interface-prob-step0.pkl'))) == 0:
        logger.error(f'[S1 model] interface prediction not found in {args.output_dir}, Error')
        sys.exit(1)
    
    ### check paddle version
    if args.distributed:
        assert paddle.fluid.core.is_compiled_with_dist(), "Please using the paddle version compiled with distribute."
    args.distributed = args.distributed and dist.get_world_size() > 1
    dp_rank, dp_nranks = init_distributed_env(args)
    logger.info(f'>>> dp_rank: {dp_rank}, dp_nranks: {dp_nranks}')

    ### set seed for reproduce experiment results
    if args.seed is not None:
        args.seed += dp_rank
        init_seed(args.seed)

    model_config = config.model_config(args.model_names)
    model_config.model.num_recycle = extra_infos['model']['recycle'] - 1
    model_config.model.global_config.subbatch_size = args.subbatch_size
    update_model_config_quickly_debug(model_config, args)

    diff_batch_size = model_config.model.heads.diffusion_module.test_diff_batch_size
    ensemble_times = extra_infos['model']['ensemble_times']
    logger.info(f" diffusion batch size {diff_batch_size}...")
    logger.info(f' Inference {ensemble_times} Times...')


    ## get model params path and load model
    if args.if_use_pretrained_weights and not os.path.exists(args.init_model):
        raise FileNotFoundError(f'No such model params file: {args.init_model}') 

    model = RunModel(train_config={}, model_config=model_config)
    if len(args.init_model) > 0 and os.path.exists(args.init_model):
        t0 = time.time()
        pd_params = paddle.load(args.init_model)
        has_opt = 'optimizer' in pd_params
        if has_opt:
            if model_config.model.global_config.fuse_attention:
                pd_params['model'] = fuse_attention_key_update(pd_params['model'])
            model.helixfold.set_state_dict(pd_params['model'])
        else:
            if model_config.model.global_config.fuse_attention:
                pd_params = fuse_attention_key_update(pd_params)
            model.helixfold.set_state_dict(pd_params)
        _t1 = time.time()
        logger.info('Loading model {}; use {}'.format(args.init_model, _t1 - t0))
    else:
        logger.warning('No model checkpoint found; use initialize weights.')
    logger.info('Model name: {}'.format(args.model_names))
    
    ## prediction
    ## NOTE: interface sampling config, TODO: move to config file rather than hardcode here
    interface_type = INFERENCE_INTERFACE_CONFIG.get('interface_type', None)
    interface_indirect_thres = INFERENCE_INTERFACE_CONFIG.get("interface_indirect_thres", 0) 
    interface_weight_b = INFERENCE_INTERFACE_CONFIG.get("interface_weight_b", 1) 
    interface_weight_c = INFERENCE_INTERFACE_CONFIG.get("interface_weight_c", 0.4) 
    conf_metric_name = INFERENCE_INTERFACE_CONFIG.get("interface_conf_metric_name", None) # 
    conf_metric_accumulate_method = INFERENCE_INTERFACE_CONFIG.get("conf_metric_accumulate_method", "max")
    max_prob_decay_times = INFERENCE_INTERFACE_CONFIG.get("max_prob_decay_times", None)
    min_prob = INFERENCE_INTERFACE_CONFIG.get("min_prob", None)
    min_prob_rank = INFERENCE_INTERFACE_CONFIG.get("min_prob_rank", None)
    enable_replacement_sampling = INFERENCE_INTERFACE_CONFIG.get("enable_replacement_sampling", False)
    topn = INFERENCE_INTERFACE_CONFIG.get("topn", None)
    sampling_method = INFERENCE_INTERFACE_CONFIG.get('interface_sampling_method', 'top1')
    previous_sampled_interface_dir = os.path.join(args.output_dir, 'previous_sampled_interface')
    if os.path.exists(previous_sampled_interface_dir):
        shutil.rmtree(previous_sampled_interface_dir)
    os.makedirs(previous_sampled_interface_dir)
    print(f'[DEBUG-interface]: interface_type is {interface_type}')
    print(f'[DEBUG-interface]: sample_interface_from_prob, weight_c={interface_weight_c}')

    format_job_name = extra_infos['job_name']
    all_predictions_path = {}
    all_predictions_results = {}
    batch_feats = load_features_from_pkl(args, extra_infos, dp_rank, dp_nranks)
    update_subbatch_size_by_seqLen(batch_feats, model)
    logger.info("\n[model_config]:")
    logger.info(model.model_config)

    for repeat_i in range(ensemble_times):
        logger.info(f"[repeat_i]: {repeat_i}")
        if interface_type == 'dynamic_sampling':
            batch_feats = sample_interface_from_prob(batch_feats, previous_sampled_interface_dir, repeat_i, sampling_method, 
                                                conf_metric_name, conf_metric_accumulate_method, interface_weight_b, interface_weight_c,
                                                max_prob_decay_times, min_prob=min_prob, min_prob_rank=min_prob_rank,
                                                enable_replacement_sampling=enable_replacement_sampling, topn=topn, 
                                                total_repeat_nums=ensemble_times)
        else:
            raise NotImplementedError(f'interface_type {interface_type} is not supported')
        
        results_obj = predict_structure(
                        dp_rank=dp_rank,
                        dp_nranks=dp_nranks,
                        sub_dir_name=f'{format_job_name}-pred-{str(repeat_i + 1)}',
                        diff_batch_size=diff_batch_size,
                        output_dir_base=os.path.join(args.output_dir, 'module2'),
                        batch_feats=batch_feats,
                        model_runner=model,
                        extra_infos=extra_infos)
        all_predictions_path[repeat_i] = results_obj[0]['path']
        all_predictions_results[repeat_i] = [ _res['results_wo_diffsize'] for _res in results_obj]

        if interface_type == 'dynamic_sampling' and interface_indirect_thres > 0:
            _diff_size = 0
            # update the map of previous indirectly sampled interfaces
            update_previous_indirect_sampled_map(batch_feats, all_predictions_results[repeat_i][_diff_size], 
                                                 previous_sampled_interface_dir, repeat_i,
                                                 interface_indirect_thres)


    ## Final dump the interface info and sample history, also save the overall csv.
    dump_interface_and_summary_info(
        batch_feats=batch_feats,
        results=all_predictions_results,
        history_dir=os.path.join(args.output_dir),
        out_dir=os.path.join(args.output_dir, 'interface_infos')
    )

    ## Final rename results, rename the output dir to *-<ensemble_times>-rank1
    for repeat_i, output_dir in all_predictions_path.items():
        basename_prefix = os.path.basename(output_dir).split('-pred-')[0]
        target_path = os.path.join(os.path.dirname(output_dir), f'{basename_prefix}-{repeat_i + 1}-rank1')
        os.rename(output_dir, target_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict protein structure')
    parser.add_argument('--json_path', type=str,
                        default=None, required=True,
                        help='Paths to json file, each containing '
                        'entity information including sequence, smiles or CCD, copies etc.')
    parser.add_argument('--output_dir', type=str,
                        default=None, required=True,
                        help='Path to a directory that will store results.')
    parser.add_argument('--model_names', type=str,
                        default=None, required=True,
                        help='Names of models to use.')
    parser.add_argument('--init_model', type=str,
                        default='', required=True,
                        help='Names of models to use.')
    parser.add_argument('--ccd_preprocessed_path', type=str,
                        default=None, required=True,
                        help='Path to CCD preprocessed files.')

    parser.add_argument('--random_seed', type=int,
                        help='The random seed for the data pipeline. '
                        'By default, this is randomly generated.')

    parser.add_argument('--distributed',
                        action='store_true', default=False,
                        help='Whether to use distributed DAP inference')
    parser.add_argument("--dap_degree", type=int, default=1)
    parser.add_argument("--dap_comm_sync", action='store_true', default=False)
    parser.add_argument("--bp_degree", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None, help="set seed for reproduce experiment results, None is do not set seed")

    parser.add_argument('--subbatch_size', type=int, default=48)
    parser.add_argument('--if_use_pretrained_weights', default=True, action='store_false')

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        ## revised the task status
        json_status = os.path.join(args.output_dir, 'job_status.json')
        if not os.path.exists(json_status):
            status_dict = {}
            status_dict["status"] = 'failed'
            status_dict["job_fail_reason"] = f'S1-module2 fail with exception'
            with open(json_status, 'w') as f:
                json.dump(status_dict, f, indent=4)
        else:
            with open(json_status, 'r') as f:
                status_dict = json.load(f)
                status_dict["status"] = 'failed'
                if len(status_dict["job_fail_reason"]) > 0:
                    status_dict["job_fail_reason"] += f';S1-module2 fail with exception'
                else:
                    status_dict["job_fail_reason"] = f'S1-module2 fail with exception'
            with open(json_status, 'w') as f:
                json.dump(status_dict, f, indent=4)
        
        sys.exit(1)
