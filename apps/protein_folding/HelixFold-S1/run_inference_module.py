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
from collections import defaultdict

import paddle
from paddle import distributed as dist
from utils.model import RunModel
from utils.misc import set_logging_level
from utils.utils import get_custom_amp_list
from utils.interface_utils import dump_interface_prob

from helixfold.model import config, utils
from helixfold.data.utils import atom_level_keys, map_to_continuous_indices
from helixfold.common import all_atom_pdb_save
from ppfleetx.distributed.protein_folding.scg import scg

from infer_scripts.tools.post_calculate import (
    calculate_chain_pair_pae_matrix,
    calculate_token_plddts)
from infer_scripts.tools.utils import (
    read_json, 
    filter_job_name, 
    write_format_json, 
    convert_to_json_compatible,
    get_extra_infos_for_mmcif)
from infer_scripts.tools import mmcif_writer


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


def update_model_config_quickly_debug(model_config, args):
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

def dump_cif_common_feat_masked(batch, confidence_results={}):
    required_keys = copy.deepcopy(mmcif_writer.required_keys_for_saving)
    required_keys += ['token_bonds_type', 'ref_element', 'is_ligand', 'atom_plddts']

    # 1 feat extraction
    common_feat = {k: batch['feat'][k][0]
            for k in required_keys if k in batch['feat']}
    common_feat.update(
        {k: batch['label'][k][0]
            for k in required_keys if k in batch['label']}
    )
    if 'atom_plddts' in confidence_results:
        common_feat.update(
            {'atom_plddts': confidence_results['atom_plddts'][0]})

    common_feat['all_chain_ids'] = str(common_feat['all_chain_ids']).split()
    common_feat['all_ccd_ids'] = str(common_feat['all_ccd_ids']).split()
    common_feat['all_atom_ids'] = str(common_feat['all_atom_ids']).split()
    ## asym_id start with 1
    common_feat['asym_id'] -= 1
    ## resid start with 1
    common_feat['residue_index'] += 1

    exp_dict = {
        "mask": batch['label']['all_atom_pos_mask'].numpy(),
    }

    # diffusion res final_atom_mask comes from all_atom_pos_mask
    # atom_mask = np.logical_and(pred_dict["mask"] > 0, exp_dict["mask"] > 0)[0]  # [N_atom]
    atom_mask = np.logical_and(common_feat["all_atom_pos_mask"].numpy() > 0, exp_dict["mask"] > 0)[0]  # [N_atom]
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
    return common_feat_masked, atom_mask

def dump_cif(name, batch, results, output_dir, timings, extra_infos, ligand_intra_bonds_info=None):
    """
        HF3 inference to *.cif file.
            - batch. input data
            - results. model output
            - output_dir. to save output
            - timings. timings of each step, dict
            - extra_infos. extra infos for cif wirter, dict
    """
    t0 = time.time()
    diff_results = results['diffusion_module'] 
    confidence_results = results['confidence_head']

    if isinstance(diff_results['final_atom_positions'], paddle.Tensor):
        pred_dict = {
            "pos": diff_results['final_atom_positions'].numpy(),
            "mask": diff_results['final_atom_mask'].numpy(),
        }
    else:
        pred_dict = {
            "pos": diff_results['final_atom_positions'],
            "mask": diff_results['final_atom_mask'],
        }

    common_feat_masked, atom_mask = dump_cif_common_feat_masked(batch, confidence_results=confidence_results)

    ## save prediction masked 
    pred_cif_path = f'{output_dir}/predicted_structure.cif'
    mmcif_writer.prediction_to_mmcif(   # 注意这个函数会写两个 cif，两种链名模式
        entry_name=name, 
        atom_positions=pred_dict["pos"][0][atom_mask], 
        feats_dict=common_feat_masked, 
        mmcif_path=pred_cif_path,
        extra_infos=extra_infos, ligand_intra_bonds_info=ligand_intra_bonds_info)
    
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
    confi_float_names = ['ptm', 'iptm', 'has_clash', 'mean_plddt', 'ranking_confidence']
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


def ranking_all_predictions(output_dirs_map: dict):
    """
        Ranking the all predictions.
        Args:
            output_dirs_map: dict, the map of the output directories.
    """
    for pred_t, output_dirs in output_dirs_map.items():
        ranking_score_path_map = {}
        for outpath in output_dirs:
            _results = read_json(os.path.join(outpath, 'all_results.json'))
            _rank_score = _results['ranking_confidence']
            ranking_score_path_map[outpath] = _rank_score

        ranked_map = dict(sorted(ranking_score_path_map.items(), key=lambda x: x[1], reverse=True))
        rank_id = 1
        for outpath, rank_score in ranked_map.items():
            logger.debug("[ranking_all_predictions] Ranking score of %s: %.5f", outpath, rank_score)
            basename_prefix = os.path.basename(outpath).split('-pred-')[0]
            target_path = os.path.join(os.path.dirname(outpath), f'{basename_prefix}-{pred_t + 1}-rank{rank_id}')
            if os.path.exists(target_path) and os.path.isdir(target_path):
                shutil.rmtree(target_path)
            shutil.move(outpath, target_path)
            rank_id += 1


def save_result(json_name, batch_feats, 
                prediction, output_dir, timings, extra_infos, 
                ligand_bonds=None):
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
            ligand_bonds: dict, the ligand bonds information.
    """

    base_timings = copy.deepcopy(timings)    
    
    if extra_infos['model']['model_type'] != "HelixFold-S1":
        ## 1. save pdb or mmcif. 
        dump_cif(name=json_name,
                batch=batch_feats, 
                results=prediction,
                output_dir=output_dir,
                timings=base_timings, 
                extra_infos=extra_infos['mmcif'],
                ligand_intra_bonds_info=ligand_bonds)

        ## 2. save the plddts, pae, and so on;
        dump_metric_results(batch=batch_feats,
                            results=prediction,
                            output_dir=output_dir,
                            timings=base_timings)
    else:
        logger.info(f'S1 model only need to save interface prob.')
        dump_interface_prob(results=prediction, batch_i=0, output_dir=output_dir, base_name=json_name)

    ## 4. save the timings
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
    if extra_infos['model']['model_type'] == "HelixFold-S1":
        RETURN_KEYS += ['interface_head']
    
    prediction = []
    feat_key_list = [pred[rk].keys() for rk in RETURN_KEYS]
    feat_key_table = dict(zip(RETURN_KEYS, feat_key_list))
    
    for i in range(rank):
        sub_pred = {}
        for rk in RETURN_KEYS:
            feat_keys = feat_key_table[rk]
            if rk == 'interface_head':
                sub_pred[rk] = pred[rk]
            else:
                ## Shape is [batch_size, diffusion_size, *]
                sub_feat = dict(zip(feat_keys, [pred[rk][fk][:, i] for fk in feat_keys]))
                sub_pred[rk] = sub_feat
        
        prediction.append(sub_pred)
    
    return prediction


def predict_structure(
        dp_rank: int,
        dp_nranks: int,
        sub_dir_name: str,
        pkl_dir: str,
        output_dir_base: str,
        diff_batch_size: int,
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
            model_runner: RunModel, the model runner for the prediction.
            extra_infos: dict, the extra information for the prediction.
                        It has keys: mmcif, model, job_name
        Returns:
            all_pred_paths: list, the list of paths to the sub-predictions.
    """

    logger.info("model_config:")
    logger.info(model_runner.model_config)
    timings = dict()
    output_dir_base = pathlib.Path(output_dir_base)
    pkl_dir = pathlib.Path(pkl_dir)
    
    batch_feats = None
    features_pkl = pkl_dir.joinpath('final_features.pkl')
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

    if args.distributed:
        dist.barrier()

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
    logger.info(f"setting subbatch_size to {subbatch_size} for sample with seq_len: {feat_seq_len} restype: {batch_feats['feat']['restype'].shape}")

    all_pred_paths = []
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

    logger.info('prediction done')
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
        common_feat_for_bonds, _ = dump_cif_common_feat_masked(batch_feats, confidence_results={})
        ligand_bonds = mmcif_writer._prepare_bonds(common_feat_for_bonds)
        for rank_id in range(diff_batch_size):
            sub_dir_name_extra = sub_dir_name + f'-{str(rank_id + 1)}'
            output_dir = pathlib.Path(output_dir_base).joinpath(sub_dir_name_extra)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            save_result(json_name=sub_dir_name_extra, batch_feats=batch_feats,
                        prediction=prediction[rank_id], 
                        output_dir=output_dir, 
                        timings=timings,
                        extra_infos=extra_infos)
    
            all_pred_paths.append(output_dir)
        
    return all_pred_paths


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
    
    ensemble_times = user_input_json.get('ensemble', 1)
    if model_type == "HelixFold-S1":
        logger.warning(f'S1 model only support ensemble=1 in module 1')
        ensemble_times = 1
    logger.info(f'ensemble_times: {ensemble_times}')
    _model_infos['ensemble_times'] = int(ensemble_times)

    extra_infos['model'] = _model_infos

    return extra_infos


def main(args):

    set_logging_level("INFO")
    logger.info(f'[ARG] {args}')

    extra_infos = parse_infos_from_json(args, args.json_path)
    if extra_infos['model']['model_type'] == "HelixFold-S1" and \
        len(glob.glob(os.path.join(args.output_dir, 'module1', 'job-*', '*interface-prob-step0.pkl'))) > 0:
        logger.info(f'[S1 model] module1 interface generation done, skip it')
        return

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
    logger.info(f" Inference {ensemble_times} Times...")


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
    format_job_name = extra_infos['job_name']
    all_predictions_path = {}
    for idx in range(ensemble_times):
        output_path = predict_structure(
                        dp_rank=dp_rank,
                        dp_nranks=dp_nranks,
                        sub_dir_name=f'{format_job_name}-pred-{str(idx + 1)}',
                        diff_batch_size=diff_batch_size,
                        pkl_dir=args.output_dir,
                        output_dir_base=os.path.join(args.output_dir, 'module1'),
                        model_runner=model,
                        extra_infos=extra_infos)
        all_predictions_path[idx] = output_path


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
            status_dict["job_fail_reason"] = f'model fail with exception'
            with open(json_status, 'w') as f:
                json.dump(status_dict, f, indent=4)
        else:
            with open(json_status, 'r') as f:
                status_dict = json.load(f)
                status_dict["status"] = 'failed'
                if len(status_dict["job_fail_reason"]) > 0:
                    status_dict["job_fail_reason"] += f';model fail with exception'
                else:
                    status_dict["job_fail_reason"] = f'model fail with exception'
            with open(json_status, 'w') as f:
                json.dump(status_dict, f, indent=4)
        
        sys.exit(1)
