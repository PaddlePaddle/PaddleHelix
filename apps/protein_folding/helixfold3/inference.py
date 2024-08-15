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

"""Inference scripts."""
import re
import os
import copy
import argparse
import random
import paddle
import json
import pickle
import pathlib
import shutil
import logging
import numpy as np
from helixfold.common import all_atom_pdb_save
from helixfold.model import config, utils
from helixfold.data import pipeline_parallel as pipeline
from helixfold.data import pipeline_multimer_parallel as pipeline_multimer
from helixfold.data import pipeline_rna_parallel as pipeline_rna
from helixfold.data import pipeline_rna_multimer
from helixfold.data.utils import atom_level_keys, map_to_continuous_indices
from helixfold.data.tools import hmmsearch
from helixfold.data import templates
from utils.utils import get_custom_amp_list
from utils.model import RunModel
from utils.misc import set_logging_level
from typing import Dict
from infer_scripts import feature_processing_aa, preprocess
from infer_scripts.tools import mmcif_writer

ALLOWED_LIGAND_BONDS_TYPE_MAP = preprocess.ALLOWED_LIGAND_BONDS_TYPE_MAP
INVERSE_ALLOWED_LIGAND_BONDS_TYPE_MAP = {
    v: k for k, v in ALLOWED_LIGAND_BONDS_TYPE_MAP.items()
}

DISPLAY_RESULTS_KEYS = [
    'atom_chain_ids',
    'atom_plddts',
    'pae',
    'token_chain_ids',
    'token_res_ids',
    'iptm',
    'ptm',
    'ranking_confidence',
    'has_clash', 
    'mean_plddt',
]

RETURN_KEYS = ['diffusion_module', 'confidence_head']

logger = logging.getLogger(__file__)

MAX_TEMPLATE_HITS = 4

def init_seed(seed):
    """ set seed for reproduct results"""
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def batch_convert(np_array, add_batch=True):
    np_type = {}
    other_type = {}
    # 
    for key, value in np_array.items():
        if type(value) == np.ndarray:
            np_type.update(utils.map_to_tensor({key: value}, add_batch=add_batch))
        else:
            other_type[key] = [value]  ## other type shoule be list.
    
    return {**np_type, **other_type}

def preprocess_json_entity(json_path, out_dir):
    all_entitys = preprocess.online_json_to_entity(json_path, out_dir)
    if all_entitys is None:
        raise ValueError("The json file does not contain any valid entity.")
    else:
        logger.info("The json file contains %d valid entity.", len(all_entitys))
    
    return all_entitys

def convert_to_json_compatible(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_compatible(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_compatible(i) for i in obj]
    else:
        return obj
    
def get_msa_templates_pipeline(args) -> Dict:
    use_precomputed_msas = True # FLAGS.use_precomputed_msas
    template_searcher = hmmsearch.Hmmsearch(
        binary_path=args.hmmsearch_binary_path,
        hmmbuild_binary_path=args.hmmbuild_binary_path,
        database_path=args.pdb_seqres_database_path)

    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=args.template_mmcif_dir,
        max_template_date=args.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=args.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=args.obsolete_pdbs_path)

    monomer_data_pipeline = pipeline.DataPipeline(
        jackhmmer_binary_path=args.jackhmmer_binary_path,
        hhblits_binary_path=args.hhblits_binary_path,
        hhsearch_binary_path=args.hhsearch_binary_path,
        uniref90_database_path=args.uniref90_database_path,
        mgnify_database_path=args.mgnify_database_path,
        bfd_database_path=args.bfd_database_path,
        uniclust30_database_path=args.uniclust30_database_path,
        small_bfd_database_path=args.small_bfd_database_path ,
        template_searcher=template_searcher,
        template_featurizer=template_featurizer,
        use_small_bfd=args.use_small_bfd,
        use_precomputed_msas=use_precomputed_msas)

    prot_data_pipeline = pipeline_multimer.DataPipeline(
        monomer_data_pipeline=monomer_data_pipeline,
        jackhmmer_binary_path=args.jackhmmer_binary_path,
        uniprot_database_path=args.uniprot_database_path,
        use_precomputed_msas=use_precomputed_msas)

    rna_monomer_data_pipeline = pipeline_rna.RNADataPipeline(
      hmmer_binary_path=args.nhmmer_binary_path,
      rfam_database_path=args.rfam_database_path,
      rnacentral_database_path=None,
      nt_database_path=None,     
      species_identifer_map_path=None,
      use_precomputed_msas=use_precomputed_msas)  

    rna_data_pipeline = pipeline_rna_multimer.RNADataPipeline(
      monomer_data_pipeline=rna_monomer_data_pipeline)

    return {
        'protein': prot_data_pipeline,
        'rna': rna_data_pipeline
    }

def ranking_all_predictions(output_dirs):
    ranking_score_path_map = {}
    for outpath in output_dirs:
        _results = preprocess.read_json(os.path.join(outpath, 'all_results.json'))
        _rank_score = _results['ranking_confidence']
        ranking_score_path_map[outpath] = _rank_score

    ranked_map = dict(sorted(ranking_score_path_map.items(), key=lambda x: x[1], reverse=True))
    rank_id = 1
    for outpath, rank_score in ranked_map.items():
        logger.debug("[ranking_all_predictions] Ranking score of %s: %.5f", outpath, rank_score)
        basename_prefix = os.path.basename(outpath).split('-pred-')[0]
        target_path = os.path.join(os.path.dirname(outpath), f'{basename_prefix}-rank{rank_id}')
        if os.path.exists(target_path) and os.path.isdir(target_path):
            shutil.rmtree(target_path)
        shutil.copytree(outpath, target_path)
        rank_id += 1

@paddle.no_grad()
def eval(args, model, batch):
    """evaluate a given dataset"""
    model.eval()       
        
    # inference
    def _forward_with_precision(batch):
        if args.precision == "bf16" or args.bf16_infer:
            black_list, white_list = get_custom_amp_list()
            with paddle.amp.auto_cast(enable=True,
                                        custom_white_list=white_list, 
                                        custom_black_list=black_list, 
                                        level=args.amp_level, 
                                        dtype='bfloat16'):
                return model(batch, compute_loss=False)
        elif args.precision == "fp32":
            return model(batch, compute_loss=False)
        else:
            raise ValueError("Please choose precision from bf16 and fp32! ")
        
    res = _forward_with_precision(batch)
    logger.info(f"Inference Succeeds...\n")
    return res


def postprocess_fn(batch, results, output_dir, maxit_binary=None):
    """
        postprocess function for HF3 output.
            - batch. input data
            - results. model output
            - output_dir. to save output
            - maxit_binary. path to maxit binary
    """
    diff_results = results['diffusion_module']
    confidence_results = results['confidence_head']

    required_keys = copy.deepcopy(all_atom_pdb_save.required_keys_for_saving)
    required_keys += ['token_bonds_type', 'ref_element', 'is_ligand']
    required_keys = required_keys + ['atom_plddts']

    # 1 feat extraction
    common_feat = {k: batch['feat'][k][0]
            for k in required_keys if k in batch['feat']}
    common_feat.update(
        {k: batch['label'][k][0]
            for k in required_keys if k in batch['label']}
    )
    common_feat.update(
        {'atom_plddts': confidence_results['atom_plddts'][0]})

    ## NOTE: remove "UNK-"
    common_feat['all_ccd_ids'] = re.sub(r'UNK-\w*', 'UNK', common_feat['all_ccd_ids']).split()
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

    atom_mask = np.logical_and(pred_dict["mask"] > 0, 
                exp_dict["mask"] > 0)[0]  # [N_atom]
    token_mask = batch['label']['all_centra_token_indice_mask'][0].numpy().astype('bool')
    # tensor to numpy
    for feat_key in common_feat:
        if isinstance(common_feat[feat_key], paddle.Tensor):
            common_feat[feat_key] = common_feat[feat_key].numpy()
        if feat_key in ['residue_index', 'asym_id']:
            common_feat[feat_key] = common_feat[feat_key].astype(np.int32)

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
    all_atom_pdb_save.prediction_to_mmcif(
        pred_dict["pos"][0][atom_mask], 
        common_feat_masked, 
        maxit_binary=maxit_binary, 
        mmcif_path=pred_cif_path)
    
    assert os.path.exists(pred_cif_path),\
              (f"pred: {pred_cif_path} not exists! please check it")


    #### NOTE: append some contexts to cif file, Now only support ligand-intra bond type.
    ## 1. license
    mmcif_writer.mmcif_meta_append(pred_cif_path)
    
    ## 2. post add ligand bond type;
    ## N_token, for ligand, N_token == N_atom
    ref_token2atom_idx = common_feat_masked['ref_token2atom_idx']
    is_ligand = common_feat_masked['is_ligand'].astype(bool) # N_token
    perm_is_ligand = is_ligand[ref_token2atom_idx].astype(bool)
    
    ccd_ids = common_feat_masked['all_ccd_ids'] # N_atom
    atom_ids = common_feat_masked['all_atom_ids'] # N_atom
    token_bond_type = common_feat_masked['token_bonds_type'] # N_token 
    bond_mat = token_bond_type[ref_token2atom_idx][:, ref_token2atom_idx] # N_token -> N_atom
    ligand_bond_type = bond_mat[perm_is_ligand][:, perm_is_ligand]
    index1, index2 = np.nonzero(ligand_bond_type)
    bonds = [(int(i), int(j), ligand_bond_type[i][j]) for i, j in zip(index1, index2) if i < j]
    ligand_atom_ids = atom_ids[perm_is_ligand]
    ligand_ccd_ids = ccd_ids[perm_is_ligand]

    contexts = {'_chem_comp_bond.comp_id': [], 
                '_chem_comp_bond.atom_id_1': [], 
                '_chem_comp_bond.atom_id_2 ': [],
                '_chem_comp_bond.value_order': []}
    for idx, (i, j, bd_type) in enumerate(bonds):
        _bond_type = INVERSE_ALLOWED_LIGAND_BONDS_TYPE_MAP[bd_type]
        contexts['_chem_comp_bond.comp_id'].append(ligand_ccd_ids[i])
        contexts['_chem_comp_bond.atom_id_1'].append(ligand_atom_ids[i])
        contexts['_chem_comp_bond.atom_id_2 '].append(ligand_atom_ids[j])
        contexts['_chem_comp_bond.value_order'].append(_bond_type)
        # contexts['_chem_comp_bond.pdbx_ordinal'].append(idx + 1)
    mmcif_writer.mmcif_append(pred_cif_path, contexts, rm_duplicates=True)
    #### NOTE: append some contexts to cif file


def get_display_results(batch, results):
    confidence_score_float_names = ['ptm', 'iptm', 'has_clash', 'mean_plddt', 'ranking_confidence']
    confidence_score_names = ['atom_plddts', 'pae']
    ## atom_plddts: [N_atom], pae: [N_token, N_token]
    required_atom_level_keys = atom_level_keys + ['atom_plddts']
    display_required_keys = ['all_ccd_ids', 'all_atom_ids', 
                            'ref_token2atom_idx', 'restype', 
                            'residue_index', 'asym_id',
                            'all_atom_pos_mask',]
    all_results = {k: [] for k in DISPLAY_RESULTS_KEYS}
    for k in confidence_score_float_names:
        all_results[k] = float(results['confidence_head'][k])

    diff_results = results['diffusion_module']
    # 1 feat extraction
    common_feat = {k: batch['feat'][k][0]
            for k in display_required_keys if k in batch['feat']}
    common_feat.update(
        {k: batch['label'][k][0]
            for k in  display_required_keys if k in batch['label']}
    )
    common_feat.update({k: results['confidence_head'][k][0]
                            for k in confidence_score_names})

    ## NOTE: remove "UNK-"
    common_feat['all_ccd_ids'] = re.sub(r'UNK-\w*', 'UNK', common_feat['all_ccd_ids']).split()
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

    atom_mask = np.logical_and(pred_dict["mask"] > 0, exp_dict["mask"] > 0)[0]  # [N_atom] get valid atom
    token_mask = batch['label']['all_centra_token_indice_mask'][0].numpy().astype('bool') # get valid token
    # tensor to numpy
    for feat_key in common_feat:
        if isinstance(common_feat[feat_key], paddle.Tensor):
            common_feat[feat_key] = common_feat[feat_key].numpy()
        if feat_key in ['residue_index', 'asym_id']:
            common_feat[feat_key] = common_feat[feat_key].astype(np.int32)

    def apply_mask(key, val):
        """ apply mask to val """
        val = np.array(val)
        if key in required_atom_level_keys:
            if key in ['ref_token2atom_idx']:
                return map_to_continuous_indices(val[atom_mask])
            return val[atom_mask]
        else:
            if key in ['token_bonds_type', 'pae']:
                return val[token_mask, :][:, token_mask] 
            return val[token_mask]
    common_feat_masked = {k: apply_mask(k, v) for k, v in common_feat.items()}

    ## NOTE: save display results.
    ref_token2atom_idx = common_feat_masked['ref_token2atom_idx']
    chain_ids = common_feat_masked['asym_id'][ref_token2atom_idx] # N_token -> N_atom

    ## token-level
    all_results['pae'] = common_feat_masked['pae']
    for i in common_feat_masked['asym_id']:
        all_results['token_chain_ids'].append(all_atom_pdb_save.all_chain_ids[i])
    for i in common_feat_masked['residue_index']:
        all_results['token_res_ids'].append(i)

    ## atom-level
    all_results['atom_plddts'] = common_feat_masked['atom_plddts']
    all_results['atom_chain_ids'] = [all_atom_pdb_save.all_chain_ids[ca_i] for ca_i in chain_ids]

    return all_results


def save_result(feature_dict, prediction, output_dir, maxit_bin):
    postprocess_fn(batch=feature_dict, 
                    results=prediction,
                    output_dir=output_dir,
                    maxit_binary=maxit_bin)
    
    all_results = {k: [] for k in DISPLAY_RESULTS_KEYS}
    res = get_display_results(batch=feature_dict,results=prediction)
    
    for k in all_results:
        if k in res:
            all_results[k] = convert_to_json_compatible(res[k])

    with open(output_dir.joinpath('all_results.json'), 'w') as f:
        f.write(json.dumps(all_results, indent=4))


def split_prediction(pred, rank):
    prediction = []
    feat_key_list = [pred[rk].keys() for rk in RETURN_KEYS]
    feat_key_table = dict(zip(RETURN_KEYS, feat_key_list))
    
    for i in range(rank):
        sub_pred = {}
        for rk in RETURN_KEYS:
            feat_keys = feat_key_table[rk]
            sub_feat = dict(zip(feat_keys, [pred[rk][fk][:, i] for fk in feat_keys]))
            sub_pred[rk] = sub_feat
    
        prediction.append(sub_pred)
    
    return prediction


def main(args):
    set_logging_level(args.logging_level)

    """main function"""
    new_einsum = os.getenv("FLAGS_new_einsum", True)
    print(f'>>> PaddlePaddle commit: {paddle.version.commit}')
    print(f'>>> FLAGS_new_einsum: {new_einsum}')
    print(f'>>> args:\n{args}')

    all_entitys = preprocess_json_entity(args.input_json, args.output_dir)
    ## check maxit binary path
    if args.maxit_binary is not None:
        assert os.path.exists(args.maxit_binary), \
            f"The maxit binary path {args.maxit_binary} does not exists."


    ### set seed for reproduce experiment results
    seed = args.seed
    if seed is None:
        seed = np.random.randint(10000000)
    else:
        logger.warning('seed is only used for reproduction')
    init_seed(seed)


    use_small_bfd = args.preset == 'reduced_dbs'
    setattr(args, 'use_small_bfd', use_small_bfd)
    if use_small_bfd:
        assert args.small_bfd_database_path is not None
    else:
        assert args.bfd_database_path is not None
        assert args.uniclust30_database_path is not None

    logger.info('Getting MSA/Template Pipelines...')
    if args.no_msa_templ_feats:
        logger.debug('MSA/Template Pipelines are not used.')
        msa_templ_data_pipeline_dict = {}
    else:
        msa_templ_data_pipeline_dict = get_msa_templates_pipeline(args)
        

    ### create model
    model_config = config.model_config(args.model_name)
    print(f'>>> model_config:\n{model_config}')

    model = RunModel(model_config)

    if (not args.init_model is None) and (not args.init_model == ""):
        print(f"Load pretrain model from {args.init_model}")
        pd_params = paddle.load(args.init_model)
        
        has_opt = 'optimizer' in pd_params
        if has_opt:
            model.helixfold.set_state_dict(pd_params['model'])
        else:
            model.helixfold.set_state_dict(pd_params)
    
    if args.precision == "bf16" and args.amp_level == "O2":
        raise NotImplementedError("bf16 O2 is not supported yet.")

    print(f"============ Data Loading ============")
    job_base = pathlib.Path(args.input_json).stem
    output_dir_base = pathlib.Path(args.output_dir).joinpath(job_base)
    msa_output_dir = output_dir_base.joinpath('msas')
    msa_output_dir.mkdir(parents=True, exist_ok=True)

    features_pkl = output_dir_base.joinpath('final_features.pkl')
    feature_dict = feature_processing_aa.process_input_json(
                all_entitys, 
                ccd_preprocessed_path=args.ccd_preprocessed_path,
                msa_templ_data_pipeline_dict=msa_templ_data_pipeline_dict,
                msa_output_dir=msa_output_dir,
                no_msa_templ_feats=args.no_msa_templ_feats)

    # save features
    with open(features_pkl, 'wb') as f:
        pickle.dump(feature_dict, f, protocol=4)

    feature_dict['feat'] = batch_convert(feature_dict['feat'], add_batch=True)
    feature_dict['label'] = batch_convert(feature_dict['label'], add_batch=True)
    
    print(f"============ Start Inference ============")
    
    infer_times = args.infer_times
    if args.diff_batch_size > 0:
        model_config.model.heads.diffusion_module.test_diff_batch_size = args.diff_batch_size
    diff_batch_size = model_config.model.heads.diffusion_module.test_diff_batch_size 
    logger.info(f'Inference {infer_times} Times...')
    logger.info(f" diffusion batch size {diff_batch_size}...\n")
    all_pred_path = []
    for infer_id in range(infer_times):
        
        logger.info(f'Start {infer_id}-th inference...\n')
        prediction = eval(args, model, feature_dict)
        
        # save result
        prediction = split_prediction(prediction, diff_batch_size)
        for rank_id in range(diff_batch_size):
            json_name = job_base + f'-pred-{str(infer_id + 1)}-{str(rank_id + 1)}'
            output_dir = pathlib.Path(output_dir_base).joinpath(json_name)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_result(feature_dict=feature_dict,
                        prediction=prediction[rank_id],
                        output_dir=output_dir, 
                        maxit_bin=args.maxit_binary)
            all_pred_path.append(output_dir)
    
    # final ranking
    print(f'============ Ranking ! ============')
    ranking_all_predictions(all_pred_path)
    print(f'============ Inference finished ! ============')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16_infer", action='store_true', default=False)
    parser.add_argument("--seed", type=int, default=None, help="set seed for reproduce experiment results, None is do not set seed")
    parser.add_argument("--logging_level", type=str, default="DEBUG", help="NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL")
    parser.add_argument("--model_name", type=str, help='used to choose model config')
    parser.add_argument("--init_model", type=str, default='')
    parser.add_argument("--precision", type=str, choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument("--amp_level", type=str, default='O1')
    parser.add_argument("--infer_times", type=int, default=1)
    parser.add_argument("--diff_batch_size", type=int, default=-1)
    parser.add_argument('--input_json', type=str,
                        default=None, required=True,
                        help='Paths to json file, each containing '
                        'entity information including sequence, smiles or CCD, copies etc.')
    parser.add_argument('--output_dir', type=str,
                        default=None, required=True,
                        help='Path to a directory that will store results.')
    parser.add_argument('--ccd_preprocessed_path', type=str,
                        default=None, required=True,
                        help='Path to CCD preprocessed files.')
    parser.add_argument('--jackhmmer_binary_path', type=str,
                        default='/usr/bin/jackhmmer',
                        help='Path to the JackHMMER executable.')
    parser.add_argument('--hhblits_binary_path', type=str,
                        default='/usr/bin/hhblits',
                        help='Path to the HHblits executable.')
    parser.add_argument('--hhsearch_binary_path', type=str,
                        default='/usr/bin/hhsearch',
                        help='Path to the HHsearch executable.')
    parser.add_argument('--kalign_binary_path', type=str,
                        default='/usr/bin/kalign',
                        help='Path to the Kalign executable.')
    parser.add_argument('--hmmsearch_binary_path', type=str,
                        default='/usr/bin/hmmsearch',
                        help='Path to the hmmsearch executable.')
    parser.add_argument('--hmmbuild_binary_path', type=str,
                        default='/usr/bin/hmmbuild',
                        help='Path to the hmmbuild executable.')

    # binary path of the tool for RNA MSA searching
    parser.add_argument('--nhmmer_binary_path', type=str,
                        default='/usr/bin/nhmmer',
                        help='Path to the nhmmer executable.')
    
    parser.add_argument('--uniprot_database_path', type=str,
                        default=None, required=True,
                        help='Path to the Uniprot database for use '
                        'by JackHMMER.')
    parser.add_argument('--pdb_seqres_database_path', type=str,
                        default=None, required=True,
                        help='Path to the PDB '
                        'seqres database for use by hmmsearch.')
    parser.add_argument('--uniref90_database_path', type=str,
                        default=None, required=True,
                        help='Path to the Uniref90 database for use '
                        'by JackHMMER.')
    parser.add_argument('--mgnify_database_path', type=str,
                        default=None, required=True,
                        help='Path to the MGnify database for use by '
                        'JackHMMER.')
    parser.add_argument('--bfd_database_path', type=str, default=None,
                        help='Path to the BFD database for use by HHblits.')
    parser.add_argument('--small_bfd_database_path', type=str, default=None,
                        help='Path to the small version of BFD used '
                        'with the "reduced_dbs" preset.')
    parser.add_argument('--uniclust30_database_path', type=str, default=None,
                        help='Path to the Uniclust30 database for use '
                        'by HHblits.')
    # RNA MSA searching databases
    parser.add_argument('--rfam_database_path', type=str,
                        default=None, required=True,
                        help='Path to the Rfam database for RNA MSA searching.')
    parser.add_argument('--template_mmcif_dir', type=str,
                        default=None, required=True,
                        help='Path to a directory with template mmCIF '
                        'structures, each named <pdb_id>.cif')
    parser.add_argument('--max_template_date', type=str,
                        default=None, required=True,
                        help='Maximum template release date to consider. '
                        'Important if folding historical test sets.')
    parser.add_argument('--obsolete_pdbs_path', type=str,
                        default=None, required=True,
                        help='Path to file containing a mapping from '
                        'obsolete PDB IDs to the PDB IDs of their '
                        'replacements.')
    parser.add_argument('--preset',
                        default='full_dbs', required=False,
                        choices=['reduced_dbs', 'full_dbs'],
                        help='Choose preset model configuration - '
                        'no ensembling and smaller genetic database '
                        'config (reduced_dbs), no ensembling and full '
                        'genetic database config  (full_dbs)')
    parser.add_argument('--maxit_binary', type=str, default=None)
    parser.add_argument('--no_msa_templ_feats', default=False, action='store_true')
    args = parser.parse_args()
    main(args)
