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
import os
import copy
import argparse
import random
import pathlib
import shutil
import logging
from typing import Dict, List

import numpy as np
import paddle

from helixfold.model import config, utils
from helixfold.data import msa_pipeline_protein
from helixfold.data import msa_pipeline_rna
from helixfold.data.utils import (
    ATOM_LEVEL_KEYS, 
    DISPLAY_RESULTS_KEYS,
    sorted_results_by_chain_order,
    map_to_continuous_indices)
from helixfold.data.tools import hmmsearch
from helixfold.data import templates
from utils.utils import get_custom_amp_list
from utils.model import RunModel
from utils.misc import set_logging_level
from infer_scripts import feature_processing_aa, preprocess, entity_bean
from infer_scripts.tools import mmcif_writer
from infer_scripts.tools.utils import write_format_json, convert_to_json_compatible
from infer_scripts.validation import JSON_SCHEMA_PATH
from infer_scripts.validation.input_validation import validate_input_file
from infer_scripts.tools.post_calculate import calculate_chain_pair_pae_matrix

logger = logging.getLogger(__file__)


def init_seed(seed: int):
    """set seed for reproduct results"""
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def tensor_to_numpy(common_feat: Dict[str, paddle.Tensor]) -> Dict[str, np.ndarray]:
    """Func of convert paddle(tensor) to numpy."""
    for feat_key in common_feat:
        if isinstance(common_feat[feat_key], paddle.Tensor):
            if common_feat[feat_key].dtype == paddle.bfloat16:
                common_feat[feat_key] = paddle.cast(common_feat[feat_key], 'float32').numpy()
            else:
                common_feat[feat_key] = common_feat[feat_key].numpy()
        if feat_key in ['residue_index', 'asym_id']:
            common_feat[feat_key] = common_feat[feat_key].astype(np.int32)
    
    return common_feat


def batch_convert(np_array: Dict[str, np.ndarray], add_batch=True):
    """Func of convert numpy to paddle tensor, also add batch dim."""
    np_type = {}
    other_type = {}
    
    for key, value in np_array.items():
        if type(value) == np.ndarray:
            try:
                np_type.update(utils.map_to_tensor({key: value}, add_batch=add_batch))
            except Exception as e:
                print(f"[ERROR] Failed to convert {key} to tensor: {e}")
                raise e
        else:
            other_type[key] = [value]

    return {**np_type, **other_type}


def preprocess_json_to_entity(json_path: str, out_dir: pathlib.Path) -> List[entity_bean.EntityBean]:
    """Preprocess json file to entity bean.

    Args:
        json_path: Path to input JSON file
        out_dir: Directory to write output files

    Returns:
        List of entity bean
    
    Raises:
        ValidationException: If input JSON file is not valid
    """

    # 1. Validate input JSON against schema
    logger.info(f'Validating input JSON against schema: {JSON_SCHEMA_PATH}')
    validate_input_file(json_path, schema_path=JSON_SCHEMA_PATH)
    
    # 2. simple preprocess entities
    all_entities = preprocess.online_json_parser(json_path)

    # 3. write the raw json file to out_dir
    shutil.copyfile(
        json_path, 
        out_dir.joinpath(os.path.basename(json_path))
    )

    # 4. copy the LICENSE to out_dir
    root_path = pathlib.Path(__file__).parent
    shutil.copyfile(
        pathlib.Path(root_path).joinpath('LICENSE'), 
        out_dir.joinpath('terms_of_use.md')
    )
    
    return all_entities


def get_msa_templates_pipeline(args: argparse.Namespace) -> Dict:
    """Get MSA/Template Pipelines for protein/rna"""
    
    # Check the args first
    use_reduced_bfd = args.preset == 'reduced_dbs'
    setattr(args, 'use_reduced_bfd', use_reduced_bfd)
    if use_reduced_bfd:
        assert args.reduced_bfd_database_path is not None
    else:
        assert args.bfd_database_path is not None
        assert args.uniclust30_database_path is not None

    template_searcher = hmmsearch.Hmmsearch(
        binary_path=args.hmmsearch_binary_path,
        hmmbuild_binary_path=args.hmmbuild_binary_path,
        database_path=args.pdb_seqres_database_path)

    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=args.template_mmcif_dir,
        max_template_date=args.max_template_date,
        max_hits=4,
        kalign_binary_path=args.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=args.obsolete_pdbs_path)

    protein_data_pipeline = msa_pipeline_protein.DataPipeline(
        jackhmmer_binary_path=args.jackhmmer_binary_path,
        hhblits_binary_path=args.hhblits_binary_path,
        uniref90_database_path=args.uniref90_database_path,
        mgnify_database_path=args.mgnify_database_path,
        bfd_database_path=args.bfd_database_path,
        uniclust30_database_path=args.uniclust30_database_path,
        reduced_bfd_database_path=args.reduced_bfd_database_path,
        uniprot_database_path=args.uniprot_database_path,
        template_searcher=template_searcher,
        template_featurizer=template_featurizer,
        use_reduced_bfd=args.use_reduced_bfd,
        use_precomputed_msas=True)

    rna_data_pipeline = msa_pipeline_rna.DataPipeline(
      hmmer_binary_path=args.nhmmer_binary_path,
      rfam_database_path=args.rfam_database_path,
      rnacentral_database_path=None,
      nt_database_path=None,     
      species_identifer_map_path=None,
      use_precomputed_msas=True)  

    return {
        'protein': protein_data_pipeline,
        'rna': rna_data_pipeline
    }


def ranking_all_predictions(output_dirs: List[pathlib.Path]):
    """Ranking all predictions based on ranking confidence.

    Args:
        output_dirs: List of output directories.
    """
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
def eval(args: argparse.Namespace, model: RunModel, batch: Dict) -> Dict:
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


def _get_common_feat_to_save(batch: Dict, 
                            results: Dict, 
                            cif_required_keys: List[str],
                            metric_required_keys: List[str],
                            mask_ignore_keys: List[str],
                            atom_level_mask_keys: List[str]) -> Dict:
    """Preprocess inference results, extract features, and apply masks.

    Args:
        batch: Input data.
        results: Model output.
        cif_required_keys: cif key information to be extracted from batch.
        metric_required_keys: metric key information to be extracted from confidence results.
        mask_ignore_keys: Key information to be ignored in mask.
        atom_level_mask_keys: Atom-level keys for apply different mask

    Returns:
        A dictionary of features after preprocessing and masking.
    """

    def _apply_mask(key, val, atom_mask, token_mask):
        """apply mask to val"""
        val = np.array(val)
        if key in mask_ignore_keys:
            return val
        elif key in atom_level_mask_keys:
            if key in ['ref_token2atom_idx']:
                return map_to_continuous_indices(val[atom_mask])
            return val[atom_mask]
        else:
            if key in ['token_bonds_type', 'pae']:
                return val[token_mask, :][:, token_mask] 
            return val[token_mask]

    # 0. prepare the keys first.
    diff_results = results['diffusion_module']
    confidence_results = results['confidence_head']

    # 1. keys-value extraction
    common_feat = {k: batch['feat'][k][0] for k in cif_required_keys if k in batch['feat']}
    common_feat.update({k: batch['label'][k][0] for k in cif_required_keys if k in batch['label']})
    common_feat.update({k: confidence_results[k][0] for k in metric_required_keys})

    # 2. convert string keys to list
    string_keys = ['all_chain_ids', 'all_ccd_ids', 'all_atom_ids', 'chain_ids']
    for k in string_keys:
        common_feat[k] = str(common_feat[k]).split()
    
    atom_mask = np.logical_and(
        diff_results['final_atom_mask'].numpy() > 0, 
        batch['label']['all_atom_pos_mask'].numpy() > 0)[0]
    token_mask = batch['label']['all_centra_token_indice_mask'][0].numpy().astype('bool')

    ## 3. convert to numpy and apply atom/token level mask for common features
    common_feat = tensor_to_numpy(common_feat)
    common_feat = {k: _apply_mask(k, v, atom_mask, token_mask) for k, v in common_feat.items()}
    common_feat['atom_mask'] = atom_mask
    common_feat['token_mask'] = token_mask

    ## 4. add ligand intra bond info
    common_feat['ligand_intra_bonds_info'] = mmcif_writer._prepare_bonds(common_feat)

    return common_feat 


def dump_cif(entry_name: str, 
            common_feat: Dict, 
            results: Dict, 
            output_dir: pathlib.Path, 
            mmcif_extra_infos: Dict):
    """save inference results to all_results.json file.

    Args:
        common_feat: input data
        results: model output
        output_dir: to save output
    """

    diff_results = results['diffusion_module'] 
    atom_mask = common_feat['atom_mask']
    ligand_intra_bond_info = common_feat['ligand_intra_bonds_info'] 
    pos = diff_results['final_atom_positions'].numpy()[0]
    
    pred_cif_path = f'{output_dir}/predicted_structure.cif'
    mmcif_writer.prediction_to_mmcif(
        entry_name=entry_name,
        atom_positions=pos[atom_mask],
        feats_dict=common_feat,
        mmcif_path=pred_cif_path,
        extra_infos=mmcif_extra_infos,
        ligand_intra_bonds_info=ligand_intra_bond_info,
    )
    assert os.path.exists(pred_cif_path), (f"pred: {pred_cif_path} not exists! please check it")


def dump_metric_results(common_feat: Dict, results: Dict, output_dir: pathlib.Path):
    """Save metric results to all_results.json file.

    Args:
        common_feat: common feature to save metric results
        results: model output
        output_dir: to save output
    """
    ## NOTE: save display results.
    all_results = {}
    ref_token2atom_idx = common_feat['ref_token2atom_idx']
    ori_chain_ids = common_feat['all_chain_ids'] # N_atom
    ori_chain_ids_token = common_feat['chain_ids'] # N_token

    ## 1. SINGLE
    all_results['ptm'] = float(common_feat['ptm'])
    all_results['iptm'] = float(common_feat['iptm'])
    all_results['has_clash'] = float(common_feat['has_clash'])
    all_results['mean_plddt'] = common_feat['atom_plddts'].mean()
    all_results['ranking_confidence'] = float(common_feat['ranking_confidence']) 

    ## 2. token-level
    all_results['pae'] = common_feat['pae']
    all_results['token_chain_ids'] = ori_chain_ids_token.tolist()
    all_results['token_res_ids'] = (common_feat['residue_index'] + 1).tolist()

    ## 3. atom-level
    all_results['atom_plddts'] = common_feat['atom_plddts']
    all_results['atom_chain_ids'] = ori_chain_ids.tolist()

    ## 4. chain-level
    all_results['chain_plddt'] = common_feat['chain_plddt']
    all_results['chain_pair_iptm'] = np.where(
        common_feat['chain_pair_mask'] == 1, 
        common_feat['chain_pair_iptm'], 
        np.nan)
    np.fill_diagonal(all_results['chain_pair_iptm'], np.nan)
    all_results['chain_ptm'] = np.where(
        np.diagonal(common_feat['chain_pair_mask']) == 1, 
        np.diagonal(common_feat['chain_pair_iptm']), 
        np.nan
    )
    _chain_pae_mask = np.diagonal(common_feat['chain_pair_mask']).astype('bool')
    all_results['chain_pair_pae_min'] = calculate_chain_pair_pae_matrix(
        all_results['pae'], 
        all_results['token_chain_ids'], 
        mask_chain=_chain_pae_mask, 
        mask_value=np.nan, 
        metric_type='min'
    )                      

    all_results = sorted_results_by_chain_order(
        all_results, 
        ori_chain_ids, 
        ori_chain_ids_token
    )

    ## final results and save to json file.
    final_results = {}
    for k in DISPLAY_RESULTS_KEYS:
        if k in all_results:
            final_results[k] = convert_to_json_compatible(all_results[k])
        else:
            raise ValueError(f'Key {k} not found in result; Required keys are: {DISPLAY_RESULTS_KEYS}.')

    write_format_json(
        final_results, 
        file_path=output_dir.joinpath('all_results.json'), 
        nan_to_none=True
    )


def save_result(entry_name: str, feature_dict: Dict, 
                prediction: Dict, output_dir: pathlib.Path, extra_infos: Dict = None):
    """Save the prediction results.

    Args:
        entry_name: str, the name of the json file.
        feature_dict: dict, the batch features from the input data.
        prediction: dict, the prediction from the inference.
        output_dir: pathlib.Path, the directory to save the prediction results.
        extra_infos: dict, the extra information for the prediction.
    """
    mask_ignore_keys = ['chain_plddt', 'chain_pair_iptm', 'chain_pair_mask',
                        'ptm', 'iptm', 'has_clash', 'ranking_confidence']
    metric_required_keys = ['atom_plddts', 'pae', 'ptm', 'iptm', 
                            'chain_plddt', 'chain_pair_iptm', 'chain_pair_mask',
                            'has_clash', 'ranking_confidence']
    cif_required_keys = copy.deepcopy(mmcif_writer.REQUIRED_KEYS_FOR_SAVING)

    common_feat_to_save = _get_common_feat_to_save(
        batch=feature_dict,
        results=prediction,
        cif_required_keys=cif_required_keys,
        metric_required_keys=metric_required_keys,
        mask_ignore_keys=mask_ignore_keys,
        atom_level_mask_keys=ATOM_LEVEL_KEYS,
    )

    # 1. save structure mmcif. 
    dump_cif(
        entry_name=entry_name,
        common_feat=common_feat_to_save, 
        results=prediction,
        output_dir=output_dir,
        mmcif_extra_infos=extra_infos
    )

    # 2. save the plddts, pae, and so on;
    dump_metric_results(
        common_feat=common_feat_to_save,
        results=prediction,
        output_dir=output_dir
    )


def split_prediction(pred: Dict[str, Dict[str, np.ndarray]], rank: int) -> List[Dict]:
    """split prediction to rank parts."""
    
    RETURN_KEYS = ['diffusion_module', 'confidence_head']
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
    """Main entry in HelixFold3 inference"""

    set_logging_level(args.logging_level)
    new_einsum = os.getenv("FLAGS_new_einsum", True)
    print(f'>>> PaddlePaddle commit: {paddle.version.commit}')
    print(f'>>> FLAGS_new_einsum: {new_einsum}')
    print(f'>>> args:\n{args}')

    ### set seed for reproduce experiment results
    seed = args.seed
    if seed is None:
        seed = np.random.randint(10000000)
    else:
        logger.warning('seed is only used for reproduction')
    init_seed(seed)

    job_base = pathlib.Path(args.input_json).stem
    output_dir_base = pathlib.Path(args.output_dir).joinpath(job_base)
    output_dir_base.mkdir(parents=True, exist_ok=True)
    all_entities = preprocess_json_to_entity(args.input_json, output_dir_base)

    logger.info('Getting MSA/Template Pipelines...')
    msa_templ_data_pipeline_dict = get_msa_templates_pipeline(args) \
                                if os.getenv("NO_MSA") != "1" else {}
        

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
    feature_dict = feature_processing_aa.featurize_entities(
        all_entities, 
        ccd_preprocessed_path=args.ccd_preprocessed_path,
        msa_templ_data_pipeline_dict=msa_templ_data_pipeline_dict,
        msa_output_dir=output_dir_base.joinpath('msas'),
        use_msa_templ_feats=os.getenv("NO_MSA") != "1"
    )
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
            save_result(entry_name=job_base,
                        feature_dict=feature_dict,
                        prediction=prediction[rank_id],
                        output_dir=output_dir, 
                        extra_infos={})
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
    parser.add_argument('--reduced_bfd_database_path', type=str, default=None,
                        help='Path to the reduced version of BFD used '
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
                        'no ensembling and reduced genetic database '
                        'config (reduced_dbs), no ensembling and full '
                        'genetic database config  (full_dbs)')
    args = parser.parse_args()
    main(args)
