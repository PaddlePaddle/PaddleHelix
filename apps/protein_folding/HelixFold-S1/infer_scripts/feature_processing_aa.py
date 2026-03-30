"""Functions for building the features for the AlphaFold-3 inference pipeline."""
import copy
import dataclasses
from functools import partial
import json
import os
import time, gzip, pickle
from typing import Mapping, List, Dict
import numpy as np
import pathlib
import logging
import collections
from datetime import datetime
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

from helixfold.common import residue_constants
from helixfold.data import parsers
from helixfold.data import pipeline_multimer
from helixfold.data import pipeline_rna_parallel, pipeline_rna_multimer
from helixfold.data import pipeline_conf_bonds, pipeline_token_feature, pipeline_hybrid
from helixfold.data import label_utils
from helixfold.data.templates_input import get_refer_structure_features
from utils.interface_utils import (
    GEN_INTERFACE_CONFIG, get_interface_mask, 
    get_interface_info, get_interface_sample_constraint_mask
)

from infer_scripts.colab_msa_to_feature import DataPipelineColabSearchProtein
from infer_scripts.tools.cache_utils import search_cache, search_cache_exact, update_dynamic_cache
from infer_scripts.tools.antibody_check import antibody_chain_check
from infer_scripts.tools.utils import read_json

logger = logging.getLogger(__file__)

POLYMER_STANDARD_RESI_ATOMS = residue_constants.residue_atoms
STRING_FEATURES = ['all_chain_ids', 'all_ccd_ids','all_atom_ids', 'chain_ids',
                  'release_date','label_ccd_ids','label_atom_ids']
ALL_DTYPE_ORDER = ['protein', 'rna', 'dna', 'ligand']
MAX_MSA_DEPTH = 16384

@dataclasses.dataclass(frozen=True)
class MSATaskMeta:
    """
    Class representing a MSATaskMeta.
      - data_pipeline (DataPipeline): DataPipeline对象，用于处理单个链。
      - chain_id (str): 链ID。
      - seq (Optional[str]): 链序列（可选）。
      - desc (Optional[str]): 链描述（可选）。## NOTE： 这里默认是type_chain_id.
      - msa_output_dir (PathLike): MSA输出目录。
      - features_pkl (PathLike): 特征文件路径。  
      - task_type (Literal): 任务类型, protein or rna
    """
    data_pipeline: Mapping[str, any]
    chain_id: str
    seq: str
    desc: str
    msa_output_dir: str
    features_pkl: str
    task_type: str
    def __post_init__(self):
      assert self.task_type in ['protein', 'rna'],\
          (f"task_type: {self.task_type} not in ['protein', 'rna']")


def load_ccd_dict(ccd_preprocessed_path):
    assert os.path.exists(ccd_preprocessed_path),\
              (f'[CCD] ccd_preprocessed_path: {ccd_preprocessed_path} not exist.')
    st_1 = time.time()
    if 'pkl.gz' in ccd_preprocessed_path:
        with gzip.open(ccd_preprocessed_path, "rb") as fp:
            ccd_preprocessed_dict = pickle.load(fp)
    elif '.pkl' in ccd_preprocessed_path:
        with open(ccd_preprocessed_path, "rb") as fp:
            ccd_preprocessed_dict = pickle.load(fp)
    logger.info(f'[CCD] Load ccd dataset done. use: {time.time()-st_1}s; '\
                    f'length of ccd: {len(ccd_preprocessed_dict)}')
    
    return ccd_preprocessed_dict


def crop_msa(feat, max_msa_depth=16384):
    """ pad msa and generate msa_mask. """
    msa = feat['msa']
    deletion_mat = feat['deletion_matrix']
    has_deletion = feat['has_deletion']
    deletion_value_mat = feat['deletion_value']
    msa_mask = np.ones_like(feat['msa']).astype('float32') # [msa_depth, n_token]

    msa_depth, num_token = msa_mask.shape
    if msa_depth > max_msa_depth:
        msa_mask = msa_mask[: max_msa_depth, :]
        msa = msa[: max_msa_depth, :]
        deletion_mat = deletion_mat[: max_msa_depth, :]
        has_deletion = has_deletion[: max_msa_depth, :]
        deletion_value_mat = deletion_value_mat[: max_msa_depth, :]
    
    feat['msa'] = msa.astype('int32')
    feat['msa_mask'] = msa_mask
    feat['deletion_matrix'] = deletion_mat
    feat['has_deletion'] = has_deletion
    feat['deletion_value'] = deletion_value_mat

    return feat


def get_padding_restype(ccd_id, ccd_preprocessed_dict, dtype, 
                            extra_feats=None, is_poly_point=False):

  def _set_std_token_indices(pad_feats, token_indice, key_prefix):
      """
          Helper function to set token indices and masks.
      """
      token_indice = np.where(token_indice == 1)[0]
      assert len(token_indice) <= 1, f"residue should have only one {key_prefix}-token, Got {len(token_indice)}"
      pad_feats[f'{key_prefix}_token_indice'] = token_indice if len(token_indice) == 1 else np.array([0], dtype=np.int32)
      pad_feats[f'{key_prefix}_token_indice_mask'] = np.array([1] if len(token_indice) == 1 else [0], dtype=np.int32)

  def _get_polymer_modified_ids(dtype: str, ccd_id:str, atom_ids_list: list, is_poly_point: bool):
    """
        Processing the atom_ids in common modified residues from standard CCD.
    """
    _atom_ids_list = copy.deepcopy(atom_ids_list)
    if not is_poly_point: 
      if dtype in ['dna', 'rna'] and "OP3" in _atom_ids_list:
          _atom_ids_list.remove("OP3")
      elif dtype in ['protein'] and "OXT" in _atom_ids_list:
          _atom_ids_list.remove("OXT")

    return _atom_ids_list

  if ccd_id in ccd_preprocessed_dict:
    refs = ccd_preprocessed_dict[ccd_id]  # O(1)
    if ccd_id in residue_constants.STANDARD_LIST:
      _residue_is_standard = True 
      if not is_poly_point:
          pdb_atom_ids_list = POLYMER_STANDARD_RESI_ATOMS[ccd_id]
      else:
          pdb_atom_ids_list = refs['atom_ids']
    else:
      # for ligand/ion/modified_residues. ccd_id.
      _residue_is_standard = False
      pdb_atom_ids_list = refs['atom_ids']
      ## if common modified residues from standard CCD, we should special process for the terminal atom.
      pdb_atom_ids_list = _get_polymer_modified_ids(
                        dtype, ccd_id, pdb_atom_ids_list, is_poly_point)
  else:
    # for ligand/ion. smiles.
    assert not extra_feats is None and ccd_id in extra_feats
    _residue_is_standard = False
    refs = extra_feats[ccd_id]
    pdb_atom_ids_list = refs['atom_ids']
    pdb_atom_ids_list = _get_polymer_modified_ids(
                      dtype, ccd_id, pdb_atom_ids_list, is_poly_point)

  _atom_positions_list = refs['position']
  ## NOTE: map atom_ids to original atom_ids order from ccd; 
  if dtype != 'ligand':
    ccd_ori_atom_ids_order = refs['atom_ids']
    _new_atom_ids_list = []
    _new_atom_positions_list = []
    for idx, key in enumerate(ccd_ori_atom_ids_order):
      if key in pdb_atom_ids_list:
        _new_atom_ids_list.append(key)
        _new_atom_positions_list.append(_atom_positions_list[idx])
    assert len(_new_atom_ids_list) == len(pdb_atom_ids_list) == len(_new_atom_positions_list)
    pdb_atom_ids_list = _new_atom_ids_list
    _atom_positions_list = _new_atom_positions_list

  ref_atom_ids_index = { 
    name: i for i, name in enumerate(refs['atom_ids'])
  }
  total_nums = len(ref_atom_ids_index)
  assert total_nums > 0, f'TODO filter - Got CCD <{ccd_id}>: 0 atom nums.'
  padding_atom_pos = np.zeros([total_nums, 3], dtype=np.float32) ## Dummy Atom pos
  padding_atom_mask = np.zeros([total_nums], dtype=np.int32)
  centra_token_indice = np.zeros([total_nums], dtype=np.int32) 
  pseudo_token_indice = np.zeros([total_nums], dtype=np.int32)

  for at_id in pdb_atom_ids_list:
    if at_id in ref_atom_ids_index: 
      adjust_idx = ref_atom_ids_index[at_id]
      padding_atom_mask[adjust_idx] = 1
      if _residue_is_standard:
        if at_id in residue_constants.CENTRA_TOKEN:
          centra_token_indice[adjust_idx] = 1
          if ccd_id.upper() == "GLY":
            pseudo_token_indice[adjust_idx] = 1
        elif at_id in residue_constants.PSEUDO_TOKEN:
          if at_id == 'CB': 
            pseudo_token_indice[adjust_idx] = 1
          elif at_id == 'P' and ccd_id.upper() in residue_constants.DNA_RNA_LIST: 
            pseudo_token_indice[adjust_idx] = 1
      else:
        centra_token_indice[adjust_idx] = 1
        pseudo_token_indice[adjust_idx] = 1

  frame_indice = label_utils.get_pae_frame_mask(atom_ids_list=pdb_atom_ids_list, 
                  atom_positions_list=_atom_positions_list,
                  residue_name_3=ccd_id,
                  residue_is_standard=_residue_is_standard,
                  residue_is_missing=False,
                  ref_atom_ids_index=ref_atom_ids_index)

  pad_feats = {
    **frame_indice, 
    'ccd_ids': np.array([ccd_id] * total_nums , dtype=object),  # N_atom
    'atom_ids': refs['atom_ids'], # [N_atom]
    'atom_pos': padding_atom_pos,  # [N_atom]
    'atom_mask': padding_atom_mask,  # [N_atom]
    'token_to_atom_nums': np.array([total_nums], dtype=np.int32) \
                     if _residue_is_standard else np.ones([total_nums], dtype=np.int32)
  }

  if _residue_is_standard:
      _set_std_token_indices(pad_feats, centra_token_indice, 'centra')
      _set_std_token_indices(pad_feats, pseudo_token_indice, 'pseudo')
  else:
      # if is non-standard, token_nums == atom_nums
      pad_feats['centra_token_indice'] = np.zeros_like(centra_token_indice, dtype=np.int32)
      pad_feats['centra_token_indice_mask'] = centra_token_indice
      pad_feats['pseudo_token_indice'] = np.zeros_like(pseudo_token_indice, dtype=np.int32)
      pad_feats['pseudo_token_indice_mask'] = pseudo_token_indice

  return pad_feats


def get_inference_restype_mask(all_chain_features, ccd_preprocessed_dict, extra_feats=None):
    """
      all_chain_features: <type>_<chain_id>: chain_features, chain_features should has the ccd_seqs
      ccd_preprocessed_dict: preprocessed CCD dict
    """

    # Initialize empty arrays for various features
    all_ccd_ids = np.empty((0,), dtype=object)
    all_atom_ids = np.empty((0,), dtype=object)
    all_atom_pos = np.empty((0, 3), dtype=np.float32)
    all_atom_mask = np.empty((0,), dtype=np.int32)
    all_centra_token_indice = np.empty((0,), dtype=np.int32)
    all_centra_token_indice_mask = np.empty((0,), dtype=np.int32)
    all_token_to_atom_nums = np.empty((0,), dtype=np.int32)
    all_pseudo_token_indice = np.empty((0,), dtype=np.int32)
    all_pseudo_token_indice_mask = np.empty((0,), dtype=np.int32)
    frame_ai_indice = np.empty((0,), dtype=np.int32)  # Ntoken
    frame_bi_indice = np.empty((0,), dtype=np.int32)  # Ntoken
    frame_ci_indice = np.empty((0,), dtype=np.int32)  # Ntoken
    frame_mask = np.empty((0,), dtype=np.int32)  # Ntoken

    frame_indice_offset = 0
    for type_chain_id, ccd_list in all_chain_features.items():
        dtype, chain_id = type_chain_id.rsplit('_', 1)
        for idx, ccd_id in enumerate(ccd_list):
            is_poly_point = (idx == len(ccd_list) - 1 and dtype == 'protein') or (idx == 0 and dtype in ['rna', 'dna'])
            pad_feats = get_padding_restype(ccd_id, ccd_preprocessed_dict, dtype, extra_feats=extra_feats, 
                                            is_poly_point=is_poly_point)

            # Update indices with the current frame offset
            for key in ['ai_indice', 'bi_indice', 'ci_indice']:
                pad_feats[key] += frame_indice_offset
            frame_indice_offset += pad_feats['frame_atom_offset']

            # Concatenate features
            all_atom_ids = np.concatenate((all_atom_ids, pad_feats['atom_ids']))
            all_ccd_ids = np.concatenate((all_ccd_ids, pad_feats['ccd_ids']))
            all_atom_pos = np.concatenate((all_atom_pos, pad_feats['atom_pos']))
            all_atom_mask = np.concatenate((all_atom_mask, pad_feats['atom_mask']))
            all_centra_token_indice = np.concatenate((all_centra_token_indice, pad_feats['centra_token_indice']))
            all_centra_token_indice_mask = np.concatenate((all_centra_token_indice_mask, pad_feats['centra_token_indice_mask']))
            all_token_to_atom_nums = np.concatenate((all_token_to_atom_nums, pad_feats['token_to_atom_nums']))
            all_pseudo_token_indice = np.concatenate((all_pseudo_token_indice, pad_feats['pseudo_token_indice']))
            all_pseudo_token_indice_mask = np.concatenate((all_pseudo_token_indice_mask, pad_feats['pseudo_token_indice_mask']))
            frame_ai_indice = np.concatenate((frame_ai_indice, pad_feats['ai_indice']))
            frame_bi_indice = np.concatenate((frame_bi_indice, pad_feats['bi_indice']))
            frame_ci_indice = np.concatenate((frame_ci_indice, pad_feats['ci_indice']))
            frame_mask = np.concatenate((frame_mask, pad_feats['frame_indice_mask']))

    # Calculate cumulative sums for token indices
    cumsum_array = np.cumsum(all_token_to_atom_nums)
    all_centra_token_indice += np.insert(cumsum_array[:-1], 0, 0)
    all_pseudo_token_indice += np.insert(cumsum_array[:-1], 0, 0)

    # Assertions to ensure data integrity
    assert all_atom_pos.shape[0] == all_atom_mask.shape[0] == all_atom_ids.shape[0]
    assert all_centra_token_indice.shape[0] == all_centra_token_indice_mask.shape[0]
    assert all_pseudo_token_indice.shape[0] == all_pseudo_token_indice_mask.shape[0]
    assert frame_ai_indice.shape[0] == frame_bi_indice.shape[0] == frame_ci_indice.shape[0] == frame_mask.shape[0]  # Ntoken
    assert np.max(frame_ai_indice) < all_atom_pos.shape[0] and \
           np.max(frame_bi_indice) < all_atom_pos.shape[0] and \
           np.max(frame_ci_indice) < all_atom_pos.shape[0]

    return {
        "label_ccd_ids": all_ccd_ids,  # [N_atom, ]
        "label_atom_ids": all_atom_ids,  # [N_atom,]
        "all_atom_pos": all_atom_pos,  # [N_atom, 3]
        "all_atom_pos_mask": all_atom_mask,  # [N_atom]
        "all_centra_token_indice": all_centra_token_indice,  # [N_token, ]
        "all_centra_token_indice_mask": all_centra_token_indice_mask,  # [N_token,]
        "all_token_to_atom_nums": all_token_to_atom_nums,  # [N_token,]
        "pseudo_beta": all_atom_pos[all_pseudo_token_indice],  # [N_token,]
        "pseudo_beta_mask": all_pseudo_token_indice_mask,  # [N_token, ]

        # for pae loss computation.
        "frame_ai_indice": frame_ai_indice,  # [N_token, ]
        "frame_bi_indice": frame_bi_indice,  # [N_token, ]
        "frame_ci_indice": frame_ci_indice,  # [N_token, ]
        "frame_mask": frame_mask,  # [N_token, ]
    }


def add_assembly_features(all_chain_features, ccd_preprocessed_dict, use_msa_templ_feats=True):
  '''
    ## NOTE: keep the type and chainID orders.
    all_chain_features: {
        <type>_<chain_id>: {
          'msa_templ_feats': [msa_templ_feats],
          'ccd_seqs': [ccd_list], 
          'extra_feats': [extra_mol_info],
          'raw_info': [raw_info]
          }
        }
    }
    1. include msa pair_and_merge, af2 raw processing.
        pipeline_multimer.process_with_all_chain_features
        pipeline_rna_multimer.process_with_all_chain_features
    2. all type msa/template dense joint
        pipeline_hybrid
    3. include basic feature assembly, 
        such as token_index, entity_id, asym_id, sym_id, ref_space_uid, token_bonds, 
        perm_atom_index, ref_token2atom_idx, ref_atom_count, perm_entity_id, perm_asym_id
  '''
  ## first, prepare for msa_feats and chain_group_feats.
  extra_feats_infos = {}
  type_chain_id_to_raw_info = {}
  dtype_grouped_chains = collections.defaultdict(dict)
  dtype_msa_feats = collections.defaultdict(dict)
  for type_chain_id, chain_features in all_chain_features.items():
    dtype, chain_id = type_chain_id.rsplit('_', 1) 
    dtype_msa_feats[dtype][chain_id] = chain_features.pop('msa_templ_feats')
    dtype_grouped_chains[dtype][chain_id] = chain_features
    extra_feats_infos.update(chain_features['extra_feats'])
    type_chain_id_to_raw_info[type_chain_id] = chain_features['raw_info']

  ## total features should has keys: seq_token、conf_bond、protein/dna/rna/ligand(MSA feats / ccd_seqs / extra_feats).
  total_feats = {} 
  chain_order_mapping = {}
  ## 1. msa pair_and_merge for protein/rna, use af2-msa raw processing.
  for dtype in ALL_DTYPE_ORDER:
      if dtype not in dtype_msa_feats:
        continue
      msa_feats = {}
      if dtype in ['protein', 'rna'] and use_msa_templ_feats:
          chain_group_feats = dtype_msa_feats[dtype]
          if dtype == 'rna':
              for chain_id, features in chain_group_feats.items():
                  chain_group_feats[chain_id] = pipeline_rna_multimer.process_feat_to_mapping_to_new_token_list(features)
          msa_feats, _new_order_chain_ids = pipeline_multimer.process_with_all_chain_features(chain_group_feats, return_new_order=True)
          chain_order_mapping[dtype] = _new_order_chain_ids
      else:
          chain_order_mapping[dtype] = list(dtype_grouped_chains[dtype].keys())
      total_feats[dtype] = msa_feats

      if dtype == 'protein' and not use_msa_templ_feats:
        total_feats[dtype]['assembly_num_chains'] = len(chain_order_mapping[dtype])

  ## 1.1 reorder chain_ids for each type and add ccd_seqs and extra_feats.
  new_order_chain_infos = {}
  for dtype in ALL_DTYPE_ORDER:
      if dtype in chain_order_mapping:
          _new_order_chain_ids = chain_order_mapping[dtype]
          grouped_chains_feats = dtype_grouped_chains[dtype]
          total_feats[dtype]["ccd_seqs"] = np.concatenate([
                  np.array(grouped_chains_feats[chain_id]['ccd_seqs'], dtype=object) \
                  for chain_id in _new_order_chain_ids])
          total_feats[dtype]["extra_feats"] = extra_feats_infos
          
          for chain_id in _new_order_chain_ids:
              new_order_chain_infos[dtype + '_' + chain_id] = grouped_chains_feats[chain_id]['ccd_seqs']
  
  logger.warning(f'original_order_chain_infos: {list(all_chain_features.keys())}')
  logger.warning(f'new_order_chain_infos: {list(new_order_chain_infos.keys())}')

  ## 2. make token_seq_feats and conf_bond_feats.
  ## new_order_chain_infos is the new chain_ids order; dict: <type_chain_id>: <ccd_seqs>
  ## chatype_id_to_asym_id: dict, <type_chain_id>: <asym_id>, such as: {'protein_1-1': 1, 'protein_1-2': 2, ...}
  token_features, chatype_id_to_asym_id = pipeline_token_feature.make_sequence_features(
                                          all_chain_info=new_order_chain_infos,
                                          ccd_preprocessed_dict=ccd_preprocessed_dict,
                                          extra_feats=extra_feats_infos)
  
  ## 3. Get reference features and bond features
  ref_features = pipeline_conf_bonds.make_ccd_conf_features(all_chain_info=new_order_chain_infos,
                                                      ccd_preprocessed_dict=ccd_preprocessed_dict,
                                                      extra_feats=extra_feats_infos)
  bond_features = pipeline_conf_bonds.make_bond_features(covalent_bond=[], 
                                                      all_chain_info=new_order_chain_infos, 
                                                      ccd_preprocessed_dict=ccd_preprocessed_dict,
                                                      extra_feats=extra_feats_infos)
  ## 4. post convert features
  total_feats['seq_token'] = token_features
  total_feats['conf_bond'] = {**ref_features, **bond_features}
  np_example = pipeline_hybrid.post_convert(ccd_preprocessed_dict=ccd_preprocessed_dict,
                                                  all_chain_feats_dict=total_feats,
                                                  ordered_chain_types=ALL_DTYPE_ORDER)
  np_example = pipeline_hybrid.make_pseudo_beta(np_example, prefix='template_')
  np_example = pipeline_hybrid.make_template_further_feature(np_example)
  np_example = pipeline_hybrid.make_template_inter_mask(np_example)

  np_example["seq_mask"] = np.ones_like(np_example['restype']).astype('float32')
  np_example = crop_msa(np_example, max_msa_depth=MAX_MSA_DEPTH)
  
  ## 5. get inference pos mask:
  label = get_inference_restype_mask(new_order_chain_infos, ccd_preprocessed_dict, extra_feats_infos)

  ## 6. get feature chain_mapping， such as: {'1': <raw_info>, '2': <raw_info>, ...}
  chain_id_to_raw_info = {
      _asym_id: type_chain_id_to_raw_info[k] 
      for k, _asym_id in chatype_id_to_asym_id.items()
  }

  return {"feats": np_example,
          "label": label, 
          "chain_mapping": chain_id_to_raw_info,}


def colabfold_search_single(data_pipeline: DataPipelineColabSearchProtein,
                            sequence: str, description: str,
                            msa_output_dir: str,):
  """
    colabfold_search for protein single chains.
  """     
  fasta_strings = f">{description}\n{sequence}\n"
  msa_output_p = pathlib.Path(msa_output_dir) / 'colabfold_protein.msa'
  msa_output_p.mkdir(parents=True, exist_ok=True)
  
  new_descr_fasta = msa_output_p / 'colabfold_protein.fasta'
  with open(new_descr_fasta, 'w') as f:
    f.write(fasta_strings)
  
  ## start colabfold_search and data feature pipeline.
  a3m_p, m8_p = data_pipeline.colabfold_search(
                new_descr_fasta,
                msa_output_dir=msa_output_p / 'colabfold_protein.fasta.results')

  ## NOTE: Single chain only use 101.
  feature = data_pipeline.process_chain(input_fasta_path=new_descr_fasta, 
                                        a3m_msa_path=a3m_p,
                                        template_m8_path=m8_p, 
                                        chain_name='101') 

  with open(msa_output_p / 'features.pkl', 'wb') as f:
    pickle.dump(feature, f)
  
  return feature


def process_chain_msa(use_colabfold_search: bool, msa_task: MSATaskMeta):
    """
    处理链，如果缓存了特征文件，则直接使用缓存的特征文件，否则生成新的特征文件。
    
    Args:
        msa_task: MSATaskMeta
        use_colabfold_search: bool, whether to use colabfold_search.
    Returns:
        tuple: 返回一个元组，包含以下元素：
            - chain_id (str): 链ID。
            - raw_features (dict): 处理后的特征字典，包含预处理后的特征和其他相关信息。
            - desc (str): 链描述。
            - seq (str): 链序列。
    """
    ## data_pipeline: dict
    _data_pipeline, chain_id, seq, desc, msa_output_dir, features_pkl = \
        msa_task.data_pipeline, msa_task.chain_id, msa_task.seq, msa_task.desc, msa_task.msa_output_dir, msa_task.features_pkl
    task_type = msa_task.task_type
    if task_type == 'protein':
        common_pipeline = _data_pipeline['common']
        if use_colabfold_search:
          colab_pipeline = _data_pipeline['colab']
        else:
          colab_pipeline = None
    else:
        common_pipeline = _data_pipeline['common']
        colab_pipeline = None

    if features_pkl.exists():
        logger.info('Use offline features.pkl, for debug only!')
        with open(features_pkl, 'rb') as f:
            raw_features = pickle.load(f)
    else:
        t0 = time.time()
        raw_features = {}

        cache_hit = None
        if not isinstance(common_pipeline._monomer_data_pipeline, pipeline_rna_parallel.RNADataPipeline) \
          and common_pipeline._monomer_data_pipeline.use_msa_cache:
            # 插入MSA缓存逻辑，不需要并行
            msa_cache_dir = common_pipeline._monomer_data_pipeline.msa_cache_dir
            msa_cache_dir_oas = "single_chain_oas/oas_msa"
            msa_index_oas = 'protein_oas_msa_index.fasta'
            msa_cache_dir_dynamic = os.path.join(msa_cache_dir, "single_chain_colab")
            msa_index_dynamic = os.path.join(msa_cache_dir, 'ind/online_seq_msa_mapping.json')

            try:
              # 1. search oas cache
              if common_pipeline._monomer_data_pipeline.use_oas_msa_cache and cache_hit is None:
                cache_hit = search_cache(
                  os.path.join(msa_cache_dir, msa_index_oas),
                  binary_path=common_pipeline._monomer_data_pipeline.jackhmmer_mgnify_runner.binary_path,
                  desc=desc, seq=seq, chain_id=chain_id, 
                  feat_dir=os.path.join(msa_cache_dir, msa_cache_dir_oas)
                )
                if not cache_hit is None:
                    logger.info(f"[MSA/Template] {desc} Using cache hit from {msa_index_oas}")
                    return cache_hit

              # 2. search single_chain msa
              # PDB缓存列表目前有17w条左右，检索应该可以在1s之内完成
              if cache_hit is None:
                msa_index_fasta = os.path.join(msa_cache_dir, 'protein_msa_index.fasta')
                msa_feat_dir = os.path.join(msa_cache_dir, 'single_chain')
                fasta_id_mapping = os.path.join(msa_cache_dir, 'fasta_id.mapping')
                with open(fasta_id_mapping, 'r') as f:
                    mapping_dict = json.load(f)
                    inv_mapping_dict = {v: k for k, v in mapping_dict.items()}
                if not os.path.exists(msa_index_fasta):
                    with open(msa_index_fasta, 'w') as f:
                        for k, v in mapping_dict.items():
                            f.write(f">{v}\n")
                            f.write(k)
                            f.write("\n")

                fuzzy_match = False
                if fuzzy_match:
                  # 模糊匹配
                  cache_hit = search_cache(msa_index_fasta=msa_index_fasta,
                                binary_path=common_pipeline._monomer_data_pipeline.jackhmmer_mgnify_runner.binary_path,
                                desc=desc, seq=seq, chain_id=chain_id, feat_dir=msa_feat_dir)
                else:
                  # 精确匹配
                  cache_hit = search_cache_exact(fasta_id_mapping,
                      desc=desc, seq=seq, chain_id=chain_id, feat_dir=msa_feat_dir)

                if not cache_hit is None:
                    logger.info(f"[MSA/Template] {desc} Using cache hit from PDB.")
                    return cache_hit

              # 3. search online colab msa
              if cache_hit is None:
                # search colabfold cache
                if common_pipeline._monomer_data_pipeline.use_online_msa_cache and cache_hit is None:
                  cache_hit = search_cache_exact(
                    os.path.join(msa_cache_dir, msa_index_dynamic),
                    desc=desc, seq=seq, chain_id=chain_id, 
                    feat_dir=msa_cache_dir_dynamic
                  )
                  if not cache_hit is None:
                      logger.info(f"[MSA/Template] {desc} Using cache hit from {msa_index_dynamic}")
                      return cache_hit

              if cache_hit is None:
                    raise Exception(f'[MSA/Template] {desc}; no cache hit processed successfully, search msa on the fly!')
            except Exception as e:
                logger.warning(e)
        
        if use_colabfold_search and colab_pipeline is not None:
            ## 蛋白，使用colabfold搜索
            raw_features = colabfold_search_single(
                              data_pipeline=colab_pipeline,
                              sequence=seq, description=desc,
                              msa_output_dir=msa_output_dir)
        else:
            ## RNA, or no colabfold_search or protein cache not found, should be use normal pipeline.
            raw_features = common_pipeline._process_single_chain(
                chain_id, sequence=seq, description=desc,
                msa_output_dir=msa_output_dir,
                is_homomer_or_monomer=False)
            logger.info(f'[MSA/Template] {desc}; seq length: {len(seq)}; use: {time.time() - t0}')

            with open(features_pkl, 'wb') as f:
                pickle.dump(raw_features, f, protocol=4)
        
        if cache_hit is None and not isinstance(common_pipeline._monomer_data_pipeline, pipeline_rna_parallel.RNADataPipeline):
          # save online searched msa to cache
          pkl_dir = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(0, 999):03}"
          dynamic_cache_pkl = ''
          try:
            dynamic_cache_pkl = os.path.join(
              # common_pipeline._monomer_data_pipeline.msa_cache_dir,
              msa_cache_dir_dynamic, pkl_dir , 'features.pkl')
            os.makedirs(os.path.dirname(dynamic_cache_pkl), exist_ok=True)
            os.chmod(os.path.abspath(os.path.dirname(dynamic_cache_pkl)), 0o777)
            with open(dynamic_cache_pkl, "wb") as dynamic_cache:
              pickle.dump(raw_features, dynamic_cache, protocol=4)
            os.chmod(os.path.abspath(dynamic_cache_pkl), 0o777)
            logger.info(f"feature.pkl for {desc} saved at {dynamic_cache_pkl}")
            update_dynamic_cache(msa_index_dynamic, seq=seq, pkl_dir=pkl_dir)
          except Exception as e:
           if str(e.__class__) == "<class 'PermissionError'>":
            logger.info(f"Failed to save feat for {desc}: no Permission access for {dynamic_cache_pkl}")
           else:
            logger.info(f"Failed to save feat for {desc} due to {str(e)}")

    if 'template_all_atom_mask' in raw_features:                                                                                       
        raw_features['template_all_atom_masks'] = raw_features.pop('template_all_atom_mask')                                               
                                                                                                                                                
    return chain_id, raw_features, desc, seq


def _process_reference_structure(type_chain_id, chain_features, json_path, entity_id2type_chain_id, kalign_binary_path):
    """Process reference structure features for a given chain"""
    dtype, chain_id = type_chain_id.rsplit('_', 1)
    if dtype != 'protein':
        return

    fasta_seq = chain_features['msa_seqs']
    with open(json_path, 'r') as f:
        config_json = json.load(f)
    
    if 'ref_structures' not in config_json:
        return
        
    for index_ref, ref_structure in enumerate(config_json['ref_structures']):
        cif_path = ref_structure['ref_file']
        hit_pdb_code = cif_path.split('/')[-1].split('.')[0]
        
        for refer_target in ref_structure['refer_target_pairs']:
            refer_chain_id = refer_target['refer']
            target_entity_id = refer_target['target']
            target_type_chain_id = entity_id2type_chain_id[target_entity_id]
            
            if target_type_chain_id != type_chain_id:
                continue
                
            try:
                feature_structure = get_refer_structure_features(
                    cif_path,
                    refer_chain_id=refer_chain_id, 
                    target_sequence=fasta_seq,
                    hit_pdb_code=hit_pdb_code,
                    kalign_binary_path=kalign_binary_path
                )
                feature_structure['template_sum_probs'] = np.array([0.0])
                
                # Replace original template features
                for k in chain_features['msa_templ_feats'].keys():
                    if 'template' in k:
                        # Roll existing templates one position down and insert new template at the beginning
                        chain_features['msa_templ_feats'][k] = np.roll(chain_features['msa_templ_feats'][k], 1, axis=0)
                        chain_features['msa_templ_feats'][k][0] = feature_structure[k]
                        
                logger.info(f'Use ref_structures: {target_entity_id} {refer_chain_id}')
            except Exception as e:
                logger.warning(f'Get refer_structure_features failed: {target_entity_id} {refer_chain_id}')


def get_constraint_mask_for_S1(featurs, all_chain_features, s1_constraint_infos=List[Dict], 
                                      check_interface_type: str='AbAg',
                                      allow_check_interface: bool=True) -> np.ndarray:
    """Get sample constraint mask for S1 model. 
          If s1_constraint_infos is not empty, add sample constraint mask from json info, 
                otherwise, check AbAg interface. 
          elif AbAg interface is not empty, add sample constraint mask from AbAg interface.
          
          else, Allow all interface, constraint mask will be all 1.
      
      Args:
          featurs: dict, the features from base feature pipeline.
          all_chain_features: dict, the features for all chains with the following format:
              {
                  'type_chain_id': {'msa_templ_feats': {}, 'ccd_seqs': , 'msa_seqs': , 'extra_feats': , 'raw_info': },
                  ...
                  'protein_1-1': {....},
                  'protein_2-1': {....},
                  ...
              }
          s1_constraint_infos: List[Dict], 
                  the sample constraint for S1 model from json file.
          check_interface_type: Literal['AbAg', 'Protein_Ligand'], 
                  the interface type to check when s1_constraint_infos from json is empty.
          allow_check_interface: bool, 
                  whether to check interface when s1_constraint_infos from json is empty.

      Returns:
          constraint_mask: numpy.ndarray, (n_token, n_token)
              the interface sample constraint mask.
    """
    constraint_mask = get_interface_sample_constraint_mask(featurs['interface_mask'], 
                                                chain_ids=np.array(featurs['chain_ids']),
                                                s1_sample_constraint=s1_constraint_infos)
    
    if len(s1_constraint_infos) > 0:
        logger.info(f'Done. Add sample constraint mask for S1 model from json info')
    elif allow_check_interface:
        logger.warning(f'No user-defined sample constraint, check {check_interface_type} interface.')
        if check_interface_type == 'AbAg':
          antibody_chain_ids = []
          antigen_chain_ids = []
          for type_chain_id, chain_features in all_chain_features.items():
              chain_type, chain_id = type_chain_id.rsplit('_', 1)
              if chain_type != 'protein':
                  continue
              one_letter_seq = chain_features['msa_seqs']
              is_antigen_chain = (antibody_chain_check(one_letter_seq) == 'A')
              if is_antigen_chain:
                antigen_chain_ids.append(chain_id)
              else:
                antibody_chain_ids.append(chain_id)
          if len(antibody_chain_ids) > 0 and len(antigen_chain_ids) > 0:
             logger.warning(f'Found {len(antibody_chain_ids)} antibody chains and {len(antigen_chain_ids)} antigen chains. ' \
                                  f'Construct AbAg interface constraint mask implicitly ...')
             _abag_constraint_infos = []
             for ab_chain_id in antibody_chain_ids:
               for ag_chain_id in antigen_chain_ids:
                 _abag_constraint_infos.append({'left_entity': ab_chain_id, 'right_entity': ag_chain_id})
             constraint_mask = get_interface_sample_constraint_mask(featurs['interface_mask'], 
                                                chain_ids=np.array(featurs['chain_ids']),
                                                s1_sample_constraint=_abag_constraint_infos)
          else:
            logger.warning(f'Both antibody and antigen chains are not found, skip AbAg interface constraint mask.')
        else:
          raise ValueError(f'check_interface_type {check_interface_type} is not supported yet.')
          
    return constraint_mask


def featurize_entities(all_entities, json_path, ccd_preprocessed_path, 
                          msa_templ_data_pipeline_dict, msa_output_dir,
                          use_msa_templ_feats=True,
                          use_colabfold_search=False, 
                          kalign_binary_path=None):
    
    ## Get model type and sample constraint from json file.
    raw_info_from_json = read_json(json_path)
    model_type = raw_info_from_json.get('model_type', "HelixFold3")
    s1_constraint_infos = raw_info_from_json.get('s1_sample_constraint', [])
    logger.info(f'model_type: {model_type}, s1_constraint_infos: {s1_constraint_infos}')

    all_chain_features = {}
    entity_id2type_chain_id = {}
    for entity_items in all_entities:    
      chain_id = entity_items.raw_info['asym_chain_id']
      type_chain_id = entity_items.dtype + '_' + chain_id
      ccd_list = parsers.parse_ccd_fasta(entity_items.seqs)
      chain_features = {'msa_templ_feats': {},
                        'ccd_seqs': ccd_list, 
                        'msa_seqs': entity_items.msa_seqs,
                        'extra_feats': entity_items.extra_mol_infos ,
                        'raw_info': entity_items.raw_info}
      all_chain_features[type_chain_id] = chain_features
      entity_id2type_chain_id[chain_id] = type_chain_id

    if isinstance(msa_output_dir, str):
        msa_output_dir = pathlib.Path(msa_output_dir)

    if use_msa_templ_feats:
      ## 1. get all msa_seqs for protein/rna MSA/Template search. Only for protein/rna.
      tasks = []
      fasta_seq_to_type_chain_id = {}

      for type_chain_id, chain_features in all_chain_features.items():
        dtype, chain_id = type_chain_id.rsplit('_', 1) 
        if dtype not in ['protein', 'rna']:
          continue
        
        fasta_seq = chain_features['msa_seqs']
        if fasta_seq not in fasta_seq_to_type_chain_id:
          fasta_seq_to_type_chain_id[fasta_seq] = []
          fasta_seq_to_type_chain_id[fasta_seq].append(type_chain_id)
        else:
          ## NOTE: same fasta_seq, but different chain_id. will only search once.
          fasta_seq_to_type_chain_id[fasta_seq].append(type_chain_id)
          continue
        
        if dtype == 'protein':
          _data_pipeline = {'common': msa_templ_data_pipeline_dict['protein']}
          if use_colabfold_search:
            _data_pipeline.update({'colab': msa_templ_data_pipeline_dict['colabfold_prot']})
        elif dtype == 'rna':
          _data_pipeline = {'common': msa_templ_data_pipeline_dict['rna']}

        features_pkl_dir = msa_output_dir.joinpath(f'{type_chain_id}')
        features_pkl_dir.mkdir(parents=True, exist_ok=True)
        features_pkl = features_pkl_dir.joinpath('features.pkl')

        meta_task = MSATaskMeta(
                    data_pipeline=_data_pipeline, 
                    chain_id=chain_id, 
                    seq=fasta_seq, 
                    desc=type_chain_id, 
                    msa_output_dir=features_pkl_dir, 
                    features_pkl=features_pkl,
                    task_type=dtype)
        tasks.append(meta_task)

      print('Protein use colabfold_search:', use_colabfold_search)
      print('MSA fastas:', list(fasta_seq_to_type_chain_id.items()))

      ## 2. multiprocessing for protein/rna MSA/Template search.
      seqs_to_msa_features = {}
      logger.info('[Multiprocess] starting MSA/Template search...')
      t0 = time.time()
      partial_process = partial(process_chain_msa, use_colabfold_search)
      ## NOTE: use_colabfold_search: True, MAX_MSA_WORKERS=3; False, MAX_MSA_WORKERS=1
      if use_colabfold_search:
        MAX_MSA_WORKERS = 3
      else:
        MAX_MSA_WORKERS = 1
      with ProcessPoolExecutor(max_workers=MAX_MSA_WORKERS) as executor:
          futures = [executor.submit(partial_process, task) for task in tasks]

          for future in as_completed(futures):
              try:
                  _, raw_features, type_chain_id, seqs = future.result()
                  seqs_to_msa_features[seqs] = raw_features
              except Exception as exc:
                  import traceback; traceback.print_exc()
                  logger.error(f'Task generated an exception : {exc}')
      logger.info(f'[Multiprocess] All msa/template use: {time.time() - t0}')
      ## if feature > 2G, load seqs_to_msa_features from pkl.
      for type_chain_id in all_chain_features.keys():
        dtype, chain_id = type_chain_id.rsplit('_', 1)
        if dtype not in ['protein', 'rna']:
          continue
        chain_features = all_chain_features[type_chain_id]
        fasta_seq = chain_features['msa_seqs']
        if fasta_seq not in seqs_to_msa_features: 
          features_pkl = msa_output_dir.joinpath(f'{type_chain_id}').joinpath('features.pkl')
          if os.path.exists(features_pkl):
            with open(features_pkl, 'rb') as f:
              feature = pickle.load(f) 
            seqs_to_msa_features[fasta_seq] = copy.deepcopy(feature)

      ## 3. add msa_templ_feats to all_chain_features.
      for type_chain_id in all_chain_features.keys():
        dtype, chain_id = type_chain_id.rsplit('_', 1) 
        if dtype not in ['protein', 'rna']:
          continue
        chain_features = all_chain_features[type_chain_id]
        fasta_seq = chain_features['msa_seqs']
        if fasta_seq in seqs_to_msa_features:
          chain_features['msa_templ_feats'] = copy.deepcopy(seqs_to_msa_features[fasta_seq])

      ## if add reference feature to template features.
      for type_chain_id in all_chain_features.keys():
        _process_reference_structure(
            type_chain_id,
            all_chain_features[type_chain_id],
            json_path,
            entity_id2type_chain_id,
            kalign_binary_path
        )

    ## 4. add assembly features and check.
    assert len(all_entities) == len(all_chain_features.keys())
    ccd_preprocessed_dict = load_ccd_dict(ccd_preprocessed_path)
    all_feats = add_assembly_features(all_chain_features, ccd_preprocessed_dict, 
                                        use_msa_templ_feats=use_msa_templ_feats)
    np_example, label, chain_mapping = all_feats['feats'], all_feats['label'], all_feats['chain_mapping']
    assert len(all_entities) == len(np.unique(np_example['all_chain_ids']))

    if model_type == "HelixFold-S1":
      logger.info('[S1 model] add interface/constraint mask')
      np_example['interface_mask'] = get_interface_mask(feat=np_example,label=label,
                                              interface_type=GEN_INTERFACE_CONFIG['interface_type'])
      np_example['interface_source'] = GEN_INTERFACE_CONFIG['interface_source']
      np_example.update(get_interface_info(np_example, label, GEN_INTERFACE_CONFIG['interface_type'], 
                                              np_example['interface_mask']))

      np_example['interface_sample_constraint_mask'] = get_constraint_mask_for_S1(
                                              np_example, all_chain_features, s1_constraint_infos,
                                              check_interface_type='AbAg', allow_check_interface=True)


    sample = {
      "feat": np_example,
      "label": label,
      'label_cropped': {},
    }

    print('========Feats===========')
    _log_result_shape = '\n'.join([f'{k}: {type(v)} {np.shape(v)}' for k, v in np_example.items()])
    print(_log_result_shape)
    print('========Label===========')
    _log_label_shape = '\n'.join([f'{k}: {type(v)} {np.shape(v)}' for k, v in label.items()])
    print(_log_label_shape)

    for key in STRING_FEATURES:
      if key in sample['feat'].keys():
        sample['feat'][key] = ' '.join(sample['feat'][key])
      if key in sample['label'].keys():
        sample['label'][key] = ' '.join(sample['label'][key])


    return sample

