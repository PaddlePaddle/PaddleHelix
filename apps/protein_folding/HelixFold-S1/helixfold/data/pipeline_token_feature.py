"""Functions for building the input features (reference ccd features) for the HelixFold model."""
 
import collections
import os
import time
from typing import Any, KeysView, Mapping, MutableMapping, Optional, Sequence, Union, List
from absl import logging
from helixfold.common import residue_constants
from helixfold.data import parsers
import numpy as np
import re
import json
import gzip
import pickle
from rdkit import Chem
 
# Internal import (7716).
FeatureDict = MutableMapping[str, np.ndarray]
ELEMENT_MAPPING = Chem.GetPeriodicTable()
 
def _chainID_order_check(features_chain_info, all_chain_info):
  _check_chain_order = collections.OrderedDict()
  for _i in features_chain_info:
    _check_chain_order[_i] = 1
  _check_chain_order = list(_check_chain_order.keys())
  _raw_chain_order = [k.rsplit('_', 1)[1] for k, _ in all_chain_info.items()]
 
  assert _raw_chain_order == _check_chain_order
 
def make_perm_atom_index(raw_feats: FeatureDict) -> FeatureDict:
  perm_asym_id = raw_feats['perm_asym_id']
  atom_bincount = np.bincount(perm_asym_id)
 
  raw_feats['perm_atom_index'] = np.empty((0,), dtype=np.int32)
  for count in atom_bincount:
    raw_feats['perm_atom_index'] = np.concatenate([raw_feats['perm_atom_index'],
                                            np.arange(count)])
  return raw_feats
 
def flatten_is_protein_features(is_protein_feats: np.ndarray) -> FeatureDict:
  '''
    flatten `is_protein_feats` to one-hot vector
      `is_protein`
      `is_dna`
      `is_rna`
      `is_ligand`
  '''
  res = {
    "is_protein": np.zeros(len(is_protein_feats), dtype=int),
    "is_dna": np.zeros(len(is_protein_feats), dtype=int),
    "is_rna": np.zeros(len(is_protein_feats), dtype=int),
    "is_ligand": np.zeros(len(is_protein_feats), dtype=int),
  }
  unique_values = np.unique(is_protein_feats)  
  for i, value in enumerate(unique_values):  
      type_keys = 'is_' + residue_constants.order_to_CHAIN_type[value]
      res[type_keys][is_protein_feats == value] = 1
 
  return res
 
def make_sequence_features(
    all_chain_info, ccd_preprocessed_dict, 
    extra_feats: Optional[dict]=None) -> FeatureDict:
  """Constructs a feature dict of sequence features.
    Args:
      all_chain_info: A dict of chain infos. {dtype}_{chain_id}: ccd_seq.
                      such as protein_A, rna_A, dna_B, non_polymer_c： ['SEP', 'ALA']
      ccd_preprocessed_dict: A dict of ccd_id: Token features.
      extra_feats: Optional, For user-defined ligand input; such as smiles, 
                    which ccd_seqs_name is lower-case + digit, such as 'aaa', 'aab', 'aa1', 'aa2'
      NOTE: It is only support for online inference.
        aaa, aab...: {
            "atom_symbol": elements, List [C, C, N, O]
            "charge": charge, List: [0, -1, 1]
            "atom_ids": atom_names, List: [C1, C2, N1, O1]
            "coval_bonds": bonds, List: ('C9', 'N1', 'SING'), ('N1', 'C10', 'SING')
            "position": pos, np.ndarray, shape: (N_atom, 3)
        }
  """
  
  ### NOTE: which is the best choice to identify seq, (msa sequence or ccd sequence？) 
  ### <grouped_chains> : All chain will be grouped by same <ccd> sequence.
  ### then the length of <grouped_chains> is equal to the number of entity.
  seq_to_entity_id = {} 
  grouped_chains = collections.defaultdict(list)
  for chain_type_id, ccd_seq in all_chain_info.items():
    seq = ''.join(ccd_seq)
    if seq not in seq_to_entity_id:
      seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
    grouped_chains[seq_to_entity_id[seq]].append((chain_type_id, ccd_seq))
    # entity_id : [(chain_type_id, ccd_seq)]
 
  ## NOTE: to keep the same order with all_chain_info.keys()
  chainid_to_entity_id = {}
  chainid_to_sym_id = {}
  for entity_id, group_chain_infos in grouped_chains.items():
    for sym_id, (chain_type_id, ccd_seq) in enumerate(group_chain_infos, start=1):
      chain_type, _alphabet_chain_id = chain_type_id.rsplit('_', 1) 
      chainid_to_entity_id[_alphabet_chain_id] = entity_id
      chainid_to_sym_id[_alphabet_chain_id] = sym_id
 
 
  def _append_features(feature_dict, key, value, num_atoms, dtype=np.int32):
      feature_dict[key].append(np.array([value] * num_atoms, dtype=dtype))

  # NOTE: chain_type is must be one of ['protein', 'dna', 'rna', 'ligand']
  chatype_id_to_asym_id = {}
  features = collections.defaultdict(list)
  chain_num_id = 1
  for chain_type_id, ccd_seq in all_chain_info.items():
    chatype_id_to_asym_id[chain_type_id] = chain_num_id
    chain_type, _alphabet_chain_id = chain_type_id.rsplit('_', 1) 
    entity_id = chainid_to_entity_id[_alphabet_chain_id]
    sym_id = chainid_to_sym_id[_alphabet_chain_id]
    for residue_id, ccd_id in enumerate(ccd_seq):
      if ccd_id not in ccd_preprocessed_dict:
        assert not extra_feats is None and ccd_id in extra_feats,\
                  f'<{ccd_id}> not in ccd_preprocessed_dict, But got extra_feats is None'
        _ccd_feats = extra_feats[ccd_id]
      else:
        _ccd_feats = ccd_preprocessed_dict[ccd_id]
      num_atoms = len(_ccd_feats['position'])
      assert num_atoms > 0, f'TODO filter - Got CCD <{ccd_id}>: 0 atom nums.'
      
      if ccd_id not in residue_constants.STANDARD_LIST: 
          _append_features(features, 'asym_id', chain_num_id, num_atoms)
          _append_features(features, 'sym_id', sym_id, num_atoms)
          _append_features(features, 'entity_id', entity_id, num_atoms)
          _append_features(features, 'residue_index', residue_id, num_atoms)
          _append_features(features, 'is_protein', residue_constants.CHAIN_type_order[chain_type], num_atoms)

          restype_mapping = {
              'ligand': residue_constants.AF3_restype_order['UNK'],
              'protein': residue_constants.AF3_restype_order['UNK'],
              'dna': residue_constants.AF3_restype_order['DN'],
              'rna': residue_constants.AF3_restype_order['N']
          }
          _append_features(features, 'restype', restype_mapping[chain_type], num_atoms)
          _append_features(features, 'chain_ids', _alphabet_chain_id, num_atoms, dtype=object)
      else:
          _append_features(features, 'residue_index', residue_id, 1)
          _append_features(features, 'is_protein', residue_constants.CHAIN_type_order[chain_type], 1)
          _append_features(features, 'restype', residue_constants.AF3_restype_order[ccd_id], 1)
          _append_features(features, 'asym_id', chain_num_id, 1)
          _append_features(features, 'sym_id', sym_id, 1)
          _append_features(features, 'entity_id', entity_id, 1)
          _append_features(features, 'chain_ids', _alphabet_chain_id, 1, dtype=object)

      features['perm_entity_id'].append(np.array([entity_id] * num_atoms, dtype=np.int32)) # [N_atom, ]
      features['perm_asym_id'].append(np.array([chain_num_id] * num_atoms, dtype=np.int32)) # [N_atom, ]
      features['all_chain_ids'].append(np.array([_alphabet_chain_id] * num_atoms, dtype=object))  # [N_atom]
      features['all_ccd_ids'].append(np.array([ccd_id] * num_atoms, dtype=object))  # [N_atom]
      features['all_atom_ids'].append(np.array(_ccd_feats['atom_ids'], dtype=object)) # [N_atom]
 
    chain_num_id += 1
 
  for k, v in features.items():
      features[k] = np.concatenate(v, axis=0)
  features['token_index'] = np.arange(len(features['residue_index']), dtype=np.int32)
  features = make_perm_atom_index(features)
  features.update(
    flatten_is_protein_features(features['is_protein']))
 
  ## To check same token-level features.
  assert len(set([len(v) for k, v in features.items() if not k.startswith('perm_') and not k.startswith('all_')])) == 1 
  ## To check same atom-level features.
  assert len(set([len(v) for k, v in features.items() if k.startswith('perm_') or k.startswith('all_')])) == 1  
  ## To check same order with all_chain_info.keys()
  _chainID_order_check(features['all_chain_ids'], all_chain_info)
 
  return features, chatype_id_to_asym_id
 
 
class DataPipeline:
  """Get all the ccd input features."""
 
  def __init__(self, ccd_preprocessed_path: str):
    """Initializes the data pipeline."""
    self.ccd_preprocessed_path = ccd_preprocessed_path
 
  def process(self,
              unit_json_path: str,
              assembly_json_path: Optional[str] = None,
              select_mmcif_chainID: Optional[List[str]] = None,
              ccd_preprocessed_dict: Optional[str] = None,
              ccd_output_dir: Optional[str] = None) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""

    meta_info_keys = [
      'resolution',
      'release_date',
      'method',
      'covalent_bonds',
    ]
    with open(unit_json_path, 'r') as f:
      unit_dict = json.load(f)
    
    if not assembly_json_path is None:
      with open(assembly_json_path, 'r') as f:
        assembly_dict = json.load(f)
      for meta_k in meta_info_keys:
        assembly_dict[meta_k] = unit_dict[meta_k]
    else:
      assembly_dict = unit_dict
    
    # ## For Unit TEST
    # if 'basic' not in assembly_dict:
    #   raise ValueError('Parsing Error')
    # if 'rna' in assembly_dict['basic'] or 'dna' in assembly_dict['basic']:
    #   raise ValueError(f"dna/rna not include in this test.")
    # ## For Unit TEST

    if ccd_preprocessed_dict is None:
      ccd_preprocessed_dict = {}
      st_1 = time.time()
      if 'pkl.gz' in self.ccd_preprocessed_path:
          with gzip.open(self.ccd_preprocessed_path, "rb") as fp:
              ccd_preprocessed_dict = pickle.load(fp)
      logging.info(f'load ccd dataset done. use {time.time()-st_1}s')
 
    if select_mmcif_chainID is not None:
      select_mmcif_chainID = set(select_mmcif_chainID)
 
    basic_token_dict = assembly_dict['basic']
    # NOTE: dtype must be protein, dna, rna, non_polymer
    basic_token_dict = {seq_id:basic_token_dict[seq_id] 
                for seq_id in ['protein', 'dna', 'rna', 'non_polymer', 'ligand'] if seq_id in basic_token_dict}
   
    all_chain_info = collections.OrderedDict()
    for dtype, raw_values in basic_token_dict.items():
        chain_ids = raw_values['chain_ids']
        ccd_seqs = raw_values['seqs']
        for ccd_seq, chain_id in zip(ccd_seqs, chain_ids):
            if select_mmcif_chainID is not None and chain_id not in select_mmcif_chainID:
              continue
            parsed_ccd = parsers.parse_ccd_fasta(ccd_seq)
            all_chain_info[f'{dtype}_{chain_id}'] = parsed_ccd
 
    assert len(all_chain_info) > 0, f"Invalid parsed in json [{assembly_json_path}]; select {select_mmcif_chainID}"
 
    Token_features = make_sequence_features(all_chain_info=all_chain_info,
                                            ccd_preprocessed_dict=ccd_preprocessed_dict)
 
    return {**Token_features}
 
 
 
if __name__ == '__main__':
    import json 

    ccd_preprocessed_path = '/root/paddlejob/workspace/output/yexianbin/preprocess_hf3_data/ccd_preprocessed_etkdg.pkl.gz'
    data = DataPipeline(ccd_preprocessed_path=ccd_preprocessed_path)
    # t = data.process(unit_json_path=unit_demo_json_path, assembly_json_path=assembly_demo_json_path, 
    #                     select_mmcif_chainID=['B', 'C'])
    # print(t)
    ccd_preprocessed_dict = {}
    st_1 = time.time()
    if 'pkl.gz' in ccd_preprocessed_path:
        with gzip.open(ccd_preprocessed_path, "rb") as fp:
            ccd_preprocessed_dict = pickle.load(fp)
    print(f'load ccd dataset done. use {time.time()-st_1}s')

    # unit_demo_json_path = '/root/paddlejob/workspace/output/yexianbin/file_download/1hho.cif.json'
    # assembly_demo_json_path = '/root/paddlejob/workspace/output/yexianbin/file_download/1hho-assembly1.cif.json'

    def _process(tuple_path):
      pdbid, unit_demo_json_path, assembly_demo_json_path = tuple_path
      t1 = time.time()
      try:
        t = data.process(unit_json_path=unit_demo_json_path, assembly_json_path=assembly_demo_json_path,
                            select_mmcif_chainID=None, ccd_preprocessed_dict=ccd_preprocessed_dict)
        print(f'[SUCCESS] | {pdbid} | {time.time() - t1}')
      except Exception as e:
        import traceback
        # traceback.print_exc()
        print(f'[ERROR] | {e} | [{assembly_demo_json_path}]')
    
    ## UNIT-TEST
    import os
    from glob import glob
    from tqdm import tqdm
    from multiprocessing import Pool
    all_assembly_path = sorted(glob('/root/paddlejob/workspace/output/yexianbin/preprocess_hf3_data/mmcif_assembly_0620_selected/*.cif.json'))
    all_pdbid = [os.path.basename(t).split('.')[0].split('-')[0] for t in all_assembly_path]
    all_unit_path = [f'/root/paddlejob/workspace/output/yexianbin/preprocess_hf3_data/mmcif_basic_0620/{t}.cif.json' for t in all_pdbid]

    tuple_input = list(zip(all_pdbid, all_unit_path, all_assembly_path))
    # print(tuple_input[0])
    with Pool(20) as p:
      re = list(p.imap_unordered(_process, tuple_input))
