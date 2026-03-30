"""Functions for building the input features (reference ccd features) for the HelixFold model."""

import collections
import os
import time
from typing import Any, KeysView, Mapping, MutableMapping, Optional, Sequence, Union, List
from absl import logging
from helixfold.common import residue_constants
from helixfold.data import msa_identifiers
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
ALLOWED_LIGAND_BONDS_TYPE = {
    "SING": 1,
    "DOUB": 2,
    "TRIP": 3,
    "QUAD": 4, 
    "AROM": 12,
}

def bond_convert_unit_to_assembly(unit_coval_bonds_info: List[dict],
                                  chain_mapping: dict) -> List[dict]:
  """
    TODO: Filter bond when not be appeared in different Unit.
    chain_mapping, mmcif_chainID to <type>_<mmcif>_<author>
        From *-assembly1.cif
        such as {"A": "protein_A_A",
                "B": "protein_B_B",
                "C": "ligand_C_C",
                "D": "ligand_D_D",
                "E": "ligand_E_A",
                "F": "ligand_F_B"}
    unit_coval_bonds_info, Unit bond infos from *.cif, not *-assembly1.cif
        list[dict], 
        such as :{
            "ptnr1_label_asym_id": "B",
            "ptnr1_label_comp_id": "ASN",
            "ptnr1_label_seq_id": "56",
            "ptnr1_label_atom_id": "ND2",
            "ptnr2_label_asym_id": "F",
            "ptnr2_label_comp_id": "NAG",
            "ptnr2_label_seq_id": ".",
            "ptnr2_label_atom_id": "C1",
            "pdbx_dist_value": 1.43,
            "pdbx_leaving_flag": "one",
            "ptnr1_auth_asym_id": "B",
            "ptnr1_auth_comp_id": "ASN",
            "ptnr1_auth_seq_id": "56",
            "ptnr2_auth_asym_id": "B",
            "ptnr2_auth_comp_id": "NAG",
            "ptnr2_auth_seq_id": "401",
            "bond_type": "covale"
          }
  """
  if not unit_coval_bonds_info:
    return []
  
  # Mapping, mmcif_to_author_chain. it use for assembly which has Glygan.
  mmcif_to_author_chain = {}
  # Mapping, such as: AA -> [AA-2, AA-3], B -> [], C -> [C-2, C-3]
  mmcif_chain_assembly = collections.defaultdict(set)

  for mmcif_chain_id, tyep_mmcif_author in chain_mapping.items():
    _, _, _author_chaID = tyep_mmcif_author.split('_')
    mmcif_to_author_chain[mmcif_chain_id] = _author_chaID
    if '-' in mmcif_chain_id: 
      ## NOTE: Assembly chainID will be expanded from Unit ChainID by inserting flag '-'.
      prefix = mmcif_chain_id.split('-')[0]
      mmcif_chain_assembly[prefix].add(mmcif_chain_id) 

  covert_covalent_bonds = []
  for _bond in unit_coval_bonds_info:
    unit_left_mmcif_asym, unit_right_mmcif_asym = \
                  _bond['ptnr1_label_asym_id'], _bond['ptnr2_label_asym_id']

    ## NOTE: Filter if one of the two ends of the bond infos is not in the chainID.
    if unit_left_mmcif_asym not in mmcif_to_author_chain:
      continue
    if unit_right_mmcif_asym not in mmcif_to_author_chain:
      continue
    covert_covalent_bonds.append(_bond)

    # find the Unit to assembly Mapping.
    left_mapping = mmcif_chain_assembly[unit_left_mmcif_asym]
    right_mapping = mmcif_chain_assembly[unit_right_mmcif_asym]
    for left_asym in left_mapping:
      surfix_id = left_asym.split('-')[-1]
      right_asym = unit_right_mmcif_asym + '-' + surfix_id
      if right_asym in right_mapping:
        _expand_bond_infos = _bond.copy()
        _expand_bond_infos['ptnr1_label_asym_id'] = left_asym
        _expand_bond_infos['ptnr2_label_asym_id'] = right_asym
        _expand_bond_infos['ptnr1_auth_asym_id'] = mmcif_to_author_chain[left_asym]
        _expand_bond_infos['ptnr2_auth_asym_id'] = mmcif_to_author_chain[right_asym]
        covert_covalent_bonds.append(_expand_bond_infos)
  
  return covert_covalent_bonds


def element_map_with_x(atom_symbol):
  # ## one-hot max shape == 128
  return residue_constants.ATOM_ELEMENT.get(atom_symbol, 127)

def convert_atom_id_name(atom_id: str) -> int:
  """
    Converts unique atom_id names to integer of atom_name. need to be padded to length 4.
    Each character is encoded as ord(c) − 32
  """
  atom_id_pad = atom_id.ljust(4, ' ')
  assert len(atom_id_pad) == 4
  return [ord(c) - 32 for c in atom_id_pad]


def make_ccd_conf_features(all_chain_info, ccd_preprocessed_dict,
                            extra_feats: Optional[dict]=None):
  """
      all_chain_info: dict, (chain_type_chain_id): ccd_seq (list of ccd), 
        such as: protein_A: ['ALA', 'MET', 'GLY']
      Constructs a feature dict of ccd ref_conf features.

      ccd_preprocessed_dict:
      keys for <ccd_id>.pkl.gz : 
        id: str
        smiles: List[str]
        atom_ids: List[str] # [OXT]
        atom_symbol: List[str] # [O]
        charge: List[int]
        leave_atom_flag: List[str]
        position: List[List[float]] # [[-0.456 0.028  -0.001], [-0.376 1.240  0.001]]
        coval_bonds: List[List[str]] # (C   OXT SING)
        raw_string: Any

      extra_feats: Optional, For user-defined ligand input; such as smiles, which ccd_seqs_name is 'UNK-*'
      NOTE: It is only support for Online inference.
        UNK-1, UNK2...: {
            "atom_symbol": elements, List [C, C, N, O]
            "charge": charge, List: [0, -1, 1]
            "atom_ids": atom_names, List: [C1, C2, N1, O1]
            "coval_bonds": bonds, List: ('C9', 'N1', 'SING'), ('N1', 'C10', 'SING')
            "position": pos, np.ndarray, shape: (N_atom, 3)
        }

  """
  encoding_records = {}
  uid = 0
  token_residue_id = 0
  
  features = collections.defaultdict(list)
  for type_chain_id, all_ccd_ids in all_chain_info.items():
    chain_id = type_chain_id.rsplit('_')[1]
    for residue_id, ccd_id in enumerate(all_ccd_ids):
      if ccd_id not in ccd_preprocessed_dict:
        assert not extra_feats is None and ccd_id in extra_feats, \
                  f'<{ccd_id}> not in ccd_preprocessed_dict, But got extra_feats is None'
        _ccd_feats = extra_feats[ccd_id]
      else:
        _ccd_feats = ccd_preprocessed_dict[ccd_id]
      num_atoms = len(_ccd_feats['position'])
      assert num_atoms > 0, f'TODO filter - Got CCD <{ccd_id}>: 0 atom nums.'

      features['ref_pos'].append(np.array(_ccd_feats['position'], dtype=np.float32))
      features['ref_mask'].append(np.array([1] * num_atoms, dtype=np.int32))
      features['ref_element'].append(np.array([element_map_with_x(t[0].upper() + t[1:].lower())
                                              for t in _ccd_feats['atom_symbol']], dtype=np.int32))
      features['ref_charge'].append(np.array(_ccd_feats['charge'], dtype=np.int32))
      features['ref_atom_name_chars'].append(
                              np.array([convert_atom_id_name(atom_id) for atom_id in _ccd_feats['atom_ids']]
                                                                , dtype=np.int32))
      
      # here we get ref_space_uid [ Each (chain id, residue index) tuple is assigned an integer on first appearance.]
      if (chain_id, residue_id) not in encoding_records:
          encoding_records[(chain_id, residue_id)] = uid
          uid += 1
      features['ref_space_uid'].append(np.array(
                                      [encoding_records[(chain_id, residue_id)]] * num_atoms, dtype=np.int32))
      
      # we get Each (chain id, token_residue_id) tuple is assigned an integer on first appearance
      if ccd_id in residue_constants.STANDARD_LIST:
        offset = 1
        features['ref_token2atom_idx'].append(np.array(
                                        [token_residue_id] * num_atoms, dtype=np.int32))
      else:
        offset = num_atoms
        for _i in range(offset):
          features['ref_token2atom_idx'].append(np.array(
                                          [token_residue_id + _i], dtype=np.int32))
      
      token_residue_id += offset

  for k, v in features.items():
    features[k] = np.concatenate(v, axis=0)
  features['ref_atom_count'] = np.bincount(features['ref_token2atom_idx'])

  assert np.max(features['ref_element']) < 128
  assert np.max(features['ref_atom_name_chars']) < 64
  assert len(set([len(v) for k, v in features.items() if k != 'ref_atom_count'])) == 1 ## To check same Atom-level features.
  return features


def make_bond_features(covalent_bond, all_chain_info, ccd_preprocessed_dict, 
                                      extra_feats: Optional[dict]=None):
  """
      all_chain_info: dict, (chain_type_chain_id): ccd_seq (list of ccd), such as: protein_A: ['ALA', 'MET', 'GLY']
        - the covalent_bond is all ready cut off by distance=2.4
        - bond-features is only the covalent bond between two atoms. (ligand-intra/inter, polymer-ligand)
      
      extra_feats: Optional, For user-defined ligand input; such as smiles, which ccd_seqs_name is 'aaa', 'aab', 'aac'.
      NOTE: It is only support for Online inference.
        aaa, aab, aac ...: {
            "atom_symbol": elements, List [C, C, N, O]
            "charge": charge, List: [0, -1, 1]
            "atom_ids": atom_names, List: [C1, C2, N1, O1]
            "coval_bonds": bonds, List: ('C9', 'N1', 'SING'), ('N1', 'C10', 'SING')
            "position": pos, np.ndarray, shape: (N_atom, 3)
        }
  """
  chain_id_list = [type_chain_id.rsplit('_')[1] for type_chain_id, _ in all_chain_info.items()]
  _set_chain_id_list = set(chain_id_list)
  parsed_covalent_bond = []
  for _bond in covalent_bond:
    left_bond_atomid, right_bond_atomid = _bond['ptnr1_label_atom_id'], _bond['ptnr2_label_atom_id']
    left_bond_name, right_bond_name = _bond['ptnr1_label_comp_id'], _bond['ptnr2_label_comp_id']
    left_bond, right_bond = _bond['ptnr1_label_asym_id'], _bond['ptnr2_label_asym_id']
    
    left_bond_idx, right_bond_idx = _bond['ptnr1_label_seq_id'], _bond['ptnr2_label_seq_id']
    auth_left_idx, auth_right_idx = _bond['ptnr1_auth_seq_id'], _bond['ptnr2_auth_seq_id']
    left_bond_idx = 1 if left_bond_idx == '.' else left_bond_idx
    right_bond_idx = 1 if right_bond_idx == '.' else right_bond_idx
    
    ## TODO: support others bond type; Now is only support covalent bond
    if _bond['bond_type'] != "covale":
      continue
    
    if _bond['pdbx_dist_value'] > 2.4:
      # the covalent_bond is cut off by distance=2.4
      continue
    
    ## When some chainID is filtered, bond need to be filtered too.
    if (left_bond not in _set_chain_id_list) or (right_bond not in _set_chain_id_list):
      continue

    parsed_covalent_bond.append([left_bond, left_bond_name, left_bond_idx, left_bond_atomid, auth_left_idx,
                          right_bond, right_bond_name, right_bond_idx, right_bond_atomid, auth_right_idx])
    # [A,CYS,105,SG, C,0WN, 1, C30]
    # ptnr1_label_asym_id, ptnr1_label_comp_id, ptnr1_label_seq_id, ptnr1_label_atom_id
  
  ## NOTE: be careful, this code is only used for all_different chain_id in one mmcif!!
  ## first only process the non-polymer ligand_intra bond;
  all_token_nums = 0
  all_token_nums_slot = {} ## (chain_id, residue_id): token_nums,  such as (A,1): 5
  chainId_to_ccd_list = {}
  chainId_to_type = {}
  ligand_bond_type = [] # (i, j, bond_type), represent the bond between token i and token j
  bond_index = [] # (i,j) represent the bond between token i and token j
  ccd_standard_set = residue_constants.STANDARD_LIST
  for chain_type_id, ccd_seq in all_chain_info.items():
      chain_type, chain_id = chain_type_id.rsplit('_', 1)
      assert chain_id not in chainId_to_ccd_list, 'Expect different mmcif chainID, but got same in [make_bond_features]'
      chainId_to_ccd_list[chain_id] = ccd_seq
      chainId_to_type[chain_id] = chain_type

      if chain_type in ['dna', 'rna']:
        first_name, last_name = 'P', "O3'"
      elif chain_type == 'protein':
        first_name, last_name = 'N', "OXT"

      pre_is_poly_modi = False
      for residue_id, ccd_id in enumerate(ccd_seq):
        if ccd_id in ccd_standard_set:
            offset = all_token_nums
            all_token_nums += 1
            all_token_nums_slot[(chain_id, residue_id)] = 1

            if chain_type == 'ligand':
              pre_is_poly_modi = False
              continue
            
            try:
              if pre_is_poly_modi and residue_id > 0:
                  pre_ccd_id = ccd_seq[residue_id - 1]
                  pre_ccd = ccd_preprocessed_dict[pre_ccd_id] if pre_ccd_id in ccd_preprocessed_dict else extra_feats[pre_ccd_id]
                  pred_atom_ids = pre_ccd['atom_ids']
                  pred_dict_atom_ids = {aid:idx for idx, aid in enumerate(pred_atom_ids)}
                  pre_offset = offset - len(pred_atom_ids)
                  pre_last_idx = pre_offset + pred_dict_atom_ids[last_name]
                  bond_index.append((pre_last_idx, offset))
            except:
                  print('>>> not standard common modified ccd: ', ccd_seq[residue_id - 1])
            pre_is_poly_modi = False
        else:
            offset = all_token_nums
            if ccd_id not in ccd_preprocessed_dict:
                assert not extra_feats is None and ccd_id in extra_feats, \
                          f'<{ccd_id}> not in ccd_preprocessed_dict, But got extra_feats is None'
                _ccd_feats = extra_feats[ccd_id]
            else:
                _ccd_feats = ccd_preprocessed_dict[ccd_id]
            atom_ids = _ccd_feats['atom_ids']
            assert len(atom_ids) > 0, f'TODO filter - Got CCD <{ccd_id}>: 0 atom nums.'
            
            all_token_nums += len(atom_ids)
            all_token_nums_slot[(chain_id, residue_id)] = len(atom_ids)

            intra_coval_bonds = _ccd_feats['coval_bonds'] # coval_bonds: List[List[str]] # (C   OXT SING)
            dict_atom_ids = {aid:idx for idx, aid in enumerate(atom_ids)}
            assert len(atom_ids) == len(dict_atom_ids)
            for _bd in intra_coval_bonds:
              l_atom_id, r_atom_id, _bond_type = _bd
              l_token_id = dict_atom_ids[l_atom_id] + offset
              r_token_id = dict_atom_ids[r_atom_id] + offset
              _bond_type_integer = ALLOWED_LIGAND_BONDS_TYPE.get(_bond_type, 1)
              bond_index.append((l_token_id, r_token_id))
              ## NOTE: now is only support covalent bond in ligand
              ligand_bond_type.append((l_token_id, r_token_id, _bond_type_integer)) # "C,OXT,SING",

            ### NOTE: IMPORTANT: bond_index for modified ccd/smiles in polymer chain.
            if chain_type == 'ligand':
              pre_is_poly_modi = False
              continue
            
            try:
              if pre_is_poly_modi and residue_id > 0:
                  pre_ccd_id = ccd_seq[residue_id - 1]
                  pre_ccd = ccd_preprocessed_dict[pre_ccd_id] if pre_ccd_id in ccd_preprocessed_dict else extra_feats[pre_ccd_id]
                  pred_atom_ids = pre_ccd['atom_ids']
                  pred_dict_atom_ids = {aid:idx for idx, aid in enumerate(pred_atom_ids)}
                  pre_offset = offset - len(pred_atom_ids)
                  pre_last_idx = pre_offset + pred_dict_atom_ids[last_name]
                  bond_index.append((pre_last_idx, offset + dict_atom_ids[first_name]))
              elif residue_id > 0:
                  bond_index.append((offset - 1, offset + dict_atom_ids[first_name]))
            except:
              print('>>> not standard common modified ccd: ', ccd_seq[residue_id - 1])

            pre_is_poly_modi = True
            ### NOTE: IMPORTANT: bond_index for modified ccd/smiles in polymer chain.


  accum_token_list = np.cumsum(list(all_token_nums_slot.values()))
  chainId_resID_to_slot_id = {k:idx for idx, k in enumerate(all_token_nums_slot.keys())}  # chain_id_residue_id: idx
  assert all_token_nums == accum_token_list[-1]

  ## Next, we add covalent_bond of ligand-ligand inter, liagnd-polymer inter
  for parsd_bond in parsed_covalent_bond:
      ## NOTE: ptnr1_label_seq_id is start from 1, be cafeful with the 0-indexing in chainId_to_ccd_list. 
      ptnr1_label_asym_id, ptnr1_label_comp_id, ptnr1_label_seq_id, ptnr1_label_atom_id, ptnr1_auth_seq_id = parsd_bond[:5]
      ptnr2_label_asym_id, ptnr2_label_comp_id, ptnr2_label_seq_id, ptnr2_label_atom_id, ptnr2_auth_seq_id = parsd_bond[5:]
      
      ## NOTE: It is the glycan, if bonds share the same mmcif_id and chain_type is the ligand/non_polymer
      if all([ptnr1_label_asym_id == ptnr2_label_asym_id,
              chainId_to_type[ptnr1_label_asym_id] == chainId_to_type[ptnr2_label_asym_id], 
              chainId_to_type[ptnr1_label_asym_id] in ['ligand', 'non_polymer']]):
        ptnr1_label_seq_id = ptnr1_auth_seq_id
        ptnr2_label_seq_id = ptnr2_auth_seq_id
        if ptnr1_auth_seq_id == ptnr2_auth_seq_id:
          ## NOTE: some ligand convalent-bond from mmcif is misslead, such as `103l`. It is intra-bond;
          continue
      elif all([chainId_to_type[ptnr1_label_asym_id] == chainId_to_type[ptnr2_label_asym_id], 
              chainId_to_type[ptnr1_label_asym_id] in ['ligand', 'non_polymer']]):
        ## NOTE: some glycan will be form with different mmcif chainID， such as `8cdo`
        if len(chainId_to_ccd_list[ptnr1_label_asym_id]) > 1:
          ptnr1_label_seq_id = ptnr1_auth_seq_id
        if len(chainId_to_ccd_list[ptnr2_label_asym_id]) > 1:
          ptnr2_label_seq_id = ptnr2_auth_seq_id

      try:
        assert ptnr1_label_asym_id in chainId_to_ccd_list and ptnr2_label_asym_id in chainId_to_ccd_list
        ptnr1_ccd_id = chainId_to_ccd_list[ptnr1_label_asym_id][int(ptnr1_label_seq_id) - 1]
        ptnr2_ccd_id = chainId_to_ccd_list[ptnr2_label_asym_id][int(ptnr2_label_seq_id) - 1]
        assert ptnr1_ccd_id == ptnr1_label_comp_id and ptnr2_ccd_id == ptnr2_label_comp_id
      except:
        ## some convalent-bond from mmcif is misslead, pass it.
        continue
      
      ptnr1_ccd_atoms_list = ccd_preprocessed_dict[ptnr1_ccd_id]['atom_ids']
      ptnr2_ccd_atoms_list = ccd_preprocessed_dict[ptnr2_ccd_id]['atom_ids']

      if ptnr1_ccd_id in ccd_standard_set:  
          ## if ccd_id is in the standard residue in AF3 (table 13), we didn't have to map to atom-leval index
          bond_latom_idx = 0
      else:
          try:
            bond_latom_idx = ptnr1_ccd_atoms_list.index(ptnr1_label_atom_id)
          except:
            print(f'WARNING - [Bond] Got {ptnr1_label_atom_id} not in ccd {ptnr1_ccd_id}')
            continue
      if ptnr2_ccd_id in ccd_standard_set:
          bond_ratom_idx = 0
      else:
          try:
            bond_ratom_idx = ptnr2_ccd_atoms_list.index(ptnr2_label_atom_id)
          except:
            print(f'WARNING - [Bond] Got {ptnr1_label_atom_id} not in ccd {ptnr2_ccd_id}')
            continue
      
      l_bond_token_slot_id = chainId_resID_to_slot_id[(ptnr1_label_asym_id, int(ptnr1_label_seq_id) - 1)]
      r_bond_token_slot_id = chainId_resID_to_slot_id[(ptnr2_label_asym_id, int(ptnr2_label_seq_id) - 1)]

      if l_bond_token_slot_id == 0:
          offset_l_bond = 0 + bond_latom_idx
      else:
          offset_l_bond = accum_token_list[l_bond_token_slot_id - 1] + bond_latom_idx
      
      if r_bond_token_slot_id == 0:
          offset_r_bond = 0 + bond_ratom_idx
      else:
          offset_r_bond = accum_token_list[r_bond_token_slot_id - 1] + bond_ratom_idx

      bond_index.append((offset_l_bond, offset_r_bond))
    
  ### Final, we make Ntoken * Ntoken martix
  feature = np.zeros((all_token_nums, all_token_nums), dtype=np.float32)
  bond_type = np.zeros((all_token_nums, all_token_nums), dtype=np.int32)
  for l, r in bond_index:
      feature[l, r] = 1
      feature[r, l] = 1
  for l, r, bond_type_id in ligand_bond_type:
      bond_type[l, r] = bond_type_id
      bond_type[r, l] = bond_type_id

  return {'token_bonds' : feature, 'token_bonds_type': bond_type}



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

    chain_mapping = assembly_dict['chain_mapping']
    all_chain_info = collections.OrderedDict()
    for dtype, raw_values in basic_token_dict.items():
        chain_ids = raw_values['chain_ids']
        ccd_seqs = raw_values['seqs']
        for ccd_seq, chain_id in zip(ccd_seqs, chain_ids):
            if select_mmcif_chainID is not None and chain_id not in select_mmcif_chainID:
              chain_mapping.pop(chain_id)
              continue
            parsed_ccd = parsers.parse_ccd_fasta(ccd_seq)
            all_chain_info[f'{dtype}_{chain_id}'] = parsed_ccd

    assert len(all_chain_info) > 0, f"Invalid parsed in json [{assembly_json_path}]; select {select_mmcif_chainID}"

    ## Make reference pos features.
    ref_features = make_ccd_conf_features(all_chain_info=all_chain_info,
                                          ccd_preprocessed_dict=ccd_preprocessed_dict)
    
    ## Make bond features
    coval_bonds_info = assembly_dict.get('covalent_bonds', [])
    if not assembly_json_path is None:
      coval_bonds_info = bond_convert_unit_to_assembly(coval_bonds_info, chain_mapping)
    bond_features = make_bond_features(covalent_bond=coval_bonds_info, 
                                          all_chain_info=all_chain_info, 
                                          ccd_preprocessed_dict=ccd_preprocessed_dict)

    return {**ref_features, **bond_features}



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
    with Pool(20) as p:
      re = list(p.imap_unordered(_process, tuple_input))

    # ## single test
    # name = '8cdo'
    # tuple_input = list(zip([name], [f'/root/paddlejob/workspace/output/yexianbin/preprocess_hf3_data/mmcif_basic_0620/{name}.cif.json'], 
    #                               [f'/root/paddlejob/workspace/output/yexianbin/preprocess_hf3_data/mmcif_assembly_0620_selected/{name}-assembly1.cif.json']))
    # _process(tuple_input[0])

