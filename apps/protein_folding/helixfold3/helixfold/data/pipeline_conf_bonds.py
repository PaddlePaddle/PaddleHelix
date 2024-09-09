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

"""Functions for building the input features (reference ccd features) for the HelixFold model."""

import collections
from dataclasses import dataclass
import gzip
import os
import pickle
import re
from typing import Any, List, Literal, Optional, Tuple

from absl import logging
from immutabledict import immutabledict
import numpy as np
from openbabel import openbabel

from helixfold.common import residue_constants
from helixfold.data.tools import utils



ALLOWED_LIGAND_BONDS_TYPE = {
    "SING": 1,
    "DOUB": 2,
    "TRIP": 3,
    "QUAD": 4, 
    "AROM": 12,
}

# Define the possible bond types as a Literal
# https://mmcif.wwpdb.org/dictionaries/mmcif_std.dic/Items/_struct_conn_type.id.html
BondType = Literal["covale", "covale_base", "covale_phosphate", "covale_sugar", "disulf", "hydrog",'metalc','mismat','modres','saltbr']


@dataclass
class AtomPartner:
    """
    Represents one partner atom in a covalent bond.

    Attributes:
        label_asym_id (str): The asymmetry identifier for the partner atom (i.e., chain ID).
        label_comp_id (str): The component identifier for the partner atom (i.e., residue name).
        seq_id (str): The sequence identifier for the partner atom (merged label_seq_id and auth_seq_id).
        label_atom_id (str): The atom identifier for the partner atom (i.e., atom name).
    """

    label_asym_id: str  # Chain ID
    label_comp_id: str  # Residue name
    seq_id: str         # Merged sequence ID
    label_atom_id: str  # Atom name


@dataclass
class CovalentBond:
    """
    Represents a covalent bond between two atoms in a molecular structure.

    Attributes:
        atom_1 (AtomPartner): The first partner atom in the bond.
        atom_2 (AtomPartner): The second partner atom in the bond.
        bond_type (BondType): The type of the bond.
        pdbx_dist_value (float): The distance value as defined in the PDBx/mmCIF format.
    """

    atom_1: AtomPartner
    atom_2: AtomPartner
    bond_type: BondType
    pdbx_dist_value: float

def parse_covalent_bond_input(input_string: str) -> List[CovalentBond]:
    """
    Parses a human-readable string into a list of CovalentBond objects.

    Args:
        input_string (str): A string representing covalent bonds, where each bond is
                            described by two atom partners separated by a comma,
                            and multiple bonds are separated by semicolons.
                            Example: "A,GLY,25,CA,A,GLY,25,N,covale,1.32; B,HIS,58,ND1,B,HIS,58,CE1,covale,1.39"

    Returns:
        List[CovalentBond]: A list of CovalentBond objects.
    """
    covalent_bonds = []

    # Split the input string by semicolons to separate individual covalent bonds
    bond_strings = input_string.split(';')

    for bond_str in bond_strings:
        # Split the individual bond string by commas to separate attributes
        bond_parts = bond_str.split(',')

        if len(bond_parts) != 10:
            raise ValueError(f"Invalid bond format: {bond_str}. Expected 10 fields per bond.")

        # Create AtomPartner instances for the two atoms in the bond
        atom_1 = AtomPartner(
            label_asym_id=bond_parts[0].strip(),
            label_comp_id=bond_parts[1].strip(),
            seq_id=bond_parts[2].strip(),
            label_atom_id=bond_parts[3].strip()
        )

        atom_2 = AtomPartner(
            label_asym_id=bond_parts[4].strip(),
            label_comp_id=bond_parts[5].strip(),
            seq_id=bond_parts[6].strip(),
            label_atom_id=bond_parts[7].strip()
        )

        # Create a CovalentBond instance
        covalent_bond = CovalentBond(
            atom_1=atom_1,
            atom_2=atom_2,
            bond_type=bond_parts[8].strip(),
            pdbx_dist_value=float(bond_parts[9].strip())
        )

        # Append the CovalentBond instance to the list
        covalent_bonds.append(covalent_bond)
    logging.info(f"Added {len(covalent_bonds)} bonds: {covalent_bonds}")

    return covalent_bonds

def load_ccd_dict(ccd_preprocessed_path: str) -> immutabledict[str, Any]:
    if not os.path.exists(ccd_preprocessed_path):
      raise FileNotFoundError(f'[CCD] ccd_preprocessed_path: {ccd_preprocessed_path} not exist.')
    
    if not ccd_preprocessed_path.endswith('.pkl.gz') and not ccd_preprocessed_path.endswith('.pkl'):
        raise ValueError(f'[CCD] ccd_preprocessed_path: {ccd_preprocessed_path} not endswith .pkl.gz and .pkl')
  
    with utils.timing(f'Loading CCD dataset from {ccd_preprocessed_path}'):
      if ccd_preprocessed_path.endswith('.pkl.gz'):
          with gzip.open(ccd_preprocessed_path, "rb") as fp:
              ccd_preprocessed_dict = immutabledict(pickle.load(fp))
      else:
          with open(ccd_preprocessed_path, "rb") as fp:
              ccd_preprocessed_dict = immutabledict(pickle.load(fp))
    
    logging.info(f'CCD dataset contains {len(ccd_preprocessed_dict)} entries.')
    
    return ccd_preprocessed_dict

def element_map_with_x(atom_symbol):
  # ## one-hot max shape == 128
  return residue_constants.ATOM_ELEMENT.get(atom_symbol, 127)

def convert_atom_id_name(atom_id: str) -> list[int]:
  """
    Converts unique atom_id names to integer of atom_name. need to be padded to length 4.
    Each character is encoded as ord(c) − 32
  """
  if (len_atom_id:=len(atom_id))>4:
     raise ValueError(f'atom_id: `{atom_id}` is too long, max length is 4.')
  atom_id_pad = atom_id.ljust(4, ' ')
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

      # atom checks
      for atom_id in _ccd_feats['atom_ids']:
        converted_atom_id_name=convert_atom_id_name(atom_id.upper())
        if max(converted_atom_id_name)>= 64:
          raise ValueError(f'>>> Problematic atom in ligand ({residue_id=}, {ccd_id=}, {chain_id=}) {atom_id=}, {converted_atom_id_name=}')
        # logging.debug(f'({residue_id=}, {ccd_id=}, {chain_id=}) {atom_id=}, {converted_atom_id_name=}')
      
      features['ref_atom_name_chars'].append(
                              np.array([convert_atom_id_name(atom_id.upper()) for atom_id in _ccd_feats['atom_ids']]
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

  if (len_ref_element:=np.max(features['ref_element'])) >= 128:
     raise ValueError(f'{len_ref_element=}, which is larger then 128.\n{features["ref_element"]}\n{"-"*79}')

  assert len(set([len(v) for k, v in features.items() if k != 'ref_atom_count'])) == 1 ## To check same Atom-level features. # WTF?
  return features


def make_bond_features(covalent_bond: List[CovalentBond], all_chain_info, ccd_preprocessed_dict, 
                                      extra_feats: Optional[dict]=None):
  """
      all_chain_info: dict, (chain_type_chain_id): ccd_seq (list of ccd), such as: protein_A: ['ALA', 'MET', 'GLY']
        - the covalent_bond is all ready cut off by distance=2.4
        - bond-features is only the covalent bond between two atoms. (ligand-intra/inter, polymer-ligand)
      
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
  chain_id_list = [type_chain_id.rsplit('_')[1] for type_chain_id, _ in all_chain_info.items()]
  _set_chain_id_list = set(chain_id_list)
  parsed_covalent_bond = []
  for _bond in covalent_bond:
    # Accessing the AtomPartner attributes for both atoms in the covalent bond
    left_bond_atomid, right_bond_atomid = _bond.atom_1.label_atom_id, _bond.atom_2.label_atom_id
    left_bond_name, right_bond_name = _bond.atom_1.label_comp_id, _bond.atom_2.label_comp_id
    left_bond, right_bond = _bond.atom_1.label_asym_id, _bond.atom_2.label_asym_id
    
    left_bond_idx, right_bond_idx = _bond.atom_1.seq_id, _bond.atom_2.seq_id
    auth_left_idx, auth_right_idx = _bond.atom_1.seq_id, _bond.atom_2.seq_id

    left_bond_idx = 1 if left_bond_idx == '.' else left_bond_idx
    right_bond_idx = 1 if right_bond_idx == '.' else right_bond_idx
    
    if _bond.bond_type != "covale":
      logging.warning(f'Detected non-covale bond type: {_bond.bond_type}')
      # continue
    
    if _bond.pdbx_dist_value > 2.4:
      # the covalent_bond is cut off by distance=2.4
      logging.warning(f'Ignore bonding with distance > 2.4: {_bond.pdbx_dist_value}')
      continue
    
    ## When some chainID is filtered, bond need to be filtered too.
    if (left_bond not in _set_chain_id_list) or (right_bond not in _set_chain_id_list):
      logging.warning(f'Ignore bonding with left and right out of chain list: ')
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

  ccd_id2atom_ids: dict[str, list]={}
  ccd_standard_set = residue_constants.STANDARD_LIST
  for chain_type_id, ccd_seq in all_chain_info.items():
      chain_type, chain_id = chain_type_id.rsplit('_', 1)
      assert chain_id not in chainId_to_ccd_list, 'Expect different mmcif chainID, but got same in [make_bond_features]'
      chainId_to_ccd_list[chain_id] = ccd_seq
      chainId_to_type[chain_id] = chain_type

      for residue_id, ccd_id in enumerate(ccd_seq):
        if ccd_id in ccd_standard_set:
            all_token_nums += 1
            all_token_nums_slot[(chain_id, residue_id)] = 1
        else:
            offset = all_token_nums
            if ccd_id not in ccd_preprocessed_dict:
                assert not extra_feats is None and ccd_id in extra_feats, \
                          f'<{ccd_id}> not in ccd_preprocessed_dict, But got extra_feats is None'
                _ccd_feats = extra_feats[ccd_id]
            else:
                _ccd_feats = ccd_preprocessed_dict[ccd_id]
            atom_ids = _ccd_feats['atom_ids']

            ccd_id2atom_ids[ccd_id] = atom_ids
            assert len(atom_ids) > 0, f'TODO filter - Got CCD <{ccd_id}>: 0 atom nums.'
            
            all_token_nums += len(atom_ids)
            all_token_nums_slot[(chain_id, residue_id)] = len(atom_ids)

            if 'non_polymer' in chain_type_id or 'ligand' in chain_type_id: # for non_polymer, (ligand, ion)
                intra_coval_bonds = _ccd_feats['coval_bonds'] # coval_bonds: List[List[str]] # (C   OXT SING)
                dict_atom_ids = {aid:idx for idx, aid in enumerate(atom_ids)}
                assert len(atom_ids) == len(dict_atom_ids)
                for _bd in intra_coval_bonds:
                  l_atom_id, r_atom_ids, _bond_type = _bd
                  l_token_id = dict_atom_ids[l_atom_id] + offset
                  r_token_id = dict_atom_ids[r_atom_ids] + offset
                  _bond_type_integer = ALLOWED_LIGAND_BONDS_TYPE.get(_bond_type, 1)
                  bond_index.append((l_token_id, r_token_id))
                  ## NOTE: now is only support covalent bond in ligand
                  ligand_bond_type.append((l_token_id, r_token_id, _bond_type_integer)) # "C,OXT,SING",

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
        if not (ptnr1_label_asym_id in chainId_to_ccd_list and ptnr2_label_asym_id in chainId_to_ccd_list):
           raise ValueError(f"Invalid chain id:\n{ptnr1_label_asym_id}/{ptnr2_label_asym_id}\n{chainId_to_ccd_list}")
        ptnr1_ccd_id = chainId_to_ccd_list[ptnr1_label_asym_id][int(ptnr1_label_seq_id) - 1]
        ptnr2_ccd_id = chainId_to_ccd_list[ptnr2_label_asym_id][int(ptnr2_label_seq_id) - 1]


        # renamed ligand residues


        if ptnr1_ccd_id != ptnr1_label_comp_id:
           logging.warning(f"Find ligand residue: {ptnr1_label_comp_id} -> {ptnr1_ccd_id}")
           #ptnr1_label_comp_id = ptnr1_ccd_id

        if ptnr2_ccd_id != ptnr2_label_comp_id:
           logging.warning(f"Find ligand residue: {ptnr2_label_comp_id} -> {ptnr2_ccd_id}")
           #ptnr2_label_comp_id = ptnr2_ccd_id

      except ValueError as e:
        ## some convalent-bond from mmcif is misslead, pass it.
        logging.warning(f'Error occurred during covalent bond processing: {e}')
        continue


      
      if ptnr1_ccd_id in ccd_preprocessed_dict:
        ptnr1_ccd_atoms_list = ccd_preprocessed_dict[ptnr1_ccd_id]['atom_ids']
      else:
        ptnr1_ccd_atoms_list = ccd_id2atom_ids[ptnr1_ccd_id]
        
      logging.debug(f'{ptnr1_ccd_id=}: {ptnr1_ccd_atoms_list=}')
      
      if ptnr2_ccd_id in ccd_preprocessed_dict:
        ptnr2_ccd_atoms_list = ccd_preprocessed_dict[ptnr2_ccd_id]['atom_ids']
      else:
        ptnr2_ccd_atoms_list  = ccd_id2atom_ids[ptnr2_ccd_id]

      logging.debug(f'{ptnr2_ccd_id=}: {ptnr2_ccd_atoms_list=}')

      if ptnr1_ccd_id in ccd_standard_set:  
          ## if ccd_id is in the standard residue in HF3 (table 13), we didn't have to map to atom-leval index
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
