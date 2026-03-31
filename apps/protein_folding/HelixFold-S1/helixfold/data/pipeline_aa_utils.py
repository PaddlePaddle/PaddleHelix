"""Functions for building the input features (reference ccd features) for the HelixFold model."""
 
import collections
import os
import time
from functools import partial
from typing import Any, KeysView, Mapping, MutableMapping, Optional, Tuple, Sequence, Union, List
from absl import logging
import numpy as np
from rdkit import Chem

from helixfold.common import residue_constants
from helixfold.data import templates_quat_affine as quat_affine
from helixfold.data import parsers
from helixfold.data import msa_pairing
from helixfold.data import pipeline_multimer, pipeline_rna, pipeline_rna_multimer

 
#############################
# make sequence features
#############################

# Internal import (7716).
FeatureDict = MutableMapping[str, np.ndarray]
ELEMENT_MAPPING = Chem.GetPeriodicTable()
KEYS_TO_CONTINUOUS_INCREASE_BETWEEN_CHAINS = [
    'token_index',
    'ref_space_uid',
    'ref_token2atom_idx',
]


def ccd_list_to_token(ccd_seq, ccd_preprocessed_dict):
    """ map list of ccd ids to token. """
    tokens = []
    for residue_id, ccd_id in enumerate(ccd_seq):
        if ccd_id in residue_constants.STANDARD_LIST:
            # one standard residue per token
            tokens.append(ccd_id)  
        else:
            # atom as token
            _ccd_feats = ccd_preprocessed_dict[ccd_id]
            atom_ids = _ccd_feats['atom_ids']
            assert len(atom_ids) > 0, f'TODO filter - Got CCD <{ccd_id}>: 0 atom nums.'
            for atom_id in atom_ids:
                tokens.append(atom_id)
    return tokens


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


def merge_token_bonds_list(token_bonds_list):
    """
    token_bonds_list: [[N1, N1], [N2, N2], ...]
    returns: [N1 + N2 + ..., N1 + N2 + ...]
    """
    for mat in token_bonds_list:
        assert mat.shape[0] == mat.shape[1]
    total_N = np.sum([mat.shape[0] for mat in token_bonds_list])
    merged_bonds = np.zeros([total_N, total_N], dtype=token_bonds_list[0].dtype)
    offset = 0
    for mat in token_bonds_list:
        end = offset + mat.shape[0]
        merged_bonds[offset: end, offset: end] = mat
        offset += mat.shape[0]
    return merged_bonds


def merge_value_list_to_continous_increase(value_list):
    """merge_value_list_to_continous_increase"""
    res = []
    offset = 0
    for vec in value_list:
        res.append(vec + offset)
        offset += np.max(vec) + 1
    res = np.concatenate(res, 0)
    return res


def ccd_to_restype(ccd_id, chain_type):
    if ccd_id not in residue_constants.STANDARD_LIST:
        if chain_type in ['ligand', 'non_polymer', 'protein']:
            # Ligands represented as “unknown amino acid”.
            restype = residue_constants.AF3_restype_order['UNK']
        elif chain_type == 'dna':
            restype = residue_constants.AF3_restype_order['DN']
        elif chain_type == 'rna':
            restype = residue_constants.AF3_restype_order['N']
        else:
            raise ValueError(chain_type)
    else:
        restype = residue_constants.AF3_restype_order[ccd_id]
    return restype


def make_sequence_features(chain_id,
        chain_type, ccd_seq, ccd_preprocessed_dict) -> FeatureDict:
    """
    Make intra chain features for a given chain.
    Args:
        chain_type: 'protein', 'dna', 'rna', 'ligand' or 'non_polymer',
        ccd_seq: list of ccd_code.
        ccd_preprocessed_dict: A dict of ccd_id: Token features.
    """

    features = collections.defaultdict(list)
    for residue_id, ccd_id in enumerate(ccd_seq):
        _ccd_feats = ccd_preprocessed_dict[ccd_id]
        num_atoms = len(_ccd_feats['position'])
        assert num_atoms > 0, f'TODO filter - Got CCD <{ccd_id}>: 0 atom nums.'
        is_standard = ccd_id in residue_constants.STANDARD_LIST
        num_tokens = 1 if is_standard else num_atoms
        
        ## token-level features
        features['residue_index'].append(np.array(
                [residue_id] * num_tokens, dtype=np.int64))
        for ctype in ['protein', 'dna', 'rna', 'ligand']:
            features[f'is_{ctype}'].append(np.array(
                    [chain_type == ctype] * num_tokens, dtype=np.int64))
        restype = ccd_to_restype(ccd_id, chain_type)
        features['seq_mask'].append(np.ones(num_tokens, dtype=np.float32))
        features['restype'].append(np.array(
                [restype] * num_tokens, dtype=np.int64))    
        features['token_index'].append(np.arange(num_tokens, dtype=np.int64))

        ## atom-level features
        features['ref_pos'].append(np.array(_ccd_feats['position'], dtype=np.float32))
        features['ref_mask'].append(np.array([1] * num_atoms, dtype=np.float32))
        features['ref_element'].append(np.array(
                [element_map_with_x(t[0].upper() + t[1:].lower())
                for t in _ccd_feats['atom_symbol']], dtype=np.int64))
        features['ref_charge'].append(np.array(_ccd_feats['charge'], dtype=np.int64))
        features['ref_atom_name_chars'].append(np.array([
                convert_atom_id_name(atom_id) 
                for atom_id in _ccd_feats['atom_ids']], dtype=np.int64))
        features['ref_space_uid'].append(np.zeros(num_atoms, dtype=np.int64))
        if is_standard:
            features['ref_token2atom_idx'].append(np.zeros([num_atoms], dtype=np.int64))
        else:
            features['ref_token2atom_idx'].append(np.arange(num_atoms, dtype=np.int64))
        
        features['all_chain_ids'].append(np.array(
                [chain_id] * num_atoms, dtype=object))  # [N_atom]
        features['all_ccd_ids'].append(np.array(
                [ccd_id] * num_atoms, dtype=object))  # [N_atom]
        features['all_atom_ids'].append(np.array(
                _ccd_feats['atom_ids'], dtype=object)) # [N_atom]
        features['perm_atom_index'].append(np.arange(num_atoms, dtype=np.int64))

        ## token_bond features
        ## TODO: add bond features for non-standard ccd
        token_bonds = np.zeros([num_tokens, num_tokens], dtype=np.float32)
        if not is_standard:
            intra_coval_bonds = _ccd_feats['coval_bonds'] # coval_bonds: List[List[str]] # (C   OXT SING)
            dict_atom_ids = {aid:idx for idx, aid in enumerate(_ccd_feats['atom_ids'])}
            assert len(dict_atom_ids) == num_tokens
            for _bd in intra_coval_bonds:
                l_atom_id, r_atom_ids, _bond_type = _bd
                l_token_id = dict_atom_ids[l_atom_id]
                r_token_id = dict_atom_ids[r_atom_ids]
                token_bonds[l_token_id, r_token_id] = 1
                token_bonds[r_token_id, l_token_id] = 1
        features['token_bonds'].append(token_bonds)
    
    for k, v_list in features.items():
        if k in KEYS_TO_CONTINUOUS_INCREASE_BETWEEN_CHAINS:
            v = merge_value_list_to_continous_increase(v_list)
        elif k == 'perm_atom_index':
            v = merge_value_list_to_continous_increase(v_list)
        elif k == 'token_bonds':
            v = merge_token_bonds_list(v_list)
        else:
            v = np.concatenate(v_list, axis=0)
        features[k] = v
    return features


def add_assembly_features(all_chain_features, all_chain_info_dict):
    """Add features to distinguish between chains.

    Args:
        all_chain_features: A dictionary which maps chain_id to a dictionary of
        features for each chain.

    Returns:
        all_chain_features: A dictionary which maps strings of the form
        `<seq_id>_<sym_id>` to the corresponding chain features. E.g. two
        chains from a homodimer would have keys A_1 and A_2. Two chains from a
        heterodimer would have keys A_1 and B_1.
    """
    # Group the chains by sequence
    seq_to_entity_id = {}
    grouped_chains = collections.defaultdict(list)
    for chain_id in all_chain_features:
        seq = ''.join(all_chain_info_dict[chain_id]['ccd_seq'])
        if seq not in seq_to_entity_id:
            seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
        grouped_chains[seq_to_entity_id[seq]].append(chain_id)
    
    ## NOTE: to keep the same order with all_chain_info.keys()
    #  in order to align with old code
    chainid_to_entity_id = {}
    chainid_to_sym_id = {}
    for entity_id, group_chain_infos in grouped_chains.items():
        for sym_id, chain_id in enumerate(group_chain_infos, start=1):
            chainid_to_entity_id[chain_id] = entity_id
            chainid_to_sym_id[chain_id] = sym_id

    new_all_chain_features = {}
    asym_id = 1
    for chain_id in all_chain_features:
        entity_id = chainid_to_entity_id[chain_id]
        sym_id = chainid_to_sym_id[chain_id]
        chain_features = all_chain_features[chain_id]

        num_tokens = len(chain_features['residue_index'])
        chain_features['asym_id'] = asym_id * np.ones(num_tokens, dtype=np.int64)
        chain_features['sym_id'] = sym_id * np.ones(num_tokens, dtype=np.int64)
        chain_features['entity_id'] = entity_id * np.ones(num_tokens, dtype=np.int64)
        num_atoms = len(chain_features['ref_pos'])
        chain_features['perm_asym_id'] = asym_id * np.ones(num_atoms, dtype=np.int64)
        chain_features['perm_entity_id'] = entity_id * np.ones(num_atoms, dtype=np.int64)
        
        new_all_chain_features[chain_id] = chain_features
        asym_id += 1
    return new_all_chain_features


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


def make_covalent_bonds(all_chain_info_dict, 
        ordered_chain_ids, coval_bonds_info, ccd_preprocessed_dict):
    """
    make covalent_bond between chains
    """
    parsed_covalent_bond = []
    for _bond in coval_bonds_info:
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
        if (left_bond not in ordered_chain_ids) or (right_bond not in ordered_chain_ids):
            continue

        parsed_covalent_bond.append([left_bond, left_bond_name, left_bond_idx, left_bond_atomid, auth_left_idx,
                            right_bond, right_bond_name, right_bond_idx, right_bond_atomid, auth_right_idx])
        # [A,CYS,105,SG, C,0WN, 1, C30]
        # ptnr1_label_asym_id, ptnr1_label_comp_id, ptnr1_label_seq_id, ptnr1_label_atom_id
    
    ## NOTE: be careful, this code is only used for all_different chain_id in one mmcif!!
    ## first only process the non-polymer ligand_intra bond;
    all_token_nums = 0
    all_token_nums_slot = {} ## (chain_id, residue_id): token_nums,  such as (A,1): 5
    ccd_standard_set = residue_constants.STANDARD_LIST
    for chain_id in ordered_chain_ids:
        ccd_seq = all_chain_info_dict[chain_id]['ccd_seq']
        for residue_id, ccd_id in enumerate(ccd_seq):
            if ccd_id in ccd_standard_set:
                all_token_nums += 1
                all_token_nums_slot[(chain_id, residue_id)] = 1
            else:
                _ccd_feats = ccd_preprocessed_dict[ccd_id]
                atom_ids = _ccd_feats['atom_ids']
                assert len(atom_ids) > 0, f'TODO filter - Got CCD <{ccd_id}>: 0 atom nums.'
                all_token_nums += len(atom_ids)
                all_token_nums_slot[(chain_id, residue_id)] = len(atom_ids)

    accum_token_list = np.cumsum(list(all_token_nums_slot.values()))
    chainId_resID_to_slot_id = {k:idx for idx, k in enumerate(all_token_nums_slot.keys())}  # chain_id_residue_id: idx
    assert all_token_nums == accum_token_list[-1]

    ## Next, we add covalent_bond of ligand-ligand inter, liagnd-polymer inter
    bond_index = [] # (i,j) represent the bond between token i and token j
    for parsd_bond in parsed_covalent_bond:
        ## NOTE: ptnr1_label_seq_id is start from 1, be cafeful with the 0-indexing in chainId_to_ccd_list. 
        ptnr1_label_asym_id, ptnr1_label_comp_id, ptnr1_label_seq_id, ptnr1_label_atom_id, ptnr1_auth_seq_id = parsd_bond[:5]
        ptnr2_label_asym_id, ptnr2_label_comp_id, ptnr2_label_seq_id, ptnr2_label_atom_id, ptnr2_auth_seq_id = parsd_bond[5:]
        
        ## NOTE: It is the glycan, if bonds share the same mmcif_id and chain_type is the ligand/non_polymer
        chain_type1 = all_chain_info_dict[ptnr1_label_asym_id]['chain_type']
        chain_type2 = all_chain_info_dict[ptnr2_label_asym_id]['chain_type']
        ccd_seq1 = all_chain_info_dict[ptnr1_label_asym_id]['ccd_seq']
        ccd_seq2 = all_chain_info_dict[ptnr2_label_asym_id]['ccd_seq']
        if all([ptnr1_label_asym_id == ptnr2_label_asym_id,
                chain_type1 == chain_type2, 
                chain_type1 in ['ligand', 'non_polymer']]):
            ptnr1_label_seq_id = ptnr1_auth_seq_id
            ptnr2_label_seq_id = ptnr2_auth_seq_id
            if ptnr1_auth_seq_id == ptnr2_auth_seq_id:
                ## NOTE: some ligand convalent-bond from mmcif is misslead, such as `103l`. It is intra-bond;
                continue
        elif all([chain_type1 == chain_type2, 
                chain_type1 in ['ligand', 'non_polymer']]):
            ## NOTE: some glycan will be form with different mmcif chainID， such as `8cdo`
            if len(ccd_seq1) > 1:
                ptnr1_label_seq_id = ptnr1_auth_seq_id
            if len(ccd_seq2) > 1:
                ptnr2_label_seq_id = ptnr2_auth_seq_id

        try:
            assert ptnr1_label_asym_id in all_chain_info_dict and ptnr2_label_asym_id in all_chain_info_dict
            ptnr1_ccd_id = ccd_seq1[int(ptnr1_label_seq_id) - 1]
            ptnr2_ccd_id = ccd_seq2[int(ptnr2_label_seq_id) - 1]
            assert ptnr1_ccd_id == ptnr1_label_comp_id and ptnr2_ccd_id == ptnr2_label_comp_id
        except Exception as e:
            ## some convalent-bond from mmcif is misslead, pass it.
            print(f'WARNING - [Bond]: {e}')
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
    covalent_bonds = np.zeros((all_token_nums, all_token_nums), dtype=np.float32)
    for l, r in bond_index:
        covalent_bonds[l, r] = 1
        covalent_bonds[r, l] = 1
    return covalent_bonds


def merge_and_adjust_interchain_features(all_chain_features, all_chain_info_dict,
        ordered_chain_ids, coval_bonds_info, ccd_preprocessed_dict):
    """
    merge chain features, adjust interchain features (ref_space_uid, 
        ref_token2atom_idx) and add covalent_bond between chains
    """
    merge_features = {}
    ## concat existing features
    for k in all_chain_features[ordered_chain_ids[0]].keys():
        v_list = [all_chain_features[c][k] for c in ordered_chain_ids]
        if k in KEYS_TO_CONTINUOUS_INCREASE_BETWEEN_CHAINS:
            v = merge_value_list_to_continous_increase(v_list)
        elif k == 'token_bonds':
            v = merge_token_bonds_list(v_list)
        else:
            v = np.concatenate(v_list, axis=0)
        merge_features[k] = v

    ## covalent bonds between chains
    covalent_bonds = make_covalent_bonds(all_chain_info_dict, 
            ordered_chain_ids, coval_bonds_info, ccd_preprocessed_dict)
    merge_features['token_bonds'] += covalent_bonds
    merge_features['covalent_bonds'] = covalent_bonds
    return merge_features


#############################
# make msa and template features
#############################

MAX_TEMPLATE_NUM = 4

AF2_NEED_TO_PADDING_KEYS = set([ 
  'template_aatype', 'template_all_atom_masks', 'template_all_atom_positions', 
  'msa', 'deletion_matrix', 'deletion_mean', 'profile', 'has_deletion', 'deletion_value',
  # 'template_pseudo_beta_mask',
  # 'template_backbone_frame_mask', 'template_distogram', 'template_unit_vector',
])

AF2_PADDING_FEATS = {
  'msa': residue_constants.AF3_restype_order['-'],
  'template_aatype': residue_constants.AF3_restype_order['-'],
}

AF2_PADDING_DIM = {
  # msa
  'msa': [1],
  'deletion_matrix': [1],
  'deletion_mean': [0],
  'profile': [0],
  'has_deletion': [1],
  'deletion_value': [1],

  # template
  'template_aatype': [1],
  'template_all_atom_masks': [1],
  'template_all_atom_positions': [1],
  'template_pseudo_beta_mask': [1],
  'template_backbone_frame_mask': [1],
  'template_distogram': [1, 2],
  'template_unit_vector': [1, 2],
}


def create_rna_feature_without_real_msa(seq):
    """ create feature dict for single chain rna. """
    seq = seq.replace('X', 'N')
    # NOTE: Temporary strategy to use the sequence itself as the MSA
    temp_msa = parsers.parse_stockholm_RNA('', query=seq)
    msa_features = pipeline_rna.make_msa_features((temp_msa,), max_align_depth=10)

    num_res = len(seq)
    sequence_features = pipeline_rna.make_sequence_features(
        sequence=seq,
        description='query',
        num_res=num_res)

    raw_features = {**sequence_features, **msa_features}
    return raw_features


def create_rna_msa_all_seq_without_real_msa(seq):
    """create msa_all_seq features for single chain rna. """
    # NOTE: Temporary strategy to add _all_seq features to rna chains
    seq = seq.replace('X', 'N')
    temp_msa = parsers.parse_stockholm_RNA('', query=seq)
    msa_features = pipeline_rna.make_msa_features((temp_msa,), max_align_depth=10)

    valid_feats = msa_pairing.MSA_FEATURES + (
        'msa_species_identifiers',
    )
    msa_features = {f'{k}_all_seq': v for k, v in msa_features.items()
            if k in valid_feats}
    return msa_features


def repeat_elements_along_axis(array: np.ndarray, axis: int, repeat_indices: list, repeat_counts: list) -> np.ndarray:
    """
    Repeat specific elements along the specified axis of the array a specified number of times.

    Args:
        array (np.ndarray): The numpy array to repeat elements for.
        axis (int): The axis along which to repeat elements.
        repeat_indices (list): The list of indices of the elements to repeat along the specified axis.
        repeat_counts (list): The list of repetition counts corresponding to each index in repeat_indices.

    Returns:
        np.ndarray: The resulting numpy array with repeated elements along the specified axis.
    """
    # Validate inputs
    if len(repeat_indices) != len(repeat_counts):
        raise ValueError("repeat_indices and repeat_counts must have the same length")
    
    # Calculate the new shape after repeating elements
    new_shape = list(array.shape)
    for idx, count in zip(repeat_indices, repeat_counts):
        new_shape[axis] += (count - 1)
    
    # Initialize the result array
    result = np.empty(new_shape, dtype=array.dtype)
    
    # Prepare slicing templates
    index_expansion = (slice(None),) * axis
    result_index = [slice(None)] * array.ndim
    current_pos = 0

    for i in range(array.shape[axis]):
        if i in repeat_indices:
            repetitions = repeat_counts[repeat_indices.index(i)]
        else:
            repetitions = 1
        result_index[axis] = slice(current_pos, current_pos + repetitions)
        
        # Slice the array correctly along the axis
        slice_tuple = index_expansion + (i,)
        repeated_slice = np.repeat(np.expand_dims(array[slice_tuple], axis), repetitions, axis=axis)
        
        result[tuple(result_index)] = repeated_slice
        current_pos += repetitions

    return result


def get_ccd_insert_index_and_nums(ccd_preprocessed_dict, ccd_list: np.ndarray, extra_mol_feats=None) -> Tuple[int, int]:
    """Get the insert index and value for a residue.
    Args:
      ccd_list: list of ccd sequence, ['GLY', 'ALA', 'THR']
      extra_mol_feats: list of USER-defined ccd mapping; such as:
      {
        "UNK-1": {position:}
        "UNK-2": {position:}
      }
    Returns:
      insert index and value.
    """
    index_nums = []
    for idx, ccd_id in enumerate(ccd_list):
        if ccd_id not in residue_constants.STANDARD_LIST:
            if ccd_id not in ccd_preprocessed_dict:
              assert not extra_mol_feats is None and ccd_id in extra_mol_feats
              _ccd_feats = extra_mol_feats[ccd_id]
            else:
              _ccd_feats = ccd_preprocessed_dict[ccd_id]
            num_atoms = len(_ccd_feats['position'])
            assert num_atoms > 0, f'TODO filter - Got CCD <{ccd_id}>: 0 atom nums.'
            index_nums.append((idx, num_atoms))
    return index_nums


def make_hhblits_profile(msa: np.ndarray):
    """Compute the HHblits MSA profile.
       msa shape: (N_msa, N_token)
       return: 
          hhblits_profile, shape (N_token, 22)
    """
    def _one_hot(depth, indices):
        """tbd."""
        res = np.eye(depth)[indices.reshape(-1)]
        return res.reshape(list(indices.shape) + [depth])

    # Compute the profile for every residue (over all MSA sequences).
    hhblits_profile = np.mean(_one_hot(len(residue_constants.AF3_restype_order), 
                                            msa), axis=0).astype('float32')
    return hhblits_profile


def feats_pad_and_concatenate(array_list: List[np.ndarray], padding_dim: Union[tuple, list], value=0):
    """
    Optimally pads the array to the target_shape by copying its values to a preallocated array of the target shape.
    
    Args:
        array_list list[np.ndarray]: The input array to be padded.
        padding_dim: list[int]: the dimensional 
        value (scalar): The value to use for padding.

    Returns:
        np.ndarray: The padded array with the specified target shape.
    """
    def _check_and_get_padding_shape(array_list):
        padding_shape = []
        num_dims_set = set()
        for array in array_list:
            num_dims_set.add(len(array.shape))
            if not padding_shape:
                padding_shape = list(array.shape)
            else:
                for i in range(len(array.shape)):
                  if i not in padding_dim:
                    padding_shape[i] = max(padding_shape[i], array.shape[i])
                  else:
                    padding_shape[i] = padding_shape[i] + array.shape[i]
        assert len(num_dims_set) == 1, f'Padding needs arrays of the same dimensional length, but got {num_dims_set}'
        assert all([pdim < len(padding_shape) for pdim in padding_dim]), 'Padding dim should no larger than dimensional length'
        return padding_shape

    target_shape = _check_and_get_padding_shape(array_list)
    padded_array = np.full(target_shape, value, dtype=array_list[0].dtype)

    padding_offset = [0] * len(target_shape)
    for array in array_list:
      _slices = []
      for idx, dim in enumerate(array.shape):
        _slices.append(slice(padding_offset[idx], dim + padding_offset[idx]))
        if idx in padding_dim:
          padding_offset[idx] += dim

      slices = tuple(_slices)
      padded_array[slices] = array

    return padded_array


def mock_msa_template_if_nonexist(
        raw_feats: dict, 
        num_residues: int,
        before_msa_pairing: bool = False
        ) -> dict:
    """
    mock raw_msa and template features if they don't exist.
    """
    assert num_residues > 0
    if 'msa' not in raw_feats:
        raw_feats['msa'] = np.ones(
                [1, num_residues], dtype=np.int64
                ) * residue_constants.AF3_restype_order['-']
        raw_feats['deletion_matrix'] = np.zeros([1, num_residues], dtype=np.float32)
        raw_feats['deletion_mean'] = np.zeros([num_residues, ], dtype=np.float32)
    if 'template_aatype' not in raw_feats:
        if before_msa_pairing:
            raw_feats["template_aatype"] = np.zeros(
                    (MAX_TEMPLATE_NUM, num_residues, 22), dtype=np.int32)
            raw_feats["template_aatype"][:, :, residue_constants.AF3_restype_order['-']] = 1
        else:
            raw_feats["template_aatype"] = np.ones(
                    (MAX_TEMPLATE_NUM, num_residues), dtype=np.int32
                    ) * residue_constants.AF3_restype_order['-'] 
        raw_feats["template_all_atom_masks"] = np.zeros(
                (MAX_TEMPLATE_NUM, num_residues, 37), dtype=np.float32)  
        raw_feats["template_all_atom_positions"] = np.zeros(
                (MAX_TEMPLATE_NUM, num_residues, 37, 3), dtype=np.float32)
    return raw_feats


def add_msa_extra_features(raw_feats: dict) -> dict:
    ## get basic features for raw msa from AF2-multimer output.
    raw_feats['profile'] = make_hhblits_profile(raw_feats['msa'].astype(int))
    raw_feats['has_deletion'] = np.clip(raw_feats['deletion_matrix'], np.array(0), np.array(1))
    raw_feats['deletion_value'] = np.arctan(raw_feats['deletion_matrix'] / 3.) * (2. / np.pi)
    return raw_feats


def expand_msa_template_for_non_standard_residue(
        raw_feats: dict, 
        ccd_seq: list,
        ccd_preprocessed_dict: dict,
        extra_feats: Optional[list]=None) -> dict:
    """
    expand non standard tokens
    """
    assert len(ccd_seq) == raw_feats['msa'].shape[-1] 
    assert len(ccd_seq) == raw_feats['template_aatype'].shape[-1] 
    index_nums = get_ccd_insert_index_and_nums(ccd_preprocessed_dict, ccd_seq, extra_feats)
    if len(index_nums) > 0:
        repeat_indices, repeat_counts = map(list, zip(*index_nums))
        partial_repeat = partial(repeat_elements_along_axis, repeat_indices=repeat_indices, repeat_counts=repeat_counts)

        raw_feats['msa'] = partial_repeat(raw_feats['msa'], axis=1)
        raw_feats['deletion_matrix'] = partial_repeat(raw_feats['deletion_matrix'], axis=1)
        raw_feats['deletion_mean'] = partial_repeat(raw_feats['deletion_mean'], axis=0)
        raw_feats['has_deletion'] = partial_repeat(raw_feats['has_deletion'], axis=1)
        raw_feats['deletion_value'] = partial_repeat(raw_feats['deletion_value'], axis=1)
        raw_feats['profile'] = partial_repeat(raw_feats['profile'], axis=0)

        raw_feats['template_aatype'] = partial_repeat(raw_feats['template_aatype'], axis=1)
        raw_feats['template_all_atom_masks'] = partial_repeat(raw_feats['template_all_atom_masks'], axis=1)
        raw_feats['template_all_atom_positions'] = partial_repeat(raw_feats['template_all_atom_positions'], axis=1)
    return raw_feats


def make_pseudo_beta(protein, prefix=''):
  """Create pseudo-beta (alpha for glycine) position and mask."""
  def _pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
      """Create pseudo beta features."""
      is_gly = np.equal(aatype, residue_constants.restype_order['G'])
      ca_idx = residue_constants.atom_order['CA']
      cb_idx = residue_constants.atom_order['CB']
      pseudo_beta = np.where(
          np.tile(is_gly[..., None].astype("int32"),
                  [1,] * len(is_gly.shape) + [3,]).astype("bool"),
          all_atom_positions[..., ca_idx, :],
          all_atom_positions[..., cb_idx, :])

      if all_atom_masks is not None:
          pseudo_beta_mask = np.where(is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
          pseudo_beta_mask = pseudo_beta_mask.astype(np.float32)
          return pseudo_beta, pseudo_beta_mask

      return pseudo_beta

  assert prefix in ['', 'template_']
  pseudo_beta, pseudo_beta_mask = _pseudo_beta_fn(
      protein['template_aatype' if prefix else 'all_atom_aatype'],
      protein[prefix + 'all_atom_positions'],
      protein['template_all_atom_masks' if prefix else 'all_atom_mask'])

  protein[prefix + 'pseudo_beta'] = pseudo_beta
  protein[prefix + 'pseudo_beta_mask'] = pseudo_beta_mask
  return protein


def make_template_further_feature(protein):

  def _np_unstack(a, axis=0):
      return np.moveaxis(a, axis, 0)

  dtype = np.float32

  n, ca, c = [residue_constants.atom_order[a]
              for a in ('N', 'CA', 'C')]
  rot, trans = quat_affine.make_transform_from_reference(
      n_xyz=protein['template_all_atom_positions'][..., n, :],
      ca_xyz=protein['template_all_atom_positions'][..., ca, :],
      c_xyz=protein['template_all_atom_positions'][..., c, :])
  affines = quat_affine.QuatAffine(
      quaternion=quat_affine.rot_to_quat(rot),
      translation=trans,
      rotation=rot)

  points = [np.expand_dims(x, axis=-2) for x in
            _np_unstack(affines.translation, axis=-1)]
  affine_vec = affines.invert_point(points, extra_dims=1)
  inv_distance_scalar = 1.0 / np.sqrt(
      1e-6 + sum([np.square(x) for x in affine_vec]))

  # NOTE: Backbone affine mask: whether the residue has C, CA, N
  # (the template mask defined above only considers pseudo CB).
  template_mask = (
      protein['template_all_atom_masks'][..., n] *
      protein['template_all_atom_masks'][..., ca] *
      protein['template_all_atom_masks'][..., c])
  protein['template_backbone_frame_mask'] = template_mask  # [N_template, N_token]

  template_mask_2d = template_mask[..., None] * template_mask[..., None, :]
  inv_distance_scalar *= template_mask_2d

  unit_vector = [(x * inv_distance_scalar)[..., None] for x in affine_vec]
  unit_vector = [x.astype(dtype) for x in unit_vector]
  unit_vector = np.concatenate(unit_vector, axis=-1)
  protein['template_unit_vector'] = unit_vector

  return protein


def assembly_and_pad(feats_list):
    merged_feats = {}
    for pad_key in AF2_NEED_TO_PADDING_KEYS:
        padding_dim = AF2_PADDING_DIM[pad_key]
        padding_value = AF2_PADDING_FEATS.get(pad_key, 0)
        value = feats_pad_and_concatenate([d[pad_key] for d in feats_list], 
            padding_dim, value=padding_value)
        merged_feats[pad_key] = value
    return merged_feats


#############################
# combine sequence and msa features
#############################


def convert_padded_template_aatype(seq_msa_feats):
    """
    the way template_aatype of rna is padded is different from old code,
    in order to make perfect alignment, we need to convert the padded values.
    Otherwise, please retrain the model.
    """
    ## NOTE: MSA first row should be equal the restype
    is_dna_ligand = np.logical_or(seq_msa_feats['is_dna'] == 1, 
            seq_msa_feats['is_ligand'] == 1)
    seq_msa_feats['msa'][0, is_dna_ligand] = seq_msa_feats['restype'][is_dna_ligand]

    is_rna = seq_msa_feats['is_rna']
    seq_msa_feats['template_aatype'][MAX_TEMPLATE_NUM:, is_rna == 1] = \
            residue_constants.AF3_restype_order['-']
    return seq_msa_feats
