"""
Functions for all atom design.
"""

import numpy as np
from copy import deepcopy

from utils.utils import multimer_collate_fn, tree_map, tree_flatten
from helixfold.data import pipeline_aa_utils

MSA_NAMES_0_DIM = [
    'aatype', 'between_segment_residues'
]
MSA_NAMES_1_DIM = [
    'deletion_matrix_int', 'msa', 'template_aatype',
    'template_all_atom_masks', 'template_all_atom_positions', 
    'deletion_matrix_int_all_seq', 'msa_all_seq',
]


def create_unk_ccd_dict():
    """create_unk_ccd_dict"""
    # TODO: add UNK for dna and rna
    new_ccd_dict = {
        'UNK': {
            'id': 'UNK',
            'smiles': ['NCCO'],
            'atom_ids': ['N', 'CA', 'C', 'O'], 
            'atom_symbol': ['N', 'C', 'C', 'O'], 
            'charge': [0, 0, 0, 0], 
            'leave_atom_flag': ['N', 'N', 'N', 'N'], 
            'position': [
                [0.0, 0.0, 0.0], 
                [0.0, 0.0, 0.0], 
                [0.0, 0.0, 0.0], 
                [0.0, 0.0, 0.0]],
            'back_model_position': [
                [0.0, 0.0, 0.0], 
                [0.0, 0.0, 0.0], 
                [0.0, 0.0, 0.0], 
                [0.0, 0.0, 0.0]], 
            'coval_bonds': [['N', 'CA', 'SING'], ['CA', 'C', 'SING'], ['C', 'O', 'DOUB']], 
            'raw_string': None
        }
    }
    return new_ccd_dict


def mask_all_chain_info_dict(
        all_chain_info_dict, 
        design_mask_dict,
        ccd_preprocessed_dict):
    """mask_all_chain_info_dict"""
    # TODO: add UNK for dna and rna
    masked_ccd = 'UNK'
    masked_seq = 'X'
    new_chain_info_dict = deepcopy(all_chain_info_dict)
    for chain_id, design_mask in design_mask_dict.items():
        design_mask = design_mask.astype('bool')
        ccd_seq_list = new_chain_info_dict[chain_id]['ccd_seq']
        msa_seq_list = list(new_chain_info_dict[chain_id]['msa_seq'])
        for i, flag in enumerate(design_mask):
            if flag:
                ccd_seq_list[i] = masked_ccd
                msa_seq_list[i] = masked_seq
        new_chain_info_dict[chain_id]['ccd_seq'] = ccd_seq_list
        new_chain_info_dict[chain_id]['msa_seq'] = ''.join(msa_seq_list)
        new_chain_info_dict[chain_id]['raw_msa_seq'] = all_chain_info_dict[chain_id]['msa_seq']
        new_chain_info_dict[chain_id]['n_token'] = len(pipeline_aa_utils.ccd_list_to_token(
                new_chain_info_dict[chain_id]['ccd_seq'], ccd_preprocessed_dict))    
    return new_chain_info_dict


def mask_mmcif_object(mmcif_object, design_mask_dict):
    """mask_mmcif_object"""
    # TODO: add UNK for dna and rna
    masked_ccd = 'UNK'
    for chain_id, design_mask in design_mask_dict.items():
        chain = mmcif_object.seqres_to_structure[chain_id]
        design_mask = design_mask.astype('bool')
        for i, (res_index, flag) in enumerate(zip(chain, design_mask)):
            if flag:
                res_at_position = chain[res_index]
                res_at_position.residue_name = masked_ccd
    return mmcif_object


def mask_msa_features(all_chain_features, all_chain_info_dict, design_mask_dict):
    """mask_msa_features
    TODO: pipeline_multimer will put the same sequence into one entity, 
        which may cause msa concatenatation error.
    """
    for chain_id, chain_features in all_chain_features.items():
        if not chain_features is None:
            # convert from bytes to str
            chain_features['sequence'] = chain_features['sequence'].astype('str')

    for chain_id, chain_features in all_chain_features.items():
        if chain_features is None:
            continue
        if not chain_id in design_mask_dict:
            continue
        design_mask = design_mask_dict[chain_id]
        if np.sum(design_mask) == 0:
            continue
        is_mask_all = np.sum(design_mask) == len(design_mask)
        
        design_mask = design_mask.astype('bool')
        for name in MSA_NAMES_0_DIM:
            if name in chain_features:
                chain_features[name][design_mask] = 0
        for name in MSA_NAMES_1_DIM:
            if name in chain_features:
                chain_features[name][:, design_mask] = 0
                # if np.all(design_mask == 1):
                #     chain_features[name] = chain_features[name][:1]
        
        ## other features
        chain_features['sequence'] = np.array([all_chain_info_dict[chain_id]['msa_seq'].encode('utf-8')])
        if is_mask_all:
            chain_features['domain_name'][:] = ''.encode('utf-8')
            chain_features['num_alignments'][:] = 1
            chain_features['msa_species_identifiers'][:] = ''.encode('utf-8')
            chain_features['template_domain_names'][:] = ''.encode('utf-8')
            chain_features['template_sequence'][:] = ('-' * len(design_mask)).encode('utf-8')
            chain_features['template_sum_probs'][:] = 0
            chain_features['msa_species_identifiers_all_seq'][:] = ''.encode('utf-8')
        
        all_chain_features[chain_id] = chain_features
    return all_chain_features
