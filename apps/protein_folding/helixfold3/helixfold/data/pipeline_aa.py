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
from copy import deepcopy
from typing import Mapping, Tuple, List

import numpy as np

from helixfold.common import residue_constants
from helixfold.data import (
    parsers,
    msa_pipeline_rna, 
    msa_pipeline_protein
)
from helixfold.data import pipeline_aa_utils


def get_assembly_sequence_features(
        select_chain_ids: List[str], 
        all_chain_info_dict: Mapping[str, any], 
        coval_bonds_info: List[any],
        ccd_preprocessed_dict: Mapping[str, any]) -> Mapping[str, any]:
    """
    Get sequence features for a given assembly with ordered chain ids.
    Args:
        select_chain_ids: list of chain ids.
        all_chain_info_dict: {
            chain_id: {
                'chain_type': str, 'protein', 'dna', 'rna' or 'ligand'.
                'msa_seq': str,
                'ccd_seq': list of ccd,
            }
        }, refer to input json files for the content format
        coval_bonds_info: list
        ccd_preprocessed_dict: dict
    """
    all_chain_features = {}
    for chain_id in select_chain_ids:
        chain_features = pipeline_aa_utils.make_sequence_features(
                chain_id,
                all_chain_info_dict[chain_id]['chain_type'],
                all_chain_info_dict[chain_id]['ccd_seq'],
                ccd_preprocessed_dict)
        all_chain_features[chain_id] = chain_features
    
    all_chain_features = pipeline_aa_utils.add_assembly_features(
            all_chain_features, all_chain_info_dict)

    merged_features = pipeline_aa_utils.merge_and_adjust_interchain_features(
            all_chain_features, 
            all_chain_info_dict,
            select_chain_ids, 
            coval_bonds_info, 
            ccd_preprocessed_dict)
    return merged_features


def process_with_all_msa_chain_features(
        msa_chain_features: Mapping[str, any], 
        all_chain_info_dict: Mapping[str, any], 
        ccd_preprocessed_dict: Mapping[str, any]) -> Tuple[Mapping[str, any], List[str]]:
    """Process the msa and template features and then merge.
    
    Msa of protein: must be provided in all_chain_features.
    Msa of rna: optional.
    Msa of dna and ligand: will be mocked in this func.
    
    Args:
        msa_chain_features: Mapping[str, any], with keys: <chain_id>: <msa_features>
        all_chain_info_dict: Mapping[str, any], with keys: <chain_id>: <raw_chain_info>
        ccd_preprocessed_dict: Mapping[str, any], with keys: <ccd_id>: <ccd_features>

    Returns:
        Tuple[Mapping[str, any], List[str]]: 
        - Mapping[str, any]: merged msa features for all chains
        - List[str]: new ordered chain ids for tracking after msa pairing

    Note:
        - The chain order will be changed and returned.
        - The dimension of msa and template features are expand for non-standard residues.
    """
    assert len(msa_chain_features) > 0
    new_all_chain_features = {}
    for chain_id in msa_chain_features:
        chain_type = all_chain_info_dict[chain_id]['chain_type']
        seq = all_chain_info_dict[chain_id]['msa_seq']
        ccd_seq = all_chain_info_dict[chain_id]['ccd_seq']
        
        is_prot_rna = chain_type in ['protein', 'rna']
        if is_prot_rna:
            chain_features = deepcopy(msa_chain_features[chain_id])
            if chain_features is None:
                if chain_type == 'rna':
                    unk_map_fn = lambda x: x.replace('X', 'N')
                else:
                    unk_map_fn = lambda x: x             
                _args = {
                    'seq': seq,
                    'chain_type': chain_type,
                    'unk_map_fn': unk_map_fn,
                    'parser_fn': parsers.parse_stockholm_RNA if chain_type == 'rna' \
                                        else parsers.parse_stockholm,
                    'sequence_featurizer': msa_pipeline_rna.make_sequence_features if chain_type == 'rna' \
                                        else msa_pipeline_protein.make_sequence_features,
                    'msa_featurizer': msa_pipeline_rna.make_msa_features if chain_type == 'rna' 
                                        else msa_pipeline_protein.make_msa_features,
                }
                chain_features = pipeline_aa_utils.create_chain_features_without_real_msa(**_args)

            if 'template_all_atom_mask' in chain_features:
                chain_features['template_all_atom_masks'] = chain_features.pop(
                        'template_all_atom_mask')
            # mock rna template
            chain_features = pipeline_aa_utils.mock_msa_template_if_nonexist(
                    chain_features, len(ccd_seq), before_msa_pairing=True)

        else:
            chain_features = {}
            chain_features = pipeline_aa_utils.mock_msa_template_if_nonexist(
                    chain_features, len(ccd_seq), before_msa_pairing=False)
            chain_features = pipeline_aa_utils.add_msa_extra_features(chain_features) 
            chain_features = pipeline_aa_utils.expand_msa_template_for_non_standard_residue(
                    chain_features, ccd_seq, ccd_preprocessed_dict)  
        
        new_all_chain_features[chain_id] = chain_features

    ## 2. merge chain features in order of protein -> rna -> dna -> ligand.
    protein_chain_ids = [cid for cid in new_all_chain_features 
            if all_chain_info_dict[cid]['chain_type'] == 'protein']
    rna_chain_ids = [cid for cid in new_all_chain_features 
            if all_chain_info_dict[cid]['chain_type'] == 'rna']
    dna_chain_ids = [cid for cid in new_all_chain_features 
            if all_chain_info_dict[cid]['chain_type'] == 'dna']
    ligand_chain_ids = [cid for cid in new_all_chain_features 
            if all_chain_info_dict[cid]['chain_type'] == 'ligand']

    ## 2.1 msa pairing for protein and rna
    # NOTE: the order of pro_rna_chain_ids may change after msa_pairing.
    pro_rna_chain_ids = protein_chain_ids + rna_chain_ids
    if len(pro_rna_chain_ids) > 0:
        pro_rna_chain_features = {k: new_all_chain_features[k] for k in pro_rna_chain_ids}
        prot_rna_merged_features, pro_rna_chain_ids = msa_pipeline_protein.process_with_all_chain_features(
                pro_rna_chain_features, return_new_order=True)
    
        prot_rna_merged_features = pipeline_aa_utils.add_msa_extra_features(
                prot_rna_merged_features) 
        prot_rna_ccd_seq = np.concatenate([
                np.array(all_chain_info_dict[cid]['ccd_seq'], dtype=object)
                for cid in pro_rna_chain_ids])
        prot_rna_merged_features = pipeline_aa_utils.expand_msa_template_for_non_standard_residue(
                prot_rna_merged_features, prot_rna_ccd_seq, ccd_preprocessed_dict)
    else:
        prot_rna_merged_features = None
    
    # 2.2 merge and pad
    if prot_rna_merged_features is None:
        feats_list = []
    else:
        feats_list = [prot_rna_merged_features]
    feats_list += [new_all_chain_features[k] for k in dna_chain_ids + ligand_chain_ids]
    new_chain_ids = pro_rna_chain_ids + dna_chain_ids + ligand_chain_ids
    merged_features = pipeline_aa_utils.assembly_and_pad(feats_list)

    # 2.3 add extra features
    merged_features['msa_mask'] = np.ones_like(merged_features['msa'], dtype='float32')
    return merged_features, new_chain_ids


def combine_assembly_seq_and_msa_features(seq_feats, msa_feats):
    """
    seq_feats: assembly sequence features
    msa_feats: assembly msa features
    """
    seq_msa_feats = {**seq_feats, **msa_feats}

    ## check chain orders of sequence and msa features are the same
    assert check_chain_orders_of_seq_and_msa(seq_feats, msa_feats), (
        f"The chain order of seq and msa is wrong")

    ## 5. some further features process in order to align with old code
    seq_msa_feats = pipeline_aa_utils.convert_padded_template_aatype(seq_msa_feats)

    ## for batch_size > 1
    ## remove used keys which will influence collate_fn
    removed_keywords = ['num_alignments', 'num_templates', 'cluster_bias_mask']
    for key in list(seq_msa_feats.keys()):
        for word in removed_keywords:
            if word in key:
                del seq_msa_feats[key]
                break
    return seq_msa_feats


def add_further_assembly_template_feat(features):
    """
    Add further template features for merged features.
    This function will cost a large memory if the token number is large.
    """
    features = pipeline_aa_utils.make_pseudo_beta(features, prefix='template_')
    features = pipeline_aa_utils.make_template_further_feature(features)
    return features


def check_chain_orders_of_seq_and_msa(seq_feats, msa_feats):
    """check chain orders of seq and msa
    by comparing msa and restype of protein and rna
    """
    is_prot_rna = np.logical_or(seq_feats['is_protein'] == 1, 
            seq_feats['is_rna'] == 1)
    # skip non-standard residues, since in msa, some non-standard residues
    #  may be mapped to standard residues
    mask = np.logical_and(is_prot_rna, 
            seq_feats['restype'] != residue_constants.HF3_restype_order['UNK'])
    return np.all(msa_feats['msa'][0, mask] ==
            seq_feats['restype'][mask])
