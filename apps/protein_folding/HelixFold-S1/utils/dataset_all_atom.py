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
"""Dataset for Alphafold2."""

import numpy as np
import paddle
import os
from os.path import exists, join
import sys
import random
import gzip
from glob import glob
import pickle
import time
import traceback
import re
from copy import deepcopy
import collections
import json
import pandas as pd
from paddle.io import IterableDataset, DataLoader
from collections import Counter
from scipy.spatial import distance_matrix
from typing import Any, Tuple, Union, List, Optional
import helixfold.data.mmcif_parsing_paddle as mmcif_parsing
from helixfold.data import pipeline_multimer, pipeline_rna_multimer
from helixfold.data.data_utils import a3m_to_features, sample_raw_msa, block_delete_msa
from helixfold.data.utils import get_crop_mask_all_atom, apply_crop_mask_and_pad
from helixfold.data.utils import select_from_samped_chain_center, build_pocket_info, \
        build_pocket_info_with_random_residues, build_interface_info_with_random_residues, \
        build_possible_interface_mask, mix_homo_interface_info
from helixfold.model import features
from helixfold.common import protein as protein_utils
from helixfold.common import residue_constants
from helixfold.data import parsers
# from helixfold.data.pipeline import make_sequence_features, make_msa_features
from helixfold.data.utils import is_ignored_key_multimer, is_batched_key_multimer

from utils.dataset import LoopedBatchSampler

from helixfold.data import feature_processing
from helixfold.data import msa_pairing

from utils.utils import multimer_collate_fn, tree_map, tree_flatten
from helixfold.common.protein import PDB_CHAIN_IDS
from helixfold.data.input.data_transforms import Seed_maker
from helixfold.data.input import input_pipeline
from utils.rotation_utils import get_rotation_mat, rotate_all_atom_positions
from utils.slice_array import slice_array

from helixfold.data import pipeline_aa, pipeline_aa_utils
from helixfold.data import pipeline_hybrid, pipeline_conf_bonds, pipeline_token_feature, label_utils

from helixfold.data.parsers import parse_stockholm_RNA
from helixfold.data import pipeline_rna

LABEL_NEED_OFFSET_KEYS = label_utils.NEED_OFFSET_KEYS
PBDID_DATA_CUTOFF = "2021-09-30"
MAX_SEQ_SIZE = 3_000
MAX_FEAT_FILE_SISE_MB = 2.5 # 2MB
MAX_DCU_SEQ_SIZE = 800
MAX_CROPPING_TOKEN_NUM = 10_000
MAX_SINGLE_CHAIN_SIZE = 10_000

# attr ==> seq_len_axis
NEED_FEATURES = {
    'atom14_atom_exists': 1,
    'atom37_atom_exists': 1,
    'residx_atom14_to_atom37': 1,
    'residx_atom37_to_atom14': 1,
    'msa_feat': 2
}

SEQ_LEN_DIMS = {
    'aatype': 0,
    'residue_index': 0,
    'msa': 1,
    'template_aatype': 1,
    'template_all_atom_masks': 1,
    'template_all_atom_positions': 1,
    'asym_id': 0,
    'sym_id': 0,
    'entity_id': 0,
    'deletion_matrix': 1,
    'deletion_mean': 0,
    'all_atom_mask': 0,
    'all_atom_positions': 0,
    'bert_mask': 1,
    'seq_mask': 0,
    'msa_mask': 1
}


def read_ccd_preprocessed_dict(path):
    """read_ccd_preprocessed_dict"""
    ccd_preprocessed_dict = {}
    if 'pkl.gz' in path:
        with gzip.open(path, "rb") as fp:
            ccd_preprocessed_dict = pickle.load(fp)
    elif '.pkl' in path:
        with open(path, "rb") as fp:
            ccd_preprocessed_dict = pickle.load(fp)
    return ccd_preprocessed_dict


def read_cif_to_mmcif_object(cif_file):
    """read cif file into mmcif object."""
    def _read_cif_string(cif_file):
        if cif_file.endswith('.cif.gz'):
            with gzip.open(cif_file, 'r') as f:
                cif_string = f.read().decode('utf8')
        else:
            cif_string = "".join(open(cif_file, 'r').readlines())
        return cif_string

    cif_string = _read_cif_string(cif_file)
    parse_result = mmcif_parsing.parse(file_id=cif_file, mmcif_string=cif_string)
    mmcif_object = parse_result.mmcif_object

    if mmcif_object is None :
        print(f'=> Skip [{cif_file}] due to mmcif parsing error. Got [{parse_result.errors}]')
        return None
    return mmcif_object


def assembly_json_dict_to_chain_dict(assembly_json_dict, ccd_preprocessed_dict=None):
    """convert assembly json dict to chain dict. 
    all_chain_info_dict: {
        chain_id: {
            'chain_type': str,
            'msa_seq': str,
            'ccd_seq': list of ccd,
        }
    }
    """
    all_chain_info_dict = {}
    for chain_type, content in assembly_json_dict['tokenization'].items():
        for cid, seq in zip(content['chain_ids'], content['msa_seqs']):
            all_chain_info_dict[cid] = {'chain_type': chain_type, 'msa_seq': seq}

    for _, content in assembly_json_dict['basic'].items():
        for cid, seq in zip(content['chain_ids'], content['seqs']):
            all_chain_info_dict[cid]['ccd_seq'] = parsers.parse_ccd_fasta(seq)
            if not ccd_preprocessed_dict is None:
                all_chain_info_dict[cid]['n_token'] = len(pipeline_aa_utils.ccd_list_to_token(
                        all_chain_info_dict[cid]['ccd_seq'], ccd_preprocessed_dict))    
    return all_chain_info_dict


def crop_msa(feat, max_msa_depth=16384):
    """ pad msa and generate msa_mask. """
    msa = feat['msa']
    msa_mask = np.ones_like(feat['msa']).astype('float32') # [msa_depth, n_token]
    delection_mat = feat['deletion_matrix']
    msa_depth, num_token = msa_mask.shape
    if msa_depth > max_msa_depth:
        msa_mask = msa_mask[: max_msa_depth, :]
        msa = msa[: max_msa_depth, :]
        delection_mat = delection_mat[: max_msa_depth, :]
    return msa.astype('int32'), msa_mask, delection_mat


def align_entity_id_by_ccd_id(label1, label2):
    """
    change entity_id of label2 to align with label1
    """
    def _get_ccd_mapping(label):
        """get ccd_concat -> entity_id mapping of label"""
        asym_id = label["perm_asym_id"]
        uniq_asym_ids = np.unique(asym_id)
        entity_id = label["perm_entity_id"]
        ccd_ids = np.array(label['label_ccd_ids'].split())
        ccd_entity_map = {}
        ccd_asym_map = {}
        for aid in uniq_asym_ids:
            mask = asym_id == aid
            ccd_concat = ' '.join(list(ccd_ids[mask]))
            ccd_entity_map[ccd_concat] = entity_id[mask][0]
            if not ccd_concat in ccd_asym_map:
                ccd_asym_map[ccd_concat] = []
            ccd_asym_map[ccd_concat].append(aid)
        return ccd_entity_map, ccd_asym_map

    ccd_entity_map1, ccd_asym_map1 = _get_ccd_mapping(label1)
    ccd_entity_map2, ccd_asym_map2 = _get_ccd_mapping(label2)
    
    # get old_entity_id -> new_entity_id mapping
    entity_change_map = {}
    for ccd, eid2 in ccd_entity_map2.items():
        if ccd in ccd_entity_map1:
            entity_change_map[eid2] = ccd_entity_map1[ccd]
    prev_max_entity_id = np.max(list(ccd_entity_map1.values()))
    for ccd, eid2 in ccd_entity_map2.items():
        if ccd not in ccd_entity_map1:
            entity_change_map[eid2] = prev_max_entity_id + 1
            prev_max_entity_id += 1

    # get old_asym_id -> new_asym_id mapping
    asym_change_map = {}
    for ccd, aids2 in ccd_asym_map2.items():
        if ccd in ccd_asym_map1:
            for aid2, aid1 in zip(aids2, ccd_asym_map1[ccd]):
                asym_change_map[aid2] = aid1
    prev_max_asym_id = np.max([x for l in ccd_asym_map1.values() for x in l])
    for ccd, aids2 in ccd_asym_map2.items():
        for aid2 in aids2:
            if aid2 not in asym_change_map:
                asym_change_map[aid2] = prev_max_asym_id + 1
                prev_max_asym_id += 1

    new_entity_id = np.array([entity_change_map[eid] 
            for eid in label2['perm_entity_id']])
    new_asym_id = np.array([asym_change_map[aid] 
            for aid in label2['perm_asym_id']])
    return new_entity_id, new_asym_id


class AllAtomDataset(paddle.io.Dataset):
    """tbd."""

    def __init__(self,
                 model_config,
                 data_config,
                 crop_size=None,
                 is_pad_if_crop=False,
                 delete_msa_block=False,
                 trainer_id=0,
                 trainer_num=1,
                 is_shuffle=False):
        """
        Iterate over clusts, where proteins in each clust
        will be all visited.
        """
        self.model_config = deepcopy(model_config)
        self.delete_msa_block = delete_msa_block

        self.data_config = data_config
        self.crop_size = crop_size
        self.is_pad_if_crop = is_pad_if_crop
        self.trainer_id = trainer_id
        self.trainer_num = trainer_num
        self.is_shuffle = is_shuffle

        self.sampling_map = self.data_config.get('sampling_map',
                {"uniform": 0.2, "weight": 0.8})
        self.pocket_map = self.data_config.get('pocket_map',
                {'centra_10A': 1.0})

        self.msa_rand_prob = self.model_config.data.get('msa_rand_prob', 0.0)
        self.max_epitope = self.data_config.get("max_epitope", 5)
        self.date_cutoff = self.data_config.get("date_cutoff", PBDID_DATA_CUTOFF)

        self.interface_config = self.data_config.get("interface_config", 
            {"interface_source": 'none',       # 'gen', 'gt' or 'none'
             "interface_type_map": {'any_interface': 0.7, 'none': 0.3},
             "dist_thres": 5, 
             "interface_sample": {"top_n": 40, "min_m": 1, "max_m": 20, "seed": None},
             "interface_gen": {"interface_size": 5, "seed": None}})

        self.max_acceptable_token_num = self.data_config.get("max_acceptable_token_num", None)

        def _assert_exists(path):
            assert exists(path), path

        _assert_exists(self.data_config.feature_dir)
        _assert_exists(self.data_config.structure_dir)

        ## check and filter (bad or not needed) proteins
        self.rna2seq_map = self._load_rna2seq_map()
        self.fasta2id_map = self._load_fasta2id_map()
        self.rna_clust_list = self._load_rna_clust_list()
        self.long_ligand_list = self._load_long_ligand_list()
        self.abag_clusts = self._load_abag_clusts()
        self.peptide_clusts = self._load_peptide_clusts()
        self.monomer_clusts = self._load_monomer_clusts()
        self.uniform_clusts, self.all_clusts = self._load_clusts()
        self.miss_clusters, self.all_miss_clusts = self._load_miss_clusts()
        self.uni_points = 0

        def _print_attribute(key, value):
            print(f'[{self.__class__.__name__}] {key}: {value}')

        _print_attribute('trainer_id/trainer_num',
                         f'{self.trainer_id}/{self.trainer_num}')
        _print_attribute('sampling_map', self.sampling_map)
        _print_attribute('all_clusts', len(self.all_clusts)
                if not self.all_clusts is None else None)
        _print_attribute('uniform_clusts', len(self.uniform_clusts)
                if not self.uniform_clusts is None else None)
        _print_attribute('rna_clust_list', len(self.rna_clust_list))
        _print_attribute('long_ligand_list', len(self.long_ligand_list))
        _print_attribute('abag_clusts', len(self.abag_clusts) 
                if not self.abag_clusts is None else None)
        _print_attribute('crop_size', crop_size)
        _print_attribute('is_pad_if_crop', is_pad_if_crop)
        _print_attribute('delete_msa_block', delete_msa_block)
        _print_attribute('interface_config', self.interface_config)
    
        self.ccd_preprocessed_dict = read_ccd_preprocessed_dict(
                data_config.ccd_preprocessed_path)

    def _load_rna2seq_map(self):
        rna2seq_map = {}
        if "rna_map_file" in self.data_config.keys():
            for line in open(self.data_config.rna_map_file):
                rna, rna_seqid = line.split()
                rna2seq_map[rna] = rna_seqid
        return rna2seq_map

    def _load_fasta2id_map(self):
        fasta2id_map = {}
        if "fasta_map_file" in self.data_config.keys():
            fasta2id_map.update(json.load(open(self.data_config.fasta_map_file)))
        return fasta2id_map
    
    def _load_rna_clust_list(self):
        rna_clust_list = []
        if 'rna_clust_list_file' in self.data_config:
            with open(self.data_config.rna_clust_list_file) as f:
                for line in f:
                    segs = line.strip().split()
                    clust = []
                    for seg in segs:
                        pdb_id, chain_id = seg.split("_")
                        clust.append({
                            'pdb_id': pdb_id, 'chains': chain_id})
                    rna_clust_list.append(clust)
        rna_clust_list = rna_clust_list[self.trainer_id::self.trainer_num]
        return rna_clust_list

    def _load_long_ligand_list(self):
        long_ligand_list = []
        if 'long_ligand_list_file' in self.data_config:
            with open(self.data_config.long_ligand_list_file) as f:
                for line in f:
                    pdb_id, chain_id, ccd, atom_num = line.strip().split()
                    long_ligand_list.append({
                            'pdb_id': pdb_id, 'chains': chain_id})
        long_ligand_list = long_ligand_list[self.trainer_id::self.trainer_num]
        return long_ligand_list

    def _load_abag_clusts(self):
        # abag_clusts = None
        # if 'abag_clust_file' in self.data_config:   
        #     columns_to_drop = ["chain_type","is_interface","cluster_id",
        #             "beta_r","n_prot","n_nuc","n_lig"]

        #     abag_clusts = pd.read_csv(self.data_config.abag_clust_file).dropna()
        #     abag_clusts.drop(columns=columns_to_drop, inplace=True)
        #     abag_clusts = abag_clusts[self.trainer_id::self.trainer_num].reset_index(drop=True)
        # return abag_clusts
        abag_clusts = []
        if "abag_clust_file" in self.data_config:
            with open(self.data_config.abag_clust_file) as f:
                for line in f:
                    pdb_id = line.strip().split()
                    abag_clusts.append({'pdb_id': pdb_id, 'chains': None})
        abag_clusts = abag_clusts[self.trainer_id::self.trainer_num]
        return abag_clusts

    def _load_peptide_clusts(self):
        peptide_clusts = None
        if 'peptide_clust_file' in self.data_config:   
            columns_to_drop = ["chain_types","is_interface","cluster_id",
                    "beta_r","n_prot","n_nuc","n_lig","weight_raw"]
            peptide_clusts = pd.read_csv(self.data_config.peptide_clust_file).dropna()
            peptide_clusts.drop(columns=columns_to_drop, inplace=True)
            peptide_clusts = peptide_clusts[self.trainer_id::self.trainer_num].reset_index(drop=True)
        return peptide_clusts

    def _load_monomer_clusts(self):
        monomer_clusts = []
        if 'monomer_clust_file' in self.data_config:   
            with open(self.data_config.monomer_clust_file) as f:
                for line in f:
                    segs = line.strip().split()
                    clust = []
                    for seg in segs:
                        pdb_id, chain_id = seg.split("_")
                        clust.append({
                            'pdb_id': pdb_id, 'chains': chain_id})
                    monomer_clusts.append(clust)
        monomer_clusts = monomer_clusts[self.trainer_id::self.trainer_num]
        return monomer_clusts

    def _load_clusts(self):
        df_uniform, df_weighted = None, None
        if 'protein_clust_file' in self.data_config:
            columns_to_drop = ["chain_types","is_interface","cluster_id",
                    "beta_r","n_prot","n_nuc","n_lig","weight_raw"]
            df_weighted = pd.read_csv(self.data_config.protein_clust_file).dropna()
            df_weighted.drop(columns=columns_to_drop, inplace=True)
            df_weighted = df_weighted[self.trainer_id::self.trainer_num].reset_index(drop=True)

            ## shuffle
            df_uniform = df_weighted.sample(frac=1).reset_index(drop=True)
        return df_uniform, df_weighted
    
    def _load_miss_clusts(self):
        df_uniform, df_weighted = None, None
        if 'miss_clust_file' in self.data_config:
            columns_to_drop = ["chain_types","is_interface","cluster_id",
                    "beta_r","n_prot","n_nuc","n_lig","weight_raw"]
            df_weighted = pd.read_csv(self.data_config.miss_clust_file).dropna()
            df_weighted.drop(columns=columns_to_drop, inplace=True)
            df_weighted = df_weighted[self.trainer_id::self.trainer_num].reset_index(drop=True)

            ## shuffle
            df_uniform = df_weighted.sample(frac=1).reset_index(drop=True)
        return df_uniform, df_weighted

    def _check_protein(self, protein):
        """tbd."""
        unit_json_path, assembly_json_path = self._get_protein_json_file(protein)
        # Check json_file
        if not exists(unit_json_path) or not exists(assembly_json_path):
                return 1
        
        unit_json_dict = json.load(open(unit_json_path))
        assembly_json_dict = json.load(open(assembly_json_path))
        
        release_date, resolution = unit_json_dict['release_date'], unit_json_dict['resolution']
        if release_date > self.date_cutoff:
            return 2
        if resolution == 0 or resolution > 9:
            return 3
        if "chain_mapping" in assembly_json_dict and \
                len(list(assembly_json_dict['chain_mapping'].keys())) > 300:
            return 4
        if not 'tokenization' in assembly_json_dict: return 5
        if not 'basic' in assembly_json_dict: return 6
        # Check structure file
        unit_struct_path, assembly_struct_path = self._get_protein_struct_file(protein)
        if not exists(unit_struct_path) or not exists(assembly_struct_path):
            return 7         
        return 0

    def _get_rna_feature_file(self, rna, seq):
        """tbd."""
        if rna in self.rna2seq_map: 
            rna = self.rna2seq_map[rna]
        path = join(self.data_config.rna_feature_dir, rna, 'features.pkl')
        if exists(path):
            return path

        path = os.path.join(self.data_config.rna_feature_dir, rna,
                            'features.pkl.gz')
        return path

    def _get_protein_feature_file(self, protein, seq):
        """tbd."""
        if seq in self.fasta2id_map: 
            protein = self.fasta2id_map[seq]
        path = join(self.data_config.feature_dir, protein, 'features.pkl')
        if exists(path):
            return path

        path = os.path.join(self.data_config.feature_dir, protein,
                            'features.pkl.gz')
        return path

    def _get_protein_struct_file(self, protein):
        """tbd."""
        path = join(self.data_config.structure_dir, f"{protein}.cif")
        gz_path = join(self.data_config.structure_dir, f"{protein}.cif.gz")
        assembly_path = join(self.data_config.assembly_structure_dir, f"{protein}-assembly1.cif")
        gz_assembly_path = join(self.data_config.assembly_structure_dir, f"{protein}-assembly1.cif.gz")

        unit_path = path if exists(path) else gz_path
        assembly_path = path if exists(assembly_path) else gz_assembly_path
        
        return unit_path, assembly_path

    def _get_protein_json_file(self, protein):
        """tbd."""
        return join(self.data_config.json_dir, f"{protein}.json"), \
                        join(self.data_config.assembly_json_dir, f"{protein}-assembly1.json")
    
    def get_seq_msa_feats(self, protein, selected_chain_ids, all_chain_info_dict, coval_bonds_info):
        """get_msa_feats"""
        def _load_pro_rna_feature_pkl(protein, chain_type, chain_id, seq):
            assert chain_type in ['protein', 'rna']
            if chain_type == 'rna':
                features_pkl = self._get_rna_feature_file(
                        protein + "_" + chain_id.split('_')[0], seq=seq)
            else:
                features_pkl = self._get_protein_feature_file(
                        protein + '_' + chain_id.split('_')[0], seq=seq)
            if not os.path.exists(features_pkl): 
                return None
            else:
                if features_pkl.endswith('.pkl.gz'):
                    with gzip.open(features_pkl, 'rb') as pkl:
                        raw_features = pickle.load(pkl)
                else:
                    with open(features_pkl, 'rb') as pkl:
                        raw_features = pickle.load(pkl)
            return raw_features

        ## 1. get features for each chain
        ## protein chains failed to load feature_pkl will be dropped.
        ## rna chains failed to load feature_pkl won't be dropped and will assign as None
        ## dna and ligand chains are assigned as None
        all_chain_features = {}
        seq_to_chain_features = {}  # homomer chains should run sample_raw_msa only once
        for chain_id in selected_chain_ids:
            chain_type = all_chain_info_dict[chain_id]['chain_type']
            is_prot_rna = chain_type in ['protein', 'rna']
            if is_prot_rna:
                seq = all_chain_info_dict[chain_id]['msa_seq']  # as the key to query the msa feature pkl
                if seq in seq_to_chain_features:
                    all_chain_features[chain_id] = seq_to_chain_features[seq]
                    continue
                
                raw_features = _load_pro_rna_feature_pkl(
                        protein, chain_type, chain_id, seq)
                if raw_features is None and chain_type == 'protein':
                    continue
                msa_rand_prob = np.random.rand() 
                if not raw_features is None and 'msa_sample' in self.data_config \
                        and msa_rand_prob < self.msa_rand_prob:
                    raw_features = sample_raw_msa(
                            raw_features, 
                            self.data_config.msa_sample.min_depth, 
                            self.data_config.msa_sample.max_depth)
                seq_to_chain_features[seq] = raw_features
            else:
                raw_features = None           
            all_chain_features[chain_id] = raw_features

        ## 2. msa pairing and merge
        msa_features, new_chain_ids = pipeline_aa.process_with_all_msa_chain_features(
                all_chain_features, 
                all_chain_info_dict, 
                self.ccd_preprocessed_dict)
        if self.delete_msa_block:
            msa_features = block_delete_msa(msa_features, self.model_config.data)
        
        ## 3. get sequence features
        seq_features = pipeline_aa.get_assembly_sequence_features(
                new_chain_ids, 
                all_chain_info_dict, 
                coval_bonds_info,
                self.ccd_preprocessed_dict)
        
        ## 4. combine sequence and msa features
        seq_msa_feats = pipeline_aa.combine_assembly_seq_and_msa_features(
                seq_features, msa_features)
        return seq_msa_feats, new_chain_ids

    def select_protein(self, index):
        """tbd."""
        sampling_method = random.choices(
                list(self.sampling_map.keys()),
                weights=list(self.sampling_map.values()),
                k=1)[0]

        if sampling_method == 'uniform':
            if self.uni_points >= len(self.uniform_clusts):
                self.uniform_clusts = self.uniform_clusts.sample(
                        frac=1).reset_index(drop=True)
                self.uni_points = 0
            item = self.uniform_clusts.iloc[self.uni_points]
            self.uni_points += 1
        elif sampling_method == 'weight':
            item = self.all_clusts.sample(n=1, weights="weight")
            item = item.iloc[0]
        elif sampling_method == "miss":
            assert self.miss_clusters is not None
            item = self.all_miss_clusts.sample(n=1, weights="weight")
            item = item.iloc[0]
        elif sampling_method == 'long_ligand':
            assert len(self.long_ligand_list) > 0
            if not hasattr(self, 'long_ligand_list_to_consume'):
                self.long_ligand_list_to_consume = []
            if len(self.long_ligand_list_to_consume) == 0:
                self.long_ligand_list_to_consume = self.long_ligand_list.copy()
                np.random.shuffle(self.long_ligand_list_to_consume)
            item = self.long_ligand_list_to_consume.pop()
        elif sampling_method == 'rna':
            assert len(self.rna_clust_list) > 0
            if not hasattr(self, 'rna_clust_list_to_consume'):
                self.rna_clust_list_to_consume = []
            if len(self.rna_clust_list_to_consume) == 0:
                self.rna_clust_list_to_consume = self.rna_clust_list.copy()
                np.random.shuffle(self.rna_clust_list_to_consume)
            clust = self.rna_clust_list_to_consume.pop()
            item = random.choice(clust)
        elif sampling_method == 'abag':
            assert not self.abag_clusts is None
            # item = self.abag_clusts.sample(n=1, weights="weight")
            # item = item.iloc[0]
            item = np.random.choice(self.abag_clusts)
        elif sampling_method == 'peptide':
            assert not self.peptide_clusts is None
            item = self.peptide_clusts.sample(n=1, weights="weight")
            item = item.iloc[0]
        elif sampling_method == 'monomer':
            assert len(self.monomer_clusts) > 0
            if not hasattr(self, 'monomer_clust_list_to_consume'):
                self.monomer_clusts_to_consume = []
            if len(self.monomer_clusts_to_consume) == 0:
                self.monomer_clusts_to_consume = self.monomer_clusts.copy()
                np.random.shuffle(self.monomer_clusts_to_consume)
            clust = self.monomer_clusts_to_consume.pop()
            item = random.choice(clust)
        else:
            raise ValueError(sampling_method)
        
        meta_data = {}
        meta_data['protein'] = str(item['pdb_id'])
        meta_data['chain_ids'] = (str(item['chains']).split(',') 
                                    if item['chains'] is not None else None)
        meta_data['sampling_method'] = sampling_method
        return meta_data

    def get_pocket_info(self, feat, label):
        pocket_type = random.choices(
                list(self.pocket_map.keys()),
                weights=list(self.pocket_map.values()),
                k=1)[0]
        if pocket_type == 'centra_10A':
            pocket_info = build_pocket_info(
                    feat['asym_id'], 
                    feat['is_ligand'], 
                    label['all_atom_pos'][label['all_centra_token_indice']], 
                    label['all_centra_token_indice_mask'])
        elif pocket_type == 'top20_rand1_10':
            m = random.randint(1, 10)
            pocket_info = build_pocket_info_with_random_residues(
                    feat['asym_id'], 
                    feat['is_ligand'], 
                    label['all_atom_pos'][label['all_centra_token_indice']], 
                    label['all_centra_token_indice_mask'],
                    top_n=20,
                    m=m)
        elif pocket_type == 'epitope':
            max_epitope = self.max_epitope
            num_epitope = np.random.randint(1, max_epitope + 1)
            #TODO: remove it when reliable
            print(f">>>>>> [DEBUG] num_epitope: {num_epitope}, max_epitope: {max_epitope}")
            
            n_token = len(feat['asym_id'])
            token_mask = label['all_centra_token_indice_mask']
            pocket_info = np.zeros([n_token, n_token], np.float32)
            is_polymer = np.logical_and(feat['is_ligand'] != 1, token_mask)
            unique_polymer_chains = np.unique(feat['asym_id'][is_polymer == 1])
            if len(unique_polymer_chains) < 2:
                print(f">>>>> [DEBUG] not enough polymer chains") #TODO: remove it when reliable
                return pocket_info
            
            antigen_chain = np.random.choice(unique_polymer_chains)
            is_antigen = np.logical_and(feat['asym_id'] == antigen_chain, token_mask)
            is_antibody = np.logical_and(is_polymer, feat['asym_id'] != antigen_chain)
            is_antibody = np.logical_and(is_antibody, token_mask)

            pocket_info = build_pocket_info_with_random_residues(
                asym_id=feat['asym_id'],
                binder_mask=is_antibody,
                target_mask=is_antigen,
                token_pos=label['all_atom_pos'][label['all_centra_token_indice']], 
                token_pos_mask=token_mask,
                top_n=-1,
                m=num_epitope
            )
        elif pocket_type == 'none':
            n_token = len(feat['asym_id'])
            pocket_info = np.zeros([n_token, n_token], 'float32')
        else:
            raise ValueError(pocket_type)
        return pocket_info


    def get_interface_mask(self, feat, label, interface_type):
        if interface_type == 'epitope_paratope':
            token_mask = label['all_centra_token_indice_mask']
            # NOTE: ligand-protein interfaces are excluded.
            is_polymer = np.logical_and(feat['is_ligand'] != 1, token_mask)
            unique_polymer_chains = np.unique(feat['asym_id'][is_polymer == 1])
            if len(unique_polymer_chains) < 2:
                print(f">>>>> [DEBUG] not enough polymer chains") #TODO: remove it when reliable
                n_token = len(feat['asym_id'])
                return np.zeros([n_token, n_token], np.float32)
            
            antigen_chain = np.random.choice(unique_polymer_chains)
            is_antigen = np.logical_and(feat['asym_id'] == antigen_chain, token_mask)
            is_antibody = np.logical_and(is_polymer, feat['asym_id'] != antigen_chain)
            is_antibody = np.logical_and(is_antibody, token_mask)

            interface_mask = build_possible_interface_mask(
                feat['asym_id'], 
                token_mask, 
                binder_mask=is_antibody, 
                target_mask=is_antigen)

        else: 
            interface_mask = build_possible_interface_mask(
                feat['asym_id'], 
                label['all_centra_token_indice_mask'], 
                binder_mask=None, 
                target_mask=None)

        return interface_mask


    def get_interface_info(self, feat, label, interface_type, interface_mask):

        if self.interface_config['interface_gen']['seed'] is not None:
            interface_gen_seed = self.interface_config['interface_gen']['seed']
        else:
            interface_gen_seed = random.randint(1, 10000)

        interface_gen_temperature = self.interface_config['interface_gen'].get('temperature', 1.0)
        interface_sampling_method = self.interface_config['interface_gen'].get('sampling_method', 'in_repeats')
        interface_sampling_beam = self.interface_config['interface_gen'].get('sampling_beam', 10)

        interface_info_dict = {
            'interface_source': self.interface_config['interface_source'],
            'interface_gen_repeats': self.interface_config['interface_gen']['gen_repeats'],
            'interface_gen_size': self.interface_config['interface_gen']['interface_size'],
            'interface_gen_seed': interface_gen_seed,
            'interface_gen_temperature': interface_gen_temperature,
            'interface_sampling_method': interface_sampling_method,
            'interface_sampling_beam': interface_sampling_beam
        }

        if interface_type in ['any_interface', 'epitope_paratope']:
            # random select m between min_m to max_m
            m = random.randint(self.interface_config['interface_sample']['min_m'], 
                            self.interface_config['interface_sample']['max_m'])
            seed = self.interface_config['interface_sample'].get('seed', None)
            ret_dict = build_interface_info_with_random_residues(
                    feat['asym_id'], 
                    feat['entity_id'], 
                    atom_pos=label['all_atom_pos'], 
                    atom_mask=label['all_atom_pos_mask'],
                    atom_to_token_mapping=feat['ref_token2atom_idx'],
                    token_pos=label['all_atom_pos'][label['all_centra_token_indice']],
                    token_pos_mask=label['all_centra_token_indice_mask'],
                    top_n=self.interface_config['interface_sample']['top_n'],
                    m=m,
                    interface_mask=interface_mask,
                    dist_thres=self.interface_config['dist_thres'],
                    dist_type=self.interface_config.get('dist_type', 'heavy_atom'),
                    seed=seed)
            interface_info_dict.update(ret_dict)
        elif interface_type in ['none', None]:
            n_token = len(feat['asym_id'])
            interface_info_dict.update({
                "interface_info_sample": np.zeros([n_token, n_token], dtype=np.float32),
                "interface_info_full": np.zeros([n_token, n_token], dtype=np.float32),
                "interface_info_mix": np.zeros([n_token, n_token], dtype=np.float32),
            })
        else:
            raise ValueError(interface_type)

        return interface_info_dict


    def get_sample(self, protein, sampled_chain_ids=None, sampling_method=None):
        ''' tbd. '''
        error_code = self._check_protein(protein)
        if error_code != 0 :
            print(f'=> Skip [{protein}] due to _check_protein no pass. {error_code}')
            return None

        sample_info = {'protein': protein, 'sampled_chains_ids': sampled_chain_ids, 
                    'sampling_method': sampling_method, 'max_cropping_token': MAX_CROPPING_TOKEN_NUM,
                    'sampled_chains-selected': [], 'unsampled_chains-selected': []}

        # 0. Read chain info from json file
        unit_json_path, assembly_json_path = self._get_protein_json_file(protein)
        unit_json_dict = json.load(open(unit_json_path))
        assembly_json_dict = json.load(open(assembly_json_path))
        
        all_chain_info_dict = assembly_json_dict_to_chain_dict(
                assembly_json_dict, self.ccd_preprocessed_dict)
        coval_bonds_info = pipeline_aa_utils.bond_convert_unit_to_assembly(
                unit_json_dict.get('covalent_bonds', []), assembly_json_dict['chain_mapping'])
        # drop chains with too long length 
        for cid in list(all_chain_info_dict.keys()):
            if all_chain_info_dict[cid]['n_token'] > MAX_SINGLE_CHAIN_SIZE:
                del all_chain_info_dict[cid]

        # 1. load mmcif object
        unit_struct_file, assembly_struct_file = self._get_protein_struct_file(protein)
        assembly_mmcif_object = read_cif_to_mmcif_object(assembly_struct_file)
        if assembly_mmcif_object is None:
            print(f'=> Skip [{protein}] due to all mmcif parsing error.')
            return None

        # 2. get crop mask
        if sampled_chain_ids is None:
            sampled_chain_ids = [np.random.choice(list(all_chain_info_dict.keys()))]
        if sampling_method in ['peptide', 'monomer']:
            loaded_chain_ids = list(sampled_chain_ids)
        else:
            loaded_chain_ids = select_from_samped_chain_center(all_chain_info_dict, sampled_chain_ids,
                    MAX_CROPPING_TOKEN_NUM, assembly_mmcif_object, self.ccd_preprocessed_dict)

        cropping_feat = pipeline_aa.get_assembly_sequence_features(
                loaded_chain_ids, 
                all_chain_info_dict, 
                coval_bonds_info,
                self.ccd_preprocessed_dict)
        cropping_label = pipeline_aa.get_assembly_label(
                loaded_chain_ids, 
                assembly_mmcif_object,
                self.ccd_preprocessed_dict)
        if cropping_label is None:
            return None
        assert pipeline_aa.check_chain_orders_of_seq_and_label(
                cropping_feat, cropping_label), (
                f"The chain order of seq and label of protein {protein} is wrong")

        # 2.1 start cropping
        if not self.crop_size is None and self.crop_size > 0:
            chain_lens = np.bincount(cropping_feat['asym_id'].astype('int64'))
            seq_infos = {'seq_lens': list(chain_lens)[1:]}  
            chain_to_asym = {k: v for k, v in zip(
                    cropping_feat['all_chain_ids'],
                    cropping_feat['asym_id'][cropping_feat["ref_token2atom_idx"]])} 
            if sampling_method in ['long_ligand', 'rna']:
                spatial_inter_crop_ratio = 0.0
                spatial_crop_ratio = 1.0
            elif sampling_method in ['peptide', 'monomer']:
                spatial_inter_crop_ratio = 0.0
                spatial_crop_ratio = 0.0
            else:
                spatial_inter_crop_ratio = self.data_config.get("spatial_inter_crop_ratio", 0.4)
                spatial_crop_ratio = self.data_config.get("spatial_crop_ratio", 0.4)
            cropped_mask, sample_info['crop_method'] = get_crop_mask_all_atom(
                    cropping_feat,
                    cropping_label,
                    seq_infos=seq_infos,
                    crop_size=self.crop_size,
                    spatial_inter_crop_ratio=spatial_inter_crop_ratio,
                    spatial_crop_ratio=spatial_crop_ratio,
                    targeted_asym_ids=[chain_to_asym[c] for c in sampled_chain_ids],
                    max_atom_num=self.crop_size * 14)
            cropped_chain_ids = [c for c in cropped_mask \
                              if np.sum(cropped_mask[c]) > 0]
        else:
            cropped_mask = {}
            for chain_id, chain_len in zip(all_chain_ids, chain_lens):
                cropped_mask[chain_id] = np.ones(chain_len, dtype='bool')
            cropped_chain_ids = all_chain_ids
        sample_info['cropped_chain'] = cropped_chain_ids

        # 3. Feats and labels
        np_example, selected_chains = self.get_seq_msa_feats(
                protein, 
                cropped_chain_ids, 
                all_chain_info_dict, 
                coval_bonds_info)
        raw_labels = pipeline_aa.get_assembly_label(
                selected_chains, 
                assembly_mmcif_object,
                self.ccd_preprocessed_dict)
        raw_labels['resolution'] = np.array([unit_json_dict['resolution']])
        if raw_labels is None:
            return None

        assert pipeline_aa.check_atom_nums_of_seq_and_label(
                np_example, raw_labels), (
                f"The atom num of seq and msa of protein {protein} is inequal")

        ## Interface related features
        if self.max_acceptable_token_num is not None and np_example['asym_id'].shape[0] > self.max_acceptable_token_num:
            print(f"Skip protein {protein}. Tokens number {np_example['asym_id'].shape[0]} too large (> {self.max_acceptable_token_num})")
            return None 
      
        # 4. cropping and padding
        if not self.crop_size is None and self.crop_size > 0:
            chain_lens = np.bincount(np_example['asym_id'].astype('int64'))
            seq_infos = {'seq_lens': list(chain_lens)[1:]}
            # also copy assembly features
            feat_crop_idx = np.where(np.concatenate([cropped_mask[c] 
                    for c in selected_chains], axis=0))[0]
            label_crop_idx = np.where(np.concatenate([cropped_mask[c] 
                    for c in selected_chains], axis=0))[0]
            feat, labels = apply_crop_mask_and_pad(
                np_example,
                raw_labels,
                feat_crop_idx,
                label_crop_idx,
                self.crop_size,
                pad_for_shorter_seq=self.is_pad_if_crop,
                for_recycle=False)
        else:
            labels = {k: v for k, v in raw_labels.items()}

        ## check chain orders of sequence features and labels
        assert pipeline_aa.check_chain_orders_of_seq_and_label(np_example, raw_labels), (
            f"The chain order of seq and msa of protein {protein} is wrong")
        
        # 5. extra features
        ## NOTE: get template further features
        feat = pipeline_aa.add_further_assembly_template_feat(feat)
        
        # copy assembly features
        for k in ['asym_id', 'sym_id', 'entity_id', 'residue_index']:
            labels[k] = np.copy(feat[k])
        # copy perm features
        for k in ['perm_atom_index', 'perm_entity_id', 'perm_asym_id']:
            raw_labels[k] = np.copy(np_example[k])
        # build pocket info
        feat['pocket_info'] = self.get_pocket_info(feat, labels)

        # sample a interface type
        # print(f'[DEBUG] protein {protein}, try to get interface_mask')
        interface_type_map = self.interface_config['interface_type_map']
        interface_type = random.choices(
                list(interface_type_map.keys()),
                weights=list(interface_type_map.values()),
                k=1)[0]
        # build interface mask
        interface_mask = self.get_interface_mask(feat, labels, interface_type)
        # print(f'[DEBUG] protein {protein}: got interface_mask, shape: {interface_mask.shape}, type: {interface_type}')
        feat['interface_mask'] = interface_mask
        feat['interface_source'] = self.interface_config['interface_source']
        # build interface info
        # if self.interface_config['interface_source'] in ['gt', 'gen']:
        feat.update(self.get_interface_info(feat, labels, interface_type, interface_mask)) 
        # print(f'[DEBUG] protein {protein}: got interface_info')

        ## Interface related features
        # # sample a interface type
        # interface_type_map = self.interface_config['interface_type_map']
        # interface_type = random.choices(
        #         list(interface_type_map.keys()),
        #         weights=list(interface_type_map.values()),
        #         k=1)[0]
        # # build interface mask
        # interface_mask = self.get_interface_mask(feat, labels, interface_type)
        # feat['interface_mask'] = interface_mask
        # feat['interface_source'] = self.interface_config['interface_source']
        # # build interface info
        # if self.interface_config['interface_source'] in ['gt', 'gen']:
        #     feat.update(self.get_interface_info(feat, labels, interface_type, interface_mask)) 

        # string infomations
        for key in ['all_chain_ids', 'all_ccd_ids', 'all_atom_ids']:
            feat[key] = ' '.join(feat[key])
        for key in ['label_ccd_ids', 'label_atom_ids']:
            labels[key] = ' '.join(labels[key])
            raw_labels[key] = ' '.join(raw_labels[key])

        # 6. filtering
        _, valid_atom_num = np.unique(feat['perm_asym_id'][labels['all_atom_pos_mask'] == 1], 
                return_counts=True)
        if len(valid_atom_num) == 0:
            print(f"Skip {protein} due to no valid chains")
            return None
        if np.max(valid_atom_num) < 3:
            print(
                f"Skip {protein} due to all chains contain less than 3 atoms")
            return None
        ## in case that the gpu memory may exceed limit
        if len(feat['perm_asym_id']) > self.crop_size * 14:
            print(f"Skip {protein} due to atom_num per-token larger than 14: "
                    f"perm_asym_id {len(feat['perm_asym_id'])}")
            return None

        sample = {
            'name': protein,
            'feat': feat,
            'label_cropped': labels,
            'label': raw_labels,
            'chain_ids': ''.join(selected_chains),
            'sample_info': json.dumps(sample_info),
            'unit_structure_file': unit_struct_file,
            'structure_file': assembly_struct_file
        }
        return sample

    def __len__(self):
        return 2024

    def __getitem__(self, index):
        sample = None
        while sample is None:
            items = self.select_protein(index)
            protein = items['protein']
            chain_ids = items['chain_ids']
            try:
                print(f"[AllAtomDataset] [INFO] Try to load {items}")
                sample = self.get_sample(protein, sampled_chain_ids=chain_ids,
                        sampling_method=items['sampling_method'])
            except Exception as ex:
                print(f'[DATA] index {index} prot {items} failed: {ex}')
                import traceback
                traceback.print_exc(file=sys.stdout)
            # index = np.random.randint(self.__len__())
            index = random.randint(0, self.__len__() - 1)
        return sample

# TODO: add interface operations
class AllAtomTestPosebusterDataset(IterableDataset):
    """
        Dataset for test set, PosebusterV1,V2 inference.
        Only use the unit mmcif.
    """

    def __init__(self,
                 model_config,
                 data_config,
                 trainer_id=0,
                 trainer_num=1):
        self.model_config = deepcopy(model_config)
        self.data_config = data_config
        self.trainer_id = trainer_id
        self.trainer_num = trainer_num

        def _assert_exists(path):
            assert exists(path), path

        _assert_exists(self.data_config.protein_list_file)
        _assert_exists(self.data_config.feature_dir)
        _assert_exists(self.data_config.json_dir)
        _assert_exists(self.data_config.ligand_structure_dir)

        if hasattr(self.data_config, 'structure_dir'):
            # test dataset should have this config,
            _assert_exists(self.data_config.structure_dir)

        ## check and filter (bad or not needed) proteins
        self.protein2seq_map = self._load_protein2seq_map()
        self.fasta2id_map = self._load_fasta2id_map()
        self.protein_infos, failed_proteins, failed_codes = self._get_protein_infos()

        self.interface_config = self.data_config.get("interface_config", 
            {"interface_source": 'none',          # 'gen', 'gt' or 'none'
             "interface_type": "protein_ligand", # "protein_ligand", "any_interface", 'none'
             "dist_thres": 5, 
             "interface_sample": {"top_n": 10, "min_m": 5, "max_m": 5, "seed": 1},
             "interface_gen": {"interface_size": 1, "seed": 2}})

        def _print_attribute(key, value):
            print(f'[{self.__class__.__name__}] {key}: {value}')

        _print_attribute('protein_list_file',
                         self.data_config.protein_list_file)
        _print_attribute('feature_dir', self.data_config.feature_dir)
        _print_attribute('json_dir', self.data_config.json_dir)
        _print_attribute('ligand_structure_dir', self.data_config.ligand_structure_dir)
        _print_attribute('trainer_id/trainer_num',
                         f'{self.trainer_id}/{self.trainer_num}')
        _print_attribute('protein_num', len(self.protein_infos))
        _print_attribute('failed_protein_num', len(failed_proteins))
        _print_attribute('failed_proteins', failed_proteins)
        _print_attribute('falied codes with freq', np.unique(list(failed_codes.values()), return_counts=True))

        # pipelines for non_polymer/ conf_bond
        self.pipeline_token_feature = pipeline_token_feature.DataPipeline(
            ccd_preprocessed_path=data_config.ccd_preprocessed_path)
        self.conf_bond_pipeline = pipeline_conf_bonds.DataPipeline(
            ccd_preprocessed_path=data_config.ccd_preprocessed_path)
    
        ccd_preprocessed_dict = {}
        if 'pkl.gz' in data_config.ccd_preprocessed_path:
            with gzip.open(data_config.ccd_preprocessed_path, "rb") as fp:
                ccd_preprocessed_dict = pickle.load(fp)
        elif '.pkl' in data_config.ccd_preprocessed_path:
            with open(data_config.ccd_preprocessed_path, "rb") as fp:
                ccd_preprocessed_dict = pickle.load(fp)
        self.ccd_preprocessed_dict = ccd_preprocessed_dict

    def _load_protein2seq_map(self):
        protein2seq_map = {}
        if "protein_map_file" in self.data_config.keys():
            for line in open(self.data_config.protein_map_file):
                protein, protein_seqid = line.split()
                protein2seq_map[protein] = protein_seqid
        return protein2seq_map

    def _load_fasta2id_map(self):
        fasta2id_map = {}
        if "fasta_map_file" in self.data_config.keys():
            fasta2id_map.update(json.load(open(self.data_config.fasta_map_file)))
        return fasta2id_map

    def _get_protein_infos(self):
        """tbd"""
        with open(self.data_config.protein_list_file) as r:
            proteins_list = r.readlines()[self.trainer_id :: self.trainer_num]
            proteins_list = [p.strip() for p in proteins_list]

        valid_proteins, failed_proteins = [], []
        failed_codes = {}
        for protein_info in proteins_list:
            error_code = self._check_protein(protein_info)
            if not error_code == 0:
                failed_proteins.append(protein_info)
                if not error_code == 1:
                    failed_codes[json.loads(protein_info)["protein_name"]] = error_code
            else:
                valid_proteins.append(protein_info)

        return valid_proteins, failed_proteins, failed_codes

    def _check_protein(self, protein_info):
        # read protein_name and chain_ids from protein
        # protein_info: jsonline in {"protein_name": "PROT", "chains": [{"id": "ID"},
        #  {"id": "ID", "is_ligand": TRUE}, … ]} format

        # check jsonline format
        try: 
            protein_list_item = json.loads(protein_info)
        except: 
            return 1

        protein_name = protein_list_item["protein_name"]
        targeted_chains = [item["id"] for item in protein_list_item["chains"]]

        # Check json_file
        input_json_path = self._get_protein_json_file(protein_name)
        if not exists(input_json_path):
            return 2
        with open(input_json_path) as input_json: 
            input_dict = json.load(input_json)

        if not 'tokenization' in input_dict: return 3
        if not 'basic' in input_dict: return 4

        raw_token_dict = input_dict['tokenization']

        # 1. protein feat
        if "protein" in raw_token_dict:
            seqs = raw_token_dict["protein"]["msa_seqs"]
            chain_ids = raw_token_dict["protein"]["chain_ids"]
            if len(chain_ids) == 0: 
                return 5   

            # filter out chains that are not mentioned in targeted_chains
            seqs = [s for c, s in zip(chain_ids, seqs) if c in targeted_chains]
            chain_ids = [c for c in chain_ids if c in targeted_chains]

            for seq, chain_id in zip(seqs, chain_ids):  
                if not os.path.exists(self._get_protein_feature_file(protein_name + "_" + chain_id, seq=seq)):
                    print(self._get_protein_feature_file(protein_name + "_" + chain_id, seq=seq) + "not exist")
                    return 6
        # 2. RNA/ DNA
        if "dna" in raw_token_dict: 
            pass # TODO(zhukunrui)
        if "rna" in raw_token_dict: 
            pass # TODO(zhukunui)
        # 3. non polymer
        if "ligand" in raw_token_dict: 
            pass  # TODO(zhukunui)
        return 0

    def _get_protein_feature_file(self, protein, seq):
        """tbd."""
        if protein in self.protein2seq_map: 
            protein = self.protein2seq_map[protein]
        if seq in self.fasta2id_map: 
            protein = self.fasta2id_map[seq]
        path = join(self.data_config.feature_dir, protein, 'features.pkl')
        if exists(path):
            return path

        path = os.path.join(self.data_config.feature_dir, protein,
                            'features.pkl.gz')
        return path

    def _get_protein_struct_file(self, protein):
        """tbd."""
        path = join(self.data_config.structure_dir, f"{protein}.cif")
        if exists(path):
            return path
        path = join(self.data_config.structure_dir, f"{protein}.cif.gz")
        if exists(path):
            return path
        path = join(self.data_config.structure_dir, f"{protein}.pdb")
        if exists(path):
            return path
        path = join(self.data_config.structure_dir, f"{protein}.pdb.gz")
        return path

    def _get_protein_json_file(self, protein):
        """tbd."""
        return join(self.data_config.json_dir, f"{protein}.json")

    def get_protein_feat(self, protein, seqs, chain_info_list, random_seed):
        """tbd."""
        # load from single chain and combine 
        all_chain_features = {}
        is_ligand_chain = {}
        all_chain_atom_pos = {}
        all_chain_atom_mask = {}
        for seq, chain_info in zip(seqs, chain_info_list):
            # chain_info {"id": "ID", "is_ligand": TRUE}
            chain_id = chain_info["id"]
            features_pkl = self._get_protein_feature_file(protein + '_' + chain_id, seq=seq)
            if features_pkl.endswith('.pkl.gz'):
                with gzip.open(features_pkl, 'rb') as pkl:
                    raw_features = pickle.load(pkl)
            else:
                with open(features_pkl, 'rb') as pkl:
                    raw_features = pickle.load(pkl)
            ############
            ## TODO: check why the key name are different
            if 'template_all_atom_mask' in raw_features:
                raw_features['template_all_atom_masks'] = raw_features.pop('template_all_atom_mask')
            ############

            if 'msa_sample' in self.data_config:
                raw_features = sample_raw_msa(raw_features, self.data_config)

            all_chain_features[chain_id] = raw_features

        raw_features = pipeline_multimer.process_with_all_chain_features(all_chain_features)
        return raw_features

    def get_label(self, protein, mmcif_object, chain_ids):
        ''' tbd. '''
        label_list = []
        offset = 0
        for chain_id in chain_ids:
            try:
                protein_chain_d = label_utils.get_position_label(
                        mmcif_object_unit=mmcif_object, 
                        mmcif_object_assembly=None,
                        ccd_preprocessed_dict=self.ccd_preprocessed_dict,
                        mmcif_chain_id=chain_id, check_release_date=False)
            except Exception as exception:
                print(f"Exception in loading label for {protein} {chain_id}")
                return None

            if protein_chain_d is None: 
                return None

            protein_chain_d = protein_chain_d[chain_id]

            for label_key in LABEL_NEED_OFFSET_KEYS:
               protein_chain_d[label_key] += offset
            offset += protein_chain_d['all_atom_pos'].shape[0]
            label_list.append(protein_chain_d)

        raw_labels = {}
        if len(label_list) == 0: 
            return {}
        for key in label_list[0].keys():
            raw_labels[key] = np.concatenate([l[key] for l in label_list],
                                             axis=0)
        return raw_labels
    
    def _get_chain_ca_pos(self, feats):
        gt_pos = feats['all_atom_pos']
        gt_pos_mask = feats['all_atom_pos_mask'].astype('bool')
        centra_token = feats['all_centra_token_indice']
        centra_token_mask = feats['all_centra_token_indice_mask'].astype('bool')
        gt_ca_pos = gt_pos[centra_token][centra_token_mask]
        return gt_ca_pos
 
    def _select_closet_chainID(self, mmcif_object, chain_id_infos, protein, theta):
        """tdb"""
        
        def _get_ca_distance_matrix(ca_pos, ligand_pos, theta):
            """tbd"""
            # find protein ca positions that should be kept in the pocket.
            dis_matrix = distance_matrix(ligand_pos, ca_pos)
            pro_ca_list = dis_matrix.min(axis=0) <= theta
            pro_ca_index = np.where(pro_ca_list)[0]
            return len(pro_ca_index) >= 3 # large 3 is enough to keep the pocket
            
        
        lig_chain_ids = [ chainid for chainid, item in chain_id_infos.items() if item.get('is_assigned_ligand', False)]
        assert len(lig_chain_ids) == 1, f"{protein} is not support more than one is_assigned_ligand chain"
        lig_chain_id = lig_chain_ids[0]
        others_chain_id_infos = chain_id_infos.copy()
        others_chain_id_infos.pop(lig_chain_id)

        liagnd_chain_d = label_utils.get_position_label(mmcif_object, 
                                    ccd_preprocessed_dict=self.ccd_preprocessed_dict,
                                    mmcif_chain_id=lig_chain_id, check_release_date=False)
        ligand_ca_pos = self._get_chain_ca_pos(liagnd_chain_d[lig_chain_id])

        select_chain_id_info = {}
        for chain_id, item in others_chain_id_infos.items():
            try:
                protein_chain_d = label_utils.get_position_label(mmcif_object, 
                                ccd_preprocessed_dict=self.ccd_preprocessed_dict,
                                mmcif_chain_id=chain_id, check_release_date=False)

                ca_pos = self._get_chain_ca_pos(protein_chain_d[chain_id])

                if _get_ca_distance_matrix(ca_pos, ligand_ca_pos, theta):
                    select_chain_id_info[chain_id] = item

            except Exception as exception:
                print(f"Exception in loading label for {protein} {chain_id}")
                pass
        
        select_chain_id_info[lig_chain_id] = chain_id_infos[lig_chain_id]
        return select_chain_id_info
    
    def get_pocket_info(self, feat, label):
        pocket_type = self.data_config.get('pocket_type', 'centra_10A')
        if pocket_type == 'centra_10A':
            pocket_info = build_pocket_info(
                    feat['asym_id'], 
                    feat['is_ligand'], 
                    label['all_atom_pos'][label['all_centra_token_indice']], 
                    label['all_centra_token_indice_mask'])
        elif pocket_type == 'top20_rand5':
            pocket_info = build_pocket_info_with_random_residues(
                    feat['asym_id'], 
                    feat['is_ligand'], 
                    label['all_atom_pos'][label['all_centra_token_indice']], 
                    label['all_centra_token_indice_mask'],
                    top_n=20,
                    m=5)
        else:
            raise ValueError(pocket_type)
        return pocket_info


    def get_interface_mask(self, feat, label, interface_type, chain_info_list):
        if interface_type == "protein_ligand":
            binder_mask = feat['is_ligand']
            target_mask = None
        else:
            binder_mask = None
            target_mask = None

        interface_mask = build_possible_interface_mask(
            feat['asym_id'], 
            label['all_centra_token_indice_mask'], 
            binder_mask=binder_mask, 
            target_mask=target_mask)

        return interface_mask


    def get_interface_info(self, feat, label, interface_type, interface_mask):

        if self.interface_config['interface_gen']['seed'] is not None:
            interface_gen_seed = self.interface_config['interface_gen']['seed']
        else:
            interface_gen_seed = random.randint(1, 10000)

        interface_gen_temperature = self.interface_config['interface_gen'].get('temperature', 1.0)
        interface_sampling_method = self.interface_config['interface_gen'].get('sampling_method', 'in_repeats')
        interface_sampling_beam = self.interface_config['interface_gen'].get('sampling_beam', 10)

        interface_info_dict = {
            'interface_source': self.interface_config['interface_source'],
            'interface_gen_size': self.interface_config['interface_gen']['interface_size'],
            'interface_gen_repeats': self.interface_config['interface_gen']['gen_repeats'],
            'interface_gen_seed': interface_gen_seed,
            'interface_gen_temperature': interface_gen_temperature,
            'interface_sampling_method': interface_sampling_method,
            'interface_sampling_beam': interface_sampling_beam
        }

        if interface_type in ['any_interface', 'protein_ligand']:
            # random select m between min_m to max_m
            m = random.randint(self.interface_config['interface_sample']['min_m'], 
                            self.interface_config['interface_sample']['max_m'])
            seed = self.interface_config['interface_sample'].get('seed', None)
            ret_dict = build_interface_info_with_random_residues(
                    feat['asym_id'], 
                    feat['entity_id'], 
                    atom_pos=label['all_atom_pos'], 
                    atom_mask=label['all_atom_pos_mask'],
                    atom_to_token_mapping=feat['ref_token2atom_idx'],
                    token_pos=label['all_atom_pos'][label['all_centra_token_indice']],
                    token_pos_mask=label['all_centra_token_indice_mask'],
                    top_n=self.interface_config['interface_sample']['top_n'],
                    m=m,
                    interface_mask=interface_mask,
                    dist_thres=self.interface_config['dist_thres'],
                    dist_type=self.interface_config.get('dist_type', 'heavy_atom'),
                    seed=seed)
            interface_info_dict.update(ret_dict)
        elif interface_type in ['none', None]:
            n_token = len(feat['asym_id'])
            interface_info_dict.update({
                "interface_info_sample": np.zeros([n_token, n_token], dtype=np.float32),
                "interface_info_full": np.zeros([n_token, n_token], dtype=np.float32),
                "interface_info_mix": np.zeros([n_token, n_token], dtype=np.float32),
            })
        else:
            raise ValueError(interface_type)

        return interface_info_dict

        
    def get_sample(self, protein_info, random_seed=None, 
            theta=10, check_max_seq_len=True, skip_pocket_info=False):
        """tbd."""
        # read protein_name and assigned chain_ids from protein
        # protein_info: jsonline in {"protein_name": "PROT", "chains": [{"id": "ID"}, 
        #               {"id": "ID", "is_ligand": TRUE}, … ]} format
        protein_list_item = json.loads(protein_info)
        protein_name = protein_list_item["protein_name"]
        targeted_chains = [item["id"] for item in protein_list_item["chains"]]
        chain_ids_to_info = dict(zip([item["id"] for item in protein_list_item["chains"]],
                                     [item for item in protein_list_item["chains"]]))
        
        # 1. Read chain info from json file
        input_json_path = self._get_protein_json_file(protein_name)
        with open(input_json_path) as input_json: 
            input_dict = json.load(input_json)

        all_chain_ids = []
        chain_id_2_num_token = {}
        basic_token_dict = input_dict['basic']
        raw_token_dict = input_dict['tokenization']
        # NOTE: dtype must be protein, dna, rna, non_polymer
        raw_token_dict = {seq_id:raw_token_dict[seq_id] 
                                for seq_id in ['protein', 'dna', 'rna', 'non_polymer', 'ligand'] if seq_id in raw_token_dict}
        basic_token_dict = {seq_id:basic_token_dict[seq_id] 
                                for seq_id in ['protein', 'dna', 'rna', 'non_polymer', 'ligand'] if seq_id in basic_token_dict}
        msa_chain_info = collections.OrderedDict()
        ccd_chain_info = collections.OrderedDict()
        for dtype, raw_values in raw_token_dict.items():
            msa_seqs = raw_values['msa_seqs']  ## for msa search and template matching.
            chain_ids = raw_values['chain_ids']
            for msa_seq, chain_id in zip(msa_seqs, chain_ids):
                msa_chain_info[f'{dtype}_{chain_id}'] = msa_seq

        for dtype, raw_values in basic_token_dict.items():
            ccd_seqs = raw_values['seqs']
            chain_ids = raw_values['chain_ids']
            for ccd_seq, chain_id in zip(ccd_seqs, chain_ids):
                ccd_parsed = parsers.parse_ccd_fasta(ccd_seq) # str: list
                ccd_chain_info[f'{dtype}_{chain_id}'] = ccd_parsed
                chain_id_2_num_token[chain_id] = len(pipeline_aa_utils.ccd_list_to_token(ccd_parsed, 
                        self.ccd_preprocessed_dict))
                all_chain_ids.append(chain_id)

        # try to load structure object
        if hasattr(self.data_config, 'structure_dir'):
            protein_struct_file = self._get_protein_struct_file(protein_name)
            if protein_struct_file.endswith('.cif.gz'):
                with gzip.open(protein_struct_file, 'r') as f:
                    cif_string = f.read().decode('utf8')
            else:
                cif_string = "".join(open(protein_struct_file, 'r').readlines())

            parse_result = mmcif_parsing.parse(file_id=protein_struct_file, mmcif_string=cif_string)
            mmcif_object = parse_result.mmcif_object
            selected_chain_ids = self._select_closet_chainID(
                    mmcif_object, chain_ids_to_info, protein_name, theta=theta)
            if len(selected_chain_ids) < 2:
                print(f'Skip {protein_name} due to no valid protein-ligand selected_chain_ids')
                return None 
        else:
            print(f'Skip {protein_name} due to no structure dir')
            return None 

        # 2. Feats collection
        all_feats = {}
        selected_chains = []
        chain_info_list = []
        targeted_chains = [  # keep the same chain order with input_dict
            c for c in selected_chain_ids if c in targeted_chains
        ]
        # 2.1 non_polymer
        all_feats['seq_token'] = self.pipeline_token_feature.process(
            unit_json_path=input_json_path,
            assembly_json_path=None,
            select_mmcif_chainID=targeted_chains,
            ccd_preprocessed_dict=self.ccd_preprocessed_dict)
                    
        # 2.1.1 Check full sequence len
        full_seq_len = all_feats['seq_token']['token_index'].shape[0]
        if check_max_seq_len and full_seq_len > MAX_SEQ_SIZE:
            raise Exception(f"Test sample({protein_name}) \
                    seq_length({full_seq_len}) exceeded {MAX_SEQ_SIZE}, skip")

        # 2.2 conf_bond
        all_feats['conf_bond'] = self.conf_bond_pipeline.process(
                unit_json_path=input_json_path,
                assembly_json_path=None, 
                select_mmcif_chainID=targeted_chains,
                ccd_preprocessed_dict=self.ccd_preprocessed_dict)
        
        # 2.3 Protein feats
        if random_seed is None:
            random_seed = np.random.randint(10000000)        
        
        if 'protein' in raw_token_dict:
            msa_seqs = raw_token_dict["protein"]["msa_seqs"]
            prot_chain_ids = raw_token_dict["protein"]["chain_ids"]

            # filter out chains that are not mentioned in targeted_chains
            msa_seqs = [s for c, s in zip(prot_chain_ids, msa_seqs) if c in targeted_chains]
            prot_chain_ids = [c for c in prot_chain_ids if c in targeted_chains]
            
            # add group info to chain_ids
            chain_info_list.extend([selected_chain_ids[c] for c in prot_chain_ids])
            selected_chains.extend(prot_chain_ids)

            all_feats["protein"] = self.get_protein_feat(protein_name, seqs=msa_seqs, 
                chain_info_list=chain_info_list, random_seed=random_seed)
            all_feats['protein']["ccd_seqs"] = np.concatenate([np.array(v, dtype=object)\
                    for k, v in ccd_chain_info.items() \
                    if 'protein' in k and k.split('_')[-1] in prot_chain_ids])
                        
        if "ligand" in raw_token_dict:
            ligand_chain_ids = basic_token_dict["ligand"]["chain_ids"]
            ligand_chain_ids = [c for c in ligand_chain_ids if c in targeted_chains]
            if len(ligand_chain_ids) > 0:
                all_feats['ligand'] = {}
                all_feats['ligand']["ccd_seqs"] = np.concatenate([np.array(v, dtype=object)\
                        for k, v in ccd_chain_info.items() \
                        if 'ligand' in k and k.split('_')[-1] in ligand_chain_ids])
                chain_info_list.extend([selected_chain_ids[c] for c in ligand_chain_ids])
                selected_chains.extend(ligand_chain_ids)
    
        # 2.5 post convert
        np_example = pipeline_hybrid._post_convert(self.ccd_preprocessed_dict, all_feats)
        ## NOTE: get template further features
        np_example = pipeline_hybrid.make_pseudo_beta(np_example, prefix='template_')
        np_example = pipeline_hybrid.make_template_further_feature(np_example)

        np_example["seq_mask"] = np.ones_like(
            np_example['restype']).astype('float32')
        np_example["msa"], np_example['msa_mask'], np_example['deletion_matrix'] = crop_msa(np_example)

        for key in ['all_chain_ids', 'all_ccd_ids','all_atom_ids' ]:
            if key in np_example:
                np_example[key] = ' '.join(np_example[key])

        sample = {
            'name': protein_name,
            'feat': np_example,
            'label_cropped': {},
            'label': {},
            'chain_info_list': json.dumps(chain_info_list), 
        }

        # 3. get labels
        if hasattr(self.data_config, 'structure_dir'):   
            if mmcif_object is None:
                print(f'=> Skip [{protein_struct_file}] due to mmcif parsing error. Got [{parse_result.errors}]')
                return sample
                
            raw_labels = self.get_label(protein_name, mmcif_object=mmcif_object,
                chain_ids=selected_chains)

            assert raw_labels is not None, "failed to load label for all targeted chains"

            ## copy assembly features
            for k in ['asym_id', 'sym_id', 'entity_id', 'residue_index']:
                raw_labels[k] = np.array(np_example[k])

            # copy perm features
            for k in ['perm_atom_index', 'perm_entity_id', 'perm_asym_id']:
                raw_labels[k] = np.copy(np_example[k])

            if raw_labels == {}: 
                return sample

            if np_example['ref_pos'].shape[0] != raw_labels['all_atom_pos'].shape[0]:
                print(f'Skip {protein_name} due to inequal atom num')
                return None 
        
            raw_labels['resolution'] = np.array(
            [np.mean(raw_labels['resolution'])])

            for key in ['release_date','label_ccd_ids','label_atom_ids']:
                if key in raw_labels:
                    raw_labels[key] = ' '.join(raw_labels[key])

            sample['label'] = raw_labels
        
        # build pocket info
        if not skip_pocket_info and self.data_config.get('add_pocket_info', False):
            sample['feat']['pocket_info'] = self.get_pocket_info(np_example, raw_labels)
        
        ## Interface related features
        # get interface type
        interface_type = self.interface_config['interface_type']
        print(f'[DEBUG-interface]: interface_type: {interface_type}')
        # build interface mask
        interface_mask = self.get_interface_mask(np_example, raw_labels, interface_type, protein_list_item['chains'])
        np_example['interface_mask'] = interface_mask
        np_example['interface_source'] = self.interface_config['interface_source']
        print(f'[DEBUG-interface]: interface_mask sum: {interface_mask.sum()}')

        # build interface info
        # if self.interface_config['interface_source'] in ['gt', 'gen']:
        np_example.update(self.get_interface_info(np_example, raw_labels, interface_type, interface_mask)) 
        interface_info_full = np_example['interface_info_full']
        print(f'[DEBUG-interface]: interface_info_full sum: {interface_info_full.sum()}')

        # 4. get ligand structure
        lig_chain_ids = [item['id'] for item in protein_list_item['chains']
                if item.get('is_assigned_ligand', False)]
        assert len(lig_chain_ids) == 1
        lig_chain_id = lig_chain_ids[0]
        sdf_files = glob(
                f'{self.data_config.ligand_structure_dir}/{protein_name}_{lig_chain_id}*.sdf')
        assert len(sdf_files) == 1
        sample['ligand_structure_file'] = sdf_files[0]

        return sample

    def __iter__(self):
        """tbd"""
        for x in self.protein_infos:
            try:
                protein_name = json.loads(x)['protein_name']
                print(f"[AllAtomTestPosebusterDataset] [INFO] Try to load {protein_name}......")
                sample = self.get_sample(x)
                ## Deal with posebuster_v1 special case
                if json.loads(x)['protein_name'] == '7suc':
                    sample = self.get_sample(x, theta=6)
                elif json.loads(x)['protein_name'] == '7b2c':
                    sample = self.get_sample(x, theta=5)
                ## get full label for permutation_align
                if json.loads(x)['protein_name'] == '8f4j':
                    sample_full = sample
                else:
                    sample_full = self.get_sample(x, theta=300, 
                            check_max_seq_len=False, skip_pocket_info=True)
                sample['label_full'] = sample_full['label']
                # align entity_id and asym_id for permutation_align
                new_entity_id, new_asym_id = align_entity_id_by_ccd_id(
                        sample['label'], sample['label_full'])
                sample['label_full']['perm_entity_id'] = new_entity_id
                sample['label_full']['perm_asym_id'] = new_asym_id
                yield sample
            except Exception as ex:
                import traceback
                traceback.print_exc(file=sys.stdout)
                print(f'[DATA] prot {x} failed: {ex}')


class AllAtomTestDataset(IterableDataset):
    """
        Dataset for test set, RecentDNA, RecentRNA
        Use assembly json.
    """

    def __init__(self,
                 model_config,
                 data_config,
                 trainer_id=0,
                 trainer_num=1):
        self.model_config = deepcopy(model_config)
        self.data_config = data_config
        self.trainer_id = trainer_id
        self.trainer_num = trainer_num

        def _assert_exists(path):
            assert exists(path), path

        _assert_exists(self.data_config.protein_list_file)
        _assert_exists(self.data_config.feature_dir)
        _assert_exists(self.data_config.json_dir)
        _assert_exists(self.data_config.structure_dir)
        self.use_assembly = self.data_config.use_assembly
        if self.use_assembly:
            _assert_exists(self.data_config.assembly_json_dir)
            _assert_exists(self.data_config.assembly_structure_dir)

        ## check and filter (bad or not needed) proteins
        self.rna2seq_map = self._load_rna2seq_map()
        self.fasta2id_map = self._load_fasta2id_map()
        self.protein_infos, failed_proteins, failed_codes = self._get_protein_infos()

        self.interface_config = self.data_config.get("interface_config", 
            {"interface_source": 'none',          # 'gen', 'gt' or 'none'
             "interface_type": "any_interface", # "any_interface", "epitope_paratope", 'none'
             "dist_thres": 5, 
             "interface_sample": {"top_n": 10, "min_m": 5, "max_m": 5, "seed": 1},
             "interface_gen": {"interface_size": 1, "seed": 2}})

        def _print_attribute(key, value):
            print(f'[{self.__class__.__name__}] {key}: {value}')

        _print_attribute('protein_list_file',
                         self.data_config.protein_list_file)
        _print_attribute('use_assembly', self.use_assembly)
        _print_attribute('feature_dir', self.data_config.feature_dir)
        _print_attribute('json_dir', self.data_config.json_dir)
        _print_attribute('trainer_id/trainer_num',
                         f'{self.trainer_id}/{self.trainer_num}')
        _print_attribute('protein_num', len(self.protein_infos))
        _print_attribute('failed_protein_num', len(failed_proteins))
        _print_attribute('failed_proteins', failed_proteins)
        _print_attribute('falied codes with freq', np.unique(list(failed_codes.values()), return_counts=True))
        _print_attribute('interface_config', self.interface_config)

        self.ccd_preprocessed_dict = read_ccd_preprocessed_dict(
                data_config.ccd_preprocessed_path)
        self.num_epitope = data_config.get("num_epitope", 0)

    def _load_fasta2id_map(self):
        fasta2id_map = {}
        if "fasta_map_file" in self.data_config.keys():
            fasta2id_map.update(json.load(open(self.data_config.fasta_map_file)))
        return fasta2id_map

    def _load_rna2seq_map(self):
        rna2seq_map = {}
        if "rna_map_file" in self.data_config.keys():
            for line in open(self.data_config.rna_map_file):
                rna, rna_seqid = line.split()
                rna2seq_map[rna] = rna_seqid
        return rna2seq_map

    def _get_protein_infos(self):
        """tbd"""
        with open(self.data_config.protein_list_file) as r:
            proteins_list = r.readlines()[self.trainer_id :: self.trainer_num]
            proteins_list = [p.strip() for p in proteins_list]

        valid_proteins, failed_proteins = [], []
        failed_codes = {}
        for protein_info in proteins_list:
            error_code = self._check_protein(protein_info)
            if not error_code == 0:
                failed_proteins.append(protein_info)
                if not error_code == 1:
                    failed_codes[json.loads(protein_info)["protein_name"]] = error_code
            else:
                valid_proteins.append(protein_info)

        return valid_proteins, failed_proteins, failed_codes

    def _get_protein_json_file(self, protein):
        """tbd."""
        json_file = join(self.data_config.json_dir, f"{protein}.json")
        if self.use_assembly:
            return (json_file,
                    join(self.data_config.assembly_json_dir, f"{protein}-assembly1.json"))
        else:
            return json_file, None


    def _get_protein_struct_file(self, protein):
        """tbd."""
        path = join(self.data_config.structure_dir, f"{protein}.cif")
        gz_path = join(self.data_config.structure_dir, f"{protein}.cif.gz")
        unit_path = path if exists(path) else gz_path
        if self.use_assembly:
            assembly_path = join(self.data_config.assembly_structure_dir, f"{protein}-assembly1.cif")
            gz_assembly_path = join(self.data_config.assembly_structure_dir, f"{protein}-assembly1.cif.gz")
            assembly_path = path if exists(assembly_path) else gz_assembly_path
            return unit_path, assembly_path
        else:
            return unit_path, None
    
    def _get_protein_feature_file(self, protein, seq):
        """tbd."""
        if seq in self.fasta2id_map: 
            protein = self.fasta2id_map[seq]
        path = join(self.data_config.feature_dir, protein, 'features.pkl')
        if exists(path):
            return path
        path = os.path.join(self.data_config.feature_dir, 
                protein, 'features.pkl.gz')
        return path

    def _get_rna_feature_file(self, rna, seq):
        """tbd."""
        if rna in self.rna2seq_map: 
            rna = self.rna2seq_map[rna]
        path = join(self.data_config.rna_feature_dir, rna, 'features.pkl')
        if exists(path):
            return path
        path = os.path.join(self.data_config.rna_feature_dir, 
                rna, 'features.pkl.gz')
        return path

    def get_seq_msa_feats(self, protein, selected_chain_ids, all_chain_info_dict, coval_bonds_info):
        """get_msa_feats"""
        def _load_pro_rna_feature_pkl(protein, chain_type, chain_id, seq):
            assert chain_type in ['protein', 'rna']
            if chain_type == 'rna':
                features_pkl = self._get_rna_feature_file(
                        protein + "_" + chain_id.split('_')[0], seq=seq)
            else:
                features_pkl = self._get_protein_feature_file(
                        protein + '_' + chain_id.split('_')[0], seq=seq)
            if not os.path.exists(features_pkl): 
                return None
            else:
                if features_pkl.endswith('.pkl.gz'):
                    with gzip.open(features_pkl, 'rb') as pkl:
                        raw_features = pickle.load(pkl)
                else:
                    with open(features_pkl, 'rb') as pkl:
                        raw_features = pickle.load(pkl)
            return raw_features

        ## 1. get features for each chain
        ## protein chains failed to load feature_pkl will be dropped.
        all_chain_features = {}
        for chain_id in selected_chain_ids:
            chain_type = all_chain_info_dict[chain_id]['chain_type']
            is_prot_rna = chain_type in ['protein', 'rna']
            if is_prot_rna:
                seq = all_chain_info_dict[chain_id]['msa_seq']  # as the key to query the msa feature pkl
                raw_features = _load_pro_rna_feature_pkl(
                        protein, chain_type, chain_id, seq)
                if raw_features is None and chain_type == 'protein':
                    continue
            else:
                raw_features = None           
            all_chain_features[chain_id] = raw_features

         ## 2. msa pairing and merge
        msa_features, new_chain_ids = pipeline_aa.process_with_all_msa_chain_features(
                all_chain_features, 
                all_chain_info_dict, 
                self.ccd_preprocessed_dict)
        
       ## 3. get sequence features
        seq_features = pipeline_aa.get_assembly_sequence_features(
                new_chain_ids, 
                all_chain_info_dict, 
                coval_bonds_info,
                self.ccd_preprocessed_dict)
        
        ## 4. combine sequence and msa features
        seq_msa_feats = pipeline_aa.combine_assembly_seq_and_msa_features(
                seq_features, msa_features)
        return seq_msa_feats, new_chain_ids

    def _check_protein(self, protein_info):
        """
        protein_info: jsonline in {"protein_name": "PROT", "chains": [{"id": "ID"},
         {"id": "ID", "is_ligand": TRUE}, … ]} format
        """
        item = json.loads(protein_info)
        protein = item["protein_name"]
        targeted_chains = [chain["id"] for chain in item["chains"]]

        # Check json_file
        unit_json_path, assembly_json_path = self._get_protein_json_file(protein)
        if not exists(unit_json_path):
            return 1 
        if self.use_assembly and not exists(assembly_json_path):
            return 2
        json_dict = json.load(open(assembly_json_path 
                if self.use_assembly else unit_json_path))
        if not 'tokenization' in json_dict: return 3
        if not 'basic' in json_dict: return 4

        # check protein feature_pkl
        all_chain_info_dict = assembly_json_dict_to_chain_dict(json_dict)
        for chain_id in targeted_chains:
            if not chain_id in all_chain_info_dict:
                print(f'[AllAtomTestDataset] chain {chain_id} not found in sample {protein_info}')
                return 5
            if not all_chain_info_dict[chain_id]['chain_type'] == 'protein':
                continue
            seq = all_chain_info_dict[chain_id]['msa_seq']
            features_pkl = self._get_protein_feature_file(
                    protein + '_' + chain_id.split('-')[0], seq=seq)
            if not os.path.exists(features_pkl): 
                return 6

        # Check structure file
        unit_struct_path, assembly_struct_path = self._get_protein_struct_file(protein)
        if not exists(unit_struct_path):
            return 7
        if self.use_assembly and not exists(assembly_struct_path):
            return 8
        return 0

    def get_epitope_info(self, feat, label, chain_info_list):
        """ Get epitope info like pocket info. Randomly select several residues. """
        n_token = len(feat['asym_id'])
        none_pocket_info = np.zeros([n_token, n_token], dtype=np.float32)
        
        antigen_cids = []
        antibody_cids = []
        for cid, chain_info in enumerate(chain_info_list):
            if chain_info.get('is_ligand', False):
                antibody_cids.append(cid)
            else:
                antigen_cids.append(cid)

        if len(antibody_cids) == 0:
            # speed up: skip non-abag test or no-antibody cases
            return none_pocket_info
        
        token_mask = label['all_centra_token_indice_mask']
        is_antigen = np.zeros_like(token_mask)
        for cid in antigen_cids:
            is_antigen = np.logical_or(is_antigen, feat['asym_id'] == cid)
        is_antigen = np.logical_and(is_antigen, token_mask)

        is_antibody = np.zeros_like(token_mask)
        for cid in antibody_cids:
            is_antibody = np.logical_or(is_antibody, feat['asym_id'] == cid)
        is_antibody = np.logical_and(is_antibody, token_mask)

        epitope_info = build_pocket_info_with_random_residues(
                asym_id=feat['asym_id'],
                binder_mask=is_antibody,
                target_mask=is_antigen,
                token_pos=label['all_atom_pos'][label['all_centra_token_indice']], 
                token_pos_mask=token_mask,
                top_n=-1,
                m=self.num_epitope,
                seed=11451,
            )
        
        return epitope_info

    def get_interface_mask(self, feat, label, interface_type, chain_info_list):
        if interface_type == 'epitope_paratope':
            # select interface between antigen and antibody
            antigen_cids = []
            antibody_cids = []
            for cid, chain_info in enumerate(chain_info_list):
                if chain_info.get('is_ligand', False):
                    antibody_cids.append(cid)
                else:
                    antigen_cids.append(cid)

            if len(antibody_cids) == 0:
                n_token = len(feat['asym_id'])
                interface_mask = np.zeros([n_token, n_token], dtype=np.float32)
                return interface_mask

            token_mask = label['all_centra_token_indice_mask']
            is_antigen = np.zeros_like(token_mask)
            for cid in antigen_cids:
                is_antigen = np.logical_or(is_antigen, feat['asym_id'] == cid+1)
            is_antigen = np.logical_and(is_antigen, token_mask)

            is_antibody = np.zeros_like(token_mask)
            for cid in antibody_cids:
                is_antibody = np.logical_or(is_antibody, feat['asym_id'] == cid+1)
            is_antibody = np.logical_and(is_antibody, token_mask)

            interface_mask = build_possible_interface_mask(
                feat['asym_id'], 
                token_mask, 
                binder_mask=is_antibody, 
                target_mask=is_antigen)

        else: 
            interface_mask = build_possible_interface_mask(
                feat['asym_id'], 
                label['all_centra_token_indice_mask'], 
                binder_mask=None, 
                target_mask=None)

        return interface_mask


    def get_interface_info(self, feat, label, interface_type, interface_mask):

        if self.interface_config['interface_gen']['seed'] is not None:
            interface_gen_seed = self.interface_config['interface_gen']['seed']
        else:
            interface_gen_seed = random.randint(1, 10000)

        interface_gen_temperature = self.interface_config['interface_gen'].get('temperature', 1.0)
        interface_sampling_method = self.interface_config['interface_gen'].get('sampling_method', 'in_repeats')
        interface_sampling_beam = self.interface_config['interface_gen'].get('sampling_beam', 10)

        interface_info_dict = {
            'interface_source': self.interface_config['interface_source'],
            'interface_gen_size': self.interface_config['interface_gen']['interface_size'],
            'interface_gen_repeats': self.interface_config['interface_gen']['gen_repeats'],
            'interface_gen_seed': interface_gen_seed,
            'interface_gen_temperature': interface_gen_temperature,
            'interface_sampling_method': interface_sampling_method,
            'interface_sampling_beam': interface_sampling_beam
        }

        if interface_type in ['any_interface', 'epitope_paratope']:
            # random select m between min_m to max_m
            m = random.randint(self.interface_config['interface_sample']['min_m'], 
                            self.interface_config['interface_sample']['max_m'])
            seed = self.interface_config['interface_sample'].get('seed', None)
            ret_dict = build_interface_info_with_random_residues(
                    feat['asym_id'], 
                    feat['entity_id'], 
                    atom_pos=label['all_atom_pos'], 
                    atom_mask=label['all_atom_pos_mask'],
                    atom_to_token_mapping=feat['ref_token2atom_idx'],
                    token_pos=label['all_atom_pos'][label['all_centra_token_indice']],
                    token_pos_mask=label['all_centra_token_indice_mask'],
                    top_n=self.interface_config['interface_sample']['top_n'],
                    m=m,
                    interface_mask=interface_mask,
                    dist_thres=self.interface_config['dist_thres'],
                    dist_type=self.interface_config.get('dist_type', 'heavy_atom'),
                    seed=seed)
            interface_info_dict.update(ret_dict)
        elif interface_type in ['none', None]:
            n_token = len(feat['asym_id'])
            interface_info_dict.update({
                "interface_info_sample": np.zeros([n_token, n_token], dtype=np.float32),
                "interface_info_full": np.zeros([n_token, n_token], dtype=np.float32),
                "interface_info_mix": np.zeros([n_token, n_token], dtype=np.float32),
            })
        else:
            raise ValueError(interface_type)

        return interface_info_dict

    
    def get_sample(self, protein_info):
        """tbd."""
        item = json.loads(protein_info)
        protein = item["protein_name"]
        targeted_chains = [chain["id"] for chain in item["chains"]]
        
        # 0. Read chain info from json file
        unit_json_path, assembly_json_path = self._get_protein_json_file(protein)
        unit_json_dict = json.load(open(unit_json_path))
        if self.use_assembly:
            assembly_json_dict = json.load(open(assembly_json_path))
            coval_bonds_info = pipeline_aa_utils.bond_convert_unit_to_assembly(
                    unit_json_dict.get('covalent_bonds', []), assembly_json_dict['chain_mapping'])
        else:
            coval_bonds_info = []
        
        all_chain_info_dict = assembly_json_dict_to_chain_dict(
                assembly_json_dict if self.use_assembly else unit_json_dict, 
                self.ccd_preprocessed_dict)

        # 1. load mmcif object
        unit_struct_file, assembly_struct_file = self._get_protein_struct_file(protein)
        if self.use_assembly:
            mmcif_object = read_cif_to_mmcif_object(assembly_struct_file)
        else:
            mmcif_object = read_cif_to_mmcif_object(unit_struct_file)
        assert not mmcif_object is None, (
            f'=> Skip [{protein}] due to all mmcif parsing error.')

        # 3. Feats and labels
        np_example, selected_chains = self.get_seq_msa_feats(
                protein, 
                targeted_chains, 
                all_chain_info_dict, 
                coval_bonds_info)
        # 3.1. extra features
        ## NOTE: get template further features
        np_example = pipeline_aa.add_further_assembly_template_feat(np_example)
        assert set(selected_chains) == set(targeted_chains)
        raw_labels = pipeline_aa.get_assembly_label(
                selected_chains, 
                mmcif_object,
                self.ccd_preprocessed_dict)
        assert not raw_labels is None
        assert pipeline_aa.check_atom_nums_of_seq_and_label(
                np_example, raw_labels), (
                f"The atom num of seq and msa of protein {protein} is inequal")

        ## check chain orders of sequence features and labels
        assert pipeline_aa.check_chain_orders_of_seq_and_label(np_example, raw_labels), (
            f"The chain order of seq and msa of protein {protein} is wrong")

        # mark
        np_example['pocket_info'] = self.get_epitope_info(np_example, raw_labels, item['chains'])

        ## Interface related features
        # get interface type
        interface_type = self.interface_config['interface_type']
        # build interface mask
        interface_mask = self.get_interface_mask(np_example, raw_labels, interface_type, item['chains'])
        np_example['interface_mask'] = interface_mask
        np_example['interface_source'] = self.interface_config['interface_source']
        # build interface info
        # if self.interface_config['interface_source'] in ['gt', 'gen']:
        np_example.update(self.get_interface_info(np_example, raw_labels, interface_type, interface_mask)) 

        # copy assembly features
        for k in ['asym_id', 'sym_id', 'entity_id', 'residue_index']:
            raw_labels[k] = np.copy(np_example[k])
        np_example['atom_count'] = np.bincount(np_example['ref_token2atom_idx'])
        # copy perm features
        for k in ['perm_atom_index', 'perm_entity_id', 'perm_asym_id']:
            raw_labels[k] = np.copy(np_example[k])
        # string infomations
        for key in ['all_chain_ids', 'all_ccd_ids', 'all_atom_ids']:
            np_example[key] = ' '.join(np_example[key])
        for key in ['label_ccd_ids', 'label_atom_ids']:
            raw_labels[key] = ' '.join(raw_labels[key])

        sample = {
            'name': protein,
            'feat': np_example,
            'label_cropped': {},
            'label': raw_labels,
            'chain_info_list': json.dumps(item["chains"])
        }
        return sample
    
    def __iter__(self):
        """tbd"""
        for x in self.protein_infos:
            try:
                yield self.get_sample(x)
            except Exception as ex:
                import traceback
                traceback.print_exc(file=sys.stdout)
                print(f'[DATA] prot {x} failed: {ex}')


class TestDatasetFromFinalPkl(IterableDataset):
    """
        Dataset for test set, 
        load samples from pdparams.
    """

    def __init__(self,
                 data_config,
                 trainer_id=0,
                 trainer_num=1):
        self.trainer_id = trainer_id
        self.trainer_num = trainer_num
        self.data_config = data_config
        samples = [json.loads(x) for x in \
                        open(self.data_config.protein_list_file, "r")]
        self.samples = samples[self.trainer_id :: self.trainer_num]
        print('worker samples:', len(self.samples), self.samples)
        self.use_assembly = self.data_config.use_assembly
        self.ccd_preprocessed_dict = read_ccd_preprocessed_dict(
                data_config.ccd_preprocessed_path)

    def _get_protein_struct_file(self, protein):
        """tbd."""
        path = join(self.data_config.structure_dir, f"{protein}.cif")
        gz_path = join(self.data_config.structure_dir, f"{protein}.cif.gz")
        unit_path = path if exists(path) else gz_path
        if self.use_assembly:
            assembly_path = join(self.data_config.assembly_structure_dir, f"{protein}-assembly1.cif")
            gz_assembly_path = join(self.data_config.assembly_structure_dir, f"{protein}-assembly1.cif.gz")
            assembly_path = path if exists(assembly_path) else gz_assembly_path
            return unit_path, assembly_path
        else:
            return unit_path, None

    def get_sample(self, x):
        prot = x["protein_name"]
        chains = x["chains"]
        prot_chains = f"{prot}_{'_'.join(sorted([c['id'] for c in chains]))}"
        feat_path = os.path.join(
            self.data_config.feature_dir,
            prot_chains + self.data_config._suffix, 
            self.data_config.final_pkl # "TEMP_OUTPUT/final_features.pkl"
        )
        feat = paddle.load(feat_path)

        # 1. load mmcif object
        unit_struct_file, assembly_struct_file = self._get_protein_struct_file(prot)
        if self.use_assembly:
            mmcif_object = read_cif_to_mmcif_object(assembly_struct_file)
        else:
            mmcif_object = read_cif_to_mmcif_object(unit_struct_file)
        if mmcif_object is None:
            print(f'=> Skip [{prot}] due to all mmcif parsing error.')
            return None

        raw_labels = pipeline_aa.get_assembly_label(
                [c["id"] for c in chains], 
                mmcif_object,
                self.ccd_preprocessed_dict)
        
        # copy assembly features
        for k in ['asym_id', 'sym_id', 'entity_id', 'residue_index']:
            raw_labels[k] = np.copy(feat["feat"][k])

        # copy perm features
        for k in ['perm_atom_index', 'perm_entity_id', 'perm_asym_id']:
            raw_labels[k] = np.copy(feat["feat"][k])

        for key in ['label_ccd_ids', 'label_atom_ids']:
            raw_labels[key] = ' '.join(raw_labels[key])

        assert raw_labels["all_centra_token_indice"].shape[0] == feat["feat"]["restype"].shape[0], \
            f"inequal size between label {raw_labels['all_centra_token_indice'].shape} \
              and feat {feat['feat']['restype'].shape}"

        # feat['label']['residue_index']  = feat['feat']['residue_index']
        feat["label"] = raw_labels
        feat["name"] = prot
        feat["chain_info_list"] = json.dumps(chains)
        return feat

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        """tbd"""
        for x in self.samples:
            try:
                yield self.get_sample(x)
            except Exception as ex:
                import traceback
                traceback.print_exc(file=sys.stdout)
                print(f'[DATA] prot {x} failed: {ex}')


if __name__ == '__main__':
    from tqdm import tqdm
    import json
    import os
    import ml_collections
    from helixfold.model import config
    from utils.metric_aa import draw_interface_heatmap

    # demo data: zhukunrui/data/protein_folding/multimer/demo/all_atom_demo
    Model_name = "allatom_rec0_msa128_diff_ditres_lossw4"
    model_config = config.model_config(Model_name)
    data_config = ml_collections.ConfigDict(
        # json.load(open('./data/all_atom_demo/all_atom_demo.json', 'r'))
        # json.load(open('./data/all_atom_demo/all_atom_debug.json', 'r'))
        # json.load(open('./data_configs/all_atom-multimer_clust.json'))
        # json.load(open('./data/all_atom_demo_prot_rna_ligand/all_atom_demo.json', 'r'))
        # json.load(open('./data/all_atom_demo_rna_dna/all_atom_rna_dna.json', 'r'))
        # json.load(open('./data/all_atom_demo_rna_msa/all_atom_rna_dna.json', 'r'))
        # json.load(open('./data/posebuster_demo/demo_config.json', 'r'))
        # json.load(open('./data/all_atom_demo_weightPDB/all_atom_demo_weightPDB.json', 'r'))
        # json.load(open('./data/all_atom_demo_weightPDB/all_atom_weightPDB.json', 'r'))
        # json.load(open('./data_configs/all_atom-debug.json', 'r'))
        json.load(open('./data_configs/all_atom_interface_gen_atom5A_debug.json', 'r'))
        )
    ## train set
    # dataset = AllAtomDataset(model_config=model_config,
    #                                 data_config=data_config.train.rcsb_pdb,
    #                                 crop_size=384,
    #                                 trainer_id=0,
    #                                 trainer_num=1,
    #                                 is_pad_if_crop=True,
    #                                 delete_msa_block=False)
    # from utils.dataset_all_atom_bak import AllAtomDataset as AllAtomDatasetBak
    # dataset2 = AllAtomDatasetBak(model_config=model_config,
    #                                 data_config=data_config.train.rcsb_pdb,
    #                                 crop_size=384,
    #                                 trainer_id=0,
    #                                 trainer_num=1,
    #                                 is_pad_if_crop=True,
    #                                 delete_msa_block=False)
    ## test set

    test_configs = {
        'abag_test114_select': data_config.test.abag_test114_select,
        # 'heavy_atom5': data_config.test.abag_test_2024_heavy_atom5,
        # 'central_atom10': data_config.test.abag_test_2024_central_atom10,
        # 'central_atom15': data_config.test.abag_test_2024_central_atom15,
    }

    for test_name, test_config in test_configs.items():
        dataset = iter(AllAtomTestDataset(model_config=model_config,
                                        data_config=test_config))
        output_dir = f'tmp/{test_name}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in tqdm(range(10)):
            data = next(dataset)
            if data is None:
                print('data none')
                continue

            interface_map = data['feat']['interface_info_full']
            asym_id = data['feat']['asym_id']
            output_file = os.path.join(output_dir, f'{data["name"]}_interface_map.png')
            draw_interface_heatmap(interface_map, asym_id, output_file, title=f'{data["name"]}, {test_name}')

        
    # from utils.dataset_all_atom_bak import AllAtomTestDataset as AllAtomTestDatasetBak
    # dataset1 = iter(AllAtomTestDatasetBak(model_config=model_config,
    #                                 data_config=data_config.test.recent_heter_v2))
    
    def print_mat(mat): 
        for x in mat: 
            print(' '.join([str(a) for a in x]))

    # for i in range(100):
    #     print('---------------', i)
    #     data0 = next(dataset)
    #     # data0 = dataset[0]
    #     if data0 is None:
    #         print('data none')
    #         continue
    #     # print(data0.keys())
    #     # continue
    #     data1 = next(dataset1)

    #     data0 = tree_flatten(data0)
    #     data1 = tree_flatten(data1)
    #     for k, v in data0.items():
    #         if k not in data1:
    #             print('--nonexist', k)
    #             continue
    #         if isinstance(v, np.ndarray):
    #             if np.issubdtype(v.dtype, np.number):
    #                 if v.shape != data1[k].shape:
    #                     print('--', k, v.shape, data1[k].shape)
    #                     min_size = min(v.shape[0], data1[k].shape[0])
    #                     if not np.allclose(v[:min_size], data1[k][:min_size]):
    #                         print((v - data1[k]).sum(), v - data1[k])
    #                 else:
    #                     if not np.allclose(v, data1[k]):
    #                         print('-- ', k)
    #                         print((v - data1[k]).sum(), v - data1[k])
    #                         ###################
    #                         # PDB
    #                         # if k == 'feat.token_bonds':
    #                         #     import pdb; pdb.set_trace()
    #                         ###################
    #             else:
    #                 if not np.all(v == data1[k]):
    #                     print('-- ', k)
    #                     print('[ERROR] not equal')
    #         else:
    #             pass
    #             # print('--skip', k)
        # print('>>>> success')
