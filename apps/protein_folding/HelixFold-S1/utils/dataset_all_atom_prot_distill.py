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
import pickle
import time
import traceback
import re
from copy import deepcopy
import json
from glob import glob
from paddle.io import IterableDataset, DataLoader

# from utils.modules_diffusion import Diffuser
# from omegaconf import OmegaConf
from helixfold.data import mmcif_parsing, pipeline
from helixfold.data import pipeline_multimer
from helixfold.data.data_utils import AFDB_a3m_to_msa_features, load_chain, generate_label, \
    load_pdb_chain, sample_raw_msa
from helixfold.data.utils import get_crop_mask_all_atom, apply_crop_mask_and_pad, crop_contiguous, \
    crop_spatial_all_atom
from helixfold.data.utils import make_pair_pocket_mask_by_receptor_residue, \
     make_pair_pocket_mask_by_receptor_res_idx
from helixfold.model import features
from helixfold.model import chain_align_np_aa
from helixfold.common import protein as protein_utils
from helixfold.common import residue_constants
# from helixfold.data import parsers
# from helixfold.data.pipeline import make_sequence_features, make_msa_features
from helixfold.data.utils import is_ignored_key_multimer, is_batched_key_multimer

sys.path.insert(0, '/root/paddlejob/workspace/env_run/xy/helixfold_single')
from utils.dataset import LoopedBatchSampler

from helixfold.data import feature_processing
from helixfold.data import msa_pairing
from helixfold.data import parsers
from helixfold.data.pipeline_multimer import _make_chain_id_map, convert_monomer_features, add_assembly_features, pad_msa

from utils.utils import multimer_collate_fn, tree_map
from helixfold.common.protein import PDB_CHAIN_IDS
from helixfold.data.input.data_transforms import Seed_maker
from helixfold.data.input import input_pipeline
from utils.rotation_utils import get_rotation_mat, rotate_all_atom_positions

from helixfold.data import pipeline_multimer
from helixfold.data import pipeline_hybrid, pipeline_conf_bonds, pipeline_token_feature, label_utils

MAX_SEQ_SIZE = 1600
MAX_DCU_SEQ_SIZE = 800

# attr ==> seq_len_axis
NEED_FEATURES = {
    'atom14_atom_exists': 1,
    'atom37_atom_exists': 1,
    'residx_atom14_to_atom37': 1,
    'residx_atom37_to_atom14': 1,
    'msa_feat': 2
}


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


class AllAtomProtDistillDataset(paddle.io.Dataset):
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

        self.data_config = data_config
        self.crop_size = crop_size
        self.is_pad_if_crop = is_pad_if_crop
        self.trainer_id = trainer_id
        self.trainer_num = trainer_num
        self.is_shuffle = is_shuffle
        
        def _assert_exists(path):
            assert exists(path), path

        _assert_exists(self.data_config.protein_clust_file)
        _assert_exists(self.data_config.feature_dir)
        _assert_exists(self.data_config.structure_dir)

        ## check and filter (bad or not needed) proteins
        self.clusts = self._load_clusts()
        self.clusts_to_consume = [[] for _ in range(len(self.clusts))]

        def _print_attribute(key, value):
            print(f'[{self.__class__.__name__}] {key}: {value}')

        _print_attribute('protein_clust_file',
                         self.data_config.protein_clust_file)
        _print_attribute('trainer_id/trainer_num',
                         f'{self.trainer_id}/{self.trainer_num}')
        _print_attribute('clust_num', len(self.clusts))
        _print_attribute('protein_num', np.sum([len(x) for x in self.clusts]))
        _print_attribute('crop_size', crop_size)
        _print_attribute('is_pad_if_crop', is_pad_if_crop)
        _print_attribute('delete_msa_block', delete_msa_block)
        assert np.sum([len(x) for x in self.clusts]) > 0, f"protein_num {np.sum([len(x) for x in self.clusts])} <= 0"

        ## AF3 featurization pipeline.
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

    def _load_clusts(self):
        with open(self.data_config.protein_clust_file) as r:
            lines = r.readlines()[self.trainer_id::self.trainer_num]

        clusts = []
        for line in lines:
            clust = []
            for chain_name in line.split():
                protein = chain_name.split('_')[0]
                if self._check_protein(protein):
                    clust.append(protein)
            if len(clust) > 0:
                clusts.append(clust)
        return clusts

    def _check_protein(self, protein):
        """tbd."""

        if not os.path.exists(self._get_protein_feature_file(protein)):
            return False
        if not os.path.exists(self._get_protein_struct_file(protein)):
            return False
        return True

    def _get_protein_feature_file(self, protein):
        """tbd."""
        return join(self.data_config.feature_dir, protein, 'features.pkl.gz')

    def _get_protein_struct_file(self, protein):
        """tbd."""
        return join(self.data_config.structure_dir, f"{protein}.pdb.gz")

    def _aatype_idx_to_ccd(self, aatype):
        """tbd."""
        aatype_ccds = []
        for aatype_idx in aatype:
            aatype_ccds.append(residue_constants.AF3_idx_to_restype_order[aatype_idx])
    
        return aatype_ccds

    def _convert_AF3_feats(self, af2_feats, protein):
        """tbd."""
        ## NOTE: we assert Protein monomer distillation didn't has non-standard residues.
        ccd_sequence = self._aatype_idx_to_ccd(af2_feats['aatype'])
        _check_sequence = set(ccd_sequence)
        if 'UNK' in _check_sequence or '-' in _check_sequence:
            print(f'[DATA] Drop {protein} due to have X/- residues.')
            return None
        if len(np.unique(af2_feats['entity_id'])) > 1:
            print(f'[DATA] Drop {protein} due to different entity in Protein monomer distillation')
            return None

        af2_keep_keys_in_hf3 = set([
            'msa', 'num_alignments', 'template_aatype', 
            'template_all_atom_masks', 
            'template_all_atom_positions', 
            'deletion_matrix', 'deletion_mean',
            'num_templates', 'cluster_bias_mask',
            'seq_mask', 'msa_mask'])
        
        ## Set data structure of `all_chain_info` and put it into featurization pipeline.
        all_chain_info = {
            'protein_A': ccd_sequence  # <type>_<chain_id>: list of CCD
        }

        total_feats = {}
        ## 1. Get token features
        token_features = pipeline_token_feature.make_sequence_features(all_chain_info=all_chain_info,
                                                ccd_preprocessed_dict=self.ccd_preprocessed_dict)
        
        ## 2. Get reference features and bond features
        ref_features = pipeline_conf_bonds.make_ccd_conf_features(all_chain_info=all_chain_info,
                                                            ccd_preprocessed_dict=self.ccd_preprocessed_dict)
        bond_features = pipeline_conf_bonds.make_bond_features(covalent_bond=[], 
                                                            all_chain_info=all_chain_info, 
                                                            ccd_preprocessed_dict=self.ccd_preprocessed_dict)
        ## 3. post convert features
        total_feats['protein'] = af2_feats
        total_feats['protein']['ccd_seqs'] = np.array(ccd_sequence, dtype=object)
        total_feats['seq_token'] = token_features
        total_feats['conf_bond'] = {**ref_features, **bond_features}
        np_example = pipeline_hybrid._post_convert(ccd_preprocessed_dict=self.ccd_preprocessed_dict,
                                                        all_chain_feats_dict=total_feats)

        ## NOTE: get template further features
        np_example = pipeline_hybrid.make_pseudo_beta(np_example, prefix='template_')
        np_example = pipeline_hybrid.make_template_further_feature(np_example)

        np_example["seq_mask"] = np.ones_like(np_example['restype']).astype('float32')
        np_example["msa"], np_example['msa_mask'], np_example['deletion_matrix'] = crop_msa(np_example)
        return np_example

    def get_input_feat(self, protein):
        """tbd."""
        def _random_drop(raw_features):
            """drop sample according to num_residue"""
            L = raw_features['aatype'].shape[0]
            if np.random.uniform() > max(min(512, L), 256) / 512.0:
                print(f'[DATA] Drop {protein} ({L}) by random')
                return True
            return False

        features_pkl = self._get_protein_feature_file(protein)
        with gzip.open(features_pkl, 'rb') as pkl:
            chain_feature = pickle.load(pkl)
        if self.is_shuffle and _random_drop(chain_feature):
            return None
        # assign zeros template to proteins with no template
        if not len(chain_feature["template_aatype"].shape) == 3: 
            num_residues = chain_feature['aatype'].shape[0]
            chain_feature["template_aatype"] = np.zeros((4, num_residues, 22), dtype=np.float32) # [num_templates, num_residues, amino_acids]
            chain_feature["template_all_atom_masks"] = np.zeros((4, num_residues, 37), dtype=np.float32)  # [num_templates, num_residues, bins]
            chain_feature["template_all_atom_positions"] = np.zeros((4, num_residues, 37, 3), dtype=np.float32)  # [num_templates, num_residues, bins, coords]

        # downsample msa
        if 'msa_sample' in self.data_config:
            chain_feature = sample_raw_msa(chain_feature, self.data_config)
        ## monomer features to multimer features
        all_chain_features = {'A': chain_feature}
        raw_features = pipeline_multimer.process_with_all_chain_features(
                all_chain_features)

        ## AF2 multimer features -> AF3 features
        processed_feature_dict = self._convert_AF3_feats(raw_features, protein=protein)

        return processed_feature_dict

    def get_label(self, protein, raw_features):
        protein_struct_file = self._get_protein_struct_file(protein)
        # chain_id = protein.split('_')[1] if len(protein.split('_')) >=2 else None
        chain_id = None
        with gzip.open(protein_struct_file, 'r') as f:
            pdb_file = f.read().decode('utf8')
        
        # ## old-AF2, for feature check. # NOTE: PASS
        # prot_obj = protein_utils.from_pdb_string(pdb_file, chain_id)
        # protein_chain_d = load_pdb_chain(prot_obj,
        #         confidence_threshold=self.data_config.get('confidence_threshold', 80.0))
        # protein_label = generate_label(protein_chain_d)

        ## AF3
        protein_label_hf3 = label_utils.from_distill_pdb_string(pdb_file, 
                            self.ccd_preprocessed_dict, 
                            confidence_threshold=self.data_config.get('confidence_threshold', 80.0),
                            chain_id=chain_id)

        ## check plddt score
        plddt_score = protein_label_hf3['all_b_factors'].mean()
        plddt_thres = self.data_config.get('plddt_score', 50.0)
        if plddt_score < plddt_thres:
            print(f'Skip {protein} due to pLDDT confidence smaller than {plddt_thres}')
            return None
        ## copy assembly features
        for k in ['asym_id', 'sym_id', 'entity_id', 'residue_index', 'seq_mask']:
            protein_label_hf3[k] = np.array(raw_features[k])
        
        protein_label_hf3.pop('all_chain_ids')
        protein_label_hf3.pop('all_residue_index')

        return protein_label_hf3

    def select_protein(self, clust_index):
        """tbd."""
        if len(self.clusts_to_consume[clust_index]) == 0:
            self.clusts_to_consume[clust_index] = deepcopy(
                self.clusts[clust_index])
            if self.is_shuffle:
                np.random.shuffle(self.clusts_to_consume[clust_index])
        clust = self.clusts_to_consume[clust_index]
        last_name = clust[-1]
        del clust[-1]
        return last_name

    def get_sample(self, protein):
        if self._check_protein(protein) == False:
            return None

        np_example = self.get_input_feat(protein)
        if np_example is None:
            return None
        raw_labels = self.get_label(protein, np_example)
        if raw_labels is None:
            return None
        
        if np_example['ref_pos'].shape[0] != raw_labels['all_atom_pos'].shape[0]:
            print(f'Skip {protein} due to inequal atom num')
            return None

        raw_labels['resolution'] = np.array(
            [np.mean(raw_labels['resolution'])])

        if not self.crop_size is None and self.crop_size > 0:
            ## Start Cropping:
            chain_len = np_example['restype'].shape[0]
            # crop_and_pad_multimer and also copy assembly features
            rand_drop = np.random.random()
            if rand_drop < (1. - self.data_config.get("spatial_crop_ratio", 0.75)):
                feat_crop_idx = crop_contiguous([chain_len], self.crop_size)
            else:
                feat_crop_idx, crop_method = crop_spatial_all_atom(
                    feat=np_example, 
                    label=raw_labels, list_n_k=[chain_len], \
                    crop_size=self.crop_size, for_recycle=False, 
                    targeted_asym_ids=None
                )

            label_crop_idx = feat_crop_idx
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
            feat = {k: v for k, v in np_example.items()}

        # only copy assembly features
        for k in ['asym_id', 'sym_id', 'entity_id', 'residue_index']:
            labels[k] = np.copy(feat[k])
        feat['atom_count'] = np.bincount(feat['ref_token2atom_idx'])

        # copy perm features
        for k in ['perm_atom_index', 'perm_entity_id', 'perm_asym_id']:
            raw_labels[k] = np.copy(np_example[k])

        # filter too few valid atoms
        if labels['all_atom_pos_mask'].sum() < chain_align_np_aa.MIN_NUM_FOR_ANCHOR_CHAIN:
            print(f'Skip {protein} due to valid atom less than '
                    f'{chain_align_np_aa.MIN_NUM_FOR_ANCHOR_CHAIN}')
            return None

        ## Features basic convert.
        for key in ['all_chain_ids', 'all_ccd_ids','all_atom_ids']:
            feat[key] = ' '.join(feat[key])
        for key in ['release_date','label_ccd_ids','label_atom_ids']:
            labels[key] = ' '.join(labels[key])
            raw_labels[key] = ' '.join(raw_labels[key])
        sample = {
            'name': protein,
            'feat': feat,
            'label_cropped': labels,
            'label': raw_labels,
            'chain_ids': 'A',
            'structure_file': self._get_protein_struct_file(protein),
        } 

        return sample

    def __len__(self):
        return len(self.clusts)

    def __getitem__(self, index):
        sample = None
        while sample is None:
            protein = self.select_protein(index)
            # sample = self.get_sample(protein)
            try:
                sample = self.get_sample(protein)
            except Exception as ex:
                print(f'[DATA] index {index} prot {protein} failed: {ex}')
                import traceback
                traceback.print_exc(file=sys.stdout)
            # index = np.random.randint(self.__len__())
            index = random.randint(0, self.__len__()-1)
        return sample


class AllAtomAFDBDataset(paddle.io.Dataset):
    """tbd"""
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
        Iterate over clusts, where proteins in ech clust
        will be all visited.
        """
        self.model_config = deepcopy(model_config)
        self.data_config = data_config
        self.crop_size = crop_size
        self.is_pad_if_crop = is_pad_if_crop
        self.trainer_id = trainer_id
        self.trainer_num = trainer_num
        self.is_shuffle = is_shuffle

        def _assert_exists(path):
            assert exists(path), path
        _assert_exists(self.data_config.structure_dir)
        
        _assert_exists(self.data_config.protein_clust_file)
        _assert_exists(self.data_config.structure_dir)

        ## check and filter (bad or not needed) proteins
        self.clusts = self._load_clusts()
        self.clusts_to_consume = [[] for _ in range(len(self.clusts))]

        def _print_attribute(key, value):
            print(f'[{self.__class__.__name__}] {key}: {value}')

        _print_attribute('protein_clust_file',
                         self.data_config.protein_clust_file)
        _print_attribute('trainer_id/trainer_num',
                         f'{self.trainer_id}/{self.trainer_num}')
        _print_attribute('clust_num', len(self.clusts))
        _print_attribute('protein_num', np.sum([len(x) for x in self.clusts]))
        _print_attribute('crop_size', crop_size)
        _print_attribute('is_pad_if_crop', is_pad_if_crop)
        _print_attribute('delete_msa_block', delete_msa_block)
        assert np.sum([len(x) for x in self.clusts]) > 0, f"protein_num {np.sum([len(x) for x in self.clusts])} <= 0"

        ## AF3 featurization pipeline.
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

    def _load_clusts(self):
        with open(self.data_config.protein_clust_file) as r:
            lines = r.readlines()[self.trainer_id::self.trainer_num]

        clusts = []
        for line in lines:
            clust = []
            for protein in line.split():
                if self._check_protein(protein):
                    clust.append(protein)
            if len(clust) > 0:
                clusts.append(clust)
        return clusts

    def _check_protein(self, protein):
        """tbd."""

        if self._get_protein_feature_file(protein) is None:
            return False
        if self._get_protein_struct_file(protein) is None:
            return False
        return True

    def _get_protein_feature_file(self, protein):
        """tbd."""
        feature_path = glob(join(self.data_config.structure_dir, protein) + '/*.a3m.gz')
        if len(feature_path) != 1:
            return None
        else:
            return feature_path[0]
        

    def _get_protein_struct_file(self, protein):
        """tbd."""
        struct_file_path = join(self.data_config.structure_dir, protein, f"{protein}.pkl.gz")
        if not exists(struct_file_path):
            return None
        return struct_file_path

    def select_protein(self, clust_index):
        """tbd."""
        if len(self.clusts_to_consume[clust_index]) == 0:
            self.clusts_to_consume[clust_index] = deepcopy(
                self.clusts[clust_index])
            if self.is_shuffle:
                np.random.shuffle(self.clusts_to_consume[clust_index])
        clust = self.clusts_to_consume[clust_index]
        last_name = clust[-1]
        del clust[-1]
        return last_name

    def _aatype_idx_to_ccd(self, aatype):
        """tbd."""
        aatype_ccds = []
        for aatype_idx in aatype:
            aatype_ccds.append(residue_constants.AF3_idx_to_restype_order[aatype_idx])
    
        return aatype_ccds

    def _convert_AF3_feats(self, af2_feats, protein):
        """tbd."""
        ## NOTE: we assert Protein monomer distillation didn't has non-standard residues.
        ccd_sequence = self._aatype_idx_to_ccd(af2_feats['aatype'])
        _check_sequence = set(ccd_sequence)
        if 'UNK' in _check_sequence or '-' in _check_sequence:
            print(f'[DATA] Drop {protein} due to have X/- residues.')
            return None
        if len(np.unique(af2_feats['entity_id'])) > 1:
            print(f'[DATA] Drop {protein} due to different entity in Protein monomer distillation')
            return None

        af2_keep_keys_in_hf3 = set([
            'msa', 'num_alignments', 'template_aatype', 
            'template_all_atom_masks', 
            'template_all_atom_positions', 
            'deletion_matrix', 'deletion_mean',
            'num_templates', 'cluster_bias_mask',
            'seq_mask', 'msa_mask'])
        
        ## Set data structure of `all_chain_info` and put it into featurization pipeline.
        all_chain_info = {
            'protein_A': ccd_sequence  # <type>_<chain_id>: list of CCD
        }

        total_feats = {}
        ## 1. Get token features
        token_features = pipeline_token_feature.make_sequence_features(all_chain_info=all_chain_info,
                                                ccd_preprocessed_dict=self.ccd_preprocessed_dict)
        
        ## 2. Get reference features and bond features
        ref_features = pipeline_conf_bonds.make_ccd_conf_features(all_chain_info=all_chain_info,
                                                            ccd_preprocessed_dict=self.ccd_preprocessed_dict)
        bond_features = pipeline_conf_bonds.make_bond_features(covalent_bond=[], 
                                                            all_chain_info=all_chain_info, 
                                                            ccd_preprocessed_dict=self.ccd_preprocessed_dict)
        ## 3. post convert features
        total_feats['protein'] = af2_feats
        total_feats['protein']['ccd_seqs'] = np.array(ccd_sequence, dtype=object)
        total_feats['seq_token'] = token_features
        total_feats['conf_bond'] = {**ref_features, **bond_features}
        np_example = pipeline_hybrid._post_convert(ccd_preprocessed_dict=self.ccd_preprocessed_dict,
                                                        all_chain_feats_dict=total_feats)
        ## NOTE: get template further features
        np_example = pipeline_hybrid.make_pseudo_beta(np_example, prefix='template_')
        np_example = pipeline_hybrid.make_template_further_feature(np_example)

        np_example["seq_mask"] = np.ones_like(np_example['restype']).astype('float32')
        np_example["msa"], np_example['msa_mask'], np_example['deletion_matrix'] = crop_msa(np_example)
        return np_example

    def _parsed_a3m_str(self, a3m_str) -> list:
        idx = 0
        a3m_str_list = []
        raw_list = a3m_str.split('\n')
        while idx < len(raw_list):
            if raw_list[idx].startswith('>'):
                a3m_str_list.extend([raw_list[idx], raw_list[idx + 1]])
                idx += 1
            idx += 1
        return '\n'.join(a3m_str_list)

    def get_input_feat(self, protein: str):
        '''tdb'''

        def _random_drop(raw_features):
            """drop sample according to num_residue"""
            L = raw_features['aatype'].shape[0]
            if np.random.uniform() > max(min(512, L), 256) / 512.0:
                print(f'[DATA] Drop {protein} ({L}) by random')
                return True
            return False

        a3m_str = None
        with gzip.open(self._get_protein_feature_file(protein)) as fp:
            a3m_str = fp.read().decode('utf-8')
        if a3m_str is None:
            return None

        ### gen_input_feat
        valid_a3m_string = self._parsed_a3m_str(a3m_str)
        chain_feature = AFDB_a3m_to_msa_features(valid_a3m_string)

        if self.is_shuffle and _random_drop(chain_feature):
            return None
    
        # assign zeros template to proteins with no template
        if "template_aatype" not in chain_feature: 
            num_residues = chain_feature['aatype'].shape[0]
            chain_feature["template_aatype"] = np.zeros((4, num_residues, 22))
            chain_feature["template_all_atom_masks"] = np.zeros((4, num_residues, 37))
            chain_feature["template_all_atom_positions"] = np.zeros((4, num_residues, 37, 3))

        # downsample msa
        if 'msa_sample' in self.data_config:
            chain_feature = sample_raw_msa(chain_feature, self.data_config)
        ## monomer features to multimer features
        all_chain_features = {'A': chain_feature}
        raw_features = pipeline_multimer.process_with_all_chain_features(all_chain_features)

        ## AF2 multimer features -> AF3 features
        processed_feature_dict = self._convert_AF3_feats(raw_features, protein=protein)

        return processed_feature_dict
    
    def get_label(self, protein: str, raw_features: dict):
        '''tdb'''
        ### gen_label 
        data = None
        with gzip.open(self._get_protein_struct_file(protein)) as f_read:
            data = pickle.load(f_read)
        if data is None:
            return None
        
        ## basic AF2 -> AF3 feats is need.
        order_map = residue_constants.restype_order_with_x
        aatype_idx = np.array([order_map.get(rn, order_map['X']) for rn in data['sequence']], dtype=np.int32)
        resolution = self.data_config.get('resolution', 0.5)
        protein_chain_d = {
            'aatype_index': aatype_idx,
            'ccd_ids_list': self._aatype_idx_to_ccd(aatype_idx),
            'all_atom_positions': data['all_atom_positions'], 
            'all_atom_mask': data['all_atom_mask'],
            'confidence_list': data['confidence_list'][0], # residue-level
            'resolution': np.array([resolution], dtype=np.float32)
        }

        ## AF3
        protein_label_hf3 = label_utils.from_distill_afdb_np_feats(protein_chain_d, 
                            self.ccd_preprocessed_dict, 
                            confidence_threshold=self.data_config.get('confidence_threshold', 80.0))

        ## check plddt score
        confidence_list = data['confidence_list']
        plddt_score = np.mean(confidence_list[0])
        plddt_thres = self.data_config.get('plddt_score', 50.0)
        if plddt_score < plddt_thres:
            print(f'Skip {protein} due to pLDDT confidence smaller than {plddt_thres}')
            return None

        ## copy assembly features
        for k in ['asym_id', 'sym_id', 'entity_id', 'residue_index', 'seq_mask']:
            protein_label_hf3[k] = np.array(raw_features[k])
        
        return protein_label_hf3

    def get_sample(self, protein):
        '''tdb'''
        if not self._check_protein(protein):
            return None
                
        ### get input_feat & get label
        np_example = self.get_input_feat(protein)
        if np_example is None:
            return None
        
        raw_labels = self.get_label(protein, np_example)
        if raw_labels is None:
            return None

        if np_example['ref_pos'].shape[0] != raw_labels['all_atom_pos'].shape[0]:
            print(f'Skip {protein} due to inequal atom num')
            return None

        raw_labels['resolution'] = np.array(
            [np.mean(raw_labels['resolution'])])

        if not self.crop_size is None and self.crop_size > 0:
            ## Start Cropping:
            chain_len = np_example['restype'].shape[0]
            # crop_and_pad_multimer and also copy assembly features
            rand_drop = np.random.random()
            if rand_drop < (1. - self.data_config.get("spatial_crop_ratio", 0.75)):
                feat_crop_idx = crop_contiguous([chain_len], self.crop_size)
            else:
                feat_crop_idx, crop_method = crop_spatial_all_atom(
                    feat=np_example, 
                    label=raw_labels, list_n_k=[chain_len], \
                    crop_size=self.crop_size, for_recycle=False, 
                    targeted_asym_ids=None
                )

            label_crop_idx = feat_crop_idx
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
            feat = {k: v for k, v in np_example.items()}

        # only copy assembly features
        for k in ['asym_id', 'sym_id', 'entity_id', 'residue_index']:
            labels[k] = np.copy(feat[k])
        feat['atom_count'] = np.bincount(feat['ref_token2atom_idx'])

        # copy perm features
        for k in ['perm_atom_index', 'perm_entity_id', 'perm_asym_id']:
            raw_labels[k] = np.copy(np_example[k])

        # filter too few valid atoms
        if labels['all_atom_pos_mask'].sum() < chain_align_np_aa.MIN_NUM_FOR_ANCHOR_CHAIN:
            print(f'Skip {protein} due to valid atom less than '
                    f'{chain_align_np_aa.MIN_NUM_FOR_ANCHOR_CHAIN}')
            return None

        ## Features basic convert.
        for key in ['all_chain_ids', 'all_ccd_ids', 'all_atom_ids']:
            feat[key] = ' '.join(feat[key])
        for key in ['release_date', 'label_ccd_ids', 'label_atom_ids']:
            labels[key] = ' '.join(labels[key])
            raw_labels[key] = ' '.join(raw_labels[key])
        sample = {
            'name': protein,
            'feat': feat,
            'label_cropped': labels,
            'label': raw_labels,
            'chain_ids': 'A',
            'structure_file': self._get_protein_struct_file(protein),
        } 

        return sample
    
    def __len__(self):
        return len(self.clusts)
    
    def __getitem__(self, index):
        sample = None
        while sample is None:
            protein = self.select_protein(index)
            try:
                sample = self.get_sample(protein)
            except Exception as ex:
                print(f'[DATA] index {index} prot {protein} failed: {ex}')
                import traceback
                traceback.print_exc(file=sys.stdout)
            index = random.randint(0, self.__len__() - 1)
        return sample


if __name__ == '__main__':
    ## demo_data_path, AFDB
    from tqdm import tqdm
    import json
    import ml_collections
    from helixfold.model import config
    Model_name = 'multimer_demo'
    model_config = config.model_config(Model_name)
    data_config = ml_collections.ConfigDict(
        json.load(open('./data_configs/all_atom-bicluster-distill.json', 'r')))
    dataset = AllAtomProtDistillDataset(model_config=model_config,
                              data_config=data_config.distill['uniclust30'],
                              crop_size=384, # None
                              is_pad_if_crop=True,
                              delete_msa_block=True,
                              trainer_id=0,
                              trainer_num=1)
    print('-----')
    for data in dataset:
        print(data.keys())
        print('-----')
    exit()
    # dataloader = DataLoader(dataset=dataset,
    #                         batch_sampler=LoopedBatchSampler(dataset,
    #                                                          shuffle=True,
    #                                                          batch_size=1,
    #                                                          drop_last=False),
    #                         num_workers=0)

    # distill_data_gen = next(iter(dataloader))
    # t = dataset.get_sample('AF-X6LI38-F1_A') 
    # for t, v in t['label'].items():
    #     if type(v) == np.ndarray:
    #         print(t, v.shape)
    # import pdb; pdb.set_trace()

    import paddle
    for i, data in enumerate(tqdm(dataloader)):
        for kk, vv in data.items():
            print(f'>>>> {kk}')
            if type(vv) == dict:
                for k, v in vv.items():
                    if isinstance(v, paddle.Tensor) or isinstance(v, np.ndarray):
                        print(k, type(v), v.shape)
                    else:
                        print(k, type(v))
            else:
                print(kk, vv)
        break