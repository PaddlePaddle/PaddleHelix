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
import glob
import gzip
import pickle
import time
import traceback
import re
from copy import deepcopy
import collections
import json
from paddle.io import IterableDataset, DataLoader
from collections import Counter
from scipy.spatial import distance_matrix
from rdkit import Chem
import helixfold.data.mmcif_parsing_paddle as mmcif_parsing
from helixfold.data import pipeline_multimer, pipeline_rna_multimer
from helixfold.data.data_utils import a3m_to_features, sample_raw_msa
from helixfold.data.utils import get_crop_mask_all_atom_by_center, apply_crop_mask_and_pad
from helixfold.model import features
from helixfold.model import chain_align_np_aa
from helixfold.common import protein as protein_utils
from helixfold.common import residue_constants
from helixfold.data import parsers
# from helixfold.data.pipeline import make_sequence_features, make_msa_features
from helixfold.data.utils import is_ignored_key_multimer, is_batched_key_multimer

from helixfold.data import feature_processing
from helixfold.data import msa_pairing

from utils.utils import multimer_collate_fn, tree_map
from helixfold.common.protein import PDB_CHAIN_IDS
from helixfold.data.input.data_transforms import Seed_maker
from helixfold.data.input import input_pipeline
from utils.rotation_utils import get_rotation_mat, rotate_all_atom_positions

from helixfold.data import pipeline_hybrid, pipeline_conf_bonds, pipeline_token_feature, label_utils

MAX_SEQ_SIZE = 1600 # FIXME(zhukunrui): enlarge size
MAX_DCU_SEQ_SIZE = 800
MAX_CROPPING_TOKEN_NUM = 10_000

# attr ==> seq_len_axis
NEED_FEATURES = {
    'atom14_atom_exists': 1,
    'atom37_atom_exists': 1,
    'residx_atom14_to_atom37': 1,
    'residx_atom37_to_atom14': 1,
    'msa_feat': 2
}

ligand_feat_rules = {
    'seq_mask': {'type': 'const', 'value': 1},
    'residue_index': {'type': 'const', 'value': 0},
    'restype': {'type': 'const', 'value': 20},
    'token_index': {'type': 'max+1 incr'},
    'asym_id': {'type': 'max+1'},
    'sym_id':  {'type': 'const', 'value': 1},
    'entity_id': {'type': 'max+1'},
    'is_protein': {'type': 'const', 'value': 0},
    'is_dna': {'type': 'const', 'value': 0},
    'is_rna': {'type': 'const', 'value': 0},
    'is_ligand': {'type': 'const', 'value': 1},

    'perm_entity_id': {'type': 'max+1'},
    'perm_asym_id': {'type': 'max+1'},
    'perm_atom_index': {'type': '0 incr'},
    'all_chain_ids': {'type': 'value'},
    'all_ccd_ids': {'type': 'value'},
    'all_atom_ids': {'type': 'value'},
    'ref_pos': {'type': 'const', 'value': 0},
    'ref_mask': {'type': 'const', 'value': 1},
    'ref_element': {'type': 'value'},
    'ref_charge': {'type': 'value'},
    'ref_atom_name_chars': {'type': 'value'},
    'ref_space_uid': {'type': 'max+1'},
    'ref_token2atom_idx': {'type': 'max+1'},
    'ref_atom_count': {'type': 'const', 'value': 1},

    'token_bonds': {'type': 'pair_value'},

    'msa': {'type': 'const', 'value': 20, 'dim': 1},
    'msa_mask': {'type': 'const', 'value': 1, 'dim': 1},
    'deletion_matrix': {'type': 'const', 'value': 0, 'dim': 1},
    'deletion_mean': {'type': 'const', 'value': 0},
    'profile': {'type': 'const', 'value': 0},
    'has_deletion': {'type': 'const', 'value': 0, 'dim': 1},
    'deletion_value': {'type': 'const', 'value': 0, 'dim': 1},
    'template_aatype': {'type': 'const', 'value': 21, 'dim': 1},
    'template_all_atom_masks': {'type': 'const', 'value': 0, 'dim': 1},
    'template_all_atom_positions': {'type': 'const', 'value': 0, 'dim': 1},
    'template_pseudo_beta_mask': {'type': 'const', 'value': 0, 'dim': 1},
    'template_backbone_frame_mask': {'type': 'const', 'value': 0, 'dim': 1},
    'template_distogram': {'type': 'const', 'value': 0, 'dim': [1, 2]},
    'template_unit_vector': {'type': 'const', 'value': 0, 'dim': [1, 2]},

    'label_ccd_ids': {'type': 'value'},
    'label_atom_ids': {'type': 'value'},
    'all_atom_pos': {'type': 'value'},
    'all_atom_pos_mask':  {'type': 'const', 'value': 1},

    'all_centra_token_indice': {'type': 'value'},
    'all_centra_token_indice_mask':  {'type': 'const', 'value': 1},
    'all_token_to_atom_nums':  {'type': 'const', 'value': 1},
    'pseudo_beta': {'type': 'value'},
    'pseudo_beta_mask':  {'type': 'const', 'value': 1},
}


def get_global_residue_ids(seqs):
    ''' tbd. '''
    global_res_ids = [] # res id for current chain

    local_num_res = [] # length of current chain
    global_num_res = []  # length of all chain

    local_seq_order = [] # seq id for current chain
    local_entity_id = [] # unique seq id for current chain
    global_num_seq = []  # total number of seq
    global_num_entity = []  # total number of unique seq

    seq_set = set()
    global_res_id = 0
    for i, seq in enumerate(seqs):
        seq_global_ids = []
        if not seq in seq_set: 
            seq_set.add(seq)
        for r in seq:
            seq_global_ids.append(
                np.array([global_res_id, len(seq), sum([len(s) for s in seqs]),
                i, len(seq_set), len(seqs), len(set(seqs))
                ], dtype=np.float32)
            )
            global_res_id += 1
        global_res_ids.append(seq_global_ids)
    return global_res_ids

def pad_msa(feat, max_msa_depth):
    """ pad msa and generate msa_mask. """
    msa = feat['msa']
    msa_mask = np.ones_like(feat['msa']).astype('float32') # [msa_depth, n_token]
    msa_depth, num_token = msa_mask.shape
    if msa_depth < max_msa_depth:
        padding_size = max_msa_depth - msa_depth
        msa_paddings = np.zeros(
            (padding_size, num_token), dtype='float32'
        )
        msa_mask = np.concatenate(
            [msa_mask, msa_paddings], axis=0
        )  # [msa_depth_padded, num_tokens]
        msa = np.concatenate(
            [msa, msa_paddings], axis=0
        )  # [msa_depth_padded, num_tokens]
        # TODO(zhukunrui): add deletion_matrix handling
    return msa.astype('int32'), msa_mask

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


def cat_ligand_feat(raw_feat, ligand_N, type='', 
            value=None, dim=0):
    """cat_ligand_feat_constant"""
    if type == 'const':
        new_shape = list(raw_feat.shape)
        if dim == 0:
            new_shape[0] += ligand_N
            new_feat = np.full(new_shape, value)
            new_feat[:-ligand_N] = raw_feat
        elif dim == 1:
            new_shape[1] += ligand_N
            new_feat = np.full(new_shape, value)
            new_feat[:, :-ligand_N] = raw_feat
        elif dim == [1, 2]:
            new_shape[1] += ligand_N 
            new_shape[2] += ligand_N 
            new_feat = np.full(new_shape, value)
            new_feat[:, :-ligand_N, :-ligand_N] = raw_feat
        else:
            raise ValueError(dim)
    
    elif type == 'max+1':
        m = np.max(raw_feat)
        new_feat = np.concatenate([raw_feat, 
                np.full([ligand_N], m + 1)], 0)
    
    elif type == 'max+1 incr':
        m = np.max(raw_feat)
        new_feat = np.concatenate([raw_feat, 
                np.arange(m + 1, m + 1 + ligand_N)], 0)
    
    elif type == '0 incr':
        new_feat = np.concatenate([raw_feat, 
                np.arange(0, ligand_N)], 0)
    
    elif type == 'value':
        new_feat = np.concatenate([raw_feat, 
                value], 0)
    
    elif type == 'pair_value':
        new_shape = list(raw_feat.shape)
        new_shape[0] += ligand_N
        new_shape[1] += ligand_N
        new_feat = np.zeros(new_shape, dtype=raw_feat.dtype)
        new_feat[:-ligand_N, :-ligand_N] = raw_feat
        new_feat[-ligand_N:, -ligand_N:] = value
    
    else:
        raise ValueError(type)
    
    new_feat = new_feat.astype(raw_feat.dtype)
    return new_feat


class AllAtomLigandDistillDataset(paddle.io.Dataset):
    """tbd."""

    def __init__(self,
                 model_config,
                 data_config,
                 crop_size=None,
                 is_pad_if_crop=False,
                 trainer_id=0,
                 trainer_num=1,
                 is_shuffle=False):
        self.model_config = deepcopy(model_config)

        self.data_config = data_config
        self.crop_size = crop_size
        self.is_pad_if_crop = is_pad_if_crop
        self.trainer_id = trainer_id
        self.trainer_num = trainer_num
        self.is_shuffle = is_shuffle

        def _assert_exists(path):
            assert exists(path), path

        _assert_exists(self.data_config.ligand_structure_dir)
        _assert_exists(self.data_config.feature_dir)
        _assert_exists(self.data_config.structure_dir)

        ## check and filter (bad or not needed) proteins
        self.protein2seq_map = self._load_protein2seq_map()
        self.rna2seq_map = self._load_rna2seq_map()
        self.fasta2id_map = self._load_fasta2id_map()
        self.data_list, failed_data_list, failed_codes = self._load_data_list()

        def _print_attribute(key, value):
            print(f'[{self.__class__.__name__}] {key}: {value}')

        _print_attribute('ligand_structure_dir',
                         self.data_config.ligand_structure_dir)
        _print_attribute('trainer_id/trainer_num',
                         f'{self.trainer_id}/{self.trainer_num}')
        _print_attribute('sample_num', len(self.data_list))
        _print_attribute('falied sample_num', len(failed_data_list))
        _print_attribute('falied codes with freq', np.unique(list(failed_codes.values()), return_counts=True))
        _print_attribute('crop_size', crop_size)
        _print_attribute('is_pad_if_crop', is_pad_if_crop)

        # pipelines for non_polymer/ conf_bond
        self.pipeline_token_feature = pipeline_token_feature.DataPipeline(
            ccd_preprocessed_path=data_config.ccd_preprocessed_path)
        self.conf_bond_pipeline = pipeline_conf_bonds.DataPipeline(
            ccd_preprocessed_path=data_config.ccd_preprocessed_path)
    
        ccd_preprocessed_dict = {}
        st_1 = time.time()
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

    def _load_data_list(self):
        """
        ligand_path example:
        MCE_1a9u_pocket0_qvina02_exh8/MCE_library1_MCE-BioActive00013468_S0_T0_mode1.mol2.gz
        """
        ligand_paths = glob.glob(f'{self.data_config.ligand_structure_dir}/*/*.mol2.gz')
        ligand_paths = ligand_paths[self.trainer_id::self.trainer_num]
        data_list, failed_data_list = [], []
        failed_codes = {}
        for ligand_path in ligand_paths:
            prot_name = ligand_path.split('/')[-2].split('_')[1]
            error_code = self._check_data(prot_name, ligand_path)
            if error_code == 0:
                data_list.append({
                    'prot_name': prot_name,
                    'ligand_path': ligand_path})
            else:
                failed_data_list.append(line)
                failed_codes[line] = error_code
        return data_list, failed_data_list, failed_codes

    def _check_data(self, protein, ligand_path):
        """tbd."""
        input_json_path = self._get_protein_json_file(protein)
        # Check json_file
        if not os.path.exists(input_json_path):
            return 1 
        with open(input_json_path) as input_json: 
            input_dict = json.load(input_json)

        if not 'tokenization' in input_dict: return 2
        if not 'basic' in input_dict: return 3

        raw_token_dict = input_dict['tokenization']

        # Check chain features
        # 1. protein feat
        if "protein" in raw_token_dict:
            seqs = raw_token_dict["protein"]["msa_seqs"]
            chain_ids = raw_token_dict["protein"]["chain_ids"]
            if len(chain_ids) == 0: 
                return 4 

            min_seq_len = 1000
            num_features = 0 
            for seq, chain_id in zip(seqs, chain_ids):
                if os.path.exists(self._get_protein_feature_file(protein + "_" + chain_id, seq=seq)):
                    num_features += 1
                    min_seq_len = min(len(seq), min_seq_len)
                # else: 
                #     if self.data_config.get("check_all_chain_features", True): 
                #         return 5  # Keep protein if all chain feature exists
                #     else:  
                #         pass # Keep protein if at least 1 chain feature exists

            if num_features < 1: 
                return 7 

            # Check structure file
            if not os.path.exists(self._get_protein_struct_file(protein)):
                return 9
        
        if not exists(ligand_path):
            return 10
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
        return path

    def _get_protein_json_file(self, protein):
        """tbd."""
        return join(self.data_config.json_dir, f"{protein}.json")

    def get_docking_ligand_feat(self, mol):
        """get_docking_ligand_feat"""
        ret = {
            'atomic_num': [],
            'formal_charge': [],
            'atom_name': [],
            'atom_pos': [],
        }
        conf = mol.GetConformer()
        for i, atom in enumerate(mol.GetAtoms()):
            ret['atomic_num'].append(atom.GetAtomicNum())
            ret['formal_charge'].append(atom.GetFormalCharge())
            ret['atom_name'].append(atom.GetSymbol())
            pos = conf.GetAtomPosition(i)
            ret['atom_pos'].append([pos.x, pos.y, pos.z])

        N = len(ret['atom_name'])
        ret['bonds'] = np.zeros([N, N], 'float32')
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            ret['bonds'][i, j] = 1
            ret['bonds'][j, i] = 1
                
        ret = {k: np.array(v) for k, v in ret.items()}
        return ret
    
    def add_docking_ligand_feat_to_sample(self, ligand_feat, np_example, raw_labels):
        """add_docking_ligand_feat_to_sample"""
        ligand_N = len(ligand_feat['atom_name'])
        raw_atom_N = len(raw_labels['all_atom_pos'])
        special_values = {
            'all_chain_ids': ['DOCK'] * ligand_N,
            'all_ccd_ids': ['DOCK'] * ligand_N,
            'all_atom_ids': ligand_feat['atom_name'],
            'ref_element': np.array([residue_constants.ATOM_ELEMENT[x]
                    for x in ligand_feat['atom_name']]),
            'ref_charge': np.array(ligand_feat['formal_charge']),   # TODO
            'ref_atom_name_chars': np.array([pipeline_conf_bonds.convert_atom_id_name(x)
                    for x in ligand_feat['atom_name']]),
            'token_bonds': ligand_feat['bonds'],
            'label_ccd_ids': ['DOCK'] * ligand_N,
            'label_atom_ids': ligand_feat['atom_name'],
            'all_atom_pos': ligand_feat['atom_pos'],
            'all_centra_token_indice': np.arange(raw_atom_N + 1, raw_atom_N + 1 + ligand_N),
            'pseudo_beta': ligand_feat['atom_pos'],
        }

        for d in [np_example, raw_labels]:
            for k, v in d.items():
                if k not in ligand_feat_rules:
                    continue
                rule = ligand_feat_rules[k]
                if rule['type'] in ['value', 'string', 'pair_value']:
                    rule = {'value': special_values[k], **rule}
                d[k] = cat_ligand_feat(v, ligand_N, **rule)
                # if isinstance(v, np.ndarray):
                #     print(v.shape, v.dtype, '->', d[k].shape, d[k].dtype)
                # else:
                #     print(len(v), len(d[k]))
        
        return np_example, raw_labels


    def get_protein_feat(self, protein, seqs, chain_ids, seqtype='protein'):
        """tbd."""

        def _random_drop(raw_features):
            """drop sample according to num_residue"""
            L = raw_features['aatype'].shape[0]
            if np.random.uniform() > max(min(512, L), 256) / 512.0:
                print(f'[DATA] Drop {protein} ({L}) by random')
                return True
            return False

        full_seq_length = 0
        all_chain_features = {}
        all_chain_atom_pos = {}
        all_chain_atom_mask = {}
        max_chain_len, selected_receptor_chain = 0, None
        selected_chains = []
        global_res_ids = get_global_residue_ids(seqs)

        for (chain_id, seq, global_ids) in zip(chain_ids, seqs, global_res_ids):
            seq_len = len(seq)
            if os.environ.get('DCU_MODE_SOFTMAX_CLIP', '0') == '1':
                # fix _queue.Empty error by mas_paring block_diag
                # skip feature when full sequence length exceeded
                if full_seq_length + seq_len > MAX_DCU_SEQ_SIZE:
                    print(f" {protein} {chain_id} seq_length({seq_len}) + \
                            full_seq_length({full_seq_length}) exceeded, skip")
                    break
            else: 
                if full_seq_length + seq_len > MAX_SEQ_SIZE:
                    print(f" {protein} {chain_id} seq_length({seq_len}) + \
                        full_seq_length({full_seq_length}) exceeded, skip")
                    break

            if seqtype == 'rna':
                features_pkl = self._get_rna_feature_file(protein + "_" + chain_id, seq=seq)
            else:
                features_pkl = self._get_protein_feature_file(protein + "_" + chain_id, seq=seq)
                
            # skip features whose features_pkl does not exists
            if not os.path.exists(features_pkl): 
                # print(f"skip {features_pkl}")
                continue 
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

            if not raw_features['seq_length'][0] == seq_len:
                print(f"Skip {protein}_{chain_id} due to inequal residue num \
                    between fasta({seq_len}) and features_pkl({raw_features['seq_length'][0]})")
                continue 
            
            # protein_chain_d = load_chain(mmcif_object, chain_id)
            # if not protein_chain_d['all_atom_positions'].shape[0] == seq_len: 
            #     print(f"Skip {protein}_{chain_id} due to inequal residue num \
            #           between fasta({seq_len}) and protein_chain_d['all_atom_positions'](\
            #           {protein_chain_d['all_atom_positions'].shape})")
            #     continue

            all_chain_features[chain_id] = raw_features
            all_chain_features[chain_id].update({"global_res_ids": global_ids})

            full_seq_length += seq_len
            selected_chains.append(chain_id)

        if full_seq_length == 0: 
            return None

        if seqtype == 'rna':
            raw_features = pipeline_rna_multimer.process_with_all_chain_features(all_chain_features)
        else:
            raw_features = pipeline_multimer.process_with_all_chain_features(all_chain_features)
        
        if self.is_shuffle and _random_drop(raw_features):
            return None

        # processed_feature_dict = features.np_example_to_features(
        #     np_example=raw_features,
        #     config=self.model_config)

        raw_features["global_res_ids"] = np.concatenate(
            [all_chain_features[chain_id]["global_res_ids"] for chain_id in selected_chains]
        )
        return raw_features, selected_chains


    def get_label(self, protein, selected_chains, mmcif_object, chain_ids):
        """tbd."""
        label_list = []
        offset = 0
        for chain_id in chain_ids:
            # one molecular per chain
            if not chain_id in selected_chains: 
                continue

            try:
                protein_chain_d = label_utils.get_position_label(mmcif_object, 
                        ccd_preprocessed_dict=self.ccd_preprocessed_dict,
                        mmcif_chain_id=chain_id)
            except Exception as exception:
                print(f"Exception in loading label for {protein} {chain_id}")
                return None
            
            if protein_chain_d is None: 
                return None

            protein_chain_d = protein_chain_d[chain_id]

            protein_chain_d['all_centra_token_indice'] += offset
            offset += protein_chain_d['all_atom_pos'].shape[0]
      
            # protein_label = generate_label(protein_chain_d)
            # if seq_len != protein_chain_d['seq_length']:
            #     print(f'Skip {protein} due to inequal residue num')
            #     continue
            # for key in ['seq_length']: 
            #     protein_chain_d.pop(key)
            label_list.append(protein_chain_d)

        raw_labels = {}
        if len(label_list) == 0: 
            return None
        for key in label_list[0].keys():
            raw_labels[key] = np.concatenate([l[key] for l in label_list], axis=0)
        return raw_labels

    def get_sample(self, data):
        ''' tbd. '''
        protein = data['prot_name']
        ligand_path = data['ligand_path']

        # load docking ligand
        with gzip.open(ligand_path, 'rt') as gz_file:
            mol2_content = gz_file.read()
        try:
            mol = Chem.MolFromMol2Block(mol2_content)
            if mol is None:
                return None
        except Exception as e:
            print(f'=> Skip [{ligand_path}] due to RDKit MolFromMol2Block error. Got [{str(e)}]')
            return None
        docking_ligand_feat = self.get_docking_ligand_feat(mol)

        # 0. load mmcif object
        protein_struct_file = self._get_protein_struct_file(protein)
        if protein_struct_file.endswith('.cif.gz'):
            with gzip.open(protein_struct_file, 'r') as f:
                cif_string = f.read().decode('utf8')
        else:
            cif_string = "".join(open(protein_struct_file, 'r').readlines())

        parse_result = mmcif_parsing.parse(file_id=protein_struct_file, mmcif_string=cif_string)
        mmcif_object = parse_result.mmcif_object

        if mmcif_object is None:
            print(f'=> Skip [{protein_struct_file}] due to mmcif parsing error. Got [{parse_result.errors}]')
            return None

        # 1. Read chain info from json file
        input_json_path = self._get_protein_json_file(protein)
        with open(input_json_path) as input_json: 
            input_dict = json.load(input_json)

        all_chain_ids = []
        chain_id_2_num_token = {}
        basic_token_dict = input_dict['basic']
        raw_token_dict = input_dict['tokenization']
        # NOTE: dtype must be protein, dna, rna, non_polymer TODO: add collections.OrderedDict()
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
            # if dtype in ["dna", "rna"]: 
            #     continue # FIXME(zhukunrui): dna/ rna chains not supported.

            ccd_seqs = raw_values['seqs']
            chain_ids = raw_values['chain_ids']
            for ccd_seq, chain_id in zip(ccd_seqs, chain_ids):
                ccd_parsed = parsers.parse_ccd_fasta(ccd_seq) # str: list
                ccd_chain_info[f'{dtype}_{chain_id}'] = ccd_parsed
                chain_id_2_num_token[chain_id] = len(ccd_list_to_token(ccd_parsed, 
                        self.ccd_preprocessed_dict))
                all_chain_ids.append(chain_id)

        # 2. Select cropped chains
        # if sampled_chain_ids is None: 
        #     sampled_chain_ids = all_chain_ids
        # 2.0 token num filter
        chain_within_length = []
        sample_token_num = 0
        # for chain in sampled_chain_ids:
        #     sample_token_num += chain_id_2_num_token[chain]
        #     if sample_token_num < MAX_CROPPING_TOKEN_NUM:
        #         chain_within_length.append(chain)
        # sampled_chain_ids = [c for c in sampled_chain_ids if c in chain_within_length]
        for chain in all_chain_ids:
            # if chain not in sampled_chain_ids:
            sample_token_num += chain_id_2_num_token[chain]
            if sample_token_num < MAX_CROPPING_TOKEN_NUM:
                chain_within_length.append(chain)
        all_chain_ids = [c for c in all_chain_ids if c in chain_within_length]
        cropping_feat = {}
        # 2.1 non_polymer
        cropping_feat['seq_token'] = self.pipeline_token_feature.process(input_json_path,
            select_mmcif_chainID=all_chain_ids,
            ccd_preprocessed_dict=self.ccd_preprocessed_dict,ccd_output_dir=None)

        # 2.2 conf_bond
        cropping_feat['conf_bond'] = self.conf_bond_pipeline.process(input_json_path, 
            select_mmcif_chainID=all_chain_ids,
            ccd_preprocessed_dict=self.ccd_preprocessed_dict, ccd_output_dir=None)
        # 2.3 label 
        cropping_feat['labels'] = self.get_label(protein, all_chain_ids, mmcif_object=mmcif_object, 
            chain_ids=all_chain_ids)
            # seqs=[chain_to_seq[chain] for chain in selected_chains], chain_ids=selected_chains)
        if cropping_feat['labels'] is None:
            return None
        # 2.4 map asym_id to chain_id
        asym_to_chain = {}
        chain_to_asym = {}
        for token_id, chain_id in zip(cropping_feat["conf_bond"]["ref_token2atom_idx"], 
                                    cropping_feat["seq_token"]['all_chain_ids']):
            asym_to_chain[cropping_feat["seq_token"]["asym_id"][token_id]] = chain_id
            chain_to_asym[chain_id] = cropping_feat["seq_token"]["asym_id"][token_id]
        # 2.5 start cropping
        if not self.crop_size is None and self.crop_size > 0:
            chain_lens = np.bincount(cropping_feat['seq_token']['asym_id'].astype('int64'))
            seq_infos = {'seq_lens': list(chain_lens)[1:]}  
            ligand_N = len(docking_ligand_feat['atom_pos'])
            center_coords = docking_ligand_feat['atom_pos'][np.random.randint(ligand_N)]
            cropped_mask = get_crop_mask_all_atom_by_center(
                center_coords,
                cropping_feat,
                seq_infos=seq_infos,
                crop_size=self.crop_size - ligand_N)
            cropped_chains = [c for c in cropped_mask \
                              if np.sum(cropped_mask[c]) > 0]
            cropped_chains = [c for c in all_chain_ids if c in cropped_chains]
        else:
            cropped_mask = {}
            for chain_id, chain_len in zip(all_chain_ids, chain_lens):
                cropped_mask[chain_id] = np.ones(chain_len, dtype='bool')
            cropped_chains = all_chain_ids

        # 3. Feats collection
        all_feats = {}
        selected_chains = []
        # 3.1 Protein feats
        if 'protein' in raw_token_dict:
            msa_seqs = raw_token_dict["protein"]["msa_seqs"]
            prot_chain_ids = raw_token_dict["protein"]["chain_ids"]

            msa_seqs = [s for c, s in zip(prot_chain_ids, msa_seqs) if c in cropped_chains]
            prot_chain_ids = [c for c in prot_chain_ids if c in cropped_chains]

            protein_feat_selected = self.get_protein_feat(protein, seqs=msa_seqs, \
                                chain_ids=prot_chain_ids)
            if not protein_feat_selected is None:
                protein_feat, prot_chain_ids = protein_feat_selected
                
                all_feats["protein"] = protein_feat
                all_feats['protein']["ccd_seqs"] = np.concatenate([np.array(v, dtype=object) \
                            for k, v in ccd_chain_info.items() \
                            if 'protein' in k and k.split('_')[-1] in prot_chain_ids])
                selected_chains.extend(prot_chain_ids)

        # # 3.2 DNA/ RNA feats
        # TODO: (yexianbin) open when rna msa is ready;
        if "rna" in raw_token_dict: 
            msa_seqs = raw_token_dict["rna"]["msa_seqs"]
            rna_chain_ids = basic_token_dict["rna"]["chain_ids"]

            msa_seqs = [s for c, s in zip(rna_chain_ids, msa_seqs) if c in cropped_chains]
            rna_chain_ids = [c for c in rna_chain_ids if c in cropped_chains]

            rna_feat_selected = self.get_protein_feat(protein, seqs=msa_seqs, \
                                chain_ids=rna_chain_ids, seqtype='rna')
            if not rna_feat_selected is None:
                rna_feat, rna_chain_ids = rna_feat_selected
                
                all_feats["rna"] = rna_feat
                all_feats['rna']["ccd_seqs"] = np.concatenate([np.array(v, dtype=object) \
                            for k, v in ccd_chain_info.items() \
                            if 'rna' in k and k.split('_')[-1] in rna_chain_ids])

                selected_chains.extend(rna_chain_ids)

        if "dna" in raw_token_dict: 
            dna_chain_ids = basic_token_dict["dna"]["chain_ids"]
            dna_chain_ids = [c for c in dna_chain_ids if c in cropped_chains]
            if len(dna_chain_ids) > 0:
                all_feats['dna'] = {}
                all_feats['dna']["ccd_seqs"] = np.concatenate([np.array(v, dtype=object)\
                        for k, v in ccd_chain_info.items() \
                        if 'dna' in k and k.split('_')[-1] in dna_chain_ids])
            selected_chains.extend(dna_chain_ids)

        if "ligand" in raw_token_dict:
            ligand_chain_ids = basic_token_dict["ligand"]["chain_ids"]
            ligand_chain_ids = [c for c in ligand_chain_ids if c in cropped_chains]
            if len(ligand_chain_ids) > 0:
                all_feats['ligand'] = {}
                all_feats['ligand']["ccd_seqs"] = np.concatenate([np.array(v, dtype=object)\
                        for k, v in ccd_chain_info.items() \
                        if 'ligand' in k and k.split('_')[-1] in ligand_chain_ids])
            selected_chains.extend(ligand_chain_ids)

        # print('selected_chains', selected_chains)
        if len(selected_chains) == 0:
            print(f"Skip {protein} due to no valid selected chains")
            return None

        # 3.3 non_polymer
        all_feats['seq_token'] = self.pipeline_token_feature.process(input_json_path,
            select_mmcif_chainID=selected_chains,
            ccd_preprocessed_dict=self.ccd_preprocessed_dict,ccd_output_dir=None)
            
        # 3.4 conf_bond
        all_feats['conf_bond'] = self.conf_bond_pipeline.process(input_json_path, 
            select_mmcif_chainID=selected_chains,
            ccd_preprocessed_dict=self.ccd_preprocessed_dict, ccd_output_dir=None)

        # 3.5 post convert
        # TODO(zhukunrui): load parts of the chains not supported
        # if not len(selected_chains) == len(all_chain_ids):
        #     print(f"selected_chains {selected_chains} less than protein_chains {all_chain_ids}, skipped")
        #     return None  
        np_example = pipeline_hybrid._post_convert(self.ccd_preprocessed_dict, all_feats)

        np_example["seq_mask"] = np.ones_like(
            np_example['restype']).astype('float32')
        np_example["msa"], np_example['msa_mask'], np_example['deletion_matrix'] = crop_msa(np_example)

        # 4. get labels
        raw_labels = self.get_label(protein, selected_chains, mmcif_object=mmcif_object, 
            chain_ids=all_chain_ids)
            # seqs=[chain_to_seq[chain] for chain in selected_chains], chain_ids=selected_chains)
        if raw_labels is None:
            return None

        if np_example['ref_pos'].shape[0] != raw_labels['all_atom_pos'].shape[0]:
            print(f'Skip {protein} due to inequal atom num')
            return None

        ## add docking ligand feat to sample
        np_example, raw_labels = self.add_docking_ligand_feat_to_sample(
                docking_ligand_feat, np_example, raw_labels)
      
        # 5. cropping and padding
        # for_recycle = False # len(np_example['asym_id'].shape) == 2
        raw_labels['resolution'] = np.array(
            [np.mean(raw_labels['resolution'])])
        if not self.crop_size is None and self.crop_size > 0:
            chain_lens = np.bincount(np_example['asym_id'].astype('int64'))

            seq_infos = {'seq_lens': list(chain_lens)[1:]}

            # crop_and_pad_multimer
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
                self.crop_size - ligand_N,
                pad_for_shorter_seq=self.is_pad_if_crop,
                for_recycle=False)
        else:
            labels = {k: v for k, v in raw_labels.items()}
        # only copy assembly features
        for k in ['asym_id', 'sym_id', 'entity_id', 'residue_index']:
            labels[k] = np.copy(feat[k])
        feat['atom_count'] = np.bincount(feat['ref_token2atom_idx'])

        # copy perm features
        for k in ['perm_atom_index', 'perm_entity_id', 'perm_asym_id']:
            raw_labels[k] = np.copy(np_example[k])
  
        # 6. more filtering steps from hfold-multimer
        _, valid_atom_num = np.unique(feat['perm_asym_id'][labels['all_atom_pos_mask'] == 1], 
                return_counts=True)
        if len(valid_atom_num) == 0:
            print(f"Skip {protein} due to no valid chains")
            return None
        if np.max(valid_atom_num) < chain_align_np_aa.MIN_NUM_FOR_ANCHOR_CHAIN:
            print(
                f"Skip {protein} due to all chains contain less than "
                f"{chain_align_np_aa.MIN_NUM_FOR_ANCHOR_CHAIN} atoms")
            return None
        ## in case that the gpu memory may exceed limit
        if len(feat['perm_asym_id']) > len(feat['asym_id']) * 14:
            print(f"Skip {protein} due to atom_num per-token larger than 14")
            return None
        
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
            'chain_ids': ''.join(chain_ids),
        }

        sample['structure_file'] = self._get_protein_struct_file(protein)
        return sample

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = None
        while sample is None:
            data = self.data_list[index]
            try:
                sample = self.get_sample(data)
            except Exception as ex:
                print(f'[DATA] index {index} data {data} failed: {ex}')
                import traceback
                traceback.print_exc(file=sys.stdout)
            index = random.randint(0, self.__len__() - 1)
        return sample


if __name__ == '__main__':
    from tqdm import tqdm
    import json
    import ml_collections
    from helixfold.model import config
    # demo data: zhukunrui/data/protein_folding/multimer/demo/all_atom_demo
    Model_name = "allatom_demo"
    model_config = config.model_config(Model_name)
    data_config = ml_collections.ConfigDict(
        # json.load(open('./data/all_atom_demo/all_atom_demo.json', 'r'))
        # json.load(open('./data/all_atom_demo/all_atom_debug.json', 'r'))
        json.load(open('./data_configs/all_atom-multimer_clust.json'))
        # json.load(open('./data/all_atom_demo_prot_rna_ligand/all_atom_demo.json', 'r'))
        # json.load(open('./data/all_atom_demo_rna_dna/all_atom_rna_dna.json', 'r'))
        # json.load(open('./data/all_atom_demo_rna_msa/all_atom_rna_dna.json', 'r'))
        # json.load(open('./data/posebuster_demo/demo_config.json', 'r'))
        )
    data_config.train.ligand_structure_dir = './xianbin/output_tmp'
    train_dataset = AllAtomLigandDistillDataset(model_config=model_config,
                                    data_config=data_config.train,
                                    crop_size=384,
                                    trainer_id=0,
                                    trainer_num=1,
                                    is_pad_if_crop=True)

    data = train_dataset[0]

