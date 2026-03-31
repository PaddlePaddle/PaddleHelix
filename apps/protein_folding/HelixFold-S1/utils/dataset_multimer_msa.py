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
from paddle.io import IterableDataset, DataLoader

from utils.modules_diffusion import Diffuser
from omegaconf import OmegaConf
from helixfold.data import mmcif_parsing, pipeline
from helixfold.data import pipeline_multimer
from helixfold.data.data_utils import a3m_to_features, load_chain, generate_label, load_pdb_chain, sample_raw_msa
from helixfold.data.utils import crop_and_pad_multimer, get_heterodimer_chains
from helixfold.data.utils import make_pair_pocket_mask_by_receptor_residue, \
     make_pair_pocket_mask_by_receptor_res_idx
from helixfold.model import features
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

def get_rigid_template(seq, protein_chain_d=None):
    """
    create rigid template
    regurn zeros template if protein_chain_d is None
    """
    rigid_template = {}
    seq_len = len(seq)

    # create zeros template
    rigid_template["template_sequence"] = seq
    rigid_template["template_aatype"] = residue_constants.sequence_to_onehot(
                sequence=rigid_template["template_sequence"], 
                mapping=residue_constants.HHBLITS_AA_TO_ID
            )
    rigid_template["template_all_atom_masks"] = np.zeros(shape=(seq_len, 37), dtype="float32")
    rigid_template["template_all_atom_positions"] = np.zeros(shape=(seq_len, 37, 3), dtype="float32")
    rigid_template["template_domain_names"] = "self"
    rigid_template["template_sum_probs"] = [0.]

    # get atom pos from protein_chain_d
    if protein_chain_d is not None:
        rigid_template["template_all_atom_masks"] = protein_chain_d["all_atom_mask"].astype("float32")
        rigid_template["template_all_atom_positions"] = protein_chain_d["all_atom_positions"]
    
    return rigid_template

def _process_single_chain(sequence: str, description: str):
    """Runs the monomer pipeline on a single chain."""
    a3m_str = f'>{description}\n{sequence}'
    chain_features = a3m_to_features([a3m_str])
    return chain_features


def add_num_sym(all_chain_features):
    """Add num_sym feature"""
    chains = list(all_chain_features.keys())
    num_sym = dict()
    for ch in chains:
        ch = ch.split('_')[0]
        num_sym[ch] = num_sym.get(ch, 0) + 1

    for ch in chains:
        ch_ = ch.split('_')[0]
        shape = all_chain_features[ch]['asym_id'].shape
        dtype = all_chain_features[ch]['asym_id'].dtype
        all_chain_features[ch]['num_sym'] = \
            num_sym[ch_] * np.ones(shape, dtype=dtype)

    return all_chain_features


def pair_and_merge(all_chain_features):
    """Runs processing on features to augment, pair and merge.

    Args:
        all_chain_features: A MutableMap of dictionaries of features for each chain.

    Returns:
        A dictionary of features.
    """
    feature_processing.process_unmerged_features(all_chain_features)
    np_chains_list = list(all_chain_features.values())
    MAX_TEMPLATES = 4  # actually, single doesn't need template
    np_example = msa_pairing.merge_chain_features(
        np_chains_list=np_chains_list,
        pair_msa_sequences=False,
        max_templates=MAX_TEMPLATES)
    np_example = feature_processing.process_final(np_example)
    return np_example


def read_fasta(fasta_path):
    """
    fasta_info: {
        chain_id: {
            'seq': sequence,
            'desc': description,
        },
    }
    """
    with open(fasta_path) as f:
        input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    chain_ids = [
        input_desc.split(" ")[0].split("_")[1] for input_desc in input_descs
    ]
    fasta_info = {
        c: {
            'seq': s,
            'desc': d
        }
        for c, s, d in zip(chain_ids, input_seqs, input_descs)
    }
    return fasta_info


def fasta_info_to_multimer_feature(fasta_info, chain_ids, model_config):
    """
    chain_ids: list of chain_id, selected to extract features
    """
    input_seqs = [fasta_info[c]['seq'] for c in chain_ids]
    input_descs = [fasta_info[c]['desc'] for c in chain_ids]

    chain_id_map = _make_chain_id_map(sequences=input_seqs,
                                      descriptions=input_descs)
    all_chain_features = {}
    sequence_features = {}
    processed_features = {}
    for chain_id, fasta_chain in chain_id_map.items():
        ### gen_input_feat
        a3m_str = f">tr|{fasta_chain.description}\n{fasta_chain.sequence}\n"
        raw_features = a3m_to_features([a3m_str])
        processed_feature_dict = features.np_example_to_features(
            np_example=raw_features, config=model_config)
        for k, v in processed_feature_dict.items():
            if k not in NEED_FEATURES:
                continue
            processed_features[k] = processed_features.get(k, [])
            processed_features[k].append(v)

        if fasta_chain.sequence in sequence_features:
            all_chain_features[chain_id] = deepcopy(
                sequence_features[fasta_chain.sequence])
            continue
        chain_features = _process_single_chain(
            sequence=fasta_chain.sequence, description=fasta_chain.description)
        chain_features = convert_monomer_features(chain_features,
                                                  chain_id=chain_id)
        all_chain_features[chain_id] = chain_features
        sequence_features[fasta_chain.sequence] = chain_features

    all_chain_features = add_assembly_features(all_chain_features)
    all_chain_features = add_num_sym(all_chain_features)
    np_example = pair_and_merge(all_chain_features=all_chain_features)

    for k, v in processed_features.items():
        axis = NEED_FEATURES[k]
        processed_features[k] = np.concatenate(v, axis=axis)[0, ...]
    np_example.update(processed_features)
    return np_example


def merge_chain_labels(chain_labels, chain_ids):
    raw_labels = {}
    for key in chain_labels[chain_ids[0]].keys():
        raw_labels[key] = np.concatenate(
            [chain_labels[c][key] for c in chain_ids], axis=0)
    return raw_labels

def get_pseudo_pocket(recptor_pos, recptor_mask, 
        ligand_pos_list, ligand_mask_list, ca_ca_threshold=10):
    """locate residue idx in receptor"""
    ca_idx = residue_constants.atom_order['CA']
    receptor_ca_coords = recptor_pos[..., ca_idx, :]
    receptor_ca_mask = recptor_mask[..., ca_idx].astype('bool')
    pocket_idx = []
    for ligand_pos, ligand_mask in zip(ligand_pos_list, ligand_mask_list):
        # ref to crop_spatial
        ligand_ca_coords = ligand_pos[..., ca_idx, :]
        ligand_ca_mask = ligand_mask[..., ca_idx].astype('bool')

        # random sample pocket by distance
        coord_mask = receptor_ca_mask[..., None] * ligand_ca_mask[..., None, :]  # [len_receptor, len_ligand]
        coord_diff = np.expand_dims(receptor_ca_coords, -2) \
                     - np.expand_dims(ligand_ca_coords, -3)  # [len_receptor, len_ligand, 3]

        ca_distances = np.sqrt(np.sum(coord_diff ** 2, axis=-1))  # [len_receptor, len_ligand]
        ca_distances = ca_distances * coord_mask  # [len_receptor, len_ligand]

        len_receptor, len_ligand = ca_distances.shape

        cnt_interfaces = np.sum((ca_distances > 0) & (ca_distances < ca_ca_threshold), axis=-2)  # [len_receptor]
        interface_candidates = cnt_interfaces.nonzero()[0]  # [num_candidates]

        if np.any(interface_candidates):
            target_res = int(np.random.choice(interface_candidates))
            to_target_distances = ca_distances[: , target_res]  # map to [len_receptor]

            to_target_distances[~receptor_ca_mask] = np.infty
            break_tie = (np.arange(0, to_target_distances.shape[-1]).astype('float') * 1e-3)
            to_target_distances += break_tie  # [len_receptor]

            crop_size = np.random.randint(20, min(100, len_ligand * 5))  # random pocket size
            ret = np.argsort(to_target_distances)[:crop_size]
            ret.sort()  # rest index with shortest distance from selected ligand res
            pocket_idx.append(ret)

    return pocket_idx

def get_res_is_pocket(all_chain_features, selected_receptor_chain, all_chain_atom_pos,
                       all_chain_atom_mask, train_with_pocket_ratio):
    """ highlight pocket by assigning 1.0 to resdues that are selected. """
    res_is_pocket = {}
    for chain_id, chain_features in all_chain_features.items():
        # init res_is_pocket with zeros
        res_is_pocket[chain_id] = np.zeros_like(chain_features["aatype"].argmax(-1))
        # assign pocket info
        if chain_id == selected_receptor_chain:
            other_chains = [c for c in all_chain_features.keys()  if not c == chain_id]

            # provide pocket embedding with train_with_pocket_ratio
            if random.random() < train_with_pocket_ratio: # self.data_config.get("train_with_pocket_ratio", 0):
                pocket_ids = get_pseudo_pocket(
                    recptor_pos=all_chain_atom_pos[chain_id],
                    recptor_mask=all_chain_atom_mask[chain_id],
                    ligand_pos_list=[all_chain_atom_pos[c] for c in other_chains],
                    ligand_mask_list=[all_chain_atom_mask[c] for c in other_chains]
                )
                if len(pocket_ids) > 0:
                    # merge pocket for multiple liangd chains
                    pocket_ids = list(set(np.concatenate(pocket_ids))) 
                    res_is_pocket[chain_id][pocket_ids] = 1.0
    return res_is_pocket

class MultimerDataset(paddle.io.Dataset):
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
        self.model_config.data.eval.delete_msa_block = delete_msa_block
        assert self.model_config.model.global_config.multimer_mode

        self.data_config = data_config
        self.crop_size = crop_size
        self.is_pad_if_crop = is_pad_if_crop
        self.trainer_id = trainer_id
        self.trainer_num = trainer_num
        self.is_shuffle = is_shuffle

        self.data_mode = data_config.get('data_mode', 'multimer')
        self.train_with_rigid = data_config.get("train_with_rigid", False)

        def _assert_exists(path):
            assert exists(path), path

        _assert_exists(self.data_config.protein_clust_file)
        _assert_exists(self.data_config.feature_dir)
        _assert_exists(self.data_config.structure_dir)

        ## check and filter (bad or not needed) proteins
        self.protein2seq_map = self._load_protein2seq_map()
        self.fasta2id_map = self._load_fasta2id_map()
        self.clusts, falied_clust = self._load_clusts()
        self.clusts_to_consume = [[] for _ in range(len(self.clusts))]

        def _print_attribute(key, value):
            print(f'[{self.__class__.__name__}] {key}: {value}')

        _print_attribute('protein_clust_file',
                         self.data_config.protein_clust_file)
        _print_attribute('trainer_id/trainer_num',
                         f'{self.trainer_id}/{self.trainer_num}')
        _print_attribute('clust_num', len(self.clusts))
        _print_attribute('falied clust_num', len(falied_clust))
        _print_attribute('protein_num', np.sum([len(x) for x in self.clusts]))
        _print_attribute('crop_size', crop_size)
        _print_attribute('is_pad_if_crop', is_pad_if_crop)
        _print_attribute('data_mode', self.data_mode)
        _print_attribute('delete_msa_block', delete_msa_block)
        assert np.sum([len(x) for x in self.clusts]) > 0, f"protein_num {np.sum([len(x) for x in self.clusts])} <= 0"

        if self.data_config.get("apply_rotation", False):
            diffuser_conf = OmegaConf.load("utils/frame_diff/config/multimer.yaml")
            self.diffuser = Diffuser(diffuser_conf.diffuser)

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

    def _load_clusts(self):
        with open(self.data_config.protein_clust_file) as r:
            lines = r.readlines()[self.trainer_id::self.trainer_num]
        clusts, failed_clusts = [], []
        for line in lines:
            clust = []
            for chain_name in line.split():
                protein = chain_name.split('_')[0]
                if self._check_protein(protein):
                    clust.append(protein)
            if len(clust) > 0:
                clusts.append(clust)
            else:
                failed_clusts.append(line.split()[0])
        return clusts, failed_clusts

    def _check_protein(self, protein):
        """tbd."""
        fasta_file = self._get_protein_fasta_file(protein)
        # Check fasta_file
        if not os.path.exists(fasta_file):
                return False 
        
        # Check chain features
        seqs, descs = parsers.parse_fasta(open(fasta_file, 'r').read())
        chain_ids = [i.split()[0].split('_')[1] for i in descs]  
        if len(chain_ids) == 0: return False 

        min_seq_len = 1000
        num_features = 0 
        for seq, chain_id in zip(seqs, chain_ids):
            if os.path.exists(self._get_protein_feature_file(protein+"_"+chain_id, seq=seq)):
                num_features += 1
                min_seq_len = min(len(seq), min_seq_len)
            else: 
                if self.data_config.get("check_all_chain_features",True): 
                    return False  # Keep protein if all chain feature exists
                else:  pass # Keep protein if at least 1 chain feature exists
        if self.data_config.get("peptide_only", False):
            if not min_seq_len < 50: return False  # peptide reuqires the length of the shortest chain < 50
        if num_features < 1: return False 

        if self.data_mode == 'multimer' and num_features == 1: return False
        if self.data_mode == 'monomer' and num_features > 1: return False
        if self.data_config.get("skip_homo", False): 
            if len(set(seqs)) == 1: return False

        # Check structure file
        if not os.path.exists(self._get_protein_struct_file(protein)):
            return False
        return True

    def _get_protein_feature_file(self, protein, seq):
        """tbd."""
        if protein in self.protein2seq_map: protein = self.protein2seq_map[protein] 
        if seq in self.fasta2id_map: protein = self.fasta2id_map[seq]
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

    def _get_protein_fasta_file(self, protein):
        """tbd."""
        return join(self.data_config.fasta_dir, f"{protein}.fasta")

    def get_input_feat(self, protein, seqs, chain_ids, mmcif_object=None):
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

        # c_arange = np.arange(len(chain_ids))
        # if self.is_shuffle: np.random.shuffle(c_arange)  # shuffle indices before chain selection
        # for (chain_id, seq) in zip([chain_ids[c_id] for c_id in c_arange], [seqs[c_id] for c_id in c_arange]):
        for (chain_id, seq) in zip(chain_ids, seqs):
            seq_len = len(seq)
            if os.environ.get('DCU_MODE_SOFTMAX_CLIP', '0') == '1':
                # fix _queue.Empty error by mas_paring block_diag
                # skip feature when full sequence length exceeded
                if full_seq_length + seq_len > MAX_DCU_SEQ_SIZE: 
                    print(f" {protein} {chain_id} seq_length({seq_len}) + full_seq_length({full_seq_length}) exceeded, skip")
                    break
            else: 
                if full_seq_length + seq_len > MAX_SEQ_SIZE: 
                    print(f" {protein} {chain_id} seq_length({seq_len}) + full_seq_length({full_seq_length}) exceeded, skip")
                    break

            features_pkl = self._get_protein_feature_file(protein+"_"+chain_id, seq=seq)
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
            
            protein_chain_d = load_chain(mmcif_object, chain_id)
            if not protein_chain_d['all_atom_positions'].shape[0] == seq_len: 
                print(f"Skip {protein}_{chain_id} due to inequal residue num \
                      between fasta({seq_len}) and protein_chain_d['all_atom_positions'](\
                      {protein_chain_d['all_atom_positions'].shape})")
                continue

            if self.data_config.get("train_rigid_template_ratio", 0.) > 0:
                # provide single chain rigid info with train_rigid_template_ratio
                if random.random() < self.data_config.get("train_rigid_template_ratio", 0.):
                    assert not self.data_config.get("feature_pkl_has_label", False), \
                                "label from feature_pkl not support yet"
                    rigid_template = get_rigid_template(seq, protein_chain_d=protein_chain_d)

                    # add rigid tempate to the front of raw_features template
                    for key in rigid_template: 
                        raw_features[key] = np.concatenate(
                            ([rigid_template[key]], raw_features[key]), axis=0
                        )

            all_chain_features[chain_id] = raw_features
            full_seq_length += seq_len
            selected_chains.append(chain_id)
            all_chain_atom_pos[chain_id] = protein_chain_d["all_atom_positions"]
            all_chain_atom_mask[chain_id] = protein_chain_d["all_atom_mask"]
            if seq_len > max_chain_len:
                max_chain_len = seq_len
                selected_receptor_chain = chain_id

        print(f"{protein} full_seq_length:", full_seq_length)
        if full_seq_length == 0 : return None
        raw_features = pipeline_multimer.process_with_all_chain_features(all_chain_features)
        
        if self.is_shuffle and _random_drop(raw_features):
            return None

        processed_feature_dict = features.np_example_to_features(
            np_example=raw_features,
            config=self.model_config)

        # add pocket info
        if self.data_config.get("train_with_pocket_ratio", 0) > 0:
            # init pocket info with zeros
            processed_feature_dict["res_is_pocket"] = np.zeros_like(processed_feature_dict["aatype"].argmax(-1))
            # locate pocket
            res_is_pocket = get_res_is_pocket(all_chain_features, selected_receptor_chain, all_chain_atom_pos,
                                all_chain_atom_mask, self.data_config.get("train_with_pocket_ratio", 0))
            num_repeated = processed_feature_dict['asym_id'].shape[0]
            processed_feature_dict["res_is_pocket"] = np.concatenate([res_is_pocket[c]  # [4, num_res]
                                for c in all_chain_features.keys()])[None, : ].repeat(num_repeated, axis=0)

        return processed_feature_dict, selected_chains

    def get_label(self, protein, raw_features, selected_chains, mmcif_object=None, chain_ids=None, seqs=None):
        """tbd."""
        label_list = []
        if self.data_config.get("feature_pkl_has_label",False):
            for asym_id in np.unique(raw_features['asym_id']):
                protein_chain = dict()
                for k in ['aatype', 'all_atom_positions', 'all_atom_mask']:
                    k_ = k if k != 'aatype' else 'aatype_index'
                    protein_chain[k_] = raw_features[k][
                        np.where(raw_features['asym_id'] == asym_id)[0]]

                protein_label = generate_label(protein_chain)
                label_list.append(protein_label)

        else:
            if self.data_config.get("apply_rotation", False):  
                rot_array = get_rotation_mat(self.diffuser)

            for seq, chain_id in zip(seqs, chain_ids):
                seq_len = len(seq)
                if not chain_id in selected_chains: continue
                protein_chain_d = load_chain(mmcif_object, chain_id)
                if self.data_config.get("apply_rotation", False):
                    protein_chain_d["all_atom_positions"] = rotate_all_atom_positions( \
                                protein_chain_d["all_atom_positions"], rot_array=rot_array)

                if protein_chain_d['resolution'] > 9:
                    print(
                        f'Skip {protein} due to low resolution {protein_chain_d["resolution"]}'
                    )
                    continue
                protein_label = generate_label(protein_chain_d)
                if seq_len != protein_label['aatype_index'].shape[0]:
                    print(f'Skip {protein} due to inequal residue num')
                    continue
                label_list.append(protein_label)

        raw_labels = {}
        if len(label_list) == 0: return None
        for key in label_list[0].keys():
            raw_labels[key] = np.concatenate([l[key] for l in label_list],
                                             axis=0)
        ## copy assembly features
        for k in ['asym_id', 'sym_id', 'entity_id', 'residue_index']:
            raw_labels[k] = np.array(raw_features[k][0])
        return raw_labels

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
        fasta_file = self._get_protein_fasta_file(protein)
        seqs, descs = parsers.parse_fasta(open(fasta_file, 'r').read())
        chain_ids = [i.split()[0].split('_')[1] for i in descs]
        chain_to_seq = dict(zip(chain_ids, seqs))

        # filter chain_ids that mmcif can not cover
        mmcif_object = None
        if not self.data_config.get("feature_pkl_has_label", False):
            protein_struct_file = self._get_protein_struct_file(protein)
            if protein_struct_file.endswith('.cif.gz'):
                with gzip.open(protein_struct_file, 'r') as f:
                    cif_string = f.read().decode('utf8')
            else:
                cif_string = "".join(open(protein_struct_file, 'r').readlines())
            parse_result = mmcif_parsing.parse(file_id=protein_struct_file,
                                            mmcif_string=cif_string)
            mmcif_object = parse_result.mmcif_object
            mmcif_chain_keys = mmcif_object.chain_to_seqres.keys()
            valid_seq_chains = [(seq, cid) for seq, cid in zip(seqs, chain_ids) if cid in mmcif_chain_keys]
            chain_ids = [cid for _, cid in valid_seq_chains]
            seqs = [seq for seq, _ in valid_seq_chains]

        feat_selected = self.get_input_feat(protein, seqs=seqs, chain_ids=chain_ids, mmcif_object=mmcif_object)
        if feat_selected is None:
            return None
        feat, selected_chains = feat_selected

        raw_labels = self.get_label(protein, feat, selected_chains, mmcif_object=mmcif_object, 
            seqs=seqs, chain_ids=chain_ids)
            # seqs=[chain_to_seq[chain] for chain in selected_chains], chain_ids=selected_chains)
        if raw_labels is None:
            return None

        if self.data_config.get("skip_homo", False):
            feat, raw_labels = get_heterodimer_chains(feat, raw_labels, prot_name=protein, 
                                    disable_random=self.data_config.get("fixed_heter", False))
            if (feat is None) or (raw_labels is None):
                print(f"Skip {protein} due to get_heterodimer_chains errro!")
                return None

        raw_labels['resolution'] = np.array(
            [np.mean(raw_labels['resolution'])])

        for_recycle = len(feat['asym_id'].shape) == 2
        if not self.crop_size is None and self.crop_size > 0:
            if for_recycle:
                # different recycle inputs share same chain lengths
                chain_lens = np.bincount(feat['asym_id'][0].astype('int64'))
            else:
                chain_lens = np.bincount(feat['asym_id'].astype('int64'))

            seq_infos = {'seq_lens': list(chain_lens)[1:]}

            # crop_and_pad_multimer
            # also copy assembly features
            feat, labels = crop_and_pad_multimer(
                feat,
                raw_labels,
                seq_infos=seq_infos,
                crop_size=self.crop_size,
                pad_for_shorter_seq=self.is_pad_if_crop,
                for_recycle=for_recycle, spatial_crop_ratio=self.data_config.get("spatial_crop_ratio", 0.5))

        else:
            labels = {k: v for k, v in raw_labels.items()}
            # only copy assembly features
            for k in ['asym_id', 'sym_id', 'entity_id', 'residue_index']:
                if for_recycle:
                    labels[k] = np.copy(feat[k][0])
                else:
                    labels[k] = np.copy(feat[k])

        ## create pair_pocket_mask
        pair_pocket_mask_ratio = self.model_config.model.global_config.get('use_pair_pocket_mask', 0.)
        if  pair_pocket_mask_ratio > 0:
            pocket_size = self.model_config.model.global_config.get('infer_pocket_mask_size', 20.)
            if random.random() < pair_pocket_mask_ratio:
                pair_pocket_mask = make_pair_pocket_mask_by_receptor_residue(
                        labels['all_atom_positions'], 
                        labels['all_atom_mask'], 
                        labels['asym_id'],
                        seq_mask=feat['seq_mask'][0],
                        threshold_min=(pocket_size * 0.5), 
                        threshold_max=(pocket_size * 1.5),
                        max_inter_residue_num=5)
                feat['pair_pocket_mask'] = np.tile(pair_pocket_mask[None], 
                        [feat['asym_id'].shape[0], 1, 1])
    
        ca_idx = residue_constants.atom_order['CA']
        ca_mask = labels['all_atom_mask'][:, ca_idx]
        print(
            f"After crop and pad, {protein} valid_rate is {np.mean(ca_mask)}")

        asym_id = (labels['asym_id'] * ca_mask).astype(np.int32)
        valid_chain_lens = np.bincount(asym_id)

        print(f"{protein} cropped asym_id length: {valid_chain_lens}")

        # At least one chain contains more than 10 residues
        # TODO: remove this limitation for protein-peptide complex!
        if self.data_config.get("check_shortest_chain", True) and not np.any(valid_chain_lens[1:] > 10):
            print(
                f"Skip {protein} due to all chains contain less than 10 residues"
            )
            return None
        if not np.all(valid_chain_lens[1:] != 1):
            print(f"Skip {protein} due to some chains contains 1 residues")
            return None
        if raw_labels['all_atom_positions'].shape[0] != raw_labels['asym_id'].shape[0]:
            print(f"Skip {protein} due to inequal num res in raw_labels \
                  all_atom_positions({raw_labels['all_atom_positions'].shape}) \
                  and asym_id:({raw_labels['asym_id'].shape})")
            return None
        if labels['all_atom_positions'].shape[0] != labels['asym_id'].shape[0]:
            print(f"Skip {protein} due to inequal num res in labels \
                  all_atom_positions({labels['all_atom_positions'].shape}) \
                  and asym_id:({labels['asym_id'].shape})")
            return None

        sample = {
            'name': protein,
            'feat': feat,
            'label_cropped': labels,
            'label': raw_labels,
            'chain_ids': ''.join(chain_ids),
        }

        sample['structure_file'] = self._get_protein_struct_file(protein)
        if hasattr(self.data_config, 'fasta_file'):
            sample['fasta_file'] = self._get_protein_fasta_file(protein)

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


class MultimerDistillDataset(paddle.io.Dataset):
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
        self.model_config.data.eval.delete_msa_block = delete_msa_block
        assert self.model_config.model.global_config.multimer_mode

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
            chain_feature["template_aatype"] = np.zeros((4, num_residues, 22)) # [num_templates, num_residues, amino_acids]
            chain_feature["template_all_atom_masks"] = np.zeros((4, num_residues, 37))  # [num_templates, num_residues, bins]
            chain_feature["template_all_atom_positions"] = np.zeros((4, num_residues, 37, 3))  # [num_templates, num_residues, bins, coords]

        # downsample msa
        if 'msa_sample' in self.data_config:
            chain_feature = sample_raw_msa(chain_feature, self.data_config)
        ## monomer features to multimer features
        all_chain_features = {'A': chain_feature}
        raw_features = pipeline_multimer.process_with_all_chain_features(
                all_chain_features)
        processed_feature_dict = features.np_example_to_features(
            np_example=raw_features,
            config=self.model_config)
        if self.data_config.get("train_with_pocket_ratio", 0) > 0:
            num_residues = processed_feature_dict["aatype"].shape[1]
            processed_feature_dict["res_is_pocket"] = np.zeros((4, num_residues), dtype=np.int64)
        return processed_feature_dict

    def get_label(self, protein, raw_features):
        protein_struct_file = self._get_protein_struct_file(protein)
        # chain_id = protein.split('_')[1] if len(protein.split('_')) >=2 else None
        chain_id = None
        with gzip.open(protein_struct_file, 'r') as f:
            pdb_file = f.read().decode('utf8')
        prot_obj = protein_utils.from_pdb_string(pdb_file, chain_id)
        protein_chain_d = load_pdb_chain(prot_obj,
                confidence_threshold=self.data_config.get('confidence_threshold', 80.0))
        protein_label = generate_label(protein_chain_d)
        ## check plddt score
        tmp_pdb_val = 0
        for i in range(len(prot_obj.b_factors)):
            tmp_pdb_val += prot_obj.b_factors[i][0]
        plddt_score = tmp_pdb_val / len(prot_obj.b_factors)
        plddt_thres = self.data_config.get('plddt_score', 50.0)
        if plddt_score < plddt_thres:
            print(f'Skip {protein} due to pLDDT confidence smaller than {plddt_thres}')
            return None
        ## copy assembly features
        for k in ['asym_id', 'sym_id', 'entity_id', 'residue_index']:
            protein_label[k] = np.array(raw_features[k][0])
        return protein_label

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

        feat = self.get_input_feat(protein)
        if feat is None:
            return None
        raw_labels = self.get_label(protein, feat)
        if raw_labels is None:
            return None
        if feat['aatype'].shape[1] != raw_labels['aatype_index'].shape[0]:
            print(f'Skip {protein} due to inequal residue num')
            return None

        raw_labels['resolution'] = np.array(
            [np.mean(raw_labels['resolution'])])

        for_recycle = len(feat['asym_id'].shape) == 2
        seq_infos = {'seq_lens': [feat['aatype'].shape[1]]}
        feat, labels = crop_and_pad_multimer(
                feat,
                raw_labels,
                seq_infos=seq_infos,
                crop_size=self.crop_size,
                pad_for_shorter_seq=self.is_pad_if_crop,
                for_recycle=for_recycle)

        # [num_residues, 37] -> [num_residues] -> num_unmasked_residues
        num_unmasked_residues = (labels['all_atom_mask'].sum(-1)>0).sum()
        if num_unmasked_residues < 3:
            print(f'Skip {protein} due to num_unmasked_residues smaller than 3 after crop')
            return None   

        sample = {
            'name': protein,
            'feat': feat,
            'label_cropped': labels,
            'label': raw_labels,
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


class MultimerTestDataset(IterableDataset):
    """Dataset for test set inference."""

    def __init__(self,
                 model_config,
                 data_config,
                 trainer_id=0,
                 trainer_num=1):
        self.model_config = deepcopy(model_config)
        self.model_config.data.eval.delete_msa_block = False
        self.data_config = data_config
        self.trainer_id = trainer_id
        self.trainer_num = trainer_num

        def _assert_exists(path):
            assert exists(path), path

        _assert_exists(self.data_config.protein_list_file)
        _assert_exists(self.data_config.feature_dir)
        _assert_exists(self.data_config.fasta_dir)

        if hasattr(self.data_config, 'structure_dir'):
            # test dataset should have this config,
            _assert_exists(self.data_config.structure_dir)

        ## check and filter (bad or not needed) proteins
        self.protein2seq_map = self._load_protein2seq_map()
        self.fasta2id_map = self._load_fasta2id_map()
        self.protein_infos, failed_proteins = self._get_protein_infos()

        def _print_attribute(key, value):
            print(f'[{self.__class__.__name__}] {key}: {value}')

        _print_attribute('protein_list_file',
                         self.data_config.protein_list_file)
        _print_attribute('feature_dir', self.data_config.feature_dir)
        _print_attribute('fasta_dir', self.data_config.fasta_dir)
        _print_attribute('trainer_id/trainer_num',
                         f'{self.trainer_id}/{self.trainer_num}')
        _print_attribute('protein_num', len(self.protein_infos))
        _print_attribute('failed_protein_num', len(failed_proteins))
        _print_attribute('failed_proteins', failed_proteins)

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
            proteins_list = r.readlines()[self.trainer_id::self.trainer_num]
            proteins_list = [p.strip() for p in proteins_list]

        valid_proteins, failed_proteins = [], []
        for protein_info in proteins_list:
            if not self._check_protein(protein_info):
                failed_proteins.append(protein_info)
            else:
                valid_proteins.append(protein_info)

        return valid_proteins, failed_proteins

    def _check_protein(self, protein_info):
        # read protein_name and chain_ids from protein
        # protein_info: jsonline in {"protein_name": "PROT", "chains": [{"id": "ID"},
        #  {"id": "ID", "is_ligand": TRUE}, … ]} format

        # check jsonline format
        try: protein_list_item = json.loads(protein_info)
        except: return False

        protein_list_item = protein_list_item = json.loads(protein_info)
        protein_name = protein_list_item["protein_name"]
        targeted_chains = [item["id"] for item in protein_list_item["chains"]]

        # Check structure file
        fasta_file = self._get_protein_fasta_file(protein_name)
        if not exists(fasta_file):
            return False
        seqs, descs = parsers.parse_fasta(open(fasta_file, 'r').read())
        chain_ids = [i.split()[0].split('_')[1] for i in descs]  

        # filter out chains that are not mentioned in targeted_chains
        seqs = [s for c, s in zip(chain_ids, seqs) if c in targeted_chains]
        chain_ids = [c for c in chain_ids if c in targeted_chains]

        if self.data_config.get("load_from_model_input",False):
            model_input_pkl = os.path.join(self.data_config.feature_dir, 
                            protein_name + "_" + "_".join(
                                [chain_info["id"] for chain_info in protein_list_item["chains"]]
                                ), self.data_config.model_input_pkl_name)
            if not os.path.exists(model_input_pkl): 
                model_input_pkl = os.path.join(self.data_config.feature_dir, self.data_config.model_input_pkl_name)
            if not os.path.exists(model_input_pkl): return False 
        elif self.data_config.get("load_complete_feature",False): # load complete protein feature from one pkl file
            # check feature file with complete feature
            if not os.path.exists(self._get_protein_feature_file(protein_name, seq=None)):
                return False
        else: # load protein feature from sigle chain pkl files
            for seq, chain_id in zip(seqs, chain_ids):  
                if not os.path.exists(self._get_protein_feature_file(protein_name + "_" + chain_id, seq=seq)):
                    print(self._get_protein_feature_file(protein_name + "_" + chain_id, seq=seq) + "not exist")
                    return False
        return True

    def _get_protein_feature_file(self, protein, seq):
        """tbd."""
        if protein in self.protein2seq_map: protein = self.protein2seq_map[protein]
        if seq in self.fasta2id_map: protein = self.fasta2id_map[seq]
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

    def _get_protein_fasta_file(self, protein):
        """tbd."""
        return join(self.data_config.fasta_dir, f"{protein}.fasta")

    def get_raw_feat(self, protein_name, chain_info, seq=None, mmcif_object=None):
        """tbd."""
        # check if chain is ligand
        # chain_info {"id": "ID", "is_ligand": TRUE}
        chain_id = chain_info["id"]
        protein = protein_name + "_" + chain_id
        if not self.data_config.get("load_complete_feature",False): 
            is_ligand = chain_info.get("is_ligand", False)

        features_pkl = self._get_protein_feature_file(protein, seq=seq)
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

        # set zeros template by default
        seq_len = len(seq)
        rigid_template = get_rigid_template(seq, protein_chain_d=None)
        for key in rigid_template: 
            if key in raw_features: pass
            else: raw_features[key] = np.array([rigid_template[key]])

        # add ground truth template if available 
        if self.data_config.get("test_with_rigid_template",False):
            assert not self.data_config.get("load_complete_feature",False), "test_with_rigid_template not supported in load_complete_feature mode"
            if not is_ligand:
                protein_chain_d = load_chain(mmcif_object, chain_id)
                rigid_template = get_rigid_template(seq, protein_chain_d=protein_chain_d)

                # add rigid tempate to the front of raw_features template
                for key in rigid_template: 
                    if key in raw_features:
                        raw_features[key] = np.concatenate(
                            ([rigid_template[key]], raw_features[key])
                        )
        print(f'{protein} {chain_info} is_ligand: {is_ligand} \
                test_with_rigid_template:  {self.data_config.get("test_with_rigid_template",False)}  \
                template_all_atom_positions: {raw_features["template_all_atom_positions"].mean()}')
 
        # downsample msa
        if 'ligand_msa_sample' in self.data_config:
            assert not self.data_config.get("load_complete_feature",False), "ligand_msa_sample not supported in load_complete_feature mode"
            if is_ligand:
                raw_features = sample_raw_msa(raw_features, self.data_config, 
                                                min_depth=self.data_config.ligand_msa_sample.min_depth,
                max_depth=self.data_config.ligand_msa_sample.max_depth)

        return raw_features

    def get_input_feat(self, protein, seqs, chain_info_list, random_seed, mmcif_object):
        """tbd."""
        if self.data_config.get("load_complete_feature",False): 
            # load directly from feature file with complete feature
            features_pkl = os.path.join(self.data_config.feature_dir, protein, 'features.pkl.gz')
            with gzip.open(features_pkl, 'rb') as pkl:
                raw_features = pickle.load(pkl)
            if 'template_all_atom_mask' in raw_features:
                raw_features['template_all_atom_masks'] = raw_features['template_all_atom_mask']
            # return raw_features
        else: 
            # load from single chain and combine 
            all_chain_features = {}
            is_ligand_chain = {}
            all_chain_atom_pos = {}
            all_chain_atom_mask = {}
            for seq, chain_info in zip(seqs, chain_info_list):
                # chain_info {"id": "ID", "is_ligand": TRUE}
                chain_id = chain_info["id"]
                is_ligand = chain_info.get("is_ligand", False)
                all_chain_features[chain_id] = self.get_raw_feat(protein, chain_info, 
                            seq=seq, mmcif_object=mmcif_object)

                is_ligand_chain[chain_id] = is_ligand
                if mmcif_object is None: continue
                protein_chain_d = load_chain(mmcif_object, chain_id)
                all_chain_atom_pos[chain_id] = protein_chain_d["all_atom_positions"]
                all_chain_atom_mask[chain_id] = protein_chain_d["all_atom_mask"]
            raw_features = pipeline_multimer.process_with_all_chain_features(all_chain_features)

        processed_features = features.np_example_to_features(
            np_example=raw_features,
            config=self.model_config,
            random_seed=random_seed)
        
        # add pocket infor
        if self.data_config.get("test_with_pocket", False):
            # init pocket info with zeros
            processed_features["res_is_pocket"] = np.zeros_like(processed_features["aatype"].argmax(-1))
            # locate pocket
            receptor_chains, ligand_chains = [], []
            for chain_id in all_chain_features.keys():
                if is_ligand_chain[chain_id]: ligand_chains.append(chain_id)
                else: receptor_chains.append(chain_id)

            if len(receptor_chains) > 0 and len(ligand_chains) > 0: 
                res_is_pocket = get_res_is_pocket(all_chain_features, receptor_chains[0],
                                all_chain_atom_pos, all_chain_atom_mask, np.inf)
                num_repeated = processed_features['asym_id'].shape[0]
                processed_features["res_is_pocket"] = np.concatenate([res_is_pocket[c]  # [4, num_res]
                                    for c in all_chain_features.keys()])[None, : ].repeat(num_repeated, axis=0)

        return processed_features

    def get_label(self, protein, raw_features, chain_ids_required, structure_object, is_cif):
        label_list = []

        # fasta 
        fasta_file = self._get_protein_fasta_file(protein)
        seqs, descs = parsers.parse_fasta(open(fasta_file, 'r').read())
        chain_ids = [i.split()[0].split('_')[1] for i in descs]
        seqs_lens = [len(i) for i in seqs]

        # get label
        for seq_len, chain_id in zip(seqs_lens, chain_ids):

            if not chain_id in chain_ids_required: continue # skip chains that are not required

            if is_cif:
                protein_chain_d = load_chain(structure_object, chain_id)
                if protein_chain_d['resolution'] > 9:
                    print(
                        f'Skip {protein} due to low resolution {protein_chain_d["resolution"]}'
                    )
                    continue
            else:
                prot_obj = protein_utils.from_pdb_string(structure_object, chain_id)
                protein_chain_d = load_pdb_chain(prot_obj,
                        confidence_threshold=None)

            protein_label = generate_label(protein_chain_d)
            if seq_len != protein_label['aatype_index'].shape[0]:
                print(f'Skip {protein} due to inequal residue num')
                continue
            label_list.append(protein_label)

        raw_labels = {}
        if len(label_list) == 0: return {}
        for key in label_list[0].keys():
            raw_labels[key] = np.concatenate([l[key] for l in label_list],
                                             axis=0)
        ## copy assembly features
        for k in ['asym_id', 'sym_id', 'entity_id', 'residue_index']:
            raw_labels[k] = np.array(raw_features[k][0])
        return raw_labels
 
    def get_sample(self, protein_info, random_seed=None):
        """tbd."""
        # read protein_name and assigned chain_ids from protein
        # protein_info: jsonline in {"protein_name": "PROT", "chains": [{"id": "ID"}, 
        #               {"id": "ID", "is_ligand": TRUE}, … ]} format
        protein_list_item = protein_list_item = json.loads(protein_info)
        protein_name = protein_list_item["protein_name"]
        targeted_chains = [item["id"] for item in protein_list_item["chains"]]
        chain_ids_to_info = dict(zip([item["id"] for item in protein_list_item["chains"]],
                                     [item for item in protein_list_item["chains"]]))

        # fasta 
        fasta_file = self._get_protein_fasta_file(protein_name)
        seqs, descs = parsers.parse_fasta(open(fasta_file, 'r').read())
        chain_ids = [i.split()[0].split('_')[1] for i in descs]
        seqs_lens = [len(i) for i in seqs]
        chain_to_len = dict(zip(chain_ids, seqs_lens))
        # filter out chains that are not mentioned in targeted_chains
        seqs = [s for c, s in zip(chain_ids, seqs) if c in targeted_chains]
        chain_ids = [c for c in chain_ids if c in targeted_chains]
        # add group info to chain_ids
        chain_info_list = [chain_ids_to_info[c] for c in chain_ids]

        structure_object = None
        if hasattr(self.data_config, 'structure_dir'):
            protein_struct_file = self._get_protein_struct_file(protein_name)
            # mmcif/ pdb
            is_cif = False
            if protein_struct_file.endswith('.cif.gz'):
                with gzip.open(protein_struct_file, 'r') as f:
                    cif_string = f.read().decode('utf8')
                is_cif = True
            elif protein_struct_file.endswith('.cif'):
                cif_string = "".join(open(protein_struct_file, 'r').readlines())
                is_cif = True
            elif protein_struct_file.endswith('.pdb.gz'):
                with gzip.open(protein_struct_file, 'r') as f:
                    structure_object = f.read().decode('utf8')
            else:
                with open(protein_struct_file, 'r') as f:
                    structure_object = f.read()

            if is_cif:
                parse_result = mmcif_parsing.parse(file_id=protein_struct_file,
                                                mmcif_string=cif_string)
                structure_object = parse_result.mmcif_object
            else:
                structure_object = None
                assert not self.data_config.get("test_with_rigid_template",False), "structure pdb not supported fot test_with_rigid_template"

        if self.data_config.get("load_from_model_input",False):
            model_input_pkl = os.path.join(self.data_config.feature_dir, protein_name + "_" + 
                                           "_".join([chain_info["id"] for chain_info in chain_info_list]), 
                                           self.data_config.model_input_pkl_name)
            if not os.path.exists(model_input_pkl):
                model_input_pkl = os.path.join(self.data_config.feature_dir, self.data_config.model_input_pkl_name)
            with open(model_input_pkl, 'rb') as pkl:
                processed_features = pickle.load(pkl)
        else:
            if random_seed is None:
                random_seed = np.random.randint(10000000)

            full_seq_len_exceeded = False
            full_seq_len = 0
            for chain_id in chain_ids:
                full_seq_len += chain_to_len[chain_id]
                if full_seq_len > MAX_SEQ_SIZE:
                    raise Exception(f"Test sample({protein_name} {chain_id}) \
                                    with seq_length({chain_to_len[chain_id]}) + {full_seq_len}) exceeded, skip")
                    processed_features = None
                    full_seq_len_exceeded = True
                    break
            if not full_seq_len_exceeded:
                processed_features = self.get_input_feat(protein_name, seqs=seqs, 
                    chain_info_list=chain_info_list, random_seed=random_seed, mmcif_object=structure_object)
        
        sample = {
            'name': protein_name,
            'feat': processed_features,
            'label_cropped': {},
            'label': {},
            'chain_info_list': json.dumps(chain_info_list),
        }

        if hasattr(self.data_config, 'structure_dir'):
            if protein_struct_file.endswith(".pdb"): 
                sample.update({"pdb_struct_file": protein_struct_file})
                print("pdb protein_struct_file not supported yet")
                return sample

            raw_labels = self.get_label(protein_name, processed_features, chain_ids_required=chain_ids,
                     structure_object=structure_object, is_cif=is_cif)
            if raw_labels == {}: return sample
            raw_labels['resolution'] = np.array(
            [np.mean(raw_labels['resolution'])])
            sample['label'] = raw_labels
            ## create pair_pocket_mask
            if self.model_config.model.global_config.get('use_pair_pocket_mask', 0.) > 0:
              if self.data_config.get("test_with_pocket_mask", False):
                # locate receptor
                receptor_id = 0
                for chain_idx, chain_info in zip(np.unique(raw_labels['asym_id']), chain_info_list):
                    if not chain_info.get("is_ligand", False):
                        receptor_id = chain_idx
                        break
                pocket_size = self.model_config.model.global_config.get('infer_pocket_mask_size', 20.)
                print(f"{protein_name} size-{pocket_size} pocket_mask: chain {receptor_id} selected as receptor" \
                        + "with {chain_info_list} and asym_id unique:" \
                        + " {np.unique(raw_labels['asym_id'], return_counts=True)}")
                pair_pocket_mask = make_pair_pocket_mask_by_receptor_residue(
                        raw_labels['all_atom_positions'], 
                        raw_labels['all_atom_mask'], 
                        raw_labels['asym_id'],
                        seq_mask=processed_features['seq_mask'][0],
                        receptor_asym_id=receptor_id,
                        threshold_min=pocket_size, 
                        threshold_max=pocket_size,
                        max_inter_residue_num=5)
                processed_features['pair_pocket_mask'] = np.tile(pair_pocket_mask[None], 
                        [processed_features['asym_id'].shape[0], 1, 1])

        # use pre-selected res idx as pocket
        if self.model_config.model.global_config.get('use_pair_pocket_mask', 0.) > 0:
            if self.data_config.get("test_with_pocket_mask", False):
                asym_id = processed_features["asym_id"][0]
                assert len(asym_id.shape) == 1, f"len asym_id shape {asym_id.shape} != 1"
                # locate receptor
                receptor_id = 0
                receptor_asym_id = 0
                for chain_idx, (chain_asym_idx, chain_info) in enumerate(zip(np.unique(asym_id), chain_info_list)):
                    if not chain_info.get("is_ligand", False):
                        receptor_id = chain_idx
                        receptor_asym_id = chain_asym_idx
                        break
                receptor_pocket_idx = chain_info_list[int(receptor_id)].get("pocket_idx", [])
                if len(receptor_pocket_idx) > 0:  # assign pocket mask automatically
                    pair_pocket_mask = make_pair_pocket_mask_by_receptor_res_idx(
                        receptor_res_idx=receptor_pocket_idx,
                        asym_ids=asym_id,
                        receptor_asym_id=receptor_asym_id
                    )
                    processed_features['pair_pocket_mask'] = np.tile(pair_pocket_mask[None], 
                            [asym_id.shape[0], 1, 1])
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


class MultimerRecomDataset(MultimerDataset):
    """Dataset for recombine data inference."""

    """tbd."""
    def get_input_feat(self, chain_ls, seq_ls, mmcif_object_ls):
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
        for offset, (protein_chain, seq, mmcif_object) in enumerate(zip(chain_ls, seq_ls, mmcif_object_ls)):
            protein, chain_id = protein_chain.split("_")
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
                print(f"Skip {protein}_{chain_id} due to inequal residue num between fasta({seq_len}) \
                       and features_pkl({raw_features['seq_length'][0]})")
                continue 
            
            protein_chain_d = load_chain(mmcif_object, chain_id)
            if not protein_chain_d['all_atom_positions'].shape[0] == seq_len: 
                print(f"Skip {protein}_{chain_id} due to inequal residue num between fasta({seq_len}) \
                      and protein_chain_d['all_atom_positions']({protein_chain_d['all_atom_positions'].shape})")
                continue

            # always provide single chain rigid info
            assert not self.data_config.get("feature_pkl_has_label", False), "label from feature_pkl not support yet"
            rigid_template = get_rigid_template(seq, protein_chain_d=protein_chain_d)

            # add rigid tempate to the front of raw_features template
            for key in rigid_template: 
                raw_features[key] = np.concatenate(
                    ([rigid_template[key]], raw_features[key]), axis=0
                )

            all_chain_features[protein_utils.PDB_CHAIN_IDS[offset]] = raw_features
            full_seq_length += seq_len

        if len(all_chain_features.keys()) == 1: 
            print(f"Skip {chain_ls} with only one chain {all_chain_features.keys()}")
            return None

        print(f"{protein} full_seq_length:", full_seq_length)
        if full_seq_length == 0 : return None
        raw_features = pipeline_multimer.process_with_all_chain_features(all_chain_features)
        
        if self.is_shuffle and _random_drop(raw_features):
            return None

        processed_feature_dict = features.np_example_to_features(
            np_example=raw_features,
            config=self.model_config)
        return processed_feature_dict

    def get_sample(self, protein_ls):
        chain_ls, seq_ls, mmcif_object_ls = [], [], []
        for protein in protein_ls:
            fasta_file = self._get_protein_fasta_file(protein)
            seqs, descs = parsers.parse_fasta(open(fasta_file, 'r').read())
            chain_ids = [i.split()[0].split('_')[1] for i in descs]
            # seqs_lens = [len(i) for i in seqs]

            # filter chain_ids that mmcif can not cover
            mmcif_object = None
            protein_struct_file = self._get_protein_struct_file(protein)
            if protein_struct_file.endswith('.cif.gz'):
                with gzip.open(protein_struct_file, 'r') as f:
                    cif_string = f.read().decode('utf8')
            else:
                cif_string = "".join(open(protein_struct_file, 'r').readlines())
            parse_result = mmcif_parsing.parse(file_id=protein_struct_file,
                                            mmcif_string=cif_string)
            mmcif_object = parse_result.mmcif_object
            mmcif_chain_keys = mmcif_object.chain_to_seqres.keys()
            valid_seq_chains = [(seq, cid) for seq, cid in zip(seqs, chain_ids) if cid in mmcif_chain_keys]
            chain_ids = [cid for _, cid in valid_seq_chains]
            seqs = [seq for seq, _ in valid_seq_chains]

            chain_selected = np.random.randint(0, len(chain_ids))
            chain_ls.append(protein + "_" + chain_ids[chain_selected])
            seq_ls.append(seqs[chain_selected])
            mmcif_object_ls.append(mmcif_object)

        feat = self.get_input_feat(chain_ls, seq_ls, mmcif_object_ls)
        if feat is None:
            return None

        for_recycle = len(feat['asym_id'].shape) == 2
        if for_recycle:
            all_atom_masks = np.copy(feat["template_all_atom_masks"][0])
            asym_id = np.copy(feat["asym_id"][0])
        else:
            all_atom_masks = np.copy(feat["template_all_atom_masks"])
            asym_id = np.copy(feat["asym_id"])
        asym_id = asym_id.astype(np.int64)

        ca_idx = residue_constants.atom_order['CA']
        ca_mask = all_atom_masks[:, ca_idx]
        print(
            f"After crop and pad, {protein} valid_rate is {np.mean(ca_mask)}")

        valid_chain_lens = np.bincount(asym_id)

        # At least one chain contains more than 10 residues
        # TODO: remove this limitation for protein-peptide complex!
        if not np.any(valid_chain_lens[1:] > 10):
            print(
                f"Skip {protein} due to all chains contain less than 10 residues"
            )
            return None
        if not np.all(valid_chain_lens[1:] > 1):
            print(f"Skip {protein} due to some chains contains less than 1 residues")
            return None

        sample = {
            'name': "-".join(protein_ls),
            'feat': feat,
            'label': {},
            'chain_ids': '-'.join(chain_ls),
        }

        return sample

    def __getitem__(self, index):
        index_a = index
        index_b = random.randint(0, self.__len__() - 1)
        sample = None
        while sample is None:
            protein_a = self.select_protein(index_a)
            protein_b = self.select_protein(index_b)
            # sample = self.get_sample(protein)
            try:
                sample = self.get_sample(protein_ls=[protein_a, protein_b])
            except Exception as ex:
                print(f'[DATA] index {index_a, index_b} prot {protein_a, protein_b} failed: {ex}')
                import traceback
                traceback.print_exc(file=sys.stdout)
            index_a = random.randint(0, self.__len__() - 1)
            index_b = random.randint(0, self.__len__() - 1)
        return sample

def demo_multimer():
    import json
    import ml_collections
    from helixfold.model import config
    Model_name = 'multimer_demo'
    model_config = config.model_config(Model_name)
    data_config = ml_collections.ConfigDict(
        json.load(open('./data_configs/multimer_demo.json', 'r')))
    dataset = MultimerDataset(model_config=model_config,
                              data_config=data_config.train,
                              crop_size=100,
                              is_pad_if_crop=True,
                              delete_msa_block=True,
                              trainer_id=0,
                              trainer_num=1)

    dataloader = DataLoader(dataset=dataset,
                            batch_sampler=LoopedBatchSampler(dataset,
                                                             shuffle=True,
                                                             batch_size=1,
                                                             drop_last=False),
                            collate_fn=multimer_collate_fn,
                            num_workers=0)
    s = time.time()
    for i, item in enumerate(dataloader):
        print(f'>>>>> {i} name {item["name"]}', time.time() - s)
        feat1 = item['feat']
        label1 = item['label']
        print('==== feat')
        for k in feat1:
            if type(feat1[k]) is list:
                n = len(feat1[k])
                print(f'{k} \t{feat1[k][0].shape} \tlist length: {n}')
            else:
                print(f'{k}\t{feat1[k].shape}')
        print('==== label')
        for k in label1:
            if type(label1[k]) is list:
                n = len(label1[k])
                print(f'{k}\t{label1[k][0].shape} \tlist length: {n}')
            else:
                print(f'{k}\t{label1[k].shape}')
        s = time.time()

        # FIXME: raise RuntimeError after exit dataloader
        if i > 1:
            break


def demo_multimer_test():
    import json
    import ml_collections
    from helixfold.model import config
    # from utils.dataset_multimer_msa import MultimerDataset
    Model_name = 'multimer_demo'
    model_config = config.model_config(Model_name)
    data_config = ml_collections.ConfigDict(
        # json.load(open('./data_configs/multimer_demo.json', 'r')))
        json.load(open('./data_configs/multimer_v3-train.json', 'r')))
    # dataset = MultimerTestDataset(model_config=model_config,
    dataset = MultimerDataset(model_config=model_config,
                                #   data_config=data_config.test.demo,
                                    data_config=data_config.train,
                                    crop_size=256,
                                    trainer_id=0,
                                    trainer_num=1)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=1,
                            drop_last=False,
                            collate_fn=multimer_collate_fn,
                            num_workers=0)
    s = time.time()
    for i, item in enumerate(dataloader):
        print(f'>>>>> {i} name {item["name"]}', time.time() - s)
        feat1 = item['feat']
        label1 = item['label']
        print('==== feat')
        for k in feat1:
            print(f'{k}\t{feat1[k].shape}')
        print('==== label')
        for k in label1:
            print(f'{k}\t{label1[k].shape}')
        s = time.time()
        break


if __name__ == '__main__':
    demo_multimer()
    demo_multimer_test()
