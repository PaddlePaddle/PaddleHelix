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
from copy import deepcopy

from paddle.io import IterableDataset, DataLoader

from helixfold.data import mmcif_parsing
from helixfold.data.mmcif_parsing import MmcifObject, ResidueAtPosition, ResiduePosition
from helixfold.data.data_utils import a3m_to_features, load_chain, generate_label, load_pdb_chain, sample_raw_msa
from helixfold.data.utils import crop_and_pad_multimer, crop_and_pad_split_chains, get_heterodimer_chains
from helixfold.model import features
from helixfold.common import protein as protein_utils
from helixfold.common import residue_constants
# from helixfold.data import parsers
# from helixfold.data.pipeline import make_sequence_features, make_msa_features
from helixfold.data.utils import is_ignored_key_multimer, is_batched_key_multimer
from utils.dataset import LoopedBatchSampler

from helixfold.data import feature_processing
from helixfold.data import msa_pairing
from helixfold.data import parsers
from helixfold.data.pipeline_multimer import _make_chain_id_map, convert_monomer_features, add_assembly_features, pad_msa
from utils.dataset_multimer import NEED_FEATURES, _process_single_chain, add_num_sym, pair_and_merge

from utils.utils import multimer_collate_fn
from helixfold.common.protein import PDB_CHAIN_IDS


def fasta_to_multimer_feature_single_chain(input_fasta_path, model_config):
    with open(input_fasta_path) as f:
        input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)

    ## select a single chain from the multimer
    rand_id = np.random.randint(len(input_seqs))
    input_seqs = [input_seqs[rand_id]]
    input_descs = [input_descs[rand_id]]

    chain_ids = [input_desc.split(" ")[0].split("_")[1] for input_desc in input_descs]
    seq_lens = [len(input_seq) for input_seq in input_seqs]

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
                np_example=raw_features,
                config=model_config)
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
                sequence=fasta_chain.sequence,
                description=fasta_chain.description)
        chain_features = convert_monomer_features(
                chain_features, chain_id=chain_id)
        all_chain_features[chain_id] = chain_features
        sequence_features[fasta_chain.sequence] = chain_features

    all_chain_features = add_assembly_features(all_chain_features)
    all_chain_features = add_num_sym(all_chain_features)
    np_example = pair_and_merge(all_chain_features=all_chain_features)

    for k, v in processed_features.items():
        axis = NEED_FEATURES[k]
        processed_features[k] = np.concatenate(v, axis=axis)[0, ...]
    np_example.update(processed_features)

    seq_infos = {
        "chain_ids": chain_ids,
        "seq_lens": seq_lens,
        "input_seqs": input_seqs,
        "input_descs": input_descs,
        "asym_id": [d['asym_id'][0] for d in all_chain_features.values()],
        "sym_id": [d['sym_id'][0] for d in all_chain_features.values()],
        "entity_id": [d['entity_id'][0] for d in all_chain_features.values()],
    }
    return np_example, seq_infos


class MultimerSingleChainDataset(paddle.io.Dataset):
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
        _print_attribute('protein_clust_file', self.data_config.protein_clust_file)
        _print_attribute('trainer_id/trainer_num', f'{self.trainer_id}/{self.trainer_num}')
        _print_attribute('clust_num', len(self.clusts))
        _print_attribute('protein_num', np.sum([len(x) for x in self.clusts]))
        _print_attribute('crop_size', crop_size)
        _print_attribute('is_pad_if_crop', is_pad_if_crop)
        _print_attribute('delete_msa_block', delete_msa_block)
    
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
        return os.path.join(self.data_config.feature_dir, f"{protein}.fasta")
    
    def _get_protein_struct_file(self, protein):
        """tbd."""
        if self.data_config.get('cif_format', '.cif') == ".cif":
            protein_label_file = os.path.join(self.data_config.structure_dir, protein + '.cif')
        else:
            protein_label_file = os.path.join(self.data_config.structure_dir, protein + '.cif.gz')
        return protein_label_file 

    def get_input_feat(self, protein):
        """tbd."""
        def _random_drop(raw_features):
            """drop sample according to num_residue"""
            L = raw_features['aatype'].shape[0]
            if np.random.uniform() > max(min(512, L), 256) / 512.0:
                print(f'[DATA] Drop {protein} ({L}) by random')
                return True
            return False

        input_fasta_path = self._get_protein_feature_file(protein)
        raw_features, seq_infos = fasta_to_multimer_feature_single_chain(input_fasta_path, self.model_config)
        if raw_features == None or seq_infos == None:
            return None, None
        if self.is_shuffle and _random_drop(raw_features):
            return None, None
        return raw_features, seq_infos
    
    def get_label(self, protein, raw_features, seq_infos):
        protein_struct_file = self._get_protein_struct_file(protein)
        chain_ids = seq_infos["chain_ids"]
        cif_string = "".join(open(protein_struct_file, 'r').readlines())
        parse_result = mmcif_parsing.parse(file_id=protein_struct_file, mmcif_string=cif_string)
        mmcif_object = parse_result.mmcif_object
        label_list = []
        for seq_len, chain_id in zip(seq_infos['seq_lens'], chain_ids):
            protein_chain_d = load_chain(mmcif_object, chain_id)
            protein_label = generate_label(protein_chain_d)
            if seq_len != protein_label['aatype_index'].shape[0]:
                print(f'Skip {protein} due to inequal residue num')
                return None
            label_list.append(protein_label)
        raw_labels = {}
        for key in label_list[0].keys():
            raw_labels[key] = np.concatenate([l[key] for l in label_list], axis=0)
        ## copy assembly features
        if self.model_config.model.global_config.get('multimer_mode', False):
            for k in ['asym_id', 'sym_id', 'entity_id', 'residue_index']:
                raw_labels[k] = np.array(raw_features[k])
        return raw_labels

    def select_protein(self, clust_index):
        """tbd."""
        if len(self.clusts_to_consume[clust_index]) == 0:
            self.clusts_to_consume[clust_index] = deepcopy(self.clusts[clust_index])
            if self.is_shuffle:
                np.random.shuffle(self.clusts_to_consume[clust_index])
        clust = self.clusts_to_consume[clust_index]
        last_name = clust[-1]
        del clust[-1]
        return last_name

    def get_sample(self, protein):
        raw_features, seq_infos = self.get_input_feat(protein)
        if raw_features is None:
            return None
        raw_labels = self.get_label(protein, raw_features, seq_infos)
        if raw_labels is None:
            return None
        raw_labels['resolution'] = np.array([np.mean(raw_labels['resolution'])])

        if not self.crop_size is None and self.crop_size > 0:
            feat, label, chain_aatypes, chain_seq_lengths = crop_and_pad_split_chains(
                    raw_features, raw_labels,
                    crop_size=self.crop_size,
                    pad_for_shorter_seq=self.is_pad_if_crop)
            seq_infos['seq_lens'] = chain_seq_lengths

            # # crop_and_pad_multimer
            feat, label = crop_and_pad_multimer(
                    feat, label, 
                    seq_infos=seq_infos,
                    crop_size=self.crop_size, 
                    pad_for_shorter_seq=self.is_pad_if_crop)
            
            feat['chain_aatypes'] = chain_aatypes
            feat['chain_seq_lengths'] = chain_seq_lengths
            feat['chain_nums'] = np.array(len(chain_seq_lengths))
            feat['crop_nums'] = np.bincount(feat['asym_id'])

        chain_num = len(chain_seq_lengths)
        if chain_num > 10:
            print(f"Skip {protein} due to chain_num > 10, chain_num: {chain_num}")
            return None
        
        ca_idx = residue_constants.atom_order['CA']
        ca_mask = label['all_atom_mask'][:, ca_idx]
        print(f"After crop and pad, {protein} valid_rate is {np.mean(ca_mask)}")

        # At least one chain contains more than 10 residues
        if np.any(np.bincount(label['asym_id'] * ca_mask)[1:] > 10) == False:
            print(f"Skip {protein} due to all chains contain less than 10 residues")
            return None

        chain_lens = np.bincount(feat['asym_id'].astype('int64'))
        if (chain_lens == 1).any():
            print(f"chain_lens = 1 will cause slice error in "
                    f"multi_chain_permutation_align: {chain_lens}")
            return None

        num_recycle = self.model_config.model.num_recycle + 1
        for k, v in feat.items():
            feat[k] = np.repeat(np.expand_dims(v, axis=0), num_recycle, axis=0)
        feat['bert_mask'] = feat['bert_mask'][:, 0:1, :]
        feat['msa_mask'] = feat['msa_mask'][:, 0:1, :]
        feat['true_msa'] = feat['msa'][:, 0:1, :]

        sample = {
            'name': protein,
            'feat': feat,
            'label': raw_labels,
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
            index = np.random.randint(self.__len__())
        return sample


def demo_multimer_single_chain():
    import json
    import ml_collections
    from helixfold.model import config
    Model_name = 'multimer_seq512_pair64_l8_vio0'
    model_config = config.model_config(Model_name)
    data_config = ml_collections.ConfigDict(json.load(open('./data_configs/demo_multimer.json', 'r')))
    dataset = MultimerSingleChainDataset(
        model_config=model_config,
        data_config=data_config.train,
        crop_size=100,
        is_pad_if_crop=True,
        delete_msa_block=True,
        trainer_id=0, 
        trainer_num=1)

    dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=LoopedBatchSampler(
                dataset, shuffle=True, batch_size=1, drop_last=False),
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
    demo_multimer_single_chain()
