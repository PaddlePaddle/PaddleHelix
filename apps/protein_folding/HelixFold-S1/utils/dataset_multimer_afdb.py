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
from helixfold.data.utils import crop_and_pad_multimer, crop_and_pad_split_chains
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


def fasta_to_multimer_afdb_feature(data, prot_name, model_config):
    input_seqs, input_descs = [data['sequence']], [prot_name]

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


class MultimerAFDBDataset(paddle.io.Dataset):
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
        self.trainer_id=trainer_id
        self.trainer_num=trainer_num
        self.is_shuffle = is_shuffle

        def _assert_exists(path):
            assert exists(path), path
        _assert_exists(self.data_config.data_dir)

        ## check and filter (bad or not needed) proteins
        self.clusts = self._load_clusts()
        self.clusts_to_consume = [[] for _ in range(len(self.clusts))]

        def _print_attribute(key, value):
            print(f'[{self.__class__.__name__}] {key}: {value}')
        _print_attribute('data_dir', self.data_config.data_dir)
        _print_attribute('trainer_id/trainer_num', f'{self.trainer_id}/{self.trainer_num}')
        _print_attribute('clust_num', len(self.clusts))
        _print_attribute('protein_num', np.sum([len(x) for x in self.clusts]))
        _print_attribute('crop_size', crop_size)
        _print_attribute('is_pad_if_crop', is_pad_if_crop)
        _print_attribute('delete_msa_block', delete_msa_block)
    
    def _load_clusts(self):
        partial_clusts = sorted(os.listdir(self.data_config.data_dir))
        partial_clusts = partial_clusts[self.trainer_id::self.trainer_num]
        clusts = [] 
        for clust_name in partial_clusts:
            clust = []
            clust_path = os.path.join(self.data_config.data_dir, clust_name)
            for prot_file in sorted(os.listdir(clust_path)):
                prot_path = os.path.join(clust_path, prot_file)
                clust.append(prot_path)
            if len(clust) > 0:
                clusts.append(clust)
        return clusts
    
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

    def get_input_feat(self, data, prot_name):
        """tbd."""
        def _random_drop(raw_features):
            """drop sample according to num_residue"""
            L = raw_features['aatype'].shape[0]
            if np.random.uniform() > max(min(512, L), 256) / 512.0:
                print(f'[DATA] Drop {prot_name} ({L}) by random')
                return True
            return False

        ### gen_input_feat
        raw_features, seq_infos = fasta_to_multimer_afdb_feature(data, prot_name, self.model_config)
        if self.is_shuffle and _random_drop(raw_features):
            return None, None
        return raw_features, seq_infos

    def get_label(self, data, prot_name, raw_features, seq_infos):
        order_map = residue_constants.restype_order_with_x
        aatype_idx = np.array([order_map.get(rn, order_map['X']) for rn in data['sequence']], dtype=np.int32)
        resolution = self.data_config.get('resolution', 0.5)
        protein_chain_d = {
            'aatype_index': aatype_idx,
            'all_atom_positions': data['all_atom_positions'], 
            'all_atom_mask': data['all_atom_mask'],
            'resolution': np.array([resolution])
        }
        aa_confidence_threshold = self.data_config.get('aa_confidence_threshold', 90.0)
        high_confidence = np.array(data['confidence_list'][0]) > aa_confidence_threshold
        for i, confident in enumerate(high_confidence):
            if not confident:
                protein_chain_d["all_atom_mask"][i] = 0

        protein_label = generate_label(protein_chain_d)
        if seq_infos['seq_lens'][0] != protein_label['aatype_index'].shape[0]:
            print(f'Skip {prot_name} due to inequal residue num')
            return None
        
        label_list = [protein_label]
        raw_labels = {}
        for key in label_list[0].keys():
            raw_labels[key] = np.concatenate([l[key] for l in label_list], axis=0)
        ## copy assembly features
        if self.model_config.model.global_config.get('multimer_mode', False):
            for k in ['asym_id', 'sym_id', 'entity_id', 'residue_index']:
                raw_labels[k] = np.array(raw_features[k])
        return raw_labels

    def get_sample(self, protein):
        data = None
        with gzip.open(protein) as f_read:
            data = pickle.load(f_read)
        if data == None:
            print(f"pickle load {protein} data error")
            return None
        
        ### data filter
        confidence_list = data['confidence_list']
        plddt_score = np.mean(confidence_list[0])
        plddt_thres = self.data_config.get('plddt_score', 80.0)
        if plddt_score < plddt_thres:
            print(f"{protein} plddt_score({plddt_score}) < plddt_thres({plddt_thres}))")
            return None

        ### get input_feat & get label
        prot_name = os.path.basename(protein).split('.')[0]
        raw_features, seq_infos = self.get_input_feat(data, prot_name)
        if raw_features is None:
            print(f"{protein} raw_features is None")
            return None
        raw_labels = self.get_label(data, prot_name, raw_features, seq_infos)
        if raw_labels is None:
            print(f"{protein} raw_labels is None")
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
        
        chain_num = len(np.bincount(label['asym_id'])[1:])
        if chain_num > 10:
            print(f"Skip {prot_name} due to chain_num > 10")
            return None

        ca_idx = residue_constants.atom_order['CA']
        ca_mask = label['all_atom_mask'][:, ca_idx]
        print(f"After crop and pad, {prot_name} valid_rate is {np.mean(ca_mask)}")

        # At least one chain contains more than 10 residues
        if np.any(np.bincount(label['asym_id'] * ca_mask)[1:] > 10) == False:
            print(f"Skip {prot_name} due to all chains contain less than 10 residues")
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
            'name': prot_name,
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


def demo_multimer_afdb():
    import json
    import ml_collections
    from helixfold.model import config
    Model_name = 'multimer_seq512_pair64_l8_vio0'
    model_config = config.model_config(Model_name)
    data_config = ml_collections.ConfigDict(json.load(open('./data_configs/demo_multimer.json', 'r')))
    dataset = MultimerAFDBDataset(
        model_config=model_config,
        data_config=data_config.distill.afdb,
        crop_size=100,
        is_pad_if_crop=True,
        delete_msa_block=True,
        trainer_id=0, 
        trainer_num=1)

    dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=LoopedBatchSampler(
                dataset, shuffle=True, batch_size=1, drop_last=False),
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
    demo_multimer_afdb()

