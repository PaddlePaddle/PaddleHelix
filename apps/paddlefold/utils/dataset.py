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
from copy import deepcopy

from paddle.io import IterableDataset, DataLoader

from alphafold_paddle.data import mmcif_parsing
from alphafold_paddle.data.mmcif_parsing import MmcifObject, ResidueAtPosition, ResiduePosition
from alphafold_paddle.data.data_utils import Msas_to_features, a3m_to_features, load_chain, generate_label, load_pdb_chain
from alphafold_paddle.data.utils import crop_and_pad
from alphafold_paddle.model import features
from alphafold_paddle.common import protein as protein_utils
from alphafold_paddle.data import parsers
from alphafold_paddle.data.pipeline import make_sequence_features, make_msa_features


class LoopedBatchSampler(object):
    def __init__(self,
                 dataset=None,
                 shuffle=False,
                 batch_size=1,
                 drop_last=False):

        assert not isinstance(dataset, paddle.io.IterableDataset), \
            "dataset should not be a paddle.io.IterableDataset"
        self.dataset = dataset
        assert isinstance(shuffle, bool), \
            "shuffle should be a boolean value, but got {}".format(type(shuffle))
        self.shuffle = shuffle

        assert isinstance(batch_size, int) and batch_size > 0, \
            "batch_size should be a positive integer, but got {}".format(batch_size)
        self.batch_size = batch_size
        assert isinstance(drop_last, bool), \
            "drop_last should be a boolean value, but got {}".format(type(drop_last))
        self.drop_last = drop_last

    def __iter__(self):
        while True:
            num_samples = len(self.dataset)
            indices = np.arange(num_samples).tolist()

            if self.shuffle:
                np.random.shuffle(indices)
                
            batch_indices = []
            for idx in indices:
                batch_indices.append(idx)
                if len(batch_indices) == self.batch_size:
                    yield batch_indices
                    batch_indices = []
            if not self.drop_last and len(batch_indices) > 0:
                yield batch_indices

    def __len__(self):
        num_samples = len(self.dataset)
        num_samples += int(not self.drop_last) * (self.batch_size - 1)
        return num_samples // self.batch_size


class AF2Dataset(paddle.io.Dataset):
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
        _assert_exists(self.data_config.protein_clust_file)
        _assert_exists(self.data_config.protein_map_file)
        _assert_exists(self.data_config.feature_dir)
        _assert_exists(self.data_config.structure_dir)

        ## check and filter (bad or not needed) proteins
        self.protein2seq_map = self._load_protein2seq_map()
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
    
    def _load_protein2seq_map(self):
        protein2seq_map = {}
        for line in open(self.data_config.protein_map_file):
            protein, protein_seqid = line.split()
            protein2seq_map[protein] = protein_seqid
        return protein2seq_map

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
        if not os.path.exists(self._get_protein_feature_file(protein)):
            return False
        if not os.path.exists(self._get_protein_struct_file(protein)):
            return False
        return True
    
    def _get_protein_feature_file(self, protein):
        """tbd."""
        return os.path.join(self.data_config.feature_dir, 
                self.protein2seq_map[protein], 'features.pkl.gz')
    
    def _get_protein_struct_file(self, protein):
        """tbd."""
        # protein format: "protein_name"+"_"+"chain_id"
        protein_name, chain_id = protein.split('_')
        protein_label_file = os.path.join(
                self.data_config.structure_dir, protein_name + '.cif')
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

        protein_feature_file = self._get_protein_feature_file(protein)
        with gzip.open(protein_feature_file, 'rb') as pkl:
            raw_features = pickle.load(pkl)
        if _random_drop(raw_features):
            return None
        processed_feature_dict = features.np_example_to_features(
                np_example=raw_features,
                config=self.model_config,
                random_seed=None)
        return processed_feature_dict

    def get_label(self, protein):
        """tbd."""
        protein_struct_file = self._get_protein_struct_file(protein)
        chain_id = protein.split('_')[1]
        cif_string = ''.join(open(protein_struct_file, 'r').readlines())
        parse_result = mmcif_parsing.parse(file_id=protein_struct_file, mmcif_string=cif_string)
        mmcif_object = parse_result.mmcif_object
        protein_chain_d = load_chain(mmcif_object, chain_id)
        protein_label = generate_label(protein_chain_d)
        return protein_label

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
        feat = self.get_input_feat(protein)
        if feat is None:
            return None
        label = self.get_label(protein)
        if feat['aatype'].shape[1] != label['aatype_index'].shape[0]:
            print(f'Skip {protein} due to inequal residue num')
            return None
        # crop
        if not self.crop_size is None and self.crop_size > 0:
            feat, label = crop_and_pad(
                    feat, label, crop_size=self.crop_size, pad_for_shorter_seq=self.is_pad_if_crop)
        sample = {
            'name': protein,
            'feat': feat,
            'label': label,
        }
        return sample
    
    def __len__(self):
        return len(self.clusts)
    
    def __getitem__(self, index):
        protein = self.select_protein(index)
        try:
            sample = self.get_sample(protein)
        except Exception as ex:
            print(f'[DATA] index {index} prot {protein} failed: {ex}')
            sample = None

        if sample is None:
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)
        return sample


class AF2DistillDataset(paddle.io.Dataset):
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

        def _assert_exists(path):
            assert exists(path), path
        _assert_exists(self.data_config.feature_dir)
        _assert_exists(self.data_config.structure_dir)

        ## check and filter (bad or not needed) proteins
        self.proteins = sorted(os.listdir(self.data_config.feature_dir))[trainer_id::trainer_num]

        def _print_attribute(key, value):
            print(f'[{self.__class__.__name__}] {key}: {value}')
        _print_attribute('trainer_id/trainer_num', f'{self.trainer_id}/{self.trainer_num}')
        _print_attribute('protein_num', len(self.proteins))
        _print_attribute('crop_size', crop_size)
        _print_attribute('is_pad_if_crop', is_pad_if_crop)
        _print_attribute('delete_msa_block', delete_msa_block)
    
    def _check_protein(self, protein):
        """tbd."""
        if not os.path.exists(self._get_protein_feature_file(protein)):
            return False
        if not os.path.exists(self._get_protein_struct_file(protein)):
            return False
        return True
    
    def _get_protein_feature_file(self, protein):
        """tbd."""
        return os.path.join(self.data_config.feature_dir, protein, 'a3m.gz')
    
    def _get_protein_struct_file(self, protein):
        """tbd."""
        # protein_name = protein.split('_')[0]
        protein_name = protein
        protein_label_file = os.path.join(
                self.data_config.structure_dir, protein_name + '.pdb.gz')
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

        protein_feature_file = self._get_protein_feature_file(protein)
        a3m_str = None
        with gzip.open(protein_feature_file) as f_read:
            a3m_str = f_read.read().decode('utf8')
        seq_sequences, seq_descriptions = parsers.parse_fasta(a3m_str)
        seq_sequence, seq_description = seq_sequences[0], seq_descriptions[0]
        num_res = len(seq_sequence)
        msa_sequences, msa_deletion_matrix = parsers.parse_a3m(a3m_str)
        sequence_features = make_sequence_features(seq_sequence, seq_description, num_res)
        msa_features = make_msa_features((msa_sequences, ), (msa_deletion_matrix, ))
        raw_features = {**sequence_features, **msa_features}
        if _random_drop(raw_features):
            return None
        processed_feature_dict = features.np_example_to_features(
                np_example=raw_features,
                config=self.model_config,
                random_seed=None)
        return processed_feature_dict

    def get_label(self, protein):
        """tbd."""
        protein_struct_file = self._get_protein_struct_file(protein)
        # chain_id = protein.split('_')[1] if len(protein.split('_')) >=2 else None
        chain_id = None
        with gzip.open(protein_struct_file, 'r') as f:
            pdb_file = f.read().decode('utf8')
        prot_obj = protein_utils.from_pdb_string(pdb_file, chain_id)
        protein_chain_d = load_pdb_chain(prot_obj, confidence_threshold=0.5)
        protein_label = generate_label(protein_chain_d)
        return protein_label

    def get_sample(self, protein):
        if self._check_protein(protein) == False:
            return None
        feat = self.get_input_feat(protein)
        if feat is None:
            return None
        label = self.get_label(protein)
        if feat['aatype'].shape[1] != label['aatype_index'].shape[0]:
            print(f'Skip {protein} due to inequal residue num')
            return None
        # crop
        if not self.crop_size is None and self.crop_size > 0:
            feat, label = crop_and_pad(
                    feat, label, crop_size=self.crop_size, pad_for_shorter_seq=self.is_pad_if_crop)
        sample = {
            'name': protein,
            'feat': feat,
            'label': label,
        }
        return sample
    
    def __len__(self):
        return len(self.proteins)
    
    def __getitem__(self, index):
        protein = self.proteins[index % self.__len__()]
        try:
            sample = self.get_sample(protein)
        except Exception as ex:
            print(f'[DATA] index {index} prot {protein} failed: {ex}')
            sample = None

        if sample is None:
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)
        
        return sample


class AF2TestDataset(IterableDataset):
    """tbd."""
    def __init__(self,
            model_config,
            data_config,
            delete_msa_block=False,
            trainer_id=0, 
            trainer_num=1):
        """
        Iterate over clusts, where proteins in each clust
        will be all visited.
        """
        self.model_config = deepcopy(model_config)
        self.model_config.data.eval.delete_msa_block = delete_msa_block
        self.data_config = data_config
        self.trainer_id=trainer_id
        self.trainer_num=trainer_num

        def _assert_exists(path):
            assert exists(path), path
        _assert_exists(self.data_config.feature_dir)
        _assert_exists(self.data_config.structure_dir)

        ## check and filter (bad or not needed) proteins
        self.proteins = self._get_proteins()
        
        def _print_attribute(key, value):
            print(f'[{self.__class__.__name__}] {key}: {value}')
        _print_attribute('feature_dir', self.data_config.feature_dir)
        _print_attribute('structure_dir', self.data_config.structure_dir)
        _print_attribute('trainer_id/trainer_num', f'{self.trainer_id}/{self.trainer_num}')
        _print_attribute('protein_num', len(self.proteins))
        _print_attribute('delete_msa_block', delete_msa_block)
           
    def _get_proteins(self):
        """tbd."""
        proteins = sorted(os.listdir(self.data_config.feature_dir))
        proteins = proteins[self.trainer_id::self.trainer_num]
        return proteins
    
    def _get_protein_feature_file(self, protein):
        """tbd."""
        return join(self.data_config.feature_dir, protein, 'features.pkl.gz')
    
    def _get_protein_struct_file(self, protein):
        """tbd."""
        cif_file = join(self.data_config.structure_dir, protein + '.pdb')
        return cif_file 

    def get_input_feat(self, protein):
        """tbd."""
        protein_feature_file = self._get_protein_feature_file(protein)
        with gzip.open(protein_feature_file, 'rb') as pkl:
            raw_features = pickle.load(pkl)
        processed_feature_dict = features.np_example_to_features(
                np_example=raw_features,
                config=self.model_config,
                random_seed=None)
        return processed_feature_dict
        
    def get_label(self, protein):
        """tbd."""
        protein_struct_file = self._get_protein_struct_file(protein)
        cif_string = ''.join(open(protein_struct_file, 'r').readlines())
        parse_result = mmcif_parsing.parse(file_id=protein_struct_file, mmcif_string=cif_string)
        mmcif_obj_single = parse_result.mmcif_object
        chain_id = protein.split('_')[1]
        protein_chain_d = load_chain(mmcif_obj_single, chain_id)
        protein_label = generate_label(protein_chain_d)
        return protein_label

    def get_sample(self, protein):
        feat = self.get_input_feat(protein)
        label = {}     # currently not loading labels
        sample = {
            'name': protein,
            'feat': feat,
            'label': label,
            'struct_file': self._get_protein_struct_file(protein),
        }
        return sample
    
    def __iter__(self):
        """"tbd."""
        for protein in self.proteins:
            yield self.get_sample(protein)


def dump_batch(batch, filename):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(batch, f, protocol=4)


def demo_af2():
    import json
    import ml_collections
    from alphafold_paddle.model import config
    Model_name = 'model_1'
    # Model_name = 'model_5'
    model_config = config.model_config(Model_name)
    data_config = ml_collections.ConfigDict(json.load(open('./data_configs/demo.json', 'r')))
    # data_config = ml_collections.ConfigDict(json.load(open('./data_configs/pdb-20211015.json', 'r')))
    dataset = AF2Dataset(
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
            num_workers=0)
    s = time.time()
    for i, item in enumerate(dataloader):
        print(f'>>>>> {i} name {item["name"]}', time.time() - s)
        # feat1 = item['feat']
        # label1 = item['label']
        # print('==== feat')
        # for k in feat1:
        #     print(f'{k}\t{feat1[k].shape}')
        # print('==== label')
        # for k in label1:
        #     print(f'{k}\t{label1[k].shape}')
        # print('-------')
        s = time.time()
        # break

        # from app.protein_folding.others.utils.utils import tree_map
        # item['feat'] = tree_map(lambda x: x.numpy(), item['feat'])
        # item['label'] = tree_map(lambda x: x.numpy(), item['label'])
        # dump_batch(item, './data/demo/debug_processed_batch-seq_mask.pkl')
        # exit()


def demo_af2_test():
    from alphafold_paddle.model import config
    Model_name = 'model_5'
    model_config = config.model_config(Model_name)
    input_folder = './data/cameo/CAMEO_test/20210904/features'
    struct_folder = './data/cameo/CAMEO_test/20210904/mmcif'
    data_path = f"{input_folder}:{struct_folder}"
    dataset = AF2TestDataset(
        config=model_config,
        data_path=data_path,
        delete_msa_block=True,
        trainer_id=0, 
        trainer_num=1,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        drop_last=False,
        num_workers=0)
    s = time.time()
    for i, item in enumerate(dataloader):
        print('-------')
        print(f'{i} name {item["name"]} time: {time.time() - s:.4f}')
        s = time.time()
        feat = item['feat']
        label = item['label']
        # print('==== feat')
        # for k in feat:
        #     print(f'{k}\t{feat[k].shape}')
        # print('==== label')
        # for k in label:
        #     print(f'{k}\t{label[k].shape}')
        # break


if __name__ == '__main__':
    demo_af2()
    # demo_af2_msa()
    # demo_af2_test()
