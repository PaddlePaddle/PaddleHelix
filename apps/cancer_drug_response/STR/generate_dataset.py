#!/usr/bin/python
# coding=utf-8
#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#coding=utf-8

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import random
import paddle
from pgl.utils.data import Dataset as BaseDataset
from pgl.utils.data import Dataloader
from pgl.utils.data.dataloader import  _DataLoaderIter
import pgl
from pgl.utils.logger import log
import pdb
import copy
from sklearn.model_selection import KFold

def Create_Dataset(args, dataset, logging):
    dataset_all = {}
    if len(dataset)>1:
        for key in dataset[0].keys():
            dataset_all[key] = np.concatenate([dataset[1][key],dataset[0][key]])
    else:
        dataset_all = dataset[0]
    ccle_datasetmaxmin = {}
    drug_datasetmaxmin = {}
    if args.data_norm == 'norm':
        ccle_dataset = {}
        for i in range(len(dataset_all['cancer_type_list'])):
            ccle_dataset_keys = ccle_dataset.keys()
            if dataset_all['cancer_type_list'][i][1] not in ccle_dataset_keys:
                ccle_dataset.setdefault(dataset_all['cancer_type_list'][i][1],[]).append(dataset_all['target'][i])
            else:
                ccle_dataset[dataset_all['cancer_type_list'][i][1]].append(dataset_all['target'][i])

        # returns the numbers with their index (#, index)
        ccle_datasetmaxmin ['Max'] = {ccle : max(ccle_dataset[ccle]) for ccle in ccle_dataset} #find max value (depth) for each key (animal)
        ccle_datasetmaxmin ['Min'] = {ccle : min(ccle_dataset[ccle]) for ccle in ccle_dataset} #find min value (depth) for each key (animal)
        for i in range(len(dataset_all['target'])):
            dis=ccle_datasetmaxmin ['Max'][dataset_all['cancer_type_list'][i][1]]-ccle_datasetmaxmin ['Min'][dataset_all['cancer_type_list'][i][1]]
            dataset_all['target'][i]= (dataset_all['target'][i] - ccle_datasetmaxmin ['Min'][dataset_all['cancer_type_list'][i][1]])/dis * 10 - 5
        
        logging.info('dataset normalized by ccle !')

    elif args.data_norm == 'norm_drug':
        drug_dataset = {}
        for i in range(len(dataset_all['cancer_type_list'])):
            drug_dataset_keys = drug_dataset.keys()
            if dataset_all['cancer_type_list'][i][0] not in drug_dataset_keys:
                drug_dataset.setdefault(dataset_all['cancer_type_list'][i][0],[]).append(dataset_all['target'][i])
            else:
                drug_dataset[dataset_all['cancer_type_list'][i][0]].append(dataset_all['target'][i])

        # returns the numbers with their index (#, index)
        drug_datasetmaxmin ['Max'] = {drug : max(drug_dataset[drug]) for drug in drug_dataset} #find max value (depth) for each key (animal)
        drug_datasetmaxmin ['Min'] = {drug : min(drug_dataset[drug]) for drug in drug_dataset} #find min value (depth) for each key (animal)
        for i in range(len(dataset_all['target'])):
            dis=drug_datasetmaxmin ['Max'][dataset_all['cancer_type_list'][i][0]] - drug_datasetmaxmin ['Min'][dataset_all['cancer_type_list'][i][0]]
            dataset_all['target'][i]= (dataset_all['target'][i] - drug_datasetmaxmin ['Min'][dataset_all['cancer_type_list'][i][0]])/dis * 10 - 5
        logging.info('dataset normalized by drug!')

    else:
        logging.info('dataset do not normalized !')

    kf = KFold(n_splits=5, shuffle=True, random_state=None)
    if args.split_mode == 'mix':
        TCGA_label_set = ["ALL","BLCA","BRCA","CESC","DLBC","LIHC","LUAD",
                  "ESCA","GBM","HNSC","KIRC","LAML","LCML","LGG",
                  "LUSC","MESO","MM","NB","OV","PAAD","SCLC","SKCM",
                  "STAD","THCA",'COAD/READ']
        index_list = np.load('./split/index_list.npy').tolist()
        assert index_list[-1] == 74894
        split_list = [x for x in kf.split(index_list)]
        train_idx, test_idx = [], []
        for each_type in TCGA_label_set:
            data_subtype_idx = [i for i in range(len(dataset_all['cancer_type_list'])) if dataset_all['cancer_type_list'][i][0]==each_type]
            train_list = random.sample(data_subtype_idx,int(args.split_ratio*len(data_subtype_idx)))
            # train_list = data_subtype_idx[:int(args.split_ratio*len(data_subtype_idx))]
            test_list = [item for item in data_subtype_idx if item not in train_list]
            train_idx += train_list
            test_idx += test_list
        logging.info('dataset split mode: mix !')
    elif args.split_mode == 'mix_rand':
        split_list = np.load('./split/split_list_mix.npy',allow_pickle=True).tolist()
        train_idx = split_list[args.cross_val_num][0]
        test_idx = split_list[args.cross_val_num][1]
        logging.info('dataset split mode: mix_rand! ')

    elif args.split_mode == 'all':
        train_idx = np.array([i for i in range(len(dataset_all['drug_list']))])
        test_idx = train_idx[-100:]
        logging.info('dataset split mode: all ! ')
        
    elif args.split_mode == 'ccle':
        ccle = np.load('./split/ccle_names.npy').tolist()
        assert ccle[-1] == 'ACH-000452'
        split_list = np.load('./split/split_list_ccles.npy',allow_pickle=True)
        train_ccle_index = split_list[args.cross_val_num][0]
        train_ccle = [ccle[index] for index in train_ccle_index]
        train_idx = []
        test_idx = []
        for i in range(len(dataset_all['cancer_type_list'])):
            if dataset_all['cancer_type_list'][i][1] in train_ccle:
                train_idx.append(i)
            else:
                test_idx.append(i)
        logging.info('dataset split mode: ccle !')

    elif args.split_mode == 'drug':
        drug = np.load('./split/drug_names.npy').tolist()
        split_list = np.load('./split/split_list_drugs.npy',allow_pickle=True)
        train_drug_index = split_list[args.cross_val_num][0]
        train_drug = [drug[index] for index in train_drug_index]
        print('len_drug: ', len(drug))
        # 数据按照ccle划分
        print('train_drug:',len(train_drug))
        print('test_drug:',len(drug) - len(train_drug))
        train_idx = []
        test_idx = []
        for i in range(len(dataset_all['cancer_type_list'])):
            if dataset_all['cancer_type_list'][i][2] in train_drug:
                train_idx.append(i)
            else:
                test_idx.append(i)
        logging.info('dataset split mode: drug !')
        
    elif args.split_mode == 'drug_ccle':
        drug = np.load('./split/drug_names.npy').tolist()
        print('len_drug: ', len(drug))
        # 数据按照ccle划分
        iid = int(args.split_ratio * len(drug))
        train_drug = drug[:iid]
        print('train_drug:',len(train_drug))
        print('test_drug:',len(drug) - len(train_drug))

        ccle = np.load('./split/ccle_names.npy').tolist()
        assert ccle[-1] == 'ACH-000452'
        print('len_ccle: ', len(ccle))
        # 数据按照ccle划分
        iid = int(args.split_ratio * len(ccle))
        train_ccle = ccle[:iid]
        train_idx = []
        test_idx = []
        for i in range(len(dataset_all['cancer_type_list'])):
            if dataset_all['cancer_type_list'][i][2] in train_drug and dataset_all['cancer_type_list'][i][1] in train_ccle:
                train_idx.append(i)
            elif dataset_all['cancer_type_list'][i][2] not in train_drug and dataset_all['cancer_type_list'][i][1] not in train_ccle:
                test_idx.append(i)
        logging.info('dataset split mode: drug_ccle !')
    
    train_dataset = {}
    test_dataset = {}
    for key in dataset[0].keys():
        train_dataset[key] = dataset_all[key][train_idx]
        test_dataset[key] = dataset_all[key][test_idx]
    train_ds = Dataset(train_dataset)
    test_ds = Dataset(test_dataset)
    print('train_data num: ',len(train_ds))
    print('test_data num: ',len(test_ds))
    return train_ds, test_ds, ccle_datasetmaxmin, drug_datasetmaxmin

class CrossSampler(object):
    def __init__(self, dataset, batch_size=1, num_instances=1, drop_last=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.drop_last = drop_last
        # self.index_dic = defaultdict(list)
        self.dic_ccle = defaultdict(list)
        self.dic_pubchemid = defaultdict(list)
        assert batch_size == 2*num_instances

        for index, data in enumerate(self.dataset):
            # (g, mu, gex, me, y ,cancer_type_list) = data
            ccle_name = data[-1][1]
            self.dic_ccle[ccle_name].append(index)
            pubchem_id = data[-1][2]
            self.dic_pubchemid[pubchem_id].append(index)
            # cancer_type_list.append([cancer_type, cell_line_id, pubchem_id])
        
    def __iter__(self):
        ccles = list(self.dic_ccle.keys())
        pubchemids = list(self.dic_pubchemid.keys())
        # random.shuffle(ccles)
        # random.shuffle(pubchemids)
        min_len = min(len(ccles),len(pubchemids))

        avai_ccles = copy.deepcopy(ccles)
        avai_pubchemids = copy.deepcopy(pubchemids)
        final_idxs = []

        sampled_ccle = set([])
        sampled_pubchemids = set([])

        while len(sampled_ccle) < len(avai_ccles) or len(sampled_pubchemids) < len(avai_pubchemids):
            selected_ccle = random.sample(avai_ccles, 1)
            selected_pubchemids = random.sample(avai_pubchemids, 1)

            batch_idex = []
            # pdb.set_trace()
            idxs_ccle = np.random.choice(self.dic_ccle[selected_ccle[0]], size=self.num_instances, replace=True).tolist()
            idxs_pubchemid = np.random.choice(self.dic_pubchemid[selected_pubchemids[0]], size=self.num_instances, replace=True).tolist()

            batch_idex.extend(idxs_ccle)
            batch_idex.extend(idxs_pubchemid)

            final_idxs.append(batch_idex)

            sampled_ccle.add(selected_ccle[0])
            sampled_pubchemids.add(selected_pubchemids[0])
        self.length = len(final_idxs)
        return iter(final_idxs)

class Dataloader_sampler(object):
    """
    Dataloader for CDR(cancer drug response) sampler mode.
    """
    def __init__(self,
                 dataset,
                 batch_size=1,
                 drop_last=False,
                 shuffle=False,
                 num_workers=1,
                 collate_fn=None,
                 buf_size=1000,
                 stream_shuffle_size=0,
                 sampler=None):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.buf_size = buf_size
        self.drop_last = drop_last
        self.stream_shuffle_size = stream_shuffle_size
        self.sampler = sampler

        if self.shuffle and isinstance(self.dataset, StreamDataset):
            warn_msg = "The argument [shuffle] should not be True with StreamDataset. " \
                    "It will be ignored. " \
                    "You might want to set [stream_shuffle_size] with StreamDataset."
            warnings.warn(warn_msg)

        if self.stream_shuffle_size > 0 and self.batch_size >= stream_shuffle_size:
            raise ValueError("stream_shuffle_size must be larger than batch_size," \
                    "but got [stream_shuffle_size=%s] smaller than [batch_size=%s]" \
                    % (self.stream_shuffle_size, self.batch_size))

        if self.stream_shuffle_size > 0 and isinstance(self.dataset, Dataset):
            warn_msg = "[stream_shuffle_size] should not be set with Dataset. " \
                    "It will be ignored. " \
                    "You might want to set [shuffle] with Dataset."
            warnings.warn(warn_msg)

        if self.num_workers < 1:
            raise ValueError("num_workers(default: 1) should be larger than 0, " \
                        "but got [num_workers=%s] < 1." % self.num_workers)

    def __len__(self):
        if not isinstance(self.dataset, StreamDataset):
            return len(self.sampler)
        else:
            raise "StreamDataset has no length"

    def __iter__(self):
        # random seed will be fixed when using multiprocess,
        # so set seed explicitly every time
        np.random.seed()
        if self.num_workers == 1:
            r = paddle.reader.buffered(_DataLoaderIter(self, 0), self.buf_size)
        else:
            worker_pool = [
                _DataLoaderIter(self, wid) for wid in range(self.num_workers)
            ]
            workers = mp_reader.multiprocess_reader(
                worker_pool, use_pipe=True, queue_size=1000)
            r = paddle.reader.buffered(workers, self.buf_size)

        for batch in r():
            yield batch

    def __call__(self):
        return self.__iter__()


class Dataset(BaseDataset):
    """
    Dataset for CDR(cancer drug response)
    """

    def __init__(self, processed_data):
        self.data = processed_data
        self.keys = list(processed_data.keys())
        self.num_samples = len(processed_data[self.keys[0]])

    def __getitem__(self, idx):
        
        return self.data[self.keys[0]][idx], self.data[self.keys[1]][idx], self.data[self.keys[2]][idx], \
               self.data[self.keys[3]][idx], self.data[self.keys[4]][idx], self.data[self.keys[5]][idx]

    def get_data_loader(self, batch_size, num_workers=1,
                        shuffle=False, collate_fn=None):
        """Get dataloader.
        Args:
            batch_size (int): number of data items in a batch.
            num_workers (int): number of parallel workers.
            shuffle (int): whether to shuffle yield data.
            collate_fn: callable function that processes batch data to a list of paddle tensor.
        """
        return Dataloader_sampler(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=collate_fn)

    def __len__(self):
        return self.num_samples


def collate_fn(batch_data):
    """
    Collation function to distribute data to samples
    :param batch_data: batch data
    """
    graphs = []
    mut, gexpr, met, Y, cancer_type_list = [], [], [], [], []
    for g, mu, gex, me, y ,c in batch_data:
        graphs.append(g)
        mut.append(mu)
        gexpr.append(gex)
        met.append(me)
        Y.append(y)
        cancer_type_list.append(c)
    return graphs, mut, gexpr, met, Y, cancer_type_list


class Sampler(object):
    """Sampler
    """

    def __init__(self, dataset, batch_size=1, drop_last=False, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        perm = np.arange(0, len(self.dataset))
        if self.shuffle:
            np.random.shuffle(perm)

        batch = []
        for idx in perm:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        length = len(self.dataset)
        if self.drop_last:
            length = length // self.batch_size
        else:
            length = (length + self.batch_size - 1) // self.batch_size
        return length