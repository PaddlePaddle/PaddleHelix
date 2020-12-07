#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

"""
split molecules into train, valid and tes set
"""

import random
import numpy as np
from itertools import compress
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

__all__ = [
    'RandomSplitter',
    'IndexSplitter',
    'ScaffoldSplitter',
    'RandomScaffoldSplitter',
]


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


class Splitter(object):
    """docstring for Splitter"""
    def __init__(self):
        super(Splitter, self).__init__()


class RandomSplitter(Splitter):
    """docstring for RandomSplitter"""
    def __init__(self):
        super(RandomSplitter, self).__init__()

    def split(self, 
            dataset, 
            frac_train=None, 
            frac_valid=None, 
            frac_test=None,
            seed=None):
        """tbd"""
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)

        indices = list(range(N))
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
        train_cutoff = int(frac_train * N)
        valid_cutoff = int((frac_train + frac_valid) * N)

        train_dataset = dataset[indices[:train_cutoff]]
        valid_dataset = dataset[indices[train_cutoff:valid_cutoff]]
        test_dataset = dataset[indices[valid_cutoff:]]
        return train_dataset, valid_dataset, test_dataset


class IndexSplitter(Splitter):
    """docstring for IndexSplitter"""
    def __init__(self):
        super(IndexSplitter, self).__init__()

    def split(self, 
            dataset, 
            frac_train=None, 
            frac_valid=None, 
            frac_test=None):
        """tbd"""
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)

        indices = list(range(N))
        train_cutoff = int(frac_train * N)
        valid_cutoff = int((frac_train + frac_valid) * N)

        train_dataset = dataset[indices[:train_cutoff]]
        valid_dataset = dataset[indices[train_cutoff:valid_cutoff]]
        test_dataset = dataset[indices[valid_cutoff:]]
        return train_dataset, valid_dataset, test_dataset


class ScaffoldSplitter(Splitter):
    """docstring for ScaffoldSplitter"""
    def __init__(self):
        super(ScaffoldSplitter, self).__init__()
    
    def split(self, 
            dataset, 
            frac_train=None, 
            frac_valid=None, 
            frac_test=None):
        """
        Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
        Split dataset by Bemis-Murcko scaffolds
        This function can also ignore examples containing null values for a
        selected task when splitting. Deterministic split
        :param dataset: pytorch geometric dataset obj
        :param smiles_list: list of smiles corresponding to the dataset obj
        :param task_idx: column idx of the data.y tensor. Will filter out
        examples with null value in specified task column of the data.y tensor
        prior to splitting. If None, then no filtering
        :param null_value: float that specifies null value in data.y to filter if
        task_idx is provided
        :param frac_train:
        :param frac_valid:
        :param frac_test:
        :param return_smiles:
        :return: train, valid, test slices of the input dataset obj. If
        return_smiles = True, also returns ([train_smiles_list],
        [valid_smiles_list], [test_smiles_list])
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)

        # create dict of the form {scaffold_i: [idx1, idx....]}
        all_scaffolds = {}
        for i in range(N):
            scaffold = generate_scaffold(dataset[i]['smiles'], include_chirality=True)
            if scaffold not in all_scaffolds:
                all_scaffolds[scaffold] = [i]
            else:
                all_scaffolds[scaffold].append(i)

        # sort from largest to smallest sets
        all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
        all_scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]

        # get train, valid test indices
        train_cutoff = frac_train * N
        valid_cutoff = (frac_train + frac_valid) * N
        train_idx, valid_idx, test_idx = [], [], []
        for scaffold_set in all_scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                    test_idx.extend(scaffold_set)
                else:
                    valid_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        assert len(set(train_idx).intersection(set(valid_idx))) == 0
        assert len(set(test_idx).intersection(set(valid_idx))) == 0

        # get train, valid test indices
        train_cutoff = frac_train * N
        valid_cutoff = (frac_train + frac_valid) * N
        train_idx, valid_idx, test_idx = [], [], []
        for scaffold_set in all_scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                    test_idx.extend(scaffold_set)
                else:
                    valid_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        assert len(set(train_idx).intersection(set(valid_idx))) == 0
        assert len(set(test_idx).intersection(set(valid_idx))) == 0

        train_dataset = dataset[train_idx]
        valid_dataset = dataset[valid_idx]
        test_dataset = dataset[test_idx]
        return train_dataset, valid_dataset, test_dataset


class RandomScaffoldSplitter(Splitter):
    """docstring for RandomScaffoldSplitter"""
    def __init__(self):
        super(RandomScaffoldSplitter, self).__init__()
    
    def split(self, 
            dataset, 
            frac_train=None, 
            frac_valid=None, 
            frac_test=None,
            seed=None):
        """
        Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/chainer_chemistry/dataset/splitters/scaffold_splitter.py
        Split dataset by Bemis-Murcko scaffolds
        This function can also ignore examples containing null values for a
        selected task when splitting. Deterministic split
        :param dataset: pytorch geometric dataset obj
        :param smiles_list: list of smiles corresponding to the dataset obj
        :param task_idx: column idx of the data.y tensor. Will filter out
        examples with null value in specified task column of the data.y tensor
        prior to splitting. If None, then no filtering
        :param null_value: float that specifies null value in data.y to filter if
        task_idx is provided
        :param frac_train:
        :param frac_valid:
        :param frac_test:
        :param seed;
        :return: train, valid, test slices of the input dataset obj
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)

        rng = np.random.RandomState(seed)

        scaffolds = defaultdict(list)
        for ind in range(N):
            scaffold = generate_scaffold(dataset[ind]['smiles'], include_chirality=True)
            scaffolds[scaffold].append(ind)

        scaffold_sets = rng.permutation(list(scaffolds.values()))

        n_total_valid = int(np.floor(frac_valid * len(dataset)))
        n_total_test = int(np.floor(frac_test * len(dataset)))

        train_idx = []
        valid_idx = []
        test_idx = []

        for scaffold_set in scaffold_sets:
            if len(valid_idx) + len(scaffold_set) <= n_total_valid:
                valid_idx.extend(scaffold_set)
            elif len(test_idx) + len(scaffold_set) <= n_total_test:
                test_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        train_dataset = dataset[train_idx]
        valid_dataset = dataset[valid_idx]
        test_dataset = dataset[test_idx]
        return train_dataset, valid_dataset, test_dataset
        

# def scaffold_split(data_list, smiles_list, task_idx=None, null_value=0,
#                    frac_train=0.8, frac_valid=0.1, frac_test=0.1,
#                    return_smiles=False):
#     """
#     Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
#     Split dataset by Bemis-Murcko scaffolds
#     This function can also ignore examples containing null values for a
#     selected task when splitting. Deterministic split
#     :param dataset: pytorch geometric dataset obj
#     :param smiles_list: list of smiles corresponding to the dataset obj
#     :param task_idx: column idx of the data.y tensor. Will filter out
#     examples with null value in specified task column of the data.y tensor
#     prior to splitting. If None, then no filtering
#     :param null_value: float that specifies null value in data.y to filter if
#     task_idx is provided
#     :param frac_train:
#     :param frac_valid:
#     :param frac_test:
#     :param return_smiles:
#     :return: train, valid, test slices of the input dataset obj. If
#     return_smiles = True, also returns ([train_smiles_list],
#     [valid_smiles_list], [test_smiles_list])
#     """
#     np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

#     if not task_idx is None:
#         raise NotImplementedError()
#     else:
#         non_null = np.ones(len(data_list)) == 1
#         smiles_list = list(compress(enumerate(smiles_list), non_null))

#     # create dict of the form {scaffold_i: [idx1, idx....]}
#     all_scaffolds = {}
#     for i, smiles in smiles_list:
#         scaffold = generate_scaffold(smiles, include_chirality=True)
#         if scaffold not in all_scaffolds:
#             all_scaffolds[scaffold] = [i]
#         else:
#             all_scaffolds[scaffold].append(i)

#     # sort from largest to smallest sets
#     all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
#     all_scaffold_sets = [
#         scaffold_set for (scaffold, scaffold_set) in sorted(
#             all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
#     ]

#     # get train, valid test indices
#     train_cutoff = frac_train * len(smiles_list)
#     valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
#     train_idx, valid_idx, test_idx = [], [], []
#     for scaffold_set in all_scaffold_sets:
#         if len(train_idx) + len(scaffold_set) > train_cutoff:
#             if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
#                 test_idx.extend(scaffold_set)
#             else:
#                 valid_idx.extend(scaffold_set)
#         else:
#             train_idx.extend(scaffold_set)

#     assert len(set(train_idx).intersection(set(valid_idx))) == 0
#     assert len(set(test_idx).intersection(set(valid_idx))) == 0

#     train_data_list = [data_list[i] for i in train_idx]
#     valid_data_list = [data_list[i] for i in valid_idx]
#     test_data_list = [data_list[i] for i in test_idx]

#     if not return_smiles:
#         return train_data_list, valid_data_list, test_data_list
#     else:
#         train_smiles = [smiles_list[i][1] for i in train_idx]
#         valid_smiles = [smiles_list[i][1] for i in valid_idx]
#         test_smiles = [smiles_list[i][1] for i in test_idx]
#         return [train_data_list, valid_data_list, test_data_list, \
#                 (train_smiles, valid_smiles, test_smiles)]


# def random_scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
#                    frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0):
#     """
#     Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/chainer_chemistry/dataset/splitters/scaffold_splitter.py
#     Split dataset by Bemis-Murcko scaffolds
#     This function can also ignore examples containing null values for a
#     selected task when splitting. Deterministic split
#     :param dataset: pytorch geometric dataset obj
#     :param smiles_list: list of smiles corresponding to the dataset obj
#     :param task_idx: column idx of the data.y tensor. Will filter out
#     examples with null value in specified task column of the data.y tensor
#     prior to splitting. If None, then no filtering
#     :param null_value: float that specifies null value in data.y to filter if
#     task_idx is provided
#     :param frac_train:
#     :param frac_valid:
#     :param frac_test:
#     :param seed;
#     :return: train, valid, test slices of the input dataset obj
#     """

#     np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

#     if not task_idx is None:
#         # filter based on null values in task_idx
#         # get task array
#         y_task = np.array([data.y[task_idx].item() for data in dataset])
#         # boolean array that correspond to non null values
#         non_null = y_task != null_value
#         smiles_list = list(compress(enumerate(smiles_list), non_null))
#     else:
#         non_null = np.ones(len(dataset)) == 1
#         smiles_list = list(compress(enumerate(smiles_list), non_null))

#     rng = np.random.RandomState(seed)

#     scaffolds = defaultdict(list)
#     for ind, smiles in smiles_list:
#         scaffold = generate_scaffold(smiles, include_chirality=True)
#         scaffolds[scaffold].append(ind)

#     scaffold_sets = rng.permutation(list(scaffolds.values()))

#     n_total_valid = int(np.floor(frac_valid * len(dataset)))
#     n_total_test = int(np.floor(frac_test * len(dataset)))

#     train_idx = []
#     valid_idx = []
#     test_idx = []

#     for scaffold_set in scaffold_sets:
#         if len(valid_idx) + len(scaffold_set) <= n_total_valid:
#             valid_idx.extend(scaffold_set)
#         elif len(test_idx) + len(scaffold_set) <= n_total_test:
#             test_idx.extend(scaffold_set)
#         else:
#             train_idx.extend(scaffold_set)

#     train_dataset = dataset[torch.tensor(train_idx)]
#     valid_dataset = dataset[torch.tensor(valid_idx)]
#     test_dataset = dataset[torch.tensor(test_idx)]

#     return train_dataset, valid_dataset, test_dataset


# def random_split(dataset, task_idx=None, null_value=0,
#                    frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0,
#                  smiles_list=None):
#     """

#     :param dataset:
#     :param task_idx:
#     :param null_value:
#     :param frac_train:
#     :param frac_valid:
#     :param frac_test:
#     :param seed:
#     :param smiles_list: list of smiles corresponding to the dataset obj, or None
#     :return: train, valid, test slices of the input dataset obj. If
#     smiles_list != None, also returns ([train_smiles_list],
#     [valid_smiles_list], [test_smiles_list])
#     """
#     np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

#     if not task_idx is None:
#         # filter based on null values in task_idx
#         # get task array
#         y_task = np.array([data.y[task_idx].item() for data in dataset])
#         non_null = y_task != null_value  # boolean array that correspond to non null values
#         idx_array = np.where(non_null)[0]
#         dataset = dataset[torch.tensor(idx_array)]  # examples containing non
#         # null labels in the specified task_idx
#     else:
#         pass

#     num_mols = len(dataset)
#     random.seed(seed)
#     all_idx = list(range(num_mols))
#     random.shuffle(all_idx)

#     train_idx = all_idx[:int(frac_train * num_mols)]
#     valid_idx = all_idx[int(frac_train * num_mols):int(frac_valid * num_mols)
#                                                    + int(frac_train * num_mols)]
#     test_idx = all_idx[int(frac_valid * num_mols) + int(frac_train * num_mols):]

#     assert len(set(train_idx).intersection(set(valid_idx))) == 0
#     assert len(set(valid_idx).intersection(set(test_idx))) == 0
#     assert len(train_idx) + len(valid_idx) + len(test_idx) == num_mols

#     train_dataset = dataset[torch.tensor(train_idx)]
#     valid_dataset = dataset[torch.tensor(valid_idx)]
#     test_dataset = dataset[torch.tensor(test_idx)]

#     if not smiles_list:
#         return train_dataset, valid_dataset, test_dataset
#     else:
#         train_smiles = [smiles_list[i] for i in train_idx]
#         valid_smiles = [smiles_list[i] for i in valid_idx]
#         test_smiles = [smiles_list[i] for i in test_idx]
#         return [train_dataset, valid_dataset, test_dataset, \
#                 (train_smiles, valid_smiles, test_smiles)]

