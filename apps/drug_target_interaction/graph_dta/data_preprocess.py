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
Convert Kiba and Davis datasets into npz file which can be trained directly.

Note that the dataset split is inherited from GraphDTA and DeepDTA
"""

import os
import sys
import json
import random
import pickle
import argparse
import numpy as np
from rdkit import Chem
from collections import OrderedDict

from pahelix.utils.compound_tools import smiles_to_graph_data
from pahelix.utils.protein_tools import ProteinTokenizer
from pahelix.utils.data_utils import save_data_list_to_npz


def main():
    """Entry for data preprocessing."""
    tokenizer = ProteinTokenizer()
    for dataset in ['davis', 'kiba']:
        data_dir = os.path.join(args.dataset_root, dataset)
        if not os.path.exists(data_dir):
            print('Cannot find {}'.format(data_dir))
            continue

        train_fold = json.load(
            open(os.path.join(data_dir, 'folds', 'train_fold_setting1.txt')))
        train_fold = [ee for e in train_fold for ee in e]  # flatten
        test_fold = json.load(
            open(os.path.join(data_dir, 'folds', 'test_fold_setting1.txt')))
        ligands = json.load(
            open(os.path.join(data_dir, 'ligands_can.txt')),
            object_pairs_hook=OrderedDict)
        proteins = json.load(
            open(os.path.join(data_dir, 'proteins.txt')),
            object_pairs_hook=OrderedDict)
        # Use encoding 'latin1' to load py2 pkl from py3
        # pylint: disable=E1123
        affinity = pickle.load(
            open(os.path.join(data_dir, 'Y'), 'rb'), encoding='latin1')

        smiles_lst, protein_lst = [], []
        for k in ligands.keys():
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[k]),
                                      isomericSmiles=True)
            smiles_lst.append(smiles)

        for k in proteins.keys():
            protein_lst.append(proteins[k])

        if dataset == 'davis':
            # Kd data
            affinity = [-np.log10(y / 1e9) for y in affinity]

        affinity = np.asarray(affinity)

        # pylint: disable=E1123
        os.makedirs(os.path.join(data_dir, 'processed'), exist_ok=True)
        for split in ['train', 'test']:
            print('processing {} set of {}'.format(split, dataset))

            split_dir = os.path.join(data_dir, 'processed', split)
            # pylint: disable=E1123
            os.makedirs(split_dir, exist_ok=True)

            fold = train_fold if split == 'train' else test_fold
            rows, cols = np.where(np.isnan(affinity) == False)
            rows, cols = rows[fold], cols[fold]

            data_lst = [[] for _ in range(args.npz_files)]
            for idx in range(len(rows)):
                mol_graph = smiles_to_graph_data(smiles_lst[rows[idx]])
                data = {k: v for k, v in mol_graph.items()}

                seqs = []
                for seq in protein_lst[cols[idx]].split('\x01'):
                    seqs.extend(tokenizer.gen_token_ids(seq))
                data['protein_token_ids'] = np.array(seqs)

                af = affinity[rows[idx], cols[idx]]
                if dataset == 'davis':
                    data['Log10_Kd'] = np.array([af])
                elif dataset == 'kiba':
                    data['KIBA'] = np.array([af])

                data_lst[idx % args.npz_files].append(data)

            random.shuffle(data_lst)
            for j, sub_data_lst in enumerate(data_lst):
                random.shuffle(sub_data_lst)
                npz = os.path.join(
                    split_dir, '{}_{}_{}.npz'.format(dataset, split, j))
                save_data_list_to_npz(sub_data_lst, npz)

        print('==============================')
        print('dataset:', dataset)
        print('train_fold:', len(train_fold))
        print('test_fold:', len(test_fold))
        print('unique drugs:', len(set(smiles_lst)))
        print('unique proteins:', len(set(protein_lst)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default=None, required=True)
    parser.add_argument('--npz_files', type=int, default=1)  # set it > 1 for multi trainers
    args = parser.parse_args()
    main()
