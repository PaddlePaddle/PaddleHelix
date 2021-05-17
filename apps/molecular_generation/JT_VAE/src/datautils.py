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
"""dataset utils"""
import os
import pickle
import random
import numpy as np

from src.jtnn_enc import JTNNEncoder
from src.mpn import MPN
from src.jtmpn import JTMPN
from src.mol_tree import MolTree
from src.utils import load_json_config
from pgl.utils.data.dataset import StreamDataset


class JtnnDataSet(StreamDataset):
    """JtnnDataSet"""

    def __init__(self, data_dir, shuffle=True):
        self.file_list = os.listdir(data_dir)
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.cached_len = None

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.file_list)
        for file in self.file_list:
            with open(os.path.join(self.data_dir, file), 'rb') as f:
                data_list = pickle.load(f)
            for data in data_list:
                yield data

    def __len__(self):
        if self.cached_len is not None:
            return self.cached_len
        else:
            n = 0
            for file in self.file_list:
                data_list = load_json_config(os.path.join(self.data_dir, file))
                n += len(data_list)
            self.cached_len = n
            return n


class JtnnCollateFn(object):
    """JtnnCollateFn"""

    def __init__(self, vocab, assm):
        self.vocab = vocab
        self.assm = assm

    @staticmethod
    def set_batch_nodeID(mol_batch, vocab):
        """set batch nodeID"""
        tot = 0
        for mol_tree in mol_batch:
            for node in mol_tree.nodes:
                node.idx = tot
                node.wid = vocab.get_index(node.smiles)
                tot += 1

    def __call__(self, tree_batch):
        tree_batch, vocab, assm = tree_batch, self.vocab, self.assm
        self.set_batch_nodeID(tree_batch, vocab)
        smiles_batch = [tree.smiles for tree in tree_batch]
        jtenc_holder, mess_dict = JTNNEncoder.tensorize(tree_batch)
        jtenc_holder = jtenc_holder

        mpn_holder = MPN.tensorize(smiles_batch)

        if assm is False:
            return tree_batch, jtenc_holder, mpn_holder

        cands = []
        batch_idx = []
        for i, mol_tree in enumerate(tree_batch):
            for node in mol_tree.nodes:
                if node.is_leaf or len(node.cands) == 1:
                    continue
                cands.extend([(cand, mol_tree.nodes, node)
                              for cand in node.cands])
                batch_idx.extend([i] * len(node.cands))

        jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
        batch_idx = np.array(batch_idx)
        return {'tree_batch': tree_batch,
                'jtenc_holder': jtenc_holder,
                'mpn_holder': mpn_holder,
                'jtmpn_holder': (jtmpn_holder, batch_idx)}



