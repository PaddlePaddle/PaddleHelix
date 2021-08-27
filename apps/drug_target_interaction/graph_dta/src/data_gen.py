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
data gen
"""

import random
import numpy as np
from glob import glob

import pgl
from pgl.utils.data import Dataloader
from pgl.utils.data.dataset import StreamDataset
from pahelix.utils.protein_tools import ProteinTokenizer
from pahelix.utils.data_utils import load_npz_to_data_list


class DTADataset(StreamDataset):
    """DTADataset a subclass of StreamDataset for PGL inputs.
    """
    def __init__(self, data_dir, max_protein_len=1000,
                 subset_selector=None):
        self.max_protein_len = max_protein_len
        self.subset_selector = subset_selector
        self.cached_len = None
        self.files = glob('%s/*.npz' % data_dir)

    def __iter__(self):
        random.shuffle(self.files)
        for file in self.files:
            data_list = load_npz_to_data_list(file)
            if self.subset_selector is not None:
                data_list = self.subset_selector(data_list)
            for data in data_list:
                if self.max_protein_len > 0:
                    protein_token_ids = np.zeros(self.max_protein_len, dtype=np.int64) \
                            + ProteinTokenizer.padding_token_id
                    n = min(self.max_protein_len, data['protein_token_ids'].size)
                    protein_token_ids[:n] = data['protein_token_ids'][:n]
                    data['protein_token_ids'] = protein_token_ids
                yield data

    def __len__(self):
        if self.cached_len is not None:
            return self.cached_len
        else:
            n = 0
            for file in self.files:
                data_list = load_npz_to_data_list(file)
                n += len(data_list)

            self.cached_len = n
            return n

    def get_data_loader(self, batch_size, num_workers=4,
                        shuffle=False, collate_fn=None):
        """Get dataloader.

        Args:
            batch_size (int): number of data items in a batch.
            num_workers (int): number of parallel workers.
            shuffle (int): whether to shuffle yield data.
            collate_fn: callable function that processes batch data to a list of paddle tensor.
        """
        return Dataloader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=collate_fn)


class DTACollateFunc(object):
    """Collation function to convert batch ndarray data to tensors.
    """
    def __init__(self,
                 atom_names,
                 bond_names,
                 label_name='Log10_Kd',
                 is_inference=False,
                 max_atom_degree=10,
                 max_atom_numHs=9,
                 max_atom_valence=8):
        """Collate function for PGL dataloader.

        Args:
            graph_wrapper (pgl.graph_wrapper.GraphWrapper): graph wrapper for GNN.
            label_name (str): the key in the feed dictionary for the drug-target affinity.
                For Davis, it is `Log10_Kd`; For Kiba, it is `KIBA`.
            is_inference (bool): when its value is True, there is no label in the generated feed dictionary.
            max_atom_degree (int): maximum number of atom degree.
            max_atom_numHs (int): maximum number of Hs of atom.
            max_atom_valence (int): maximum number of explicit valence of atom.

        Return:
            collate_fn: a callable function.
        """
        assert label_name in ['Log10_Kd', 'Log10_Ki', 'KIBA']
        super(DTACollateFunc, self).__init__()
        self.atom_names = atom_names
        self.bond_names = bond_names
        self.is_inference = is_inference
        self.label_name = label_name

        self.max_atom_degree = max_atom_degree
        self.max_atom_numHs = max_atom_numHs
        self.max_atom_valence = max_atom_valence

    def __call__(self, batch_data_list):
        """
        Function caller to convert a batch of data into a big batch feed dictionary.

        Args:
            batch_data_list: a batch of the compound graph data and protein sequence tokens data.
        """
        g_list = []
        for data in batch_data_list:
            atom_numeric_feat = np.concatenate([
                self._convert(data['degree'], self.max_atom_degree),
                self._convert(data['total_numHs'], self.max_atom_numHs),
                self._convert(data['explicit_valence'], self.max_atom_valence),
                data['is_aromatic'].reshape([-1, 1])
            ], axis=1).astype(np.float32)
            node_feat = {name: data[name].reshape([-1, 1])
                         for name in self.atom_names}
            node_feat['atom_numeric_feat'] = atom_numeric_feat

            edge_feat = {name: data[name].reshape([-1, 1])
                         for name in self.bond_names}
            g = pgl.Graph(
                num_nodes=len(data[self.atom_names[0]]),
                edges=data['edges'],
                node_feat=node_feat,
                edge_feat=edge_feat)
            g_list.append(g)

        join_graph = pgl.Graph.batch(g_list)
        output = [join_graph]

        proteins_token, proteins_mask = [], []
        max_protein_len = max([len(data['protein_token_ids'])
                               for data in batch_data_list])
        for data in batch_data_list:
            tokens = data['protein_token_ids']
            new_tokens = np.zeros(max_protein_len, dtype=np.int64)
            new_tokens += ProteinTokenizer.padding_token_id
            new_tokens[:len(tokens)] = tokens

            mask = np.ones(max_protein_len, dtype=np.float32)
            mask[len(tokens):] = 0.

            proteins_token.append(new_tokens)
            proteins_mask.append(mask)

        proteins_token = np.array(proteins_token, dtype=np.int64)
        proteins_mask = np.array(proteins_mask, dtype=np.float32)

        output = [join_graph, proteins_token, proteins_mask]

        if not self.is_inference:
            batch_label = np.array([data[self.label_name] for data in batch_data_list]).reshape(-1, 1)
            batch_label = batch_label.astype('float32')
            output.append(batch_label)

        return output

    def _convert(self, nums, max_num):
        illegal_ids = np.where(nums >= max_num)[0]
        nums[illegal_ids] = max_num - 1
        return np.eye(max_num)[nums]
