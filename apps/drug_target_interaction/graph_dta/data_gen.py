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

from paddle import fluid
import pgl
from pgl.utils.data.dataset import Dataset, StreamDataset
from pahelix.utils.protein_tools import ProteinTokenizer
from pahelix.utils.data_utils import load_npz_to_data_list


class DTADataset(StreamDataset):
    """DTADataset a subclass of StreamDataset for PGL inputs.
    """
    def __init__(self, data_dir, trainer_id=0, trainer_num=1, max_protein_len=1000, subset_selector=None):
        self.max_protein_len = max_protein_len
        self.subset_selector = subset_selector
        self.cached_len = None
        files = glob('%s/*_%s.npz' % (data_dir, trainer_id))
        files = sorted(files)
        self.files = []
        for (i, file) in enumerate(files):
            if i % trainer_num == trainer_id:
                self.files.append(file)

    def __iter__(self):
        random.shuffle(self.files)
        for file in self.files:
            data_list = load_npz_to_data_list(file)
            if self.subset_selector is not None:
                data_list = self.subset_selector(data_list)
            for data in data_list:
                if self.max_protein_len > 0:
                    protein_token_ids = np.zeros(self.max_protein_len, dtype=np.int64) + ProteinTokenizer.padding_token_id
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


class DTACollateFunc(object):
    def __init__(self, graph_wrapper, label_name='Log10_Kd', is_inference=False):
    """Collate function for PGL dataloader.

    Args:
        graph_wrapper (pgl.graph_wrapper.GraphWrapper): graph wrapper for GNN.
        label_name (str): the key in the feed dictionary for the drug-target affinity.
            For Davis, it is `Log10_Kd`; For Kiba, it is `KIBA`.
        is_inference (bool): when its value is True, there is no label in the generated feed dictionary.

    Return:
        collate_fn: a callable function.
    """
        assert label_name in ['Log10_Kd', 'Log10_Ki', 'KIBA']
        super(DTACollateFunc, self).__init__()
        self.graph_wrapper = graph_wrapper
        self.is_inference = is_inference
        self.label_name = label_name

    def __call__(self, batch_data_list):
        """
        Function caller to convert a batch of data into a big batch feed dictionary.

        Args:
            batch_data_list: a batch of the compound graph data and protein sequence tokens data.

        Returns:
            feed_dict: a dictionary contains `graph/xxx` inputs for PGL and `protein_xxx` for protein model.
        """
        graph_list = []
        for data in batch_data_list:
            atom_numeric_feat = np.concatenate([
                data['atom_degrees'],
                data['atom_Hs'],
                data['atom_implicit_valence'],
                data['atom_is_aromatic'].reshape([-1, 1])
            ], axis=1).astype(np.float32)
            g = pgl.graph.Graph(
                num_nodes = len(data['atom_type']),
                edges = data['edges'],
                node_feat = {
                    'atom_type': data['atom_type'].reshape([-1, 1]),
                    'chirality_tag': data['chirality_tag'].reshape([-1, 1]),
                    'atom_numeric_feat': atom_numeric_feat
                },
                edge_feat = {
                    'bond_type': data['bond_type'].reshape([-1, 1]),
                    'bond_direction': data['bond_direction'].reshape([-1, 1])
                })
            graph_list.append(g)

        join_graph = pgl.graph.MultiGraph(graph_list)
        feed_dict = self.graph_wrapper.to_feed(join_graph)

        protein_token = [data['protein_token_ids'] for data in batch_data_list]
        protein_length = [0] + [data['protein_token_ids'].size for data in batch_data_list]
        feed_dict['protein_token'] = np.concatenate(protein_token).reshape([-1, 1]).astype('int64')
        feed_dict['protein_token_lod'] = np.add.accumulate(
                protein_length).reshape([1, -1]).astype('int32')

        if not self.is_inference:
            batch_label = np.array([data[self.label_name] for data in batch_data_list]).reshape(-1, 1)
            batch_label = batch_label.astype('float32')
            feed_dict['label'] = batch_label
        return feed_dict
