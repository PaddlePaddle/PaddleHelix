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
downstream featurizer
"""

import numpy as np
import pgl
from rdkit.Chem import AllChem

from pahelix.featurizer.featurizer import Featurizer
from pahelix.utils.compound_tools import mol_to_graph_data


class DownstreamFeaturizer(Featurizer):
    """docstring for DownstreamFeaturizer"""
    def __init__(self, graph_wrapper):
        super(DownstreamFeaturizer, self).__init__()
        self.graph_wrapper = graph_wrapper
    
    def gen_features(self, raw_data):
        """tbd"""
        smiles, label = raw_data['smiles'], raw_data['label']
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            return None
        data = mol_to_graph_data(mol)
        data['label'] = label.reshape([-1])
        data['smiles'] = smiles
        return data

    def collate_fn(self, batch_data_list):
        """tbd"""
        g_list = []
        label_list = []
        for data in batch_data_list:
            g = pgl.graph.Graph(num_nodes = len(data['atom_type']),
                    edges = data['edges'],
                    node_feat = {
                        'atom_type': data['atom_type'].reshape([-1, 1]),
                        'chirality_tag': data['chirality_tag'].reshape([-1, 1]),
                    },
                    edge_feat ={
                        'bond_type': data['bond_type'].reshape([-1, 1]),
                        'bond_direction': data['bond_direction'].reshape([-1, 1]),
                    })
            g_list.append(g)
            label_list.append(data['label'])

        join_graph = pgl.graph.MultiGraph(g_list)
        feed_dict = self.graph_wrapper.to_feed(join_graph)
        batch_label = np.array(label_list)
        
        # label: -1 -> 0, 1 -> 1
        batch_label = ((batch_label + 1.0) / 2).astype('float32')
        batch_valid = (batch_label != 0.5).astype("float32")
        feed_dict['finetune_label'] = batch_label
        feed_dict['valid'] = batch_valid
        return feed_dict

