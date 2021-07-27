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

"""
featurizer for lite gem
"""

import numpy as np
import paddle
import pgl

from pahelix.utils.compound_tools import new_smiles_to_graph_data

__all__ = [
    'LiteGEMTransformFn',
    'LiteGEMCollateFn',
]


class LiteGEMTransformFn(object):
    """tbd"""
    def __init__(self, config):
        self.config = config

    def __call__(self, raw_data):
        """
        Gen features according to raw data and return a single graph data.

        Args:
            raw_data: It contains smiles and label,we convert smiles to mol
            by rdkit,then convert mol to graph data.
        
        Returns:
            data: It contains reshape label and smiles.
        """
        smiles = raw_data['smiles']
        label = np.array([0]) if 'label' not in raw_data else raw_data['label']

        feature_dict = new_smiles_to_graph_data(smiles)
        if feature_dict is None:
            return None
        feature_dict["label"] = label
        
        new_graph = {}
        new_graph["num_nodes"] = len(feature_dict['atomic_num'])
        new_graph["nfeat"] = {key: feature_dict[key] for key in self.config.atom_names + self.config.atom_float_names}
        new_graph["efeat"] = {key: feature_dict[key] for key in self.config.bond_names}
        new_graph["edges"] = feature_dict['edges']
        new_graph["label"] = feature_dict['label'] if "label" in feature_dict else None
        return new_graph


class LiteGEMCollateFn(object):
    """CollateFn for attribute mask model of pretrain gnns"""
    def __init__(self):
        pass
    
    def __call__(self, batch):
        """
        """
        graph_list = []
        labels = []
        smiles_list = []
        #for gdata in batch_data:
        for admet_d_item in batch:
            gdata = admet_d_item.get_feature()
            g = pgl.Graph(edges=gdata['edges'],
                    num_nodes=gdata['num_nodes'],
                    node_feat=gdata['nfeat'],
                    edge_feat=gdata['efeat'])
            graph_list.append(g)
            labels.append(gdata['label'])
            smiles_list.append(gdata['smiles'])

        labels = paddle.to_tensor(np.array(labels, dtype="float32"))
        g = pgl.Graph.batch(graph_list).tensor()
        return {'graph': g, 'labels': labels}
