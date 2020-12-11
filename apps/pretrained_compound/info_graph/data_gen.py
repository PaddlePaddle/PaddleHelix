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

import numpy as np
import pgl
from pgl import graph


class MoleculeCollateFunc(object):
    """
    Collate function for molecule dataloader.
    """
    def __init__(self,
                 graph_wrapper,
                 task_type='cls',
                 num_cls_tasks=1,
                 reg_target_id=0,
                 with_graph_label=True,
                 with_pos_neg_mask=False):
        assert task_type in ['cls', 'reg']
        self.graph_wrapper = graph_wrapper
        self.task_type = task_type
        self.num_cls_tasks = num_cls_tasks
        self.reg_target_id = reg_target_id
        self.with_graph_label = with_graph_label
        self.with_pos_neg_mask = with_pos_neg_mask

    def __call__(self, batch_data_list):
        g_list = []
        label_list = []
        for data in batch_data_list:
            g = graph.Graph(
                num_nodes=len(data['atom_type']),
                edges=data['edges'],
                node_feat={
                    'atom_type': data['atom_type'].reshape([-1, 1]),
                    'chirality_tag': data['chirality_tag'].reshape([-1, 1]),
                },
                edge_feat={
                    'bond_type': data['bond_type'].reshape([-1, 1]),
                    'bond_direction': data['bond_direction'].reshape([-1, 1]),
                })
            g_list.append(g)
            if self.with_graph_label:
                label_list.append(data['label'])

        join_graph = pgl.graph.MultiGraph(g_list)
        feed_dict = self.graph_wrapper.to_feed(join_graph)

        if self.with_graph_label:
            if self.task_type == 'cls':
                batch_label = np.array(label_list).reshape(
                    -1, self.num_cls_tasks)
            elif self.task_type == 'reg':
                label_list = [label[self.reg_target_id]
                              for label in label_list]
                batch_label = np.array(label_list).reshape(-1, 1)

            # label: -1 -> 0, 1 -> 1
            batch_label = ((batch_label + 1.0) / 2).astype('float32')
            batch_valid = (batch_label != 0.5).astype("float32")
            feed_dict['label'] = batch_label
            feed_dict['valid'] = batch_valid

        if self.with_pos_neg_mask:
            pos_mask, neg_mask = self.get_pos_neg_mask(g_list)
            feed_dict['pos_mask'] = pos_mask
            feed_dict['neg_mask'] = neg_mask

        return feed_dict

    def get_pos_neg_mask(self, g_list):
        """Get the mask"""
        num_nodes = np.cumsum([0] + [g.num_nodes for g in g_list])
        num_graphs = len(g_list)

        pos_mask = np.zeros([num_nodes[-1], num_graphs]).astype(np.float32)
        neg_mask = np.ones([num_nodes[-1], num_graphs]).astype(np.float32)

        for i in range(1, num_graphs + 1):
            node_ids = np.arange(num_nodes[i-1], num_nodes[i])
            graph_id = i - 1
            pos_mask[node_ids, graph_id] = 1.0
            neg_mask[node_ids, graph_id] = 0.0

        return pos_mask, neg_mask
