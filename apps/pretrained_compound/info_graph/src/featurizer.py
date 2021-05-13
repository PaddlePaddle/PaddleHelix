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


class MoleculeCollateFunc(object):
    """
    Collate function for molecule dataloader.
    """
    def __init__(self,
                 atom_names,
                 bond_names,
                 task_type='cls',
                 num_cls_tasks=1,
                 reg_target_id=0,
                 with_graph_label=True,
                 with_pos_neg_mask=False):
        assert task_type in ['cls', 'reg']
        self.atom_names = atom_names
        self.bond_names = bond_names
        self.task_type = task_type
        self.num_cls_tasks = num_cls_tasks
        self.reg_target_id = reg_target_id
        self.with_graph_label = with_graph_label
        self.with_pos_neg_mask = with_pos_neg_mask

    def __call__(self, batch_data_list):
        """
        Function caller to convert a batch of data into a big batch feed dictionary.

        Args:
            batch_data_list: a batch of the compound graph data.

        Returns:
            feed_dict: a dictionary contains `graph/xxx` inputs for PGL.
        """
        g_list, label_list = [], []
        for data in batch_data_list:
            g = pgl.Graph(
                num_nodes=len(data[self.atom_names[0]]),
                edges=data['edges'],
                node_feat={name: data[name].reshape([-1, 1])
                           for name in self.atom_names},
                edge_feat={name: data[name].reshape([-1, 1])
                           for name in self.bond_names})
            g_list.append(g)
            if self.with_graph_label:
                label_list.append(data['label'])

        join_graph = pgl.Graph.batch(g_list)
        output = [join_graph]

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
            output.extend([batch_label, batch_valid])

        if self.with_pos_neg_mask:
            pos_mask, neg_mask = MoleculeCollateFunc.get_pos_neg_mask(g_list)
            output.extend([pos_mask, neg_mask])

        return output

    @staticmethod
    def get_pos_neg_mask(g_list):
        """Get the mask.

        Positive mask records the nodes in a batch from the same molecule.
        Negative mask records the nodes in a batch from the other molecule.
        """
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
