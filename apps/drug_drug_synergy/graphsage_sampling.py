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
| Sampling from  DDI Heterogenous graph.
"""
import os
import paddle
import numpy as np
import networkx as nx

import pgl
from pgl.sampling import graphsage_sample

def graphsage_sampling(hg, start_nodes, num_neighbours=10, etype='dti'):
    """
    Sampling radomly based on the neighbours selected nodes.
    This function only for one-degree neighbours
    Args:
    graph: pgl.Graph
    num_neighbours: max value of one-degree neighbous
    etype: the exact edge type you want to sample, default is `dti`
    """
    qualified_neighs = []
    qualified_eids = []
    all_neigs, all_eids = hg[etype].sample_predecessor(start_nodes, 5, return_eids=True)

    while num_neighbours != 0:
        starts_idx = np.random.choice(len(all_neigs))
        starts_chosen = start_nodes[starts_idx]
        if all_neigs[starts_idx].size != 0:
            selected_neighs = all_neigs[starts_idx]
            selected_eids = all_eids[starts_idx]
            successor = np.random.choice(selected_neighs)
            if (starts_chosen, successor) in qualified_eids:
                pass
            else:
                qualified_neighs.append(successor)
                qualified_eids.append((starts_chosen, successor))
                num_neighbours -= 1
        else:
            pass
            
    return qualified_neighs, qualified_eids

def subgraph_gen(hg, label_idx, neighbours=[10, 10]):
    """
    Subgraph sampling by graphsage_sampling
    """
    nt = hg.node_types
    drugs_idx = np.where(nt == 'drug')[0]
    layer1_neighs, layer1_eids = graphsage_sampling(hg, 
                                                drugs_idx, 
                                                num_neighbours = neighbours[0], 
                                                etype='dti')
    layer2_neighs, layer2_eids = graphsage_sampling(hg, 
                                                layer1_neighs, 
                                                num_neighbours = neighbours[1], 
                                                etype='ppi')
    sub_nodes = drugs_idx.tolist() + layer1_neighs + layer2_neighs
    sub_nodes_reidx = dict(zip(sub_nodes, range(len(sub_nodes))))

    label_mat = np.zeros((len(sub_nodes), len(sub_nodes))).astype('float32')
    for p in hg['dds'].edges.tolist():
        label_mat[sub_nodes_reidx[p[0]], sub_nodes_reidx[p[1]]] = label_idx[tuple(p)]
    
    sub_eids = {}
    sub_eids['dds'] = [(sub_nodes_reidx[src], sub_nodes_reidx[dst]) for (src, dst) in hg['dds'].edges.tolist()]
    sub_eids['dti'] = [(sub_nodes_reidx[src], sub_nodes_reidx[dst]) for (src, dst) in layer1_eids]
    sub_eids['dti'] += [(sub_nodes_reidx[dst], sub_nodes_reidx[src]) for (src, dst) in layer1_eids]

    sub_eids['ppi'] = [(sub_nodes_reidx[src], sub_nodes_reidx[dst]) for (src, dst) in layer2_eids]
    sub_eids['ppi'] += [(sub_nodes_reidx[dst], sub_nodes_reidx[src]) for (src, dst) in layer2_eids]

    sub_nodes_feat = hg['dds'].node_feat['features'][sub_nodes, :]
    sub_graph = pgl.HeterGraph(edges=sub_eids, num_nodes=len(sub_nodes), node_feat={'features': sub_nodes_feat})

    return {'sub_graph': (sub_graph, len(sub_nodes), sub_eids, sub_nodes_feat, label_mat)}