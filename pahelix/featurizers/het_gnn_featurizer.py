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
| Featurizers for DDI Heterogenous graph.
"""

import numpy as np
import pandas as pd
import networkx as nx
import pgl


from sklearn.preprocessing import StandardScaler

__all__ = ['DDiFeaturizer']

class DDiFeaturizer(object):
    """Featurizer for drugs"""
    def __init__(self):
        super(DDiFeaturizer, self).__init__()
        
    def collate_fn(self, ddi_data, dti_data, ppi_data, features):
        """Aggregate all needed nodes into a Hetrogenous graph"""

        drug_feat = pd.read_csv(features, index_col=0)
        drug_feat = drug_feat[~drug_feat.index.duplicated()]
        drug_feat = drug_feat.fillna(0)
        drug_feat.replace([np.inf, -np.inf], 0, inplace=True)

        nm = StandardScaler() 
        scaled_feat = pd.DataFrame(nm.fit_transform(drug_feat))
        scaled_feat = scaled_feat.fillna(0)
        scaled_feat.index = drug_feat.index

        edges = {'dds': [], 'dti': [], 'ppi': []}
        ddi_nn, ddi_nodes = num_nodes_stat(ddi_data) 
        selected_drugs_feat = scaled_feat[scaled_feat.index.isin(ddi_nodes)]
        ddi_nodes = set(selected_drugs_feat.index)
        total_nodes = set()
        label = {}

        for d in ddi_data:
            if d['pair'][0] in ddi_nodes and d['pair'][1] in ddi_nodes:
                edges['dds'].append((d['pair'][0], d['pair'][1]))
                edges['dds'].append((d['pair'][1], d['pair'][0]))
                total_nodes.add(d['pair'][0])
                total_nodes.add(d['pair'][1])
                label[d['pair'][0], d['pair'][1]] = d['label']
                label[d['pair'][1], d['pair'][0]] = d['label']
       
        for d in dti_data:
            if d['pair'][0] in ddi_nodes:
                edges['dti'].append((d['pair'][0], d['pair'][1]))
                edges['dti'].append((d['pair'][1], d['pair'][0]))
                total_nodes.add(d['pair'][0])
                total_nodes.add(d['pair'][1])

        for d in ppi_data:
            edges['ppi'].append((d['pair'][0], d['pair'][1]))
            edges['ppi'].append((d['pair'][1], d['pair'][0]))  
            total_nodes.add(d['pair'][0])
            total_nodes.add(d['pair'][1])

        num_nodes = len(total_nodes)   
        nodes_dict = dict(zip(total_nodes, range(num_nodes)))  
        node_feat = np.zeros((num_nodes, 2325)).astype('float32')  
        selected_drugs_feat.index = [nodes_dict[x] for x in selected_drugs_feat.index]
        
        for d in selected_drugs_feat.index:
            node_feat[d, :] = selected_drugs_feat.loc[d, :].values.astype('float32')
        node_feats = {'features': node_feat}
       
        ek = {'dds':[], 'dti':[], 'ppi':[]}
        for edge_type in edges.keys():
            for p in edges[edge_type]:
                p1, p2 = nodes_dict[p[0]], nodes_dict[p[1]]
                ek[edge_type].append((p1, p2))

        node_types = []
        for m in nodes_dict.keys():
            if m.startswith('CID'):
                node_types.append((nodes_dict[m], 'drug'))
            else:
                node_types.append((nodes_dict[m], 'protein'))
                
        hg = pgl.HeterGraph(num_nodes=num_nodes,
                            edges=ek,
                            node_types=node_types,
                            node_feat=node_feats)
        label_idx = {}
        for key in label.keys():
            label_idx[(nodes_dict[key[0]], nodes_dict[key[1]])] = label[key]
        
        return {'rt': (hg, nodes_dict, label, label_idx)}
        

def num_nodes_stat(data):
    """count the number of nodes from data
        
    Examples:
        data: {'pair': (a, b)}
    """
    nodes = set()
    for d in data:
        nodes.add(d['pair'][0])
        nodes.add(d['pair'][1])
    return len(nodes), nodes

def nx_graph_build(hg, nodes_dict, label):
    """
    Build Heterogenous graph with node name not idx.
    """
    nodes_dict = {v:k for k, v in nodes_dict.items()}
    g = nx.Graph()
    for i in hg['dds'].edges:
        edge = [(nodes_dict[i[0]], nodes_dict[i[1]]) + ({'weight': label[(nodes_dict[i[0]], nodes_dict[i[1]])]}, )]
        g.add_edges_from(list(edge))
    
    for etype in ['dti', 'ppi']:
        for p in hg[etype].edges:
            edge = [(nodes_dict[i[0]], nodes_dict[i[1]])]
            g.add_edges_from(list(edge))
    
    return g
