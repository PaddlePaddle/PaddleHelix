#!/usr/bin/python
#-*-coding:utf-8-*-
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
Transformer featurizer
"""

import numpy as np
from rdkit.Chem import AllChem

import os
from pahelix.utils.compound_tools import Compound3DKit
from pahelix.utils.compound_tools import mol_to_transformer_data, mol_to_trans_data_w_meta_path, mol_to_trans_data_w_pair_dist
from pahelix.utils.compound_tools import mol_to_trans_data_w_rdkit3d

from .utils import sequence_pad, edge_to_pair, pair_pad


class OptimusTransformFn(object):
    """Gen features for mol regression model"""
    def __init__(self, model_config, encoder_config, is_inference=False):
        self.model_config = model_config
        self.encoder_config = encoder_config
        self.is_inference = is_inference

    def __call__(self, raw_data):
        """
        Gen features according to raw data and return a single graph data.
        Args:
            raw_data: It contains smiles and label,we convert smiles 
            to mol by rdkit,then convert mol to graph data.
        Returns:
            data: It contains reshape label and smiles.
        """
        smiles = raw_data['smiles']
        print('processing', smiles, flush=True)
        if not 'mol' in raw_data:
            mol = AllChem.MolFromSmiles(smiles)
            if mol is None:
                print('processing', smiles, 'failed', flush=True)
                return None
        else:
            mol = raw_data['mol']
        
        data = {}
        data['smiles'] = smiles

        if self.model_config.data.gen_raw3d:
            if mol.GetNumConformers() == 0:
                data['raw_atom_pos'] = np.zeros([len(mol.GetAtoms()), 3], 'float32')
            else:
                data['raw_atom_pos'] = np.array(Compound3DKit.get_atom_poses(
                        mol, mol.GetConformer()), 'float32')
                # Guarentee that the rdkit3d generation is not influenced.
                mol.RemoveAllConformers()

        if self.model_config.data.gen_rdkit3d:
            data.update(mol_to_trans_data_w_rdkit3d(mol))
            data['rdkit_atom_pos'] = data.pop('atom_pos')
        else:
            data.update(mol_to_transformer_data(mol))

        if not self.is_inference:
            data['label'] = raw_data['label'].reshape([-1])
        return data


class OptimusCollateFn(object):
    """CollateFn for mol regression model"""
    def __init__(self, model_config, encoder_config, is_inference=False):
        self.model_config = model_config
        self.encoder_config = encoder_config
        self.is_inference = is_inference
        
    def _get_node_num_list(self, data_list):
        return [len(x[self.encoder_config.embedding_layer.atom_names[0]])
                for x in data_list]

    def _get_max_node_num(self, data_list):
        return np.max(self._get_node_num_list(data_list))

    def _get_node_mask(self, data_list, max_len):
        node_mask = np.zeros([len(data_list), max_len], 'float32')
        for i, l in enumerate(self._get_node_num_list(data_list)):
            node_mask[i, :l] = 1
        return node_mask

    def _process_atom_features(self, data_list, max_len):
        features = {}
        for name in self.encoder_config.embedding_layer.atom_names:
            features[name] = sequence_pad([x[name] for x in data_list], max_len).astype('int64')
        return features

    def _process_bond_features(self, data_list, max_len):
        features = {}
        edges_list = [x['edges'] for x in data_list]
        for name in self.encoder_config.embedding_layer.bond_names:
            if name in ["hop_num"]:
                continue
            
            value_list = [x[name] for x in data_list]
            pair_list = [edge_to_pair(e, v, max_len) for e, v in zip(edges_list, value_list)]
            features[name] = pair_pad(pair_list, max_len).astype('int64')
        return features
    
    def _process_extra_features(self, data_list, max_len):
        features = {}
        hop_num_list = [x['dist_matrix'] for x in data_list]
        hop_num = pair_pad(hop_num_list, max_len).astype('int64')
        features['hop_num'] = np.clip(hop_num, 0, self.model_config.data.max_hop)

        features['node_mask'] = self._get_node_mask(data_list, max_len).astype('float32')

        if self.model_config.data.gen_raw3d:
            features['raw_atom_pos'] = sequence_pad([x['raw_atom_pos'] for x in data_list], max_len).astype('float32')
        if self.model_config.data.gen_rdkit3d:
            if not 'rdkit_atom_pos' in data_list[0]:
                features['rdkit_atom_pos'] = sequence_pad([x['atom_pos'] for x in data_list], max_len).astype('float32')
            else:
                features['rdkit_atom_pos'] = sequence_pad([x['rdkit_atom_pos'] for x in data_list], max_len).astype('float32')
        return features

    def _process_label(self, data_list, max_len):
        features = {}
        label_list = [x['label'] for x in data_list]
        features['label'] = np.array(label_list).astype('float32')
        return features

    def __call__(self, data_list):
        max_len = self._get_max_node_num(data_list)
        batch = {}
        batch.update(self._process_atom_features(data_list, max_len))
        batch.update(self._process_bond_features(data_list, max_len))
        batch.update(self._process_extra_features(data_list, max_len))
        if not self.is_inference:
            batch.update(self._process_label(data_list, max_len))
        return batch


if __name__ == "__main__":
    import json
    from pahelix.datasets.inmemory_dataset import InMemoryDataset
    from .config import make_updated_config, MOL_REGRESSION_MODEL_CONFIG, OPTIMUS_MODEL_CONFIG
    path = "../data/pcqm4m-v2-rdkit3d/train"
    dataset = InMemoryDataset(npz_data_path=path)

    model_config = "configs/model_configs/mol_regr-optimus-mae.json"
    model_config = make_updated_config(
            MOL_REGRESSION_MODEL_CONFIG,
            json.load(open(model_config, 'r')))

    encoder_config = "configs/model_configs/opt3d_l12_c128.json"
    encoder_config = make_updated_config(
            OPTIMUS_MODEL_CONFIG,
            json.load(open(encoder_config, 'r')))
    transform_fn = LiteOptimusTransformFn(model_config, encoder_config)
    collate_fn = LiteOptimusCollateFn(model_config, encoder_config)
    data = transform_fn(dataset[0])
    collate_fn([data, data])

