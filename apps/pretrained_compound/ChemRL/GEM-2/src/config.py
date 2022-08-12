#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
config
"""
import numpy as np
import copy
import ml_collections


def make_updated_config(base_config, updated_dict):
    config = copy.deepcopy(base_config)
    config.update_from_flattened_dict(updated_dict)
    return config


MOL_REGRESSION_MODEL_CONFIG = ml_collections.ConfigDict({
    "data": {
        "gen_rdkit3d": True,
        "gen_raw3d": False,
        "max_hop": 5,
    },

    "model": {
        "atom_pos_source": "rdkit3d",
        "encoder_type": "optimus",

        "heads": {
            "pair_dist_diff": {
                "loss_scale": 0.0,
                "pretrain_steps": 20,
                "bin_start": -5.0,
                "bin_end": 5.0,
                "num_bins": 50
            },
            "property_regr": {
                "pool_type": "mean",
                "hidden_size": [128, 64],
                "output_size": 1,
                "loss_type": "l1loss",
                "label_mean": [0.0],
                "label_std": [1.0],
            },
        },
    },
})


OPTIMUS_MODEL_CONFIG = ml_collections.ConfigDict({
    "node_channel": 128,
    "pair_channel": 128,
    "triple_channel": 128,

    "embedding_layer": {
        "atom_names": ["atomic_num", "formal_charge", "degree", 
            "chiral_tag", "total_numHs", "is_aromatic", 
            "hybridization"],
        "bond_names": ["bond_dir", "bond_type", "is_in_ring", "hop_num"],
        "bond_float_names": ["bond_length"],
        "triple_names": ["hop_num_ij", "hop_num_ik", "hop_num_jk"],
        "triple_float_names": ["angle_i", "angle_j", "angle_k"],

        "rbf_params": {
            "bond_length": [0, 5, 0.1, 10.0],
            "angle_i": [0, np.pi, 0.1, 10.0],
            "angle_j": [0, np.pi, 0.1, 10.0],
            "angle_k": [0, np.pi, 0.1, 10.0],
        },
    },

    "init_dropout_rate": 0.05,
    "attention_max_hop_clip": -1,

    "optimus_block_num": 12,
    "optimus_block": {
        "node_dropout_rate": 0.05,
        "pair_dropout_rate": 0.05,
        "node_attention": {
            "use_pair_layer_norm": True,
            "num_head": 8,
            "dropout_rate": 0.05,
            "virtual_node": True
        },
        "node_ffn": {
            "hidden_factor": 4,
            "dropout_rate": 0.1
        },
        "outer_product": {
            "inner_channel": 32
        },
        "triangle_attention_start_node": {
            "num_head": 8,
            "dropout_rate": 0.05,
            "is_start": True
        },
        "triangle_attention_end_node": {
            "num_head": 8,
            "dropout_rate": 0.05,
            "is_start": False
        },
        "pair_ffn": {
            "hidden_factor": 4,
            "dropout_rate": 0.1
        },
    },
})