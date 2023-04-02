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

"""Model config."""

import copy
import ml_collections


NUM_RES = 'num residues placeholder'
NUM_MSA_SEQ = 'msa placeholder'
NUM_EXTRA_SEQ = 'extra msa placeholder'
NUM_TEMPLATES = 'num templates placeholder'


def model_config(name: str) -> ml_collections.ConfigDict:
  """Get the ConfigDict of a CASP14 model."""
  if name not in CONFIG_DIFFS:
    raise ValueError(f'Invalid model name {name}.')
  cfg = copy.deepcopy(CONFIG)
  cfg.update_from_flattened_dict(CONFIG_DIFFS[name])
  return cfg


CONFIG_DIFFS = {
    'model_1': {
        # Jumper et al. (2021) Suppl. Table 5, Model 1.1.1
        'data.common.max_extra_msa': 5120,
        'data.common.reduce_msa_clusters_by_max_templates': True,
        'data.common.use_templates': True,
        'model.embeddings_and_evoformer.template.embed_torsion_angles': True,
        'model.embeddings_and_evoformer.template.enabled': True
    },
    'model_2': {
        # Jumper et al. (2021) Suppl. Table 5, Model 1.1.2
        'data.common.reduce_msa_clusters_by_max_templates': True,
        'data.common.use_templates': True,
        'model.embeddings_and_evoformer.template.embed_torsion_angles': True,
        'model.embeddings_and_evoformer.template.enabled': True
    },
    'model_3': {
        # Jumper et al. (2021) Suppl. Table 5, Model 1.2.1
        'data.common.max_extra_msa': 5120,
    },
    'model_4': {
        # Jumper et al. (2021) Suppl. Table 5, Model 1.2.2
        'data.common.max_extra_msa': 5120,
    },
    'model_5': {
        # Jumper et al. (2021) Suppl. Table 5, Model 1.2.3
        'model.global_config.subbatch_size': 24,
        'model.global_config.fuse_attention': False,
        'model.global_config.origin_evoformer_structure': True,
    },
    'initial_model_5_dcu': {
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 512,
        'model.global_config.subbatch_size': 64,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.experimentally_resolved.weight': 0.0,
    },
    'initial': {
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 1024,
        'data.common.reduce_msa_clusters_by_max_templates': True,
        'data.common.use_templates': True,
        'model.embeddings_and_evoformer.template.embed_torsion_angles': True,
        'model.embeddings_and_evoformer.template.enabled': True,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.experimentally_resolved.weight': 0.0,
        'model.embeddings_and_evoformer.template.template_pair_stack.recompute_start_block_index': 2,
    },
    'finetune': {
        'data.eval.max_msa_clusters': 512,
        'data.common.max_extra_msa': 5120,
        'data.common.reduce_msa_clusters_by_max_templates': True,
        'data.common.use_templates': True,
        'model.embeddings_and_evoformer.template.embed_torsion_angles': True,
        'model.embeddings_and_evoformer.template.enabled': True,
        'model.embeddings_and_evoformer.evoformer.msa_row_attention_with_pair_bias.use_subbatch': True,
        'model.embeddings_and_evoformer.evoformer.msa_row_attention_with_pair_bias.subbatch_size': 512,
    },
    'msa1_ex1_vio0': {
        'data.eval.max_msa_clusters': 1,
        'data.common.max_extra_msa': 1,
        'model.global_config.fuse_attention': False,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.experimentally_resolved.weight': 0.0,
    },
    'msa128_ex1024_vio0': {
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 1024,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.experimentally_resolved.weight': 0.0,
    },
    'msa128_ex1024_temp_vio0': {
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 1024,
        'data.common.reduce_msa_clusters_by_max_templates': True,
        'data.common.use_templates': True,
        'model.embeddings_and_evoformer.template.embed_torsion_angles': True,
        'model.embeddings_and_evoformer.template.enabled': True,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.experimentally_resolved.weight': 0.0,
    },
    'msa128_ex1024_ft': {
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 1024,
        'model.heads.structure_module.structural_violation_loss_weight': 1.0,
        'model.heads.experimentally_resolved.weight': 0.01,
    },
    'msa512_ex1024': {
        'data.eval.max_msa_clusters': 512,
        'data.common.max_extra_msa': 1024,
    },
    'msa512_ex1024_rec6': {
        'data.common.num_recycle': 6,
        'model.num_recycle': 6,
        'data.eval.max_msa_clusters': 512,
        'data.common.max_extra_msa': 1024,
    },

    'seq512_pair64_l8_vio0': {
        'data.common.max_extra_msa': 1,
        'data.eval.max_msa_clusters': 1,
        'model.embeddings_and_evoformer.evoformer_num_block': 8,
        'model.embeddings_and_evoformer.msa_channel': 512,
        'model.embeddings_and_evoformer.pair_channel': 64,
        'model.embeddings_and_evoformer.mute_extra_msa': True,
        'model.embeddings_and_evoformer.evoformer.mute_msa_column': True,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing.num_intermediate_channel': 64,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming.num_intermediate_channel': 64,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
    },
    'seq512_pair64_l24_vio0': {
        'data.common.max_extra_msa': 1,
        'data.eval.max_msa_clusters': 1,
        'model.embeddings_and_evoformer.evoformer_num_block': 24,
        'model.embeddings_and_evoformer.msa_channel': 512,
        'model.embeddings_and_evoformer.pair_channel': 64,
        'model.embeddings_and_evoformer.mute_extra_msa': True,
        'model.embeddings_and_evoformer.evoformer.mute_msa_column': True,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing.num_intermediate_channel': 64,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming.num_intermediate_channel': 64,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
    },
    'seq512_pair128_l48_vio0': {
        'data.common.max_extra_msa': 1,
        'data.eval.max_msa_clusters': 1,
        'model.embeddings_and_evoformer.evoformer_num_block': 48,
        'model.embeddings_and_evoformer.msa_channel': 512,
        'model.embeddings_and_evoformer.mute_extra_msa': True,
        'model.embeddings_and_evoformer.evoformer.mute_msa_column': True,
        'model.global_config.subbatch_size': 4,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
    },

    # The following models are fine-tuned from the corresponding models above
    # with an additional predicted_aligned_error head that can produce
    # predicted TM-score (pTM) and predicted aligned errors.
    'model_1_ptm': {
        'data.common.max_extra_msa': 5120,
        'data.common.reduce_msa_clusters_by_max_templates': True,
        'data.common.use_templates': True,
        'model.embeddings_and_evoformer.template.embed_torsion_angles': True,
        'model.embeddings_and_evoformer.template.enabled': True,
        'model.heads.predicted_aligned_error.weight': 0.1
    },
    'model_2_ptm': {
        'data.common.reduce_msa_clusters_by_max_templates': True,
        'data.common.use_templates': True,
        'model.embeddings_and_evoformer.template.embed_torsion_angles': True,
        'model.embeddings_and_evoformer.template.enabled': True,
        'model.heads.predicted_aligned_error.weight': 0.1
    },
    'model_3_ptm': {
        'data.common.max_extra_msa': 5120,
        'model.heads.predicted_aligned_error.weight': 0.1
    },
    'model_4_ptm': {
        'data.common.max_extra_msa': 5120,
        'model.heads.predicted_aligned_error.weight': 0.1
    },
    'model_5_ptm': {
        'model.heads.predicted_aligned_error.weight': 0.1
    }
}

CONFIG = ml_collections.ConfigDict({
    'data': {
        'common': {
            'masked_msa': {
                'profile_prob': 0.1,
                'same_prob': 0.1,
                'uniform_prob': 0.1
            },
            'max_extra_msa': 1024,
            'msa_cluster_features': True,
            'num_recycle': 3,
            'reduce_msa_clusters_by_max_templates': False,
            'resample_msa_in_recycling': True,
            'template_features': [
                'template_all_atom_positions', 'template_sum_probs',
                'template_aatype', 'template_all_atom_masks',
                'template_domain_names'
            ],
            'unsupervised_features': [
                'aatype', 'residue_index', 'sequence', 'msa', 'domain_name',
                'num_alignments', 'seq_length', 'between_segment_residues',
                'deletion_matrix'
            ],
            'use_templates': False,
        },
        'eval': {
            'feat': {
                'aatype': [NUM_RES],
                'all_atom_mask': [NUM_RES, None],
                'all_atom_positions': [NUM_RES, None, None],
                'alt_chi_angles': [NUM_RES, None],
                'atom14_alt_gt_exists': [NUM_RES, None],
                'atom14_alt_gt_positions': [NUM_RES, None, None],
                'atom14_atom_exists': [NUM_RES, None],
                'atom14_atom_is_ambiguous': [NUM_RES, None],
                'atom14_gt_exists': [NUM_RES, None],
                'atom14_gt_positions': [NUM_RES, None, None],
                'atom37_atom_exists': [NUM_RES, None],
                'backbone_affine_mask': [NUM_RES],
                'backbone_affine_tensor': [NUM_RES, None],
                'bert_mask': [NUM_MSA_SEQ, NUM_RES],
                'chi_angles': [NUM_RES, None],
                'chi_mask': [NUM_RES, None],
                'extra_deletion_value': [NUM_EXTRA_SEQ, NUM_RES],
                'extra_has_deletion': [NUM_EXTRA_SEQ, NUM_RES],
                'extra_msa': [NUM_EXTRA_SEQ, NUM_RES],
                'extra_msa_mask': [NUM_EXTRA_SEQ, NUM_RES],
                'extra_msa_row_mask': [NUM_EXTRA_SEQ],
                'is_distillation': [],
                'msa_feat': [NUM_MSA_SEQ, NUM_RES, None],
                'msa_mask': [NUM_MSA_SEQ, NUM_RES],
                'msa_row_mask': [NUM_MSA_SEQ],
                'pseudo_beta': [NUM_RES, None],
                'pseudo_beta_mask': [NUM_RES],
                'random_crop_to_size_seed': [None],
                'residue_index': [NUM_RES],
                'residx_atom14_to_atom37': [NUM_RES, None],
                'residx_atom37_to_atom14': [NUM_RES, None],
                'resolution': [],
                'rigidgroups_alt_gt_frames': [NUM_RES, None, None],
                'rigidgroups_group_exists': [NUM_RES, None],
                'rigidgroups_group_is_ambiguous': [NUM_RES, None],
                'rigidgroups_gt_exists': [NUM_RES, None],
                'rigidgroups_gt_frames': [NUM_RES, None, None],
                'seq_length': [],
                'seq_mask': [NUM_RES],
                'target_feat': [NUM_RES, None],
                'template_aatype': [NUM_TEMPLATES, NUM_RES],
                'template_all_atom_masks': [NUM_TEMPLATES, NUM_RES, None],
                'template_all_atom_positions': [
                    NUM_TEMPLATES, NUM_RES, None, None],
                'template_backbone_affine_mask': [NUM_TEMPLATES, NUM_RES],
                'template_backbone_affine_tensor': [
                    NUM_TEMPLATES, NUM_RES, None],
                'template_mask': [NUM_TEMPLATES],
                'template_pseudo_beta': [NUM_TEMPLATES, NUM_RES, None],
                'template_pseudo_beta_mask': [NUM_TEMPLATES, NUM_RES],
                'template_sum_probs': [NUM_TEMPLATES, None],
                'true_msa': [NUM_MSA_SEQ, NUM_RES]
            },
            'fixed_size': True,
            'subsample_templates': False,  # We want top templates.
            'masked_msa_replace_fraction': 0.15,
            'max_msa_clusters': 512,
            'max_templates': 4,
            'num_ensemble': 1,
            'num_blocks': 5,    # for msa block deletion
            'randomize_num_blocks': False,
            'msa_fraction_per_block': 0.3,
        },
    },
    'model': {
        'embeddings_and_evoformer': {
            'evoformer_num_block': 48,
            'evoformer_recompute_start_block_index': 0,
            'evoformer': {
                'msa_row_attention_with_pair_bias': {
                    'dropout_rate': 0.15,
                    'gating': True,
                    'num_head': 8,
                    'orientation': 'per_row',
                    'shared_dropout': True,
                    'use_subbatch': False,
                    'subbatch_size': 48,
                },
                'msa_column_attention': {
                    'dropout_rate': 0.0,
                    'gating': True,
                    'num_head': 8,
                    'orientation': 'per_column',
                    'shared_dropout': True
                },
                'msa_transition': {
                    'dropout_rate': 0.0,
                    'num_intermediate_factor': 4,
                    'orientation': 'per_row',
                    'shared_dropout': True
                },
                'outer_product_mean': {
                    'chunk_size': 128,
                    'dropout_rate': 0.0,
                    'num_outer_channel': 32,
                    'orientation': 'per_row',
                    'shared_dropout': True
                },
                'triangle_attention_starting_node': {
                    'dropout_rate': 0.25,
                    'gating': True,
                    'num_head': 4,
                    'orientation': 'per_row',
                    'shared_dropout': True
                },
                'triangle_attention_ending_node': {
                    'dropout_rate': 0.25,
                    'gating': True,
                    'num_head': 4,
                    'orientation': 'per_column',
                    'shared_dropout': True
                },
                'triangle_multiplication_outgoing': {
                    'dropout_rate': 0.25,
                    'equation': 'ikc,jkc->ijc',
                    'num_intermediate_channel': 128,
                    'orientation': 'per_row',
                    'shared_dropout': True
                },
                'triangle_multiplication_incoming': {
                    'dropout_rate': 0.25,
                    'equation': 'kjc,kic->ijc',
                    'num_intermediate_channel': 128,
                    'orientation': 'per_row',
                    'shared_dropout': True
                },
                'pair_transition': {
                    'dropout_rate': 0.0,
                    'num_intermediate_factor': 4,
                    'orientation': 'per_row',
                    'shared_dropout': True
                }
            },
            'extra_msa_channel': 64,
            'extra_msa_stack_num_block': 4,
            'extra_msa_stack_recompute_start_block_index': 0,
            'max_relative_feature': 32,
            'msa_channel': 256,
            'pair_channel': 128,
            'prev_pos': {
                'min_bin': 3.25,
                'max_bin': 20.75,
                'num_bins': 15
            },
            'recycle_features': True,
            'recycle_pos': True,
            'seq_channel': 384,
            'template': {
                'attention': {
                    'gating': False,
                    'key_dim': 64,
                    'num_head': 4,
                    'value_dim': 64
                },
                'dgram_features': {
                    'min_bin': 3.25,
                    'max_bin': 50.75,
                    'num_bins': 39
                },
                'embed_torsion_angles': False,
                'enabled': False,
                'template_pair_stack': {
                    'num_block': 2,
                    'recompute_start_block_index': 0,
                    'triangle_attention_starting_node': {
                        'dropout_rate': 0.25,
                        'gating': True,
                        'key_dim': 64,
                        'num_head': 4,
                        'orientation': 'per_row',
                        'shared_dropout': True,
                        'value_dim': 64
                    },
                    'triangle_attention_ending_node': {
                        'dropout_rate': 0.25,
                        'gating': True,
                        'key_dim': 64,
                        'num_head': 4,
                        'orientation': 'per_column',
                        'shared_dropout': True,
                        'value_dim': 64
                    },
                    'triangle_multiplication_outgoing': {
                        'dropout_rate': 0.25,
                        'equation': 'ikc,jkc->ijc',
                        'num_intermediate_channel': 64,
                        'orientation': 'per_row',
                        'shared_dropout': True
                    },
                    'triangle_multiplication_incoming': {
                        'dropout_rate': 0.25,
                        'equation': 'kjc,kic->ijc',
                        'num_intermediate_channel': 64,
                        'orientation': 'per_row',
                        'shared_dropout': True
                    },
                    'pair_transition': {
                        'dropout_rate': 0.0,
                        'num_intermediate_factor': 2,
                        'orientation': 'per_row',
                        'shared_dropout': True
                    }
                },
                'max_templates': 4,
                'subbatch_size': 48,
                'use_template_unit_vector': False,
            }
        },
        'global_config': {
            'deterministic': False,
            'subbatch_size': 48,
            'use_remat': False,
            'zero_init': True,
            'fuse_attention': True,
            'use_dropout_nd': True,
            'origin_evoformer_structure': False,
        },
        'heads': {
            'distogram': {
                'first_break': 2.3125,
                'last_break': 21.6875,
                'num_bins': 64,
                'weight': 0.3
            },
            'predicted_aligned_error': {
                # `num_bins - 1` bins uniformly space the
                # [0, max_error_bin A] range.
                # The final bin covers [max_error_bin A, +infty]
                # 31A gives bins with 0.5A width.
                'max_error_bin': 31.,
                'num_bins': 64,
                'num_channels': 128,
                'filter_by_resolution': True,
                'min_resolution': 0.1,
                'max_resolution': 3.0,
                'weight': 0.0,
            },
            'experimentally_resolved': {
                'filter_by_resolution': True,
                'max_resolution': 3.0,
                'min_resolution': 0.1,
                'weight': 0.01
            },
            'structure_module': {
                'num_layer': 8,
                'fape': {
                    'clamp_distance': 10.0,
                    'clamp_type': 'relu',
                    'loss_unit_distance': 10.0
                },
                'angle_norm_weight': 0.01,
                'chi_weight': 0.5,
                'clash_overlap_tolerance': 1.5,
                'compute_in_graph_metrics': True,
                'dropout': 0.1,
                'num_channel': 384,
                'num_head': 12,
                'num_layer_in_transition': 3,
                'num_point_qk': 4,
                'num_point_v': 8,
                'num_scalar_qk': 16,
                'num_scalar_v': 16,
                'position_scale': 10.0,
                'sidechain': {
                    'atom_clamp_distance': 10.0,
                    'num_channel': 128,
                    'num_residual_block': 2,
                    'weight_frac': 0.5,
                    'length_scale': 10.,
                },
                'structural_violation_loss_weight': 1.0,
                'violation_tolerance_factor': 12.0,
                'weight': 1.0
            },
            'predicted_lddt': {
                'filter_by_resolution': True,
                'max_resolution': 3.0,
                'min_resolution': 0.1,
                'num_bins': 50,
                'num_channels': 128,
                'weight': 0.01
            },
            'masked_msa': {
                'num_output': 23,
                'weight': 2.0
            },
        },
        'num_recycle': 3,
        'resample_msa_in_recycling': True
    },
})
