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
  # if 'multimer' in name:
  #   return CONFIG_MULTIMER

  # if name not in CONFIG_DIFFS:
  #   raise ValueError(f'Invalid model name {name}.')

  if 'multimer' in name:
    cfg = copy.deepcopy(CONFIG_MULTIMER)
    # NOTE: there is no `data` config for CONFIG_MULTIMER
    cfg.data = CONFIG.data
    cfg.update_from_flattened_dict({
      'data.common.use_templates': True,
      'data.eval.feat.asym_id': [NUM_RES],
      'data.eval.feat.sym_id': [NUM_RES],
      'data.eval.feat.entity_id': [NUM_RES],
      'data.eval.feat.entity_mask': [NUM_RES],
    })
  elif 'allatom' in name or 'all_atom' in name:
    cfg = copy.deepcopy(CONFIG_ALLATOM)
  else:
    cfg = copy.deepcopy(CONFIG)

  if name in CONFIG_DIFFS:
    cfg.update_from_flattened_dict(CONFIG_DIFFS[name])

  return cfg


MODEL_PRESETS = {
    'monomer': (
        'model_1',
        'model_2',
        'model_3',
        'model_4',
        'model_5',
    ),
    'monomer_ptm': (
        'model_1_ptm',
        'model_2_ptm',
        'model_3_ptm',
        'model_4_ptm',
        'model_5_ptm',
    ),
    'multimer': (
        'model_1_multimer_v2',
        'model_2_multimer_v2',
        'model_3_multimer_v2',
        'model_4_multimer_v2',
        'model_5_multimer_v2',
    ),
}
MODEL_PRESETS['monomer_casp14'] = MODEL_PRESETS['monomer']

CONFIG_TINY = {
  'max_msa_clusters': 64,
  'max_extra_msa': 128,
  'structure_module.num_channel': 128,
  'structure_module.num_layer': 6,
  'num_recycle':0,
  'evoformer_num_block':10,
  'msa_channel': 128
}

CONFIG_DIFFUSION = {
    'max_diffuse_step': 1000,
    'beta_1': 0.0001,
    'beta_T': 0.02,
    'ddim_sample_step': 20,
    'ddim_eta': 0.0,
}

CONFIG_DIFFS = {
    'helixfold_3': {
    },
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
    'multimer_seq512_pair64_l8_vio0': {
        'data.common.max_extra_msa': 1,
        'data.eval.max_msa_clusters': 1,
        'model.embeddings_and_evoformer.evoformer_num_block': 8,
        'model.embeddings_and_evoformer.evoformer_recompute_start_block_index': 0,
        'model.embeddings_and_evoformer.max_relative_feature': 32,
        'model.embeddings_and_evoformer.r_max': 32,
        'model.embeddings_and_evoformer.s_max': 2,
        'model.embeddings_and_evoformer.msa_channel': 512,
        'model.embeddings_and_evoformer.pair_channel': 64,
        'model.embeddings_and_evoformer.mute_extra_msa': True,
        'model.embeddings_and_evoformer.template.enabled': False,
        'model.embeddings_and_evoformer.evoformer.mute_msa_column': True,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing.num_intermediate_channel': 64,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming.num_intermediate_channel': 64,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.compute_in_graph_metrics': True,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.global_config.use_dropout_nd': True,
        'model.global_config.fuse_attention': True,
        'model.global_config.origin_evoformer_structure': False,
        'model.heads.masked_msa.num_output': 23,
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
        'model.global_config.subbatch_size': 12,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
    },
    'model_1_multimer_v3-chainaffine_v2-pae0': {
        # af2multimer v3
        # 'data.eval.max_msa_clusters': 508,
        # 'data.common.max_extra_msa': 2048,
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 256,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.template.template_pair_stack.'
            + 'triangle_multiplication_incoming.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.template.template_pair_stack.'
            + 'triangle_multiplication_outgoing.fuse_projection_weights': True,
        # chainaffine v2
        'model.heads.structure_module.use_chain_affine_v2': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        # adapted from demo3
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        # 'model.heads.predicted_aligned_error.weight': 0.0
        # global config modified
        'model.global_config.fuse_linear': False,
        'model.global_config.use_flash_attn': False,
        # 'model.heads.structure_module.center_of_mass_weight': 1.0,
        # 'model.heads.structure_module.center_of_mass_loss_type': 'origin'
    },
    'model_1_multimer_v3-chainaffine_v2-vio0.1-pae0.1': {
        # af2multimer v3
        # 'data.eval.max_msa_clusters': 508,
        # 'data.common.max_extra_msa': 2048,
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 256,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.template.template_pair_stack.'
                + 'triangle_multiplication_incoming.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.template.template_pair_stack.'
                + 'triangle_multiplication_outgoing.fuse_projection_weights': True,
        # chainaffine v2
        'model.heads.structure_module.use_chain_affine_v2': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        # adapted from demo3
        'model.heads.structure_module.structural_violation_loss_weight': 0.1,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.1,
        # global config modified
        'model.global_config.fuse_linear': False,
        'model.global_config.use_flash_attn': False,
        # 'model.heads.structure_module.center_of_mass_weight': 1.0,
        # 'model.heads.structure_module.center_of_mass_loss_type': 'origin'
    },
    'model_1_multimer_v3-chainaffine_v2-pae0-pair_pocket': {
        'model.global_config.use_pair_pocket_mask': 0.5,
        'model.global_config.infer_pocket_mask_size': 20,
        # af2multimer v3
        # 'data.eval.max_msa_clusters': 508,
        # 'data.common.max_extra_msa': 2048,
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 256,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.template.template_pair_stack.triangle_multiplication_incoming.' \
            + 'fuse_projection_weights': True,
        'model.embeddings_and_evoformer.template.template_pair_stack.triangle_multiplication_outgoing.' \
            + 'fuse_projection_weights': True,
        # chainaffine v2
        'model.heads.structure_module.use_chain_affine_v2': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        # adapted from demo3
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.0,
        # global config modified
        'model.global_config.fuse_linear': False,
        'model.global_config.use_flash_attn': False,
        # 'model.heads.structure_module.center_of_mass_weight': 1.0,
        # 'model.heads.structure_module.center_of_mass_loss_type': 'origin'
    },
    'model_1_multimer_v3-chainaffine_v2-pae0.01-vio0.01-pair_pocket': {
        'model.global_config.use_pair_pocket_mask': 0.5,
        'model.global_config.infer_pocket_mask_size': 20,
        # af2multimer v3
        # 'data.eval.max_msa_clusters': 508,
        # 'data.common.max_extra_msa': 2048,
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 256,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.template.template_pair_stack.triangle_multiplication_incoming.' \
            + 'fuse_projection_weights': True,
        'model.embeddings_and_evoformer.template.template_pair_stack.triangle_multiplication_outgoing.' \
            + 'fuse_projection_weights': True,
        # chainaffine v2
        'model.heads.structure_module.use_chain_affine_v2': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        # adapted from demo3
        'model.heads.structure_module.structural_violation_loss_weight':  0.05,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.01,
        # global config modified
        'model.global_config.fuse_linear': False,
        'model.global_config.use_flash_attn': False,
        # 'model.heads.structure_module.center_of_mass_weight': 0.01,
        # 'model.heads.structure_module.center_of_mass_loss_type': 'origin'
    },
    'model_1_multimer_v3-chainaffine_v2-pae0.05-vio0.05-pair_pocket12-v2_full_msa': {
        'model.global_config.use_pair_pocket_mask': 0.2,
        'model.global_config.infer_pocket_mask_size': 12,
        # af2multimer v3
        'data.eval.max_msa_clusters': 512,
        'data.common.max_extra_msa': 1024,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.template.template_pair_stack.triangle_multiplication_incoming.' \
            + 'fuse_projection_weights': True,
        'model.embeddings_and_evoformer.template.template_pair_stack.triangle_multiplication_outgoing.' \
            + 'fuse_projection_weights': True,
        # chainaffine v2
        'model.heads.structure_module.use_chain_affine_v2': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        # adapted from demo3
        'model.heads.structure_module.structural_violation_loss_weight':  0.05,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.05,
        # global config modified
        'model.global_config.fuse_linear': False,
        'model.global_config.use_flash_attn': False,
        # 'model.heads.structure_module.center_of_mass_weight': 0.01,
        # 'model.heads.structure_module.center_of_mass_loss_type': 'origin'
    },
    'model_1_multimer_v3-chainaffine_v2-pae0-pair_pocket12': {
        'model.global_config.use_pair_pocket_mask': 0.5,
        'model.global_config.infer_pocket_mask_size': 12,
        # af2multimer v3
        # 'data.eval.max_msa_clusters': 508,
        # 'data.common.max_extra_msa': 2048,
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 256,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.template.template_pair_stack.triangle_multiplication_incoming.' \
            + 'fuse_projection_weights': True,
        'model.embeddings_and_evoformer.template.template_pair_stack.triangle_multiplication_outgoing.' \
            + 'fuse_projection_weights': True,
        # chainaffine v2
        'model.heads.structure_module.use_chain_affine_v2': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        # adapted from demo3
        'model.heads.structure_module.structural_violation_loss_weight':  0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.0,
        # global config modified
        'model.global_config.fuse_linear': False,
        'model.global_config.use_flash_attn': False,
        # 'model.heads.structure_module.center_of_mass_weight': 0.01,
        # 'model.heads.structure_module.center_of_mass_loss_type': 'origin'
    },
    'model_1_multimer_v3-chainaffine_v2-pae0-ipa_pair_pocket12': {
        'model.global_config.use_pair_pocket_mask': 0.5,
        'model.global_config.infer_pocket_mask_size': 12,
        'model.global_config.ipa_pair_pocket_mask': True,
        # af2multimer v3
        # 'data.eval.max_msa_clusters': 508,
        # 'data.common.max_extra_msa': 2048,
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 256,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.template.template_pair_stack.triangle_multiplication_incoming.' \
            + 'fuse_projection_weights': True,
        'model.embeddings_and_evoformer.template.template_pair_stack.triangle_multiplication_outgoing.' \
            + 'fuse_projection_weights': True,
        # chainaffine v2
        'model.heads.structure_module.use_chain_affine_v2': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        # adapted from demo3
        'model.heads.structure_module.structural_violation_loss_weight':  0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.0,
        # global config modified
        'model.global_config.fuse_linear': False,
        'model.global_config.use_flash_attn': False,
        # 'model.heads.structure_module.center_of_mass_weight': 0.01,
        # 'model.heads.structure_module.center_of_mass_loss_type': 'origin'
    },
    'multimer_demo': {
        'model.global_config.subbatch_size': 32,
    },
    'multimer_peptide_ranking': {
        'model.heads.ranking.num_bins': 1,
        'model.heads.ranking.num_channels': 1,
        'model.heads.ranking.weight': 10.0,
    },
    ## lihang
    'multimer_demo2': {
        'data.eval.max_msa_clusters': 32,
        'data.common.max_extra_msa': 64,
        'model.embeddings_and_evoformer.msa_channel': 64,
        'model.embeddings_and_evoformer.pair_channel': 32,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing.num_intermediate_channel': 32,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming.num_intermediate_channel': 32,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
    },
    'multimer_demo3': {
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 256,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.0
    },
    'multimer_initial': {
        # 'data.eval.max_msa_clusters': 128,
        # 'data.common.max_extra_msa': 1152,
        'model.global_config.fuse_attention': True,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.0
    },
    'multimer_initial-msa128_extra1152': {
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 1152,
        'model.global_config.fuse_attention': True,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.0
    },
    'multimer_chainaffine_initial': {
        'model.heads.structure_module.use_chain_affine': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        # 'data.eval.max_msa_clusters': 128,
        # 'data.common.max_extra_msa': 1152,
         'model.global_config.fuse_attention': True,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.001 
    },
    'multimer_demo3_evo24': {
        'model.embeddings_and_evoformer.evoformer_num_block': 24,
        # adapted from demo3
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 256,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.0
    },
    'multimer_demo3_evo12': {
        'model.embeddings_and_evoformer.evoformer_num_block': 12,
        # adapted from demo3
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 256,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.0
    },
    'multimer_demo3_chainaffine': {
        'model.heads.structure_module.use_chain_affine': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        # adapted from demo3
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 256,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.0
    },
    'multimer_demo3_chainaffine_v2': {
        'model.heads.structure_module.use_chain_affine_v2': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        # adapted from demo3
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 256,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.0
    },
    'multimer_demo3_chainaffine-pae0.01': {
        'model.heads.structure_module.use_chain_affine': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        # adapted from demo3
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 256,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.001
    },
    'multimer_chainaffine-pae0.5': {
        'model.heads.structure_module.use_chain_affine': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        # adapted from demo3
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.5
    },
    'multimer_chainaffine-ablation': {
        # adapted from demo3
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        # 'model.heads.predicted_aligned_error.weight': 0.5
    },
    'multimer_chainaffine-use_pocket': {
        'model.heads.structure_module.use_chain_affine': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        # adapted from demo3
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        # 'model.heads.predicted_aligned_error.weight': 0.5, # use defaulted weight 0.1
        'model.embeddings_and_evoformer.use_pocket_info': True
    },
    'multimer_chainaffine-pae_vio_defaulted': {
        'model.heads.structure_module.use_chain_affine': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        # adapted from demo3
        # 'model.heads.structure_module.structural_violation_loss_weight': 0.0, # use defaulted weight 1.0
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        # 'model.heads.predicted_aligned_error.weight': 0.5, # use defaulted weight 0.1
    },
    'multimer_chainaffine-pae0.5-vio1.0': {
        'model.heads.structure_module.use_chain_affine': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        # adapted from demo3
        'model.heads.structure_module.structural_violation_loss_weight': 1.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.5
    },
    'multimer_demo3_chainaffine-pae0.5': {
        'model.heads.structure_module.use_chain_affine': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        # adapted from demo3
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 256,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.5
    },
    'multimer_demo3_chainaffine-pae0.5-train_subatch64': {
        'model.heads.structure_module.use_chain_affine': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        'model.global_config.train_subbatch_size': 64,
        # adapted from demo3
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 256,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.5    
    },
    'multimer_chainaffine-pae0.01': {
        'model.heads.structure_module.use_chain_affine': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        # adapted from demo3
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.001    
    },
    'multimer_chainaffine-pae0.01-train_subbatch64': {
        'model.heads.structure_module.use_chain_affine': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        'model.global_config.train_subbatch_size': 64,
        # adapted from demo3
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.001    
    },
    'multimer_demo3_evo24_chainaffine': {
        'model.heads.structure_module.use_chain_affine': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        'model.embeddings_and_evoformer.evoformer_num_block': 24,
        # adapted from demo3
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 256,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.0
    },
    'multimer_demo3_evo12_chainaffine': {
        'model.heads.structure_module.use_chain_affine': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        'model.embeddings_and_evoformer.evoformer_num_block': 12,
        # adapted from demo3
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 256,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.0
    },
    'multimer_chainaffine': {
            'model.heads.structure_module.use_chain_affine': True,
            'model.heads.structure_module.chain_affine': {"num_channel": 128},  
    },
    "multimer_chainaffine-predic_align_error_0-violation_loss_weight0-atom_clamp_distance30-sidechain_length_scale10": {
        'model.heads.structure_module.use_chain_affine': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        # adapted from demo3
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.0    
    },
    "multimer_chainaffine-violation_loss_weight0-atom_clamp_distance30-sidechain_length_scale10": {
        'model.heads.structure_module.use_chain_affine': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        # adapted from demo3
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
    },
    # Diffusion
    # x0
    'multimer_evo12_recy0-x0_diffusion': {
        'model.heads.structure_module.use_diffusion': True,
        # ref https://github.com/RosettaCommons/RFdiffusion/blob/main/config/inference/base.yaml
        # ref http://gitlab.baidu.com/liulihang/pytorch-ddpm/tree/master
        'model.heads.structure_module.diffusion_weight': 1,
        'model.heads.structure_module.diffusion_loss': "x0",
        'model.heads.structure_module.use_timeemb': True,
        # multimer size config
        # evo12 recycle0
        'model.embeddings_and_evoformer.evoformer_num_block': 12,
        'model.num_recycle': 0,
        # 'model.embeddings_and_evoformer.pair_channel': 64,
        # adapted from demo3
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 256,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.0
    },
    'multimer_demo3_chainaffine-x0_diffusion': {
        'model.heads.structure_module.use_diffusion': True,
        # ref https://github.com/RosettaCommons/RFdiffusion/blob/main/config/inference/base.yaml
        # ref http://gitlab.baidu.com/liulihang/pytorch-ddpm/tree/master
        'model.heads.structure_module.diffusion_weight': 1,
        'model.heads.structure_module.diffusion_loss': "x0",
        'model.heads.structure_module.use_timeemb': True,
        # multimer_chainaffine-pae0.5
        'model.heads.structure_module.use_chain_affine': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        # recycle0
        'model.num_recycle': 0,
        'data.common.num_recycle': 0,
        # adapted from demo3
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 256,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.0,
        # global config modified
        'model.global_config.fuse_linear': False,
        'model.global_config.use_flash_attn': False,
    },
    'model_1_multimer_v3-chainaffine_v2-pae0-x0_diffusion': {
        'model.heads.structure_module.use_diffusion': True,
        # ref https://github.com/RosettaCommons/RFdiffusion/blob/main/config/inference/base.yaml
        # ref http://gitlab.baidu.com/liulihang/pytorch-ddpm/tree/master
        'model.heads.structure_module.diffusion_weight': 1,
        'model.heads.structure_module.diffusion_loss': "x0",
        'model.heads.structure_module.use_timeemb': True,
        # recycle0
        'model.num_recycle': 0,
        'data.common.num_recycle': 0,
        # model_1_multimer_v3-chainaffine_v2-pae0
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 256,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.template.template_pair_stack.'
            + 'triangle_multiplication_incoming.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.template.template_pair_stack.'
            + 'triangle_multiplication_outgoing.fuse_projection_weights': True,
        # chainaffine v2
        'model.heads.structure_module.use_chain_affine_v2': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        # adapted from demo3
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.0,
        # global config modified
        'model.global_config.fuse_linear': False,
        'model.global_config.use_flash_attn': False,
    },
    'model_1_multimer_v3-chainaffine_v2-pae0-x0_diffusion-diff_pos_as_templ_prev': {
        'model.heads.structure_module.use_diffusion': True,
        # ref https://github.com/RosettaCommons/RFdiffusion/blob/main/config/inference/base.yaml
        # ref http://gitlab.baidu.com/liulihang/pytorch-ddpm/tree/master
        'model.use_diff_pos_as_templ': True,
        'model.use_diff_pos_as_prev': True,
        'model.heads.structure_module.diffusion_weight': 1,
        'model.heads.structure_module.diffusion_loss': "x0",
        'model.heads.structure_module.use_timeemb': True,
        # recycle0
        'model.num_recycle': 0,
        'data.common.num_recycle': 0,
        # model_1_multimer_v3-chainaffine_v2-pae0
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 256,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.template.template_pair_stack.'
            + 'triangle_multiplication_incoming.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.template.template_pair_stack.'
            + 'triangle_multiplication_outgoing.fuse_projection_weights': True,
        # chainaffine v2
        'model.heads.structure_module.use_chain_affine_v2': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        # adapted from demo3
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.0,
        # global config modified
        'model.global_config.fuse_linear': False,
        'model.global_config.use_flash_attn': False,
    },
    'model_1_multimer_v3-chainaffine_v2-pae0-x0_diffusion-struct_transformer': {
        'model.heads.structure_module.use_diffusion': True,
        # ref https://github.com/RosettaCommons/RFdiffusion/blob/main/config/inference/base.yaml
        # ref http://gitlab.baidu.com/liulihang/pytorch-ddpm/tree/master
        'model.heads.structure_module.diffusion_weight': 1,
        'model.heads.structure_module.diffusion_loss': "x0",
        'model.heads.structure_module.use_timeemb': True,
        # extra transformer encoders in structure module
        'model.heads.structure_module.num_transformers': 8,
        # recycle0
        'model.num_recycle': 0,
        'data.common.num_recycle': 0,
        # model_1_multimer_v3-chainaffine_v2-pae0
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 256,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.template.template_pair_stack.'
            + 'triangle_multiplication_incoming.fuse_projection_weights': True,
        'model.embeddings_and_evoformer.template.template_pair_stack.'
            + 'triangle_multiplication_outgoing.fuse_projection_weights': True,
        # chainaffine v2
        'model.heads.structure_module.use_chain_affine_v2': True,
        'model.heads.structure_module.chain_affine': {"num_channel": 128},
        # adapted from demo3
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.0,
        # global config modified
        'model.global_config.fuse_linear': False,
        'model.global_config.use_flash_attn': False,
    },
    # score
    'multimer_evo12_recy0-rot_score_diffusion': {
        'model.heads.structure_module.use_diffusion': True,
        # ref https://github.com/RosettaCommons/RFdiffusion/blob/main/config/inference/base.yaml
        # ref http://gitlab.baidu.com/liulihang/pytorch-ddpm/tree/master
        'model.heads.structure_module.diffusion_weight': 1,
        'model.heads.structure_module.diffusion_loss': "rot_score",
        'model.heads.structure_module.use_timeemb': True,
        # multimer size config
        # evo12 recycle0
        'model.embeddings_and_evoformer.evoformer_num_block': 12,
        'model.num_recycle': 0,
        # 'model.embeddings_and_evoformer.pair_channel': 64,
        # adapted from demo3
        'data.eval.max_msa_clusters': 128,
        'data.common.max_extra_msa': 256,
        'model.heads.structure_module.structural_violation_loss_weight': 0.0,
        'model.heads.structure_module.interface_fape.atom_clamp_distance': 30.0,
        'model.heads.structure_module.sidechain.length_scale': 10.0,
        'model.heads.predicted_aligned_error.weight': 0.0
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
    },
    'allatom_demo': {
        # 'model.embeddings_and_pairformer.pairformer.num_block': 4,
    },
    'allatom_distow03_msa4096_forcen_test_n200_08': {
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.test_diff_batch_size': 1,
        'model.heads.diffusion_module.step_num': 200,
        'model.heads.diffusion_module.gamma0': 0.8,
        'model.heads.diffusion_module.force_centering': True,
        'model.heads.confidence_head.weight': 0.0,
    },
    'allatom_distow03_msa4096_forcen_conf001': {
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': True,
        'model.heads.confidence_head.weight': 0.01,
    },
    'allatom_distow03_msa4096_forcen_conf001_smlddt0': {
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': True,
        'model.heads.diffusion_module.loss_smooth_lddt_weight': 0.0,
        'model.heads.confidence_head.weight': 0.01,
    },
    'allatom_distow03_msa4096_forcen_conf001_smlddt0_delTemp': {
        'model.embeddings_and_pairformer.template_module.zero_prob': 0.3,
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': True,
        'model.heads.diffusion_module.loss_smooth_lddt_weight': 0.0,
        'model.heads.confidence_head.weight': 0.01,
    },
    'allatom_distow03_msa4096_forcen_conf001_smlddt0_delTemp_randSampMSA': {
        'data.msa_rand_prob': 0.5,
        'model.embeddings_and_pairformer.template_module.zero_prob': 0.3,
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': True,
        'model.heads.diffusion_module.loss_smooth_lddt_weight': 0.0,
        'model.heads.confidence_head.weight': 0.01,
    },
    'allatom_distow03_msa4096_forcen_conf001_smlddt0_delTemp_recycle10': {
        'model.num_recycle': 9,
        'model.embeddings_and_pairformer.template_module.zero_prob': 0.3,
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': True,
        'model.heads.diffusion_module.loss_smooth_lddt_weight': 0.0,
        'model.heads.confidence_head.weight': 0.01,
    },
    'allatom_distow03_msa4096_forcen_conf001_pocket05_randTemp_randMSA': {
        'model.global_config.h100': True,
        'model.embeddings_and_pairformer.template_module.zero_prob': 0.3,
        'model.input_embedder.add_pocket': True,
        'model.input_embedder.pocket_train_prob': 0.5,
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': True,
        'model.heads.confidence_head.weight': 0.01,
    },
    'allatom_distow03_msa4096_forcen_conf001_pocket05_randTempMSA_bond': {
        'model.global_config.h100': True,
        # loss bond
        'model.heads.diffusion_module.loss_bond_weight': 1.0,
        # random sample MSA
        'data.msa_rand_prob': 0.5,
        # random sample template
        'model.embeddings_and_pairformer.template_module.zero_prob': 0.3,
        # pocket
        'model.input_embedder.add_pocket': True,
        'model.input_embedder.pocket_train_prob': 0.5,
        # other
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': True,
        'model.heads.confidence_head.weight': 0.01,
    },
    'allatom_distow03_msa4096_forcen_conf001_pocket05_randTempMSA_bond01': {
        'model.global_config.h100': True,
        # loss bond
        'model.heads.diffusion_module.loss_bond_weight': 0.1,
        # random sample MSA
        'data.msa_rand_prob': 0.5,
        # random sample template
        'model.embeddings_and_pairformer.template_module.zero_prob': 0.3,
        # pocket
        'model.input_embedder.add_pocket': True,
        'model.input_embedder.pocket_train_prob': 0.5,
        # other
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': True,
        'model.heads.confidence_head.weight': 0.01,
    },
    'allatom_distow03_msa4096_forcen_conf001_pocket05_randTemp06MSA_bond01_fape05': {
        'model.global_config.h100': True,
        # loss bond
        'model.heads.diffusion_module.loss_bond_weight': 0.1,
        # loss fape
        'model.heads.diffusion_module.loss_fape_weight': 0.5,
        'model.heads.diffusion_module.fape_z': 15.0,
        # random sample MSA
        'data.msa_rand_prob': 0.5,
        # random sample template
        'model.embeddings_and_pairformer.template_module.zero_prob': 0.6,
        # pocket
        'model.input_embedder.add_pocket': True,
        'model.input_embedder.pocket_train_prob': 0.5,
        # other
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': True,
        'model.heads.confidence_head.weight': 0.01,
    },
    'allatom_distow03_msa4096_conf001_pocket05_randTemp06MSA_bond01_fape05': {
        'model.global_config.h100': True,
        # loss bond
        'model.heads.diffusion_module.loss_bond_weight': 0.1,
        # loss fape
        'model.heads.diffusion_module.loss_fape_weight': 0.5,
        'model.heads.diffusion_module.fape_z': 15.0,
        # random sample MSA
        'data.msa_rand_prob': 0.5,
        # random sample template
        'model.embeddings_and_pairformer.template_module.zero_prob': 0.6,
        # pocket
        'model.input_embedder.add_pocket': True,
        'model.input_embedder.pocket_train_prob': 0.5,
        # other
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': False,
        'model.heads.confidence_head.weight': 0.01,
    },
    'allatom_distow03_msa4096_conf001_pocket05_randTemp06MSA_bond01_fape05_strans50': {
        'model.global_config.h100': True,
        # loss bond
        'model.heads.diffusion_module.loss_bond_weight': 0.1,
        # loss fape
        'model.heads.diffusion_module.loss_fape_weight': 0.5,
        'model.heads.diffusion_module.fape_z': 15.0,
        # random sample MSA
        'data.msa_rand_prob': 0.5,
        # random sample template
        'model.embeddings_and_pairformer.template_module.zero_prob': 0.6,
        # pocket
        'model.input_embedder.add_pocket': True,
        'model.input_embedder.pocket_train_prob': 0.5,
        # other
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': False,
        'model.heads.confidence_head.weight': 0.01,
        'model.heads.diffusion_module.s_trans': 50.0,
    },
    'allatom_distow03_msa4096_conf001_pocket05_randTemp06MSA_bond01_fape05_st50_uniform': {
        'model.global_config.h100': True,
        # loss bond
        'model.heads.diffusion_module.loss_bond_weight': 0.1,
        # loss fape
        'model.heads.diffusion_module.loss_fape_weight': 0.5,
        'model.heads.diffusion_module.fape_z': 15.0,
        # random sample MSA
        'data.msa_rand_prob': 0.5,
        # random sample template
        'model.embeddings_and_pairformer.template_module.zero_prob': 0.6,
        # pocket
        'model.input_embedder.add_pocket': True,
        'model.input_embedder.pocket_train_prob': 0.5,
        # central random augmentation
        'model.heads.diffusion_module.s_trans': 50.0,
        'model.heads.diffusion_module.uniform_aug': True,
        # other
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': False,
        'model.heads.confidence_head.weight': 0.01,
    },
    'allatom_distow03_msa4096_conf001_pocket05_randTemp06MSA_bond01_fape05_eta2': {
        'model.global_config.h100': True,
        # loss bond
        'model.heads.diffusion_module.loss_bond_weight': 0.1,
        # loss fape
        'model.heads.diffusion_module.loss_fape_weight': 0.5,
        'model.heads.diffusion_module.fape_z': 15.0,
        # random sample MSA
        'data.msa_rand_prob': 0.5,
        # random sample template
        'model.embeddings_and_pairformer.template_module.zero_prob': 0.6,
        # pocket
        'model.input_embedder.add_pocket': True,
        'model.input_embedder.pocket_train_prob': 0.5,
        # other
        'model.heads.diffusion_module.eta': 2.0,
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': False,
        'model.heads.confidence_head.weight': 0.01,
    },
    'allatom_distow03_msa4096_forcen_conf001_pocket05_randTempMSA_fape_bond': {
        'model.global_config.h100': True,
        # loss bond
        'model.heads.diffusion_module.loss_bond_weight': 1.0,
        # loss fape
        'model.heads.diffusion_module.loss_fape_weight': 1.0,
        'model.heads.diffusion_module.fape_z': 15.0,
        # random sample MSA
        'data.msa_rand_prob': 0.5,
        # random sample template
        'model.embeddings_and_pairformer.template_module.zero_prob': 0.3,
        # pocket
        'model.input_embedder.add_pocket': True,
        'model.input_embedder.pocket_train_prob': 0.5,
        # other
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': True,
        'model.heads.confidence_head.weight': 0.01,
    },
    'allatom_distow03_msa4096_forcen_conf001_test_n200': {
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.test_diff_batch_size': 1,
        'model.heads.diffusion_module.step_num': 200,
        'model.heads.diffusion_module.force_centering': True,
        'model.heads.confidence_head.weight': 0.01,
    },
    'allatom_distow03_msa4096_forcen_conf001_test_n200_08': {
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.test_diff_batch_size': 1,
        'model.heads.diffusion_module.step_num': 200,
        'model.heads.diffusion_module.gamma0': 0.8,
        'model.heads.diffusion_module.force_centering': True,
        'model.heads.confidence_head.weight': 0.01,
    },
    'allatom_distow03_msa4096_forcen_conf001_test_db5_n200_08_15': {
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.test_diff_batch_size': 5,
        'model.heads.diffusion_module.step_num': 200,
        'model.heads.diffusion_module.gamma0': 0.8,
        'model.heads.diffusion_module.eta': 1.5,
        'model.heads.diffusion_module.force_centering': True,
        'model.heads.confidence_head.weight': 0.01,
        'model.embeddings_and_pairformer.msa_module.use_msa_dynamic_subbatch': True,
        'model.heads.diffusion_module.atom_encoder.atom_transformer.diffusion_transformer.use_rotary': True,
        'model.heads.diffusion_module.diffusion_transformer.use_rotary': True,
        'model.heads.diffusion_module.atom_decoder.atom_transformer.diffusion_transformer.use_rotary': True,
    },
    'allatom_distow03_msa4096_conf001_randTemp06MSA_bond01_fape03_st50Uni_rotary-constrains': {
        'model.global_config.h100': True,
        'model.input_embedder.add_pocket': True, # pocket
        'model.input_embedder.add_constrains': True, # add_constrains
        # loss
        'model.heads.diffusion_module.loss_bond_weight': 0.1,
        'model.heads.diffusion_module.loss_fape_weight': 0.3,
        'model.heads.diffusion_module.fape_z': 15.0,
        # random sample MSA and template
        'data.msa_rand_prob': 0.5,
        'model.embeddings_and_pairformer.template_module.zero_prob': 0.6,
        # central random augmentation
        'model.heads.diffusion_module.s_trans': 50.0,
        'model.heads.diffusion_module.uniform_aug': True,
        # rotary
        'model.heads.diffusion_module.atom_encoder.atom_transformer.diffusion_transformer.use_rotary': True,
        'model.heads.diffusion_module.diffusion_transformer.use_rotary': True,
        'model.heads.diffusion_module.atom_decoder.atom_transformer.diffusion_transformer.use_rotary': True,
        # other
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': False,
        'model.heads.confidence_head.weight': 0.01,

        'model.heads.diffusion_module.step_num': 200,
        'model.heads.diffusion_module.gamma0': 0.8,
        'model.heads.diffusion_module.eta': 1.5,
        'model.heads.diffusion_module.test_diff_batch_size': 5,
    },
    'allatom_distow03_msa4096_forcen_conf001_rec9': {
        'model.num_recycle': 9,
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': True,
        'model.heads.confidence_head.weight': 0.01,
    },
    'allatom_distow03_msa4096_forcen_conf001_smlddt0_pocket': {
        'model.input_embedder.add_pocket': True,
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': True,
        'model.heads.diffusion_module.loss_smooth_lddt_weight': 0.0,
        'model.heads.confidence_head.weight': 0.01,
    },
    'allatom_distow03_msa4096_forcen_conf001_rec9': {
        'model.num_recycle': 9,
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': True,
        'model.heads.confidence_head.weight': 0.01,
    },
    'allatom_distow03_msa4096_forcen_conf001_smlddt0_pocket': {
        'model.input_embedder.add_pocket': True,
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': True,
        'model.heads.diffusion_module.loss_smooth_lddt_weight': 0.0,
        'model.heads.confidence_head.weight': 0.01,
    },
    'allatom_distow03_msa4096_forcen_conf001_interface': {
        'model.input_embedder.add_interface': True,
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': True,
        'model.heads.confidence_head.weight': 0.01,
    },
    'allatom_interface_gen': {
        'model.global_config.h100': False,
        # loss bond
        'model.heads.diffusion_module.loss_bond_weight': 0.1,
        # loss fape
        'model.heads.diffusion_module.loss_fape_weight': 0.5,
        'model.heads.diffusion_module.fape_z': 15.0,
        # random sample MSA
        'data.msa_rand_prob': 0.5,
        # random sample template
        'model.embeddings_and_pairformer.template_module.zero_prob': 0.6,
        # interface gen
        'model.add_interface': True,
        'model.gen_interface': True,
        'model.heads.interface_head.weight': 1.0,
        # other
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': False,
        'model.heads.confidence_head.weight': 0.01,
    },
    'allatom_interface_gen_homomix': {
        'model.global_config.h100': False,
        # loss bond
        'model.heads.diffusion_module.loss_bond_weight': 0.1,
        # loss fape
        'model.heads.diffusion_module.loss_fape_weight': 0.5,
        'model.heads.diffusion_module.fape_z': 15.0,
        # random sample MSA
        'data.msa_rand_prob': 0.5,
        # random sample template
        'model.embeddings_and_pairformer.template_module.zero_prob': 0.6,
        # interface gen
        'model.add_interface': True,
        'model.gen_interface': True,
        'model.train_with_mix_interface': True,
        'model.heads.interface_head.weight': 1.0,
        # other
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': False,
        'model.heads.confidence_head.weight': 0.01,
    },
    'allatom_interface_gen_homomix_fixrepr': {
        'model.global_config.h100': False,
        # loss bond
        'model.heads.diffusion_module.loss_bond_weight': 0.1,
        # loss fape
        'model.heads.diffusion_module.loss_fape_weight': 0.5,
        'model.heads.diffusion_module.fape_z': 15.0,
        # random sample MSA
        'data.msa_rand_prob': 0.5,
        # random sample template
        'model.embeddings_and_pairformer.template_module.zero_prob': 0.6,
        # interface gen
        'model.add_interface': True,
        'model.gen_interface': True,
        'model.heads.interface_head.train_with_mix_interface': True,
        'model.heads.interface_head.update_repr_during_interface_gen': False,
        'model.heads.interface_head.weight': 1.0,
        # other
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': False,
        'model.heads.confidence_head.weight': 0.01,
    },
    'allatom_interface_gen_homomix_fixrepr_save_intermediate_interface': {
        'model.global_config.h100': False,
        # loss bond
        'model.heads.diffusion_module.loss_bond_weight': 0.1,
        # loss fape
        'model.heads.diffusion_module.loss_fape_weight': 0.5,
        'model.heads.diffusion_module.fape_z': 15.0,
        # random sample MSA
        'data.msa_rand_prob': 0.5,
        # random sample template
        'model.embeddings_and_pairformer.template_module.zero_prob': 0.6,
        # interface gen
        'model.add_interface': True,
        'model.gen_interface': True,
        'model.heads.interface_head.train_with_mix_interface': True,
        'model.heads.interface_head.update_repr_during_interface_gen': False,
        'model.heads.interface_head.save_intermediate_interface_infos': True,
        'model.heads.interface_head.weight': 1.0,
        # other
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': False,
        'model.heads.confidence_head.weight': 0.01,
    },
    'allatom_interface_gen_fixrepr_save_intermediate_interface': {
        'model.global_config.h100': False,
        # loss bond
        'model.heads.diffusion_module.loss_bond_weight': 0.1,
        # loss fape
        'model.heads.diffusion_module.loss_fape_weight': 0.5,
        'model.heads.diffusion_module.fape_z': 15.0,
        # random sample MSA
        'data.msa_rand_prob': 0.5,
        # random sample template
        'model.embeddings_and_pairformer.template_module.zero_prob': 0.6,
        # interface gen
        'model.add_interface': True,
        'model.gen_interface': True,
        'model.heads.interface_head.train_with_mix_interface': False,
        'model.heads.interface_head.update_repr_during_interface_gen': False,
        'model.heads.interface_head.save_intermediate_interface_infos': True,
        'model.heads.interface_head.weight': 1.0,
        # other
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': False,
        'model.heads.confidence_head.weight': 0.01,

        # diffusion
        'model.heads.diffusion_module.test_diff_batch_size': 1,
        'model.heads.diffusion_module.step_num': 10,
        'model.heads.diffusion_module.gamma0': 0,
    },
    'allatom_interface': {
        'model.global_config.h100': False,
        # loss bond
        'model.heads.diffusion_module.loss_bond_weight': 0.1,
        # loss fape
        'model.heads.diffusion_module.loss_fape_weight': 0.5,
        'model.heads.diffusion_module.fape_z': 15.0,
        # random sample MSA
        'data.msa_rand_prob': 0.5,
        # random sample template
        'model.embeddings_and_pairformer.template_module.zero_prob': 0.6,
        # interface, NOTE: @liuyang, need to check if this is correct, model.add_interface is True ??
        'model.add_interface': True,
        'model.input_embedder.add_interface': True,
        # other
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': False,
        'model.heads.confidence_head.weight': 0.01,

        # diffusion
        'model.heads.diffusion_module.test_diff_batch_size': 1,
        'model.heads.diffusion_module.step_num': 200, ## NOTE: 200 steps for interface module2 inference.
        'model.heads.diffusion_module.gamma0': 0,
    },
    'allatom_interface_gen_homomix_fixrepr_save_intermediate_interface_diffstep10': {
        'model.global_config.h100': False,
        # loss bond
        'model.heads.diffusion_module.loss_bond_weight': 0.1,
        # loss fape
        'model.heads.diffusion_module.loss_fape_weight': 0.5,
        'model.heads.diffusion_module.fape_z': 15.0,
        # random sample MSA
        'data.msa_rand_prob': 0.5,
        # random sample template
        'model.embeddings_and_pairformer.template_module.zero_prob': 0.6,
        # interface gen
        'model.add_interface': True,
        'model.gen_interface': True,
        'model.heads.interface_head.train_with_mix_interface': True,
        'model.heads.interface_head.update_repr_during_interface_gen': False,
        'model.heads.interface_head.save_intermediate_interface_infos': True,
        'model.heads.interface_head.weight': 1.0,
        # other
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': False,
        'model.heads.confidence_head.weight': 0.01,

        # diffusion
        'model.heads.diffusion_module.test_diff_batch_size': 1,
        'model.heads.diffusion_module.step_num': 10,
        'model.heads.diffusion_module.gamma0': 0,
    },
    'allatom_interface_gen_homomix_fixrepr_diffusionless': {
        'model.global_config.h100': False,
        # loss bond
        'model.heads.diffusion_module.loss_bond_weight': 0.1,
        # loss fape
        'model.heads.diffusion_module.loss_fape_weight': 0.5,
        'model.heads.diffusion_module.fape_z': 15.0,
        # random sample MSA
        'data.msa_rand_prob': 0.5,
        # random sample template
        'model.embeddings_and_pairformer.template_module.zero_prob': 0.6,
        # interface gen
        'model.add_interface': True,
        'model.gen_interface': True,
        'model.heads.interface_head.train_with_mix_interface': True,
        'model.heads.interface_head.update_repr_during_interface_gen': False,
        'model.heads.interface_head.weight': 1.0,
        # other
        'model.embeddings_and_pairformer.msa_module.msa_depth': 4096,
        'model.heads.distogram.weight': 0.3,
        'model.heads.diffusion_module.force_centering': False,
        'model.heads.confidence_head.weight': 0.01,
        # diffusion
        'model.heads.diffusion_module.test_diff_batch_size': 1,
        'model.heads.diffusion_module.step_num': 1,
        'model.heads.diffusion_module.gamma0': 0,
    },
    ####################
    # For AllAtom Design
    'allatomdesign_c384_db8_dit8_forcen_test_db5_n200_08': {
        'model.global_config.design_mode': True,
        'model.num_recycle': 0,
        'model.channel_num.diffusion_token_channel': 384,
        'model.heads.diffusion_module.diff_batch_size': 8,
        'model.heads.diffusion_module.test_diff_batch_size': 5,
        'model.heads.diffusion_module.step_num': 200,
        'model.heads.diffusion_module.gamma0': 0.8,
        'model.heads.diffusion_module.force_centering': True,
        'model.heads.diffusion_module.diffusion_transformer.n_block': 8,
    },
    'allatomdesign_c384_db8_posemb_dit8_forcen_test_db5_n200_08': {
        'model.global_config.design_mode': True,
        'model.num_recycle': 0,
        'model.add_position_encoding': True,
        'model.channel_num.diffusion_token_channel': 384,
        'model.heads.diffusion_module.diff_batch_size': 8,
        'model.heads.diffusion_module.test_diff_batch_size': 5,
        'model.heads.diffusion_module.step_num': 200,
        'model.heads.diffusion_module.gamma0': 0.8,
        'model.heads.diffusion_module.force_centering': True,
        'model.heads.diffusion_module.diffusion_transformer.n_block': 8,
    },
    'allatomdesign_c384_db8_rotary_dit8_forcen_test_db5_n200_08': {
        'model.global_config.design_mode': True,
        'model.num_recycle': 0,
        'model.channel_num.diffusion_token_channel': 384,
        'model.heads.diffusion_module.diff_batch_size': 8,
        'model.heads.diffusion_module.test_diff_batch_size': 5,
        'model.heads.diffusion_module.step_num': 200,
        'model.heads.diffusion_module.gamma0': 0.8,
        'model.heads.diffusion_module.force_centering': True,
        'model.heads.diffusion_module.diffusion_transformer.n_block': 8,
        'model.heads.diffusion_module.atom_encoder.atom_transformer.diffusion_transformer.use_rotary': True,
        'model.heads.diffusion_module.diffusion_transformer.use_rotary': True,
        'model.heads.diffusion_module.atom_decoder.atom_transformer.diffusion_transformer.use_rotary': True,
    },
    'allatomdesign_pform8_c384_db8_dit8_forcen_test_db5_n200_08': {
        'model.global_config.design_mode': True,
        'model.num_recycle': 0,
        'model.use_pairformer': True,
        'model.input_embedder.atom_encoder.atom_transformer.diffusion_transformer.n_block': 1,
        'model.embeddings_and_pairformer.pairformer.num_block': 8,
        'model.channel_num.diffusion_token_channel': 384,
        'model.heads.diffusion_module.diff_batch_size': 8,
        'model.heads.diffusion_module.test_diff_batch_size': 5,
        'model.heads.diffusion_module.step_num': 200,
        'model.heads.diffusion_module.gamma0': 0.8,
        'model.heads.diffusion_module.force_centering': True,
        'model.heads.diffusion_module.diffusion_transformer.n_block': 8,
    },
    # For AllAtom Design END
    ####################
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
            'r_max': 32,
            's_max': 2,
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
            'subbatch_size': 384,
            'use_remat': False,
            'zero_init': True,
            'low_memory': False,
            'fuse_linear': True,
            'fuse_attention': True,
            'use_flash_attn': True,
            'use_dropout_nd': True,
            'origin_evoformer_structure': False,
            'outer_product_mean_position': 'origin',
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
            }
        },
        'num_recycle': 3,
        'resample_msa_in_recycling': True
    },
})


CONFIG_MULTIMER = ml_collections.ConfigDict({
    'model': {
        'embeddings_and_evoformer': {
            'evoformer_num_block': 48,
            'evoformer': {
                'msa_column_attention': {
                    'dropout_rate': 0.0,
                    'gating': True,
                    'num_head': 8,
                    'orientation': 'per_column',
                    'shared_dropout': True
                },
                'msa_row_attention_with_pair_bias': {
                    'dropout_rate': 0.15,
                    'gating': True,
                    'num_head': 8,
                    'orientation': 'per_row',
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
                    'first': True,
                    'num_outer_channel': 32,
                    'orientation': 'per_row',
                    'shared_dropout': True
                },
                'pair_transition': {
                    'dropout_rate': 0.0,
                    'num_intermediate_factor': 4,
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
                'triangle_attention_starting_node': {
                    'dropout_rate': 0.25,
                    'gating': True,
                    'num_head': 4,
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
                'triangle_multiplication_outgoing': {
                    'dropout_rate': 0.25,
                    'equation': 'ikc,jkc->ijc',
                    'num_intermediate_channel': 128,
                    'orientation': 'per_row',
                    'shared_dropout': True
                }
            },
            'extra_msa_channel': 64,
            'extra_msa_stack_num_block': 4,
            'num_msa': 252,
            'num_extra_msa': 1152,
            'masked_msa': {
                'profile_prob': 0.1,
                'replace_fraction': 0.15,
                'same_prob': 0.1,
                'uniform_prob': 0.1
            },
            'use_chain_relative': True,
            'max_relative_chain': 2,
            'max_relative_idx': 32,
            'seq_channel': 384,
            'msa_channel': 256,
            'pair_channel': 128,
            'prev_pos': {
                'max_bin': 20.75,
                'min_bin': 3.25,
                'num_bins': 15
            },
            'recycle_features': True,
            'recycle_pos': True,
            'template': {
                'attention': {
                    'gating': False,
                    'num_head': 4
                },
                'dgram_features': {
                    'max_bin': 50.75,
                    'min_bin': 3.25,
                    'num_bins': 39
                },
                'enabled': True,
                'max_templates': 4,
                'num_channels': 64,
                'subbatch_size': 128,
                'template_pair_stack': {
                    'num_block': 2,
                    'pair_transition': {
                        'dropout_rate': 0.0,
                        'num_intermediate_factor': 2,
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
                    'triangle_attention_starting_node': {
                        'dropout_rate': 0.25,
                        'gating': True,
                        'num_head': 4,
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
                    'triangle_multiplication_outgoing': {
                        'dropout_rate': 0.25,
                        'equation': 'ikc,jkc->ijc',
                        'num_intermediate_channel': 64,
                        'orientation': 'per_row',
                        'shared_dropout': True
                    }
                }
            },
        },
        'global_config': {
            'deterministic': False,
            'multimer_mode': True,
            'subbatch_size': 96,
            'use_remat': False,
            'zero_init': True,
            'low_memory': False,
            'fuse_linear': True,
            'fuse_attention': False,
            'use_flash_attn': True,
            'outer_product_mean_position': 'first',
        },
        'heads': {
            'distogram': {
                'first_break': 2.3125,
                'last_break': 21.6875,
                'num_bins': 64,
                'weight': 0.3
            },
            'experimentally_resolved': {
                'filter_by_resolution': True,
                'max_resolution': 3.0,
                'min_resolution': 0.1,
                'weight': 0.01
            },
            'masked_msa': {
                'weight': 2.0
            },
            'predicted_aligned_error': {
                'filter_by_resolution': True,
                'max_error_bin': 31.0,
                'max_resolution': 3.0,
                'min_resolution': 0.1,
                'num_bins': 64,
                'num_channels': 128,
                'weight': 0.1
            },
            'predicted_lddt': {
                'filter_by_resolution': True,
                'max_resolution': 3.0,
                'min_resolution': 0.1,
                'num_bins': 50,
                'num_channels': 128,
                'weight': 0.01
            },
            'structure_module': {
                'angle_norm_weight': 0.01,
                'chi_weight': 0.5,
                'clash_overlap_tolerance': 1.5,
                'dropout': 0.1,
                'interface_fape': {
                    'atom_clamp_distance': 1000.0,
                    'loss_unit_distance': 20.0
                },
                'intra_chain_fape': {
                    'atom_clamp_distance': 10.0,
                    'loss_unit_distance': 10.0
                },
                'num_channel': 384,
                'num_head': 12,
                'num_layer': 8,
                'num_layer_in_transition': 3,
                'num_point_qk': 4,
                'num_point_v': 8,
                'num_scalar_qk': 16,
                'num_scalar_v': 16,
                'position_scale': 20.0,
                'sidechain': {
                    'atom_clamp_distance': 10.0,
                    'loss_unit_distance': 10.0,
                    'num_channel': 128,
                    'num_residual_block': 2,
                    'weight_frac': 0.5
                },
                'structural_violation_loss_weight': 1.0,
                'violation_tolerance_factor': 12.0,
                'weight': 1.0
            },
        },
        'num_ensemble_eval': 1,
        'num_recycle': 3,
        'resample_msa_in_recycling': True
    }
})


CONFIG_ALLATOM = ml_collections.ConfigDict({
  'data': {
    # TODO: @yexianbin
    
    'num_blocks': 5,    # for msa block deletion
    'randomize_num_blocks': True,
    'msa_fraction_per_block': 0.3,
  },
  'model': {
    'channel_num': {
        'token_channel': 384,
        'token_pair_channel': 128,
        'atom_channel': 128,
        'atom_pair_channel': 16,
        'msa_channel': 64,
        'diffusion_token_channel': 768,
        'pair_channel': 128,    # for modules.OuterProductMean
    },
    'input_embedder': {
      'atom_encoder': {
        'in_token_channel_name': 'token_channel',
        'out_token_channel_name': 'token_channel',
        'use_dense_mode': True,
        'atom_transformer': {
          'diffusion_transformer': {
            'a_channel_name': 'atom_channel',
            's_channel_name': 'atom_channel',
            'z_channel_name': 'atom_pair_channel',
            'n_block': 3,
            'n_head': 4,
            # 'local': False,    # TODO: turning this on will be very slow?
          },
          'n_query': 32,
          'n_key': 128
        },
      },
      'relative_position_encoding': {
        'relative_token_max': 32,
        'relative_chain_max': 2,
      },
    },
    'embeddings_and_pairformer': {
      'template_module': {
        'num_channel': 64,
        'max_templates': 4,
        'subbatch_size': 128,
        'pairformer_stack': {
          'num_block': 2,
          # NOTE: copy from `embeddings_and_pairformer.pairformer`, but they may diff
          'triangle_multiplication_outgoing': {
            'equation': 'ikc,jkc->ijc',
            'num_intermediate_channel': 128,
            'orientation': 'per_row',
            'dropout_rate': 0.25,
            'shared_dropout': True,
          },
          'triangle_multiplication_incoming': {
            'equation': 'kjc,kic->ijc',
            'num_intermediate_channel': 128,
            'orientation': 'per_row',
            'dropout_rate': 0.25,
            'shared_dropout': True,
          },
          'triangle_attention_starting_node': {
            'gating': True,
            'num_head': 4,
            'num_intermediate_channel': 32,
            'orientation': 'per_row',
            'dropout_rate': 0.25,
            'shared_dropout': True,
          },
          'triangle_attention_ending_node': {
            'gating': True,
            'num_head': 4,
            'num_intermediate_channel': 32,
            'orientation': 'per_column',
            'dropout_rate': 0.25,
            'shared_dropout': True,
          },
          'pair_transition': {
            'num_intermediate_factor': 4,
            'orientation': 'per_row',
            'dropout_rate': 0.0,
            'shared_dropout': True
          },
          'single_attention_with_pair_bias': {
            # NOTE: same as row-wise attention used in AlphaFold 2,
            # but only applied to a single sequence, which corresponds to
            # the single represenation
            'num_head': 16,
            'gating_bias_beta': 0,  # beta_ij
          },
          'single_transition': {
            'num_intermediate_factor': 4,
            'orientation': 'per_row',
            'dropout_rate': 0.0,
            'shared_dropout': True
          },
        }
      },
      'msa_module': {
        'num_block': 4,
        'msa_depth': 4096,
        # 'max_msa_depth': 16384, # TODO: should be used for data_loader
        'msa_channel': 64,
        'outer_product_mean': {
          'chunk_size': 128,
          'num_outer_channel': 32,
          'orientation': 'per_row',
          'dropout_rate': 0.0,
          'shared_dropout': True
        },
        'msa_pair_weighted_averaging': {
          'num_head': 8,
          'num_channel': 32,
          'orientation': 'per_row',
          'dropout_rate': 0.15,
          'shared_dropout': True,
        },
        'msa_transition': {
          'num_intermediate_factor': 4,
          'orientation': 'per_row',
          'dropout_rate': 0.0,
          'shared_dropout': True
        },
        'triangle_multiplication_outgoing': {
          'equation': 'ikc,jkc->ijc',
          'num_intermediate_channel': 128,
          'orientation': 'per_row',
          'dropout_rate': 0.25,
          'shared_dropout': True,
        },
        'triangle_multiplication_incoming': {
          'equation': 'kjc,kic->ijc',
          'num_intermediate_channel': 128,
          'orientation': 'per_row',
          'dropout_rate': 0.25,
          'shared_dropout': True,
        },
        'triangle_attention_starting_node': {
          'gating': True,
          'num_head': 4,
          'num_intermediate_channel': 32,
          'orientation': 'per_row',
          'dropout_rate': 0.25,
          'shared_dropout': True,
        },
        'triangle_attention_ending_node': {
          'gating': True,
          'num_head': 4,
          'num_intermediate_channel': 32,
          'orientation': 'per_column',
          'dropout_rate': 0.25,
          'shared_dropout': True,
        },
        'pair_transition': {
          'num_intermediate_factor': 4,
          'orientation': 'per_row',
          'dropout_rate': 0.0,
          'shared_dropout': True
        },
      },
      'pairformer': {
        'num_block': 48,
        'triangle_multiplication_outgoing': {
          'equation': 'ikc,jkc->ijc',
          'num_intermediate_channel': 128,
          'orientation': 'per_row',
          'dropout_rate': 0.25,
          'shared_dropout': True,
        },
        'triangle_multiplication_incoming': {
          'equation': 'kjc,kic->ijc',
          'num_intermediate_channel': 128,
          'orientation': 'per_row',
          'dropout_rate': 0.25,
          'shared_dropout': True,
        },
        'triangle_attention_starting_node': {
          'gating': True,
          'num_head': 4,
          'num_intermediate_channel': 32,
          'orientation': 'per_row',
          'dropout_rate': 0.25,
          'shared_dropout': True,
        },
        'triangle_attention_ending_node': {
          'gating': True,
          'num_head': 4,
          'num_intermediate_channel': 32,
          'orientation': 'per_column',
          'dropout_rate': 0.25,
          'shared_dropout': True,
        },
        'pair_transition': {
          'num_intermediate_factor': 4,
          'orientation': 'per_row',
          'dropout_rate': 0.0,
          'shared_dropout': True
        },
        'single_attention_with_pair_bias': {
          # NOTE: same as row-wise attention used in AlphaFold 2,
          # but only applied to a single sequence, which corresponds to
          # the single represenation
          'num_head': 16,
          'gating_bias_beta': 0,  # beta_ij
        },
        'single_transition': {
          'num_intermediate_factor': 4,
          'orientation': 'per_row',
          'dropout_rate': 0.0,
          'shared_dropout': True
        },
      },
    },
    'heads': {
      'diffusion_module': {
        'weight': 4.0,
        'diff_batch_size': 32,
        'test_diff_batch_size': 5,
        'step_num': 200,
        'gamma0': 0.8,
        'gamma_min': 1.0,
        'lambda': 1.0,
        'eta': 1.5,
        'loss_type': 'huber',
        'loss_smooth_lddt_weight': 1.0,
        'fape_z': 15.0,
        'loss_fape_weight': 0.0,
        'loss_bond_weight': 0.0,
        'diffusion_conditioning': {
          'relative_position_encoding': {
            'relative_token_max': 32,
            'relative_chain_max': 2,
          },
        },
        'atom_encoder': {
          'in_token_channel_name': 'token_channel',
          'out_token_channel_name': 'diffusion_token_channel',
          'use_dense_mode': True,
          'atom_transformer': {
            'diffusion_transformer': {
                'a_channel_name': 'atom_channel',
                's_channel_name': 'atom_channel',
                'z_channel_name': 'atom_pair_channel',
                'n_block': 3, 
                'n_head': 4,
            },
            'n_query': 32,
            'n_key': 128
          },
        },
        'diffusion_transformer': {
            'a_channel_name': 'diffusion_token_channel',
            's_channel_name': 'token_channel',
            'z_channel_name': 'token_pair_channel',
            'n_block': 24, 
            'n_head': 16, 
        },
        'atom_decoder': {
          'in_token_channel_name': 'diffusion_token_channel',
          'atom_transformer': {
            'diffusion_transformer': {
                'a_channel_name': 'atom_channel',
                's_channel_name': 'atom_channel',
                'z_channel_name': 'atom_pair_channel',
                'n_block': 3,
                'n_head': 4,
            },
            'n_query': 32,
            'n_key': 128
          },
        },
      },
      # TODO:
      'distogram': {
        'first_break': 2.3125,
        'last_break': 21.6875,
        'num_bins': 64,
        'weight': 1.0
      },
      'confidence_head': {
        'weight': 0.0,
        'filter_by_resolution': True,
        'min_resolution': 0.1,
        'max_resolution': 3.0,
        'b_pae': 64,
        'stride_pae': 0.5,  # Angstrom
        'b_pde': 64,
        'stride_pde': 0.5,  # Angstrom
        'b_plddt': 50,
        'sigma_data': 16,
        'atom_encoder': {
          'in_token_channel_name': 'token_channel',
          'out_token_channel_name': 'token_channel',
          'use_dense_mode': True,
          'atom_transformer': {
            'diffusion_transformer': {
                'a_channel_name': 'atom_channel',
                's_channel_name': 'atom_channel',
                'z_channel_name': 'atom_pair_channel',
                'n_block': 3, 
                'n_head': 4,
            },
            'n_query': 32,
            'n_key': 128
          },
        },
        'pairformer': {
          'num_block': 4,
          'triangle_multiplication_outgoing': {
            'equation': 'ikc,jkc->ijc',
            'num_intermediate_channel': 128,
            'orientation': 'per_row',
            'dropout_rate': 0.25,
            'shared_dropout': True,
          },
          'triangle_multiplication_incoming': {
            'equation': 'kjc,kic->ijc',
            'num_intermediate_channel': 128,
            'orientation': 'per_row',
            'dropout_rate': 0.25,
            'shared_dropout': True,
          },
          'triangle_attention_starting_node': {
            'gating': True,
            'num_head': 4,
            'num_intermediate_channel': 32,
            'orientation': 'per_row',
            'dropout_rate': 0.25,
            'shared_dropout': True,
          },
          'triangle_attention_ending_node': {
            'gating': True,
            'num_head': 4,
            'num_intermediate_channel': 32,
            'orientation': 'per_column',
            'dropout_rate': 0.25,
            'shared_dropout': True,
          },
          'pair_transition': {
            'num_intermediate_factor': 4,
            'orientation': 'per_row',
            'dropout_rate': 0.0,
            'shared_dropout': True
          },
          'single_attention_with_pair_bias': {
            # NOTE: same as row-wise attention used in AlphaFold 2,
            # but only applied to a single sequence, which corresponds to
            # the single represenation
            'num_head': 16,
            'gating_bias_beta': 0,  # beta_ij
          },
          'single_transition': {
            'num_intermediate_factor': 4,
            'orientation': 'per_row',
            'dropout_rate': 0.0,
            'shared_dropout': True
          },
        },
        'atom_decoder': {
          'in_token_channel_name': 'token_channel',
          'out_channel_name': 'token_pair_channel',
          'final_zero_init': False,
          'atom_transformer': {
            'diffusion_transformer': {
                'a_channel_name': 'atom_channel',
                's_channel_name': 'atom_channel',
                'z_channel_name': 'atom_pair_channel',
                'n_block': 3,
                'n_head': 4,
            },
            'n_query': 32,
            'n_key': 128
          },
        },
      },
      'interface_head': {
        'weight': 0.0,
        'pairformer': {
            'num_block': 1,
            'triangle_multiplication_outgoing': {
            'equation': 'ikc,jkc->ijc',
            'num_intermediate_channel': 128,
            'orientation': 'per_row',
            'dropout_rate': 0.25,
            'shared_dropout': True,
            },
            'triangle_multiplication_incoming': {
            'equation': 'kjc,kic->ijc',
            'num_intermediate_channel': 128,
            'orientation': 'per_row',
            'dropout_rate': 0.25,
            'shared_dropout': True,
            },
            'triangle_attention_starting_node': {
            'gating': True,
            'num_head': 4,
            'num_intermediate_channel': 32,
            'orientation': 'per_row',
            'dropout_rate': 0.25,
            'shared_dropout': True,
            },
            'triangle_attention_ending_node': {
            'gating': True,
            'num_head': 4,
            'num_intermediate_channel': 32,
            'orientation': 'per_column',
            'dropout_rate': 0.25,
            'shared_dropout': True,
            },
            'pair_transition': {
            'num_intermediate_factor': 4,
            'orientation': 'per_row',
            'dropout_rate': 0.0,
            'shared_dropout': True
            },
            'single_attention_with_pair_bias': {
            # NOTE: same as row-wise attention used in AlphaFold 2,
            # but only applied to a single sequence, which corresponds to
            # the single represenation
            'num_head': 16,
            'gating_bias_beta': 0,  # beta_ij
            },
            'single_transition': {
            'num_intermediate_factor': 4,
            'orientation': 'per_row',
            'dropout_rate': 0.0,
            'shared_dropout': True
            },
        },
      }
    },
    'global_config': {
      'deterministic': False,
      'all_atom_mode': True,
      'subbatch_size': 96,
      'use_remat': False,
      'zero_init': True,
      'low_memory': False,
      'fuse_linear': False,  # NOTE: paddlepaddle-gpu 2.4.1.post112 doesn't have fused_gemm_epilogue op
      'fuse_attention': True,
      'use_flash_attn': True,
      'outer_product_mean_position': 'first',
      'h100': False
    },
    'num_recycle': 3,
    'resample_msa_in_recycling': True,
    'add_interface': False,
    'gen_interface': False,
    'train_with_mix_interface': False,
    'update_repr_during_interface_gen': True
  },
})
