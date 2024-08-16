#   Copyright (c) 2024 PaddleHelix Authors. All Rights Reserved.
#
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# International License (the "License");  you may not use this file  except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://creativecommons.org/licenses/by-nc-sa/4.0/
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
  """Get the ConfigDict of a model."""

  cfg = copy.deepcopy(CONFIG_ALLATOM)
  if name in CONFIG_DIFFS:
    cfg.update_from_flattened_dict(CONFIG_DIFFS[name])

  return cfg


CONFIG_DIFFS = {
    'allatom_demo': {
        'model.heads.confidence_head.weight': 0.01
    },
    'allatom_subbatch_64_recycle_1': {
        'model.global_config.subbatch_size': 64,
        'model.num_recycle': 1,
    },
}

CONFIG_ALLATOM = ml_collections.ConfigDict({
  'data': {   
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
            # NOTE: same as row-wise attention used in HelixFold 2,
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
          # NOTE: same as row-wise attention used in HelixFold 2,
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
        'test_diff_batch_size': 5,
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
            # NOTE: same as row-wise attention used in HelixFold 2,
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
    },
    'global_config': {
      'deterministic': False,
      'all_atom_mode': True,
      'subbatch_size': 96,
      'use_remat': False,
      'zero_init': True,
      'low_memory': False,
      'fuse_linear': False,  # NOTE: paddlepaddle-gpu 2.4.1.post112 doesn't have fused_gemm_epilogue op
      'fuse_attention': False,
      'use_flash_attn': True,
      'outer_product_mean_position': 'first',
    },
    'num_recycle': 3,
    'resample_msa_in_recycling': True,
  },
})
