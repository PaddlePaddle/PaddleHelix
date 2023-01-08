import pdb
from layers.static_backbones import (
  StaticEmbeddings,
  StaticExtraMsa,
  StaticSingleTemplateEmbedding,
  StaticEvoformer
)
import paddle as pd
from paddle import nn
from tools import all_atom
import numpy as np


class StaticEmbeddingsAndEvoformer(object):
  def __init__(self,
    config:dict, # cfg['model']['embeddings_and_evoformer']
    global_config:dict,
    seq_len:int,
    channel_num:dict,
    n_cpus:int,
    module_prefix:str='embeddingsandevoformer',
    root_weights:str='static_modules',
    is_pdinfer_init:bool=False) -> None:

    # [INFO] build sample & configuration
    self.c = config
    self.gc = global_config
    feed_dict = {
      'target_feat': np.ones([1, seq_len, 22], dtype='float32'),
      'msa_feat': np.ones([1, 508, seq_len, 49], dtype='float32'),
      'seq_mask': np.ones([1, seq_len], dtype='float32'),
      'aatype': np.ones([1, seq_len], dtype='float32'),
      'residue_index': np.ones([1, seq_len], dtype='float32'),
      'template_mask': np.ones([1, 4], dtype='float32'),
      'template_aatype': np.ones([1, 4, seq_len], dtype="int32"), # define
      'template_pseudo_beta_mask': np.ones([1, 4, seq_len], dtype='float32'),
      'template_pseudo_beta': np.ones([1, 4, seq_len, 3], dtype='float32'),
      'template_all_atom_positions': np.ones([1, 4, seq_len, 37, 3], dtype='float32'),
      'template_all_atom_masks': np.ones([1, 4, seq_len, 37], dtype='float32'),
      'extra_msa': np.ones([1, 5120, seq_len], dtype='float32'),
      'extra_has_deletion': np.ones([1, 5120, seq_len], dtype='float32'),
      'extra_deletion_value': np.ones([1, 5120, seq_len], dtype='float32'),
      'extra_msa_mask': np.ones([1, 5120, seq_len], dtype='float32'),
      'msa_mask': np.ones([1, 508, seq_len], dtype='float32'),
      'prev_pos': np.ones([1, seq_len, 37, 3], dtype='float32'),
      'prev_msa_first_row': np.ones([1, seq_len, 256], dtype='float32'),
      'prev_pair': np.ones([1, seq_len, seq_len, 128], dtype='float32')
    }

    # [INFO] build embedding alyer
    feed2embeddings = {k:feed_dict[k] for k in [
      'target_feat', 
      'msa_feat', 
      'seq_mask', 
      'aatype', 
      'residue_index', 
      'template_mask', 
      'template_aatype', 
      'template_pseudo_beta_mask', 
      'template_pseudo_beta', 
      'template_all_atom_positions', 
      'template_all_atom_masks', 
      'extra_msa', 
      'extra_has_deletion', 
      'extra_deletion_value', 
      'prev_pos', 
      'prev_msa_first_row', 
      'prev_pair']
    }
    self.embeddings = StaticEmbeddings(
      config,
      global_config,
      feed2embeddings,
      channel_num,
      n_cpus,
      module_prefix='%s.embeddings' % module_prefix,
      root_weights=root_weights,
      is_pdinfer_init=is_pdinfer_init
    )

    # [INFO] build ExtraMSA layer
    feed2extramsa = {
      'extra_msa_act':np.ones([1, 5120, seq_len, 64], dtype='float32'),
      'extra_pair_act':np.ones([1, seq_len, seq_len, 128], dtype='float32'),
      'extra_msa_mask':feed_dict['extra_msa_mask'],
      'mask_2d': np.ones([1, seq_len, seq_len], dtype='float32')
    }
    self.extra_msa = StaticExtraMsa(
      config,
      global_config,
      feed2extramsa,
      channel_num,
      n_cpus,
      module_prefix='%s.extramsa' % module_prefix,
      root_weights=root_weights,
      is_pdinfer_init=is_pdinfer_init
    )

    # [INFO] build single template embedding layer
    feed2singletemplate = {
      'msa_mask': np.ones([1, 508, seq_len], dtype='float32'),
      'torsion_angles_mask': np.ones([1, 4, seq_len, 7], dtype='float32'),
      'msa_activations_raw': np.ones([1, 508, seq_len, 256], dtype='float32'),
      'template_features': np.ones([1, 4, seq_len, 57], dtype='float32')
    }
    self.single_template_embedding = StaticSingleTemplateEmbedding(
      config,
      global_config,
      feed2singletemplate,
      channel_num,
      n_cpus,
      module_prefix='%s.singletemplateembedding' % module_prefix,
      root_weights=root_weights,
      is_pdinfer_init=is_pdinfer_init
    )

    # [INFO] build evoformer stack
    feed2evoformer = {
      'msa_activations': np.ones([1, 508, seq_len, 256], dtype='float32'),
      'extra_pair_act': np.ones([1, seq_len, seq_len, 128], dtype='float32'),
      'msa_mask': np.ones([1, 508, seq_len], dtype='float32'),
      'mask_2d': np.ones([1, seq_len, seq_len], dtype='float32')
    }
    self.evoformer = StaticEvoformer(
      config,
      global_config,
      feed2evoformer,
      channel_num,
      n_cpus,
      module_prefix='%s.evoformer' % module_prefix,
      root_weights=root_weights,
      is_pdinfer_init=is_pdinfer_init
    )

  def pd2np(self, d:dict) -> dict:
    res = {}
    for k,v in d.items():
      if isinstance(v, pd.Tensor):
        res[k] = v.detach().numpy()
      else:
        res[k] = v
    return res

  def __call__(self, feed_dict:dict) -> dict:
    feed_dict = self.pd2np(feed_dict)

    # [INFO] embeddings
    res_embeddings = self.embeddings(feed_dict)
    msa_activations_raw, extra_msa_act, extra_pair_act, mask_2d = res_embeddings.values()

    # [INFO] extra_msa
    feed_to_extra_msa = {
      'extra_msa_act':extra_msa_act, # (1, 5120, len_dim, 64)
      'extra_pair_act':extra_pair_act, # (1, len_dim, len_dim, 128)
      'extra_msa_mask':feed_dict['extra_msa_mask'], # [1, 5120, len_dim]
      'mask_2d':mask_2d # (1, len_dim, len_dim)
    }
    feed_to_extra_msa = self.pd2np(feed_to_extra_msa)
    res_extra_msa = self.extra_msa(feed_to_extra_msa) # [OK] I/O valid
    extra_msa_act, extra_pair_act = res_extra_msa.values()

    # [INFO] template angle features
    template_aatype = pd.Tensor(feed_dict['template_aatype'])
    template_all_atom_positions = pd.Tensor(feed_dict['template_all_atom_positions'])
    template_all_atom_masks = pd.Tensor(feed_dict['template_all_atom_masks'])

    if self.c.template.enabled and self.c.template.embed_torsion_angles:
      num_templ, num_res = template_aatype.shape[1:]
      
      aatype_one_hot = nn.functional.one_hot(template_aatype, 22)
      # Embed the templates aatype, torsion angles and masks.
      # Shape (templates, residues, msa_channels)
      ret = all_atom.atom37_to_torsion_angles(
          aatype=template_aatype,
          all_atom_pos=template_all_atom_positions,
          all_atom_mask=template_all_atom_masks,
          # Ensure consistent behaviour during testing:
          placeholder_for_undefined=not self.gc.zero_init)

      template_features = pd.concat([
          aatype_one_hot,
          pd.reshape(ret['torsion_angles_sin_cos'],
                         [-1, num_templ, num_res, 14]),
          pd.reshape(ret['alt_torsion_angles_sin_cos'],
                         [-1, num_templ, num_res, 14]),
          ret['torsion_angles_mask']], axis=-1)

      res_single_template = self.single_template_embedding(self.pd2np({
        'msa_mask': feed_dict['msa_mask'],
        'torsion_angles_mask': ret['torsion_angles_mask'].detach().numpy(),
        'msa_activations_raw': msa_activations_raw,
        'template_features': template_features.detach().numpy()
      }))
      msa_activations, msa_mask = res_single_template.values()

    # [INFO] evoformer
    feed_to_evoformer = self.pd2np({
      'msa_activations':msa_activations,
      'extra_pair_act':extra_pair_act,
      'msa_mask':msa_mask,
      'mask_2d':mask_2d
    })
    res_evoformer = self.evoformer(feed_to_evoformer)
    single_activations, pair_activations, msa_activations = res_evoformer.values()

    num_seq = feed_dict['msa_feat'].shape[1]
    return {
      'single':single_activations,
      'pair':pair_activations,
      'msa':msa_activations[:, :num_seq],
      'msa_first_row':msa_activations[:, 0]
    }
