import pdb
from layers.backbones import (
  EvoformerIteration, 
  Embeddings, 
  SingleTemplateEmbedding,
  SingleActivations
)
from layers.static_basics import StaticModule, JitModule
import paddle
from paddle.distributed.fleet.utils import recompute
from config import model_config
import paddle as pd
from paddle.jit import to_static, save
from paddle.static import InputSpec
from paddle import inference as pdinfer
from paddle import nn
from tools import all_atom
import time
import os
from joblib import delayed, Parallel
import numpy as np
from argparse import ArgumentParser as Parser


class StaticEvoformerIteration(JitModule):
  def __init__(self, 
    config:dict, # cfg['model']['embeddings_and_evoformer']['evoformer']
    global_config:dict,
    feed_dict:dict, 
    channel_num:dict, 
    n_cpus:int,
    is_extra_msa:bool, 
    module_prefix:str = 'evoformeriteration',
    root_weights:str = 'static_modules',
    is_pdinfer_init:bool = True,
  ) -> None:
    self.c = config
    self.gc = global_config
    super(StaticEvoformerIteration, self).__init__(
      config=self.c, 
      global_config=self.gc, 
      pdmodule=EvoformerIteration(channel_num, self.c, self.gc, is_extra_msa), 
      feed_dict=feed_dict, 
      module_prefix=module_prefix,
      n_cpus=n_cpus,
      channel_num=channel_num,
      root_weights=root_weights,
      is_pdinfer_init=is_pdinfer_init
    )
    # basic hyper-params
    self.is_extra_msa = is_extra_msa


class StaticEmbeddings(JitModule):
  '''
    Embedding layer in EmbeddingsAndEvoformer
    input: 
      target_feat, # [1, len_dim, 22], dtype='float32'
      msa_feat, # [1, 508, len_dim, 49], dtype='float32'
      seq_mask, # [1, len_dim], dtype='float32'
      aatype, # [1, len_dim], dtype='int32'
      residue_index, # [1, len_dim], dtype='float32'
      template_mask, # [1, 4], dtype='float32'
      template_aatype, # [1, 4, len_dim], dtype="int32"
      template_pseudo_beta_mask, # [1, 4, len_dim], dtype='float32'
      template_pseudo_beta, # [1, 4, len_dim, 3], dtype='float32'
      template_all_atom_positions, # [1, 4, len_dim, 37, 3], dtype='float32'
      template_all_atom_masks, # [1, 4, len_dim, 37], dtype='float32'
      extra_msa, # [1, 5120, len_dim], dtype='float32'
      extra_has_deletion, # [1, 5120, len_dim], dtype='float32'
      extra_deletion_value, # [1, 5120, len_dim], dtype='float32'
      prev_pos=None, # [1, len_dim, 37, 3], dtype='float32'
      prev_msa_first_row=None, # [1, len_dim, 256], dtype='float32'
      prev_pair=None # [1, len_dim, len_dim, 128], dtype='float32'
    output:
      msa_activations_raw, # (1, 508, len_dim, 256)
      extra_msa_act, # (1, 5120, len_dim, 64)
      extra_pair_act, # (1, len_dim, len_dim, 128)
      mask_2d # (1, len_dim, len_dim)
  '''
  def __init__(self,
    config:dict, # cfg['model']['embeddings_and_evoformer']
    global_config:dict,
    feed_dict:dict,
    channel_num:dict,
    n_cpus:int,
    module_prefix:str='embeddings',
    root_weights:str='static_modules',
    is_pdinfer_init:bool=False) -> None:
    self.c = config
    self.gc = global_config
    super(StaticEmbeddings, self).__init__(
      config=self.c, 
      global_config=self.gc, 
      pdmodule=Embeddings(channel_num, self.c, self.gc),
      feed_dict=feed_dict,
      module_prefix=module_prefix,
      n_cpus=n_cpus,
      channel_num=channel_num,
      root_weights=root_weights,
      is_pdinfer_init=is_pdinfer_init
    )

    
class StaticSingleTemplateEmbedding(JitModule):
  '''
    Embedding layer in EmbeddingsAndEvoformer
    input: 
        msa_mask, # [1, 508, len_dim], dtype='float32'
        torsion_angles_mask, # ret from folding, [1, 4, len_dim, 7], dtype='float32'
        msa_activations_raw, # [1, 508, len_dim, 256], dtype='float32'
        template_features # [1, 4, len_dim, 57], dtype='float32'
    output:
      msa_activations, # 
      msa_mask # 
  '''
  def __init__(self,
    config:dict, # cfg['model']['embeddings_and_evoformer']
    global_config:dict,
    feed_dict:dict,
    channel_num:dict,
    n_cpus:int,
    module_prefix:str='singletemplateembedding',
    root_weights:str='static_modules',
    is_pdinfer_init:bool=False) -> None:
    self.c = config
    self.gc = global_config
    if self.c.template.enabled:
      channel_num['template_angle'] = 57
      channel_num['template_pair'] = 88
    super(StaticSingleTemplateEmbedding, self).__init__(
      config=self.c, 
      global_config=self.gc, 
      pdmodule=SingleTemplateEmbedding(channel_num, self.c, self.gc),
      feed_dict=feed_dict,
      module_prefix=module_prefix,
      n_cpus=n_cpus,
      channel_num=channel_num,
      root_weights=root_weights,
      is_pdinfer_init=is_pdinfer_init
    )


class StaticSingleActivations(JitModule):
  def __init__(self,
    config:dict, # cfg['model']['embeddings_and_evoformer']
    global_config:dict,
    feed_dict:dict,
    channel_num:dict,
    n_cpus:int,
    module_prefix:str='single_activations',
    root_weights:str='static_modules',
    is_pdinfer_init:bool=False) -> None:
    self.c = config
    self.gc = global_config
    super(StaticSingleActivations, self).__init__(
      config=self.c, 
      global_config=self.gc, 
      pdmodule=SingleActivations(channel_num, self.c, self.gc),
      feed_dict=feed_dict,
      module_prefix=module_prefix,
      n_cpus=n_cpus,
      channel_num=channel_num,
      root_weights=root_weights,
      is_pdinfer_init=is_pdinfer_init
    )


class StaticExtraMsa(object):
  def __init__(self,
    config:dict, # cfg['model']['embeddings_and_evoformer']
    global_config:dict,
    feed_dict:dict,
    channel_num:dict,
    n_cpus:int,
    module_prefix:str='extramsa',
    root_weights:str='static_modules',
    is_pdinfer_init:bool=False) -> None:

    self.c = config
    self.gc = global_config
    n_layers = self.c['extra_msa_stack_num_block']
    self.extra_msa_stack = []
    self.is_extra_msa = True
    for i in range(n_layers):
      self.extra_msa_stack.append(StaticEvoformerIteration(
        self.c['evoformer'],
        self.gc,
        feed_dict,
        channel_num,
        n_cpus,
        is_extra_msa=self.is_extra_msa,
        module_prefix='%s.evoformeriteration_%d' % (module_prefix, i),
        root_weights=root_weights,
        is_pdinfer_init=is_pdinfer_init
      ))

  def pd2np(self, d:dict) -> dict:
    res = {}
    for k,v in d.items():
      if isinstance(v, pd.Tensor):
        res[k] = v.detach().numpy()
      else:
        res[k] = v
    return res

  def __call__(self,
    feeddict:dict
  ) -> dict:
    extra_msa_act = feeddict['extra_msa_act'] # (1, 5120, len, 64)
    extra_pair_act = feeddict['extra_pair_act'] # (1, len, len, 128)
    extra_msa_mask = feeddict['extra_msa_mask'] # (1, 5120, len)
    mask_2d = feeddict['mask_2d'] # (1, len, len)
    for i, extra_msa_stack_iteration in enumerate(self.extra_msa_stack):
      print('# [INFO] extra_msa_stack_iteration_%d' % i)
      res = extra_msa_stack_iteration(self.pd2np({
        'extra_msa_act':extra_msa_act,
        'extra_pair_act':extra_pair_act,
        'extra_msa_mask':extra_msa_mask,
        'mask_2d':mask_2d
      }))
      ks = list(res.keys())
      extra_msa_act = res[ks[0]] # ['extra_msa_act_new']
      extra_pair_act = res[ks[1]] # ['extra_pair_act_new']
    return {
      'extra_msa_act':extra_msa_act, # (1, 5120, len, 64)
      'extra_pair_act':extra_pair_act # (1, len, len, 128)
    }


class StaticEvoformer(object):
  def __init__(self,
    config:dict, # cfg['model']['embeddings_and_evoformer']
    global_config:dict,
    feed_dict:dict,
    channel_num:dict,
    n_cpus:int,
    module_prefix:str='evoformer',
    root_weights:str='static_modules',
    is_pdinfer_init:bool=False) -> None:

    self.c = config
    self.gc = global_config
    n_layers = self.c['evoformer_num_block']
    self.is_extra_msa = False

    feed2evoformer = {
      'msa_act':feed_dict['msa_activations'],
      'pair_act':feed_dict['extra_pair_act'],
      'msa_mask':feed_dict['msa_mask'],
      'pair_mask':feed_dict['mask_2d']
    }
    if not os.path.isfile('{}/{}.evoformeriteration_0.pdiparams'.format(
      root_weights, module_prefix)):
      Parallel(n_jobs=-1)(
        delayed(self._create_layer)(
          feed2evoformer,
          channel_num,
          n_cpus,
          is_extra_msa=self.is_extra_msa,
          module_prefix='%s.evoformeriteration_%d' % (module_prefix, i),
          root_weights=root_weights,
          is_pdinfer_init=is_pdinfer_init
        ) for i in range(n_layers)
      )

    print('# [INFO] parallel compilation of iteration layers were done')
    # sequential compilation of evoformer iterations
    self.evoformer_iteration = []
    for i in range(n_layers):
      self.evoformer_iteration.append(StaticEvoformerIteration(
        self.c['evoformer'],
        self.gc,
        feed2evoformer,
        channel_num,
        n_cpus,
        is_extra_msa=self.is_extra_msa,
        module_prefix='%s.evoformeriteration_%d' % (module_prefix, i),
        root_weights=root_weights,
        is_pdinfer_init=is_pdinfer_init
      ))

    len_dims = list(feed_dict['msa_activations'][:,0].shape)
    len_dims[1] = 1
    self.single_activations = StaticSingleActivations(
      self.c,
      self.gc,
      {'msa_activation': np.ones(len_dims, dtype='float32')},
      channel_num,
      n_cpus,
      module_prefix='%s.single_activations' % module_prefix,
      root_weights=root_weights,
      is_pdinfer_init=is_pdinfer_init
    )

  def _create_layer(self, feed_dict, channel_num, n_cpus, is_extra_msa, module_prefix, root_weights, is_pdinfer_init):
    StaticEvoformerIteration(
      self.c['evoformer'],
      self.gc,
      feed_dict,
      channel_num,
      n_cpus,
      is_extra_msa,
      module_prefix,
      root_weights,
      is_pdinfer_init
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
    feed2block = {
      'msa_act':feed_dict['msa_activations'], # (1, 508, len, 256)
      'pair_act':feed_dict['extra_pair_act'], # (1, len, len, 128)
      'msa_mask':feed_dict['msa_mask'], # (1, 508, len)
      'pair_mask':feed_dict['mask_2d'] # (1, len, len)
    }
    for i, evoformer_block in enumerate(self.evoformer_iteration):
      print('# [INFO] evoformer_iteration_%d' % i)
      res = evoformer_block(self.pd2np(feed2block))
      ks = list(res.keys())
      msa_activations = res[ks[0]] # ['msa_act']
      extra_pair_act = res[ks[1]] # ['pair_act']
      feed2block['msa_act'] = msa_activations
      feed2block['pair_act'] = extra_pair_act
    
    single_acts = self.single_activations(self.pd2np({
      'msa_activation':msa_activations[:, 0]
    }))
    k = list(single_acts.keys())[0]
    single_activations = single_acts[k]
    return {
      'single_activations':single_activations, 
      'pair_activations':extra_pair_act,
      'msa_activations':msa_activations
    }
