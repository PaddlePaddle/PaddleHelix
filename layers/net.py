import pdb
import numpy as np
import paddle
import paddle.nn as nn
from paddle.fluid.framework import _dygraph_tracer
from paddle.distributed.fleet.utils import recompute
from tools import residue_constants
from tools import folding
from layers.subnets import EmbeddingsAndEvoformer
from layers.head import (
  MaskedMsaHead, 
  DistogramHead, 
  PredictedLDDTHead, 
  PredictedAlignedErrorHead, 
  ExperimentallyResolvedHead)

# Map head name in config to head name in model params
Head_names = {
    'masked_msa': 'masked_msa_head',
    'distogram': 'distogram_head',
    'predicted_lddt': 'predicted_lddt_head',
    'predicted_aligned_error': 'predicted_aligned_error_head',
    'experimentally_resolved': 'experimentally_resolved_head',   # finetune loss
}


def recompute_wrapper(func, *args, is_recompute=True):
    """Function wrapper for recompute"""
    if is_recompute:
        return recompute(func, *args)
    else:
        return func(*args)


class AlphaFold(nn.Layer):
  """AlphaFold model with recycling.

  Jumper et al. (2021) Suppl. Alg. 2 "Inference"
  """
  def __init__(self, config):
    super(AlphaFold, self).__init__()
    self.channel_num = {
        'target_feat': 22,
        'msa_feat': 49,
    }
    self.config = config
    self.global_config = config.global_config

    self.alphafold_iteration = AlphaFoldIteration(
        self.channel_num, 
        self.config, 
        self.global_config)

  def forward(self,
      batch,
      ensemble_representations=False,
    ):
      """Run the AlphaFold model.

      Arguments:
        batch: Dictionary with inputs to the AlphaFold model.
        ensemble_representations: Whether to use ensembling of representations.

      Returns:
          The output of AlphaFoldIteration is a nested dictionary containing
          predictions from the various heads.

      """
      inner_batch, num_residues = batch['aatype'].shape[1:]

      def _get_prev(ret):
          new_prev = {
              'prev_pos': ret['structure_module']['final_atom_positions'],
              'prev_msa_first_row': ret['representations']['msa_first_row'],
              'prev_pair': ret['representations']['pair'],
          }

          for k in new_prev.keys():
              new_prev[k].stop_gradient = True

          return new_prev

      def _run_single_recycling(prev, recycle_idx):
          print(f'########## recycle id: {recycle_idx} ##########')

          if self.config.resample_msa_in_recycling:
              # (B, (R+1)*E, N, ...)
              # B: batch size, R: recycling number,
              # E: ensemble number, N: residue number
              num_ensemble = inner_batch // (self.config.num_recycle + 1)
              ensembled_batch = dict()
              for k in batch.keys():
                  start = recycle_idx * num_ensemble
                  end = start + num_ensemble
                  ensembled_batch[k] = batch[k][:, start:end]
          else:
              # (B, E, N, ...)
              num_ensemble = inner_batch
              ensembled_batch = batch

          non_ensembled_batch = prev
          return self.alphafold_iteration(
              ensembled_batch, 
              non_ensembled_batch,
              ensemble_representations=ensemble_representations)

      if self.config.num_recycle:
          # aatype: (B, E, N), zeros_bn: (B, N)
          zeros_bn = paddle.zeros_like(paddle.Tensor(batch['aatype'][:, 0]), dtype='float32')

          emb_config = self.config.embeddings_and_evoformer
          prev = {
              'prev_pos': paddle.tile(
                  zeros_bn[..., None, None],
                  [1, 1, residue_constants.atom_type_num, 3]),
              'prev_msa_first_row': paddle.tile(
                  zeros_bn[..., None],
                  [1, 1, emb_config.msa_channel]),
              'prev_pair': paddle.tile(
                  zeros_bn[..., None, None],
                  [1, 1, num_residues, emb_config.pair_channel]),
          }

          if 'num_iter_recycling' in batch:
              # Training trick: dynamic recycling number
              num_iter = batch['num_iter_recycling'].numpy()[0, 0]
              num_iter = min(int(num_iter), self.config.num_recycle)
          else:
              num_iter = self.config.num_recycle

          for recycle_idx in range(num_iter):
              ret = _run_single_recycling(prev, recycle_idx)
              prev = _get_prev(ret)

      else:
          prev = {}
          num_iter = 0

      return _run_single_recycling(prev, num_iter)


class AlphaFoldIteration(nn.Layer):
    """A single recycling iteration of AlphaFold architecture.

    Computes ensembled (averaged) representations from the provided features.
    These representations are then passed to the various heads
    that have been requested by the configuration file. Each head also returns a
    loss which is combined as a weighted sum to produce the total loss.

    Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 3-22
    """

    def __init__(self, 
        channel_num, 
        config, 
        global_config, 
    ):
        super(AlphaFoldIteration, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        # copy these config for later usage
        self.channel_num['extra_msa_channel'] = config.embeddings_and_evoformer.extra_msa_channel
        self.channel_num['msa_channel'] = config.embeddings_and_evoformer.msa_channel
        self.channel_num['pair_channel'] = config.embeddings_and_evoformer.pair_channel
        self.channel_num['seq_channel'] = config.embeddings_and_evoformer.seq_channel

        self.evoformer = EmbeddingsAndEvoformer(
          channel_num=self.channel_num,
          config=self.config['embeddings_and_evoformer'],
          global_config=self.global_config)

        Head_modules = {
            'masked_msa': MaskedMsaHead,
            'distogram': DistogramHead,
            'structure_module': folding.StructureModule,
            'predicted_lddt': PredictedLDDTHead,
            'predicted_aligned_error': PredictedAlignedErrorHead,
            'experimentally_resolved': ExperimentallyResolvedHead,   # finetune loss
        }

        self.used_heads = []
        self.heads = {}
        for head_name, head_config in sorted(self.config.heads.items()):
            if head_name not in Head_modules.keys():
                continue

            self.used_heads.append(head_name)
            module = Head_modules[head_name](
                self.channel_num, head_config, self.global_config)
            # setattr(self, head_name_, module)
            self.heads[head_name] = module

    def filtered_inputs(self, d:dict, ks:list):
        ret = []
        for k, v in d.items():
            if k in ks:
                ret.append(v)
        return ret

    def __call__(self,
                ensembled_batch,
                non_ensembled_batch,
                ensemble_representations=False):
        num_ensemble = ensembled_batch['seq_length'].shape[1]
        print(ensembled_batch['seq_length'].shape)
        if not ensemble_representations:
            assert num_ensemble == 1

        def _slice_batch(i):
            b = {k: v[:, i] for k, v in ensembled_batch.items()}
            b.update(non_ensembled_batch)
            return b

        batch0 = _slice_batch(0)
        res_evoformer = self.evoformer(*self.filtered_inputs(batch0, ks=[
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
            'extra_msa_mask',
            'msa_mask',
            'prev_pos',
            'prev_msa_first_row',
            'prev_pair'
        ]))

        representations = {
            'single': res_evoformer[0],
            'pair': res_evoformer[1],
            'msa': res_evoformer[2],
            'msa_first_row': res_evoformer[3]
        }

        # MSA representations are not ensembled
        msa_representation = representations['msa']
        del representations['msa']

        if ensemble_representations:
            for i in range(1, num_ensemble):
                batch = _slice_batch(i)
                representations_update = self.evoformer(batch)
                for k in representations.keys():
                    representations[k] += representations_update[k]

            for k in representations.keys():
                representations[k] /= num_ensemble + 0.0

        representations['msa'] = msa_representation
        ret = {'representations': representations}

        def _forward_heads(representations, ret, batch0):
            for head_name, head_config in self._get_heads():
                # Skip PredictedLDDTHead and PredictedAlignedErrorHead until
                # StructureModule is executed.
                if head_name in ('predicted_lddt', 'predicted_aligned_error'):
                    continue
                else:
                    # ret[head_name] = getattr(self, head_name_)(representations, batch0)
                    if head_name == 'structure_module':
                        ret[head_name] = self.heads[head_name](representations, batch0)
                    else:
                        ret[head_name] = self.heads[head_name](representations)
                    if 'representations' in ret[head_name]:
                    # Extra representations from the head. Used by the
                    # structure module to provide activations for the PredictedLDDTHead.
                        representations.update(ret[head_name].pop('representations'))

            if self.config.heads.get('predicted_lddt.weight', 0.0):
                # Add PredictedLDDTHead after StructureModule executes.
                head_name = 'predicted_lddt'
                # Feed all previous results to give access to structure_module result.
                head_config = self.config.heads[head_name]
                # ret[head_name] = getattr(self, head_name_)(representations, batch0)
                if head_name == 'structure_module':
                    ret[head_name] = self.heads[head_name](representations, batch0)
                else:
                    ret[head_name] = self.heads[head_name](representations)

            if ('predicted_aligned_error' in self.config.heads
                    and self.config.heads.get('predicted_aligned_error.weight', 0.0)):
                # Add PredictedAlignedErrorHead after StructureModule executes.
                head_name = 'predicted_aligned_error'
                # Feed all previous results to give access to structure_module result.
                head_config = self.config.heads[head_name]
                # ret[head_name] = getattr(self, head_name_)(representations, batch0)
                if head_name == 'structure_module':
                    ret[head_name] = self.heads[head_name](representations, batch0)
                else:
                    ret[head_name] = self.heads[head_name](representations)

            return ret

        tracer = _dygraph_tracer()
        if tracer._amp_dtype == "bfloat16":
            raise NotImplementedError("Currently CPU optimized inference is unsupported on bfloat16.")
        else:
            with paddle.no_grad():
                ret = _forward_heads(representations, ret, batch0)

        return ret

    def _get_heads(self):
        assert 'structure_module' in self.used_heads
        head_names = [h for h in self.used_heads]

        for k in head_names:
            yield k, self.config.heads[k]
