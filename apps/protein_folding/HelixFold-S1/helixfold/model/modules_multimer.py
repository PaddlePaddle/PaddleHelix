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

"""Updated modules for multimer."""

import numpy as np
import gc
import paddle
import paddle.nn as nn
FLUID_DEPRECATED = not hasattr(paddle, 'fluid')
if FLUID_DEPRECATED:
    from paddle.base.framework import _dygraph_tracer
else:
    from paddle.fluid.framework import _dygraph_tracer

from helixfold.common import residue_constants
from helixfold.model.utils import mask_mean, subbatch
from helixfold.model import modules, folding, lddt, \
    quat_affine, all_atom, folding_multimer
from helixfold.model.utils import init_gate_linear, init_final_linear
from helixfold.model.chain_align import multi_chain_permutation_align
from helixfold.model import chain_align_numpy
from helixfold.model import ranking_head
from utils.utils import get_structure_module_bf16_op_list

class HelixFold(nn.Layer):
    """HelixFold-Multimer model with recycling.
    """

    def __init__(self, channel_num, config):
        super(HelixFold, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = config.global_config

        self.helixfold_iteration = HelixFoldIteration(
            self.channel_num, self.config, self.global_config)

    def forward(self,
                batch,
                label,
                return_representations=False,
                ensemble_representations=True,
                compute_loss=True):
        """Run the HelixFold-Multimer model.

        Arguments:
            batch: Dictionary with inputs to the HelixFold model.
            label: Dictionary with labels, only used for training.
            return_representations: Whether to also return the intermediate
                representations.
            ensemble_representations: placeholder, to make `forward` accept
                same inputs as `modules.HelixFold`.

        Returns:
            The output of HelixFoldIteration is a nested dictionary containing
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

        def _run_single_recycling(prev, recycle_idx, compute_loss):
            if not self.training:
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
            return self.helixfold_iteration(
                ensembled_batch, label, non_ensembled_batch,
                compute_loss=compute_loss, ensemble_representations=ensemble_representations)

        if self.config.num_recycle:
            # aatype: (B, E, N), zeros_bn: (B, N)
            zeros_bn_shape = batch['aatype'].shape[0:1] + batch['aatype'].shape[2:]

            emb_config = self.config.embeddings_and_evoformer

            # if not self.training: # for inference
            if not self.training and self.global_config.low_memory is True:
                prev = {
                    'prev_pos': paddle.zeros(
                        zeros_bn_shape + [residue_constants.atom_type_num, 3], dtype="float32") if not "rigid_pos" in batch else batch["rigid_pos"],
                    'prev_msa_first_row': paddle.zeros(
                        zeros_bn_shape + [emb_config.msa_channel], dtype="float32"),
                    'prev_pair': paddle.zeros(
                        zeros_bn_shape + [num_residues, emb_config.pair_channel], dtype=paddle.bfloat16),
                }
            else:
                prev = {
                    'prev_pos': paddle.zeros(
                        zeros_bn_shape + [residue_constants.atom_type_num, 3], dtype="float32") if not "rigid_pos" in batch else batch["rigid_pos"],
                    'prev_msa_first_row': paddle.zeros(
                        zeros_bn_shape + [emb_config.msa_channel], dtype="float32"),
                    'prev_pair': paddle.zeros(
                        zeros_bn_shape + [num_residues, emb_config.pair_channel], dtype="float32"),
                }
            
            # add diffused pos to prev_pos
            if self.config.get("use_diff_pos_as_prev", False):
                prev.update({
                    'prev_pos': batch["pos_diffused"] # [batch, num_res, 37, 3]
                    })

            if 'num_iter_recycling' in batch:
                # Training trick: dynamic recycling number
                num_iter = batch['num_iter_recycling'].numpy()[0, 0]
                num_iter = min(int(num_iter), self.config.num_recycle)
            else:
                num_iter = self.config.num_recycle

            for recycle_idx in range(num_iter):
                ret = _run_single_recycling(prev, recycle_idx, compute_loss=False)
                prev = _get_prev(ret)
                # if not self.training:
                if not self.training and self.global_config.low_memory is True:
                    del ret
                    gc.collect()

        else:
            prev = {}
            num_iter = 0
            # add diffused pos to prev_pos
            if self.config.get("use_diff_pos_as_prev", False):
                prev.update({
                    'prev_pos': batch["pos_diffused"] # [batch, num_res, 37, 3]
                    })

        ret = _run_single_recycling(
            prev, num_iter, compute_loss=compute_loss)
        if compute_loss:
            ret, loss = ret

        if not return_representations:
            del ret['representations']

        if compute_loss:
            return ret, loss
        else:
            return ret


class HelixFoldIteration(nn.Layer):
    """A single recycling iteration of HelixFold architecture.

    Computes ensembled (averaged) representations from the provided features.
    These representations are then passed to the various heads
    that have been requested by the configuration file.
    """

    def __init__(self, channel_num, config, global_config):
        super(HelixFoldIteration, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        # copy these config for later usage
        self.channel_num['extra_msa_channel'] = config.embeddings_and_evoformer.extra_msa_channel
        self.channel_num['msa_channel'] = config.embeddings_and_evoformer.msa_channel
        self.channel_num['pair_channel'] = config.embeddings_and_evoformer.pair_channel
        self.channel_num['seq_channel'] = config.embeddings_and_evoformer.seq_channel

        self.evoformer = EmbeddingsAndEvoformer(
            self.channel_num, self.config.embeddings_and_evoformer,
            self.global_config)

        Head_modules = {
            'masked_msa': modules.MaskedMsaHead,
            'distogram': modules.DistogramHead,
            'structure_module': folding.StructureModule,
            'predicted_lddt': modules.PredictedLDDTHead,
            'predicted_aligned_error': modules.PredictedAlignedErrorHead,
            'experimentally_resolved': modules.ExperimentallyResolvedHead,   # finetune loss
        }

        self.used_heads = []
        self.heads = []
        for head_name, head_config in sorted(self.config.heads.items()):
            if head_name not in Head_modules:
                continue

            if head_config.get('weight', 0.) == 0.:
                continue

            self.used_heads.append(head_name)
            module = Head_modules[head_name](
                self.channel_num, head_config, self.global_config)

            head_name_ = modules.Head_names.get(head_name, head_name)
            setattr(self, head_name_, module)
            self.heads.append(module)

        self.use_diffusion = self.config.heads.structure_module.get("use_diffusion",False)

    def forward(self,
                ensembled_batch,
                label,
                non_ensembled_batch,
                compute_loss=False, ensemble_representations=False):
        num_ensemble = ensembled_batch['seq_length'].shape[1]

        if not ensemble_representations:
            assert num_ensemble == 1

        def _slice_batch(i):
            b = {k: v[:, i] for k, v in ensembled_batch.items()}
            b.update(non_ensembled_batch)
            return b

        batch0 = _slice_batch(0)
        representations = self.evoformer(batch0)

        # MSA representations are not ensembled
        msa_representation = representations['msa']
        del representations['msa']
        # # MaskedMSAHead is apply on batch0, put after permutation_alignment
        # label['bert_mask'] = batch0['bert_mask']
        # label['true_msa'] = batch0['true_msa']
        # label['residue_index'] = batch0['residue_index']

        # Multimer always uses ensemble
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

        def loss(head_name_, head_config, ret, head_name, aligned_label, filter_ret=True):
            if filter_ret:
                value = ret[head_name]
            else:
                value = ret
            loss_output = getattr(self, head_name_).loss(value, aligned_label)
            ret[head_name].update(loss_output)
            loss = head_config.weight * ret[head_name]['loss']
            return loss

        def _forward_heads(representations, ret, batch0, label):
            total_loss = 0.
            # execute structure_module firstly
            head_name = 'structure_module'
            head_name_ = modules.Head_names.get(head_name, head_name)
            head_config = self.config.heads[head_name]
            ret[head_name] = getattr(self, head_name_)(representations, batch0)
            if 'representations' in ret[head_name]:
                representations.update(ret[head_name].pop('representations'))
            if compute_loss:
                if self.use_diffusion:
                    aligned_label = label # skip permutation align
                else:
                    ### do multi chain permutation align to generate new label
                    use_numpy = True
                    if use_numpy:
                        aligned_label = chain_align_numpy.multi_chain_permutation_align(
                                batch0, label, ret[head_name]) 
                    else:
                        aligned_label = multi_chain_permutation_align(batch0, label, ret[head_name]) 
                # MaskedMSAHead is apply on batch0
                aligned_label['bert_mask'] = batch0['bert_mask']
                aligned_label['true_msa'] = batch0['true_msa']
                aligned_label['residue_index'] = batch0['residue_index']

                total_loss += loss(head_name_, head_config, ret, head_name, aligned_label)

            for head_name, head_config in self._get_heads():
                head_name_ = modules.Head_names.get(head_name, head_name)
                # Skip PredictedLDDTHead and PredictedAlignedErrorHead until
                # StructureModule is executed.
                if head_name in ('structure_module', 'predicted_lddt', 'predicted_aligned_error'):
                    continue
                else:
                    ret[head_name] = getattr(self, head_name_)(representations, batch0)
                    if 'representations' in ret[head_name]:
                    # Extra representations from the head. Used by the
                    # structure module to provide activations for the PredictedLDDTHead.
                        representations.update(ret[head_name].pop('representations'))
                if compute_loss:
                    total_loss += loss(head_name_, head_config, ret, head_name, aligned_label)

            if self.config.heads.get('predicted_lddt.weight', 0.0):
                # Add PredictedLDDTHead after StructureModule executes.
                head_name = 'predicted_lddt'
                # Feed all previous results to give access to structure_module result.
                head_name_ = modules.Head_names.get(head_name, head_name)
                head_config = self.config.heads[head_name]
                ret[head_name] = getattr(self, head_name_)(representations, batch0)
                if compute_loss:
                    total_loss += loss(head_name_, head_config, ret, head_name, aligned_label, filter_ret=False)

            if ('predicted_aligned_error' in self.config.heads
                    and self.config.heads.get('predicted_aligned_error.weight', 0.0)):
                # Add PredictedAlignedErrorHead after StructureModule executes.
                head_name = 'predicted_aligned_error'
                # Feed all previous results to give access to structure_module result.
                head_config = self.config.heads[head_name]
                head_name_ = modules.Head_names.get(head_name, head_name)
                ret[head_name] = getattr(self, head_name_)(representations, batch0)
                if compute_loss:
                    total_loss += loss(head_name_, head_config, ret, head_name, aligned_label, filter_ret=False)

            return ret, total_loss

        # if not self.training:
        if not self.training and self.global_config.low_memory is True:
            black_list, white_list = get_structure_module_bf16_op_list()
            with paddle.amp.auto_cast(level='O1', custom_white_list=white_list, custom_black_list=black_list, dtype='bfloat16'):
                ret, total_loss = _forward_heads(representations, ret, batch0, label)

        else:
            tracer = _dygraph_tracer()
            if tracer._amp_dtype == "bfloat16":
                with paddle.amp.auto_cast(enable=False):
                    bf16 = paddle.base.core.VarDesc.VarType.BF16 if FLUID_DEPRECATED else paddle.fluid.core.VarDesc.VarType.BF16
                    for key, value in representations.items():
                        if value.dtype in [bf16]:
                            temp_value = value.cast('float32')
                            temp_value.stop_gradient = value.stop_gradient
                            representations[key] = temp_value
                    for key, value in batch0.items():
                        if value.dtype in [bf16]:
                            temp_value = value.cast('float32')
                            temp_value.stop_gradient = value.stop_gradient
                            batch0[key] = temp_value
                    ret, total_loss = _forward_heads(representations, ret, batch0, label)

            else:
                ret, total_loss = _forward_heads(representations, ret, batch0, label)

        if compute_loss:
            return ret, total_loss
        else:
            return ret

    def _get_heads(self):
        assert 'structure_module' in self.used_heads
        head_names = [h for h in self.used_heads]

        for k in head_names:
            yield k, self.config.heads[k]


class EmbeddingsAndEvoformer(nn.Layer):
    """Embeds the input data and runs Evoformer.

    Produces the MSA, single and pair representations.
    Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 5-18
    Richard et al. (2021)
    """

    def __init__(self, channel_num, config, global_config):
        super(EmbeddingsAndEvoformer, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        Linear = paddle.incubate.nn.FusedLinear if self.global_config.fuse_linear else paddle.nn.Linear

        # InputEmbedder
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 5
        # Jumper et al. (2021) Suppl. Alg. 3 "InputEmbedder"
        self.preprocess_1d = Linear(channel_num['target_feat'],
                                       self.config.msa_channel, name='preprocess_1d')
        self.preprocess_msa = Linear(channel_num['msa_feat'],
                                        self.config.msa_channel, name='preprocess_msa')
        self.left_single = Linear(channel_num['target_feat'], self.config.pair_channel,
                                     name='left_single')
        self.right_single = Linear(channel_num['target_feat'], self.config.pair_channel,
                                      name='right_single')

        # RecyclingEmbedder
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 6
        # Jumper et al. (2021) Suppl. Alg. 32 "RecyclingEmbedder"
        if self.config.recycle_pos:
            self.prev_pos_linear = Linear(self.config.prev_pos.num_bins,
                                             self.config.pair_channel)

        if self.config.recycle_features:
            self.prev_msa_first_row_norm = nn.LayerNorm(
                self.config.msa_channel)
            self.prev_pair_norm = nn.LayerNorm(self.config.pair_channel)

        if self.config.template.enabled:
            self.channel_num['template_pair'] = self.config.pair_channel
            self.template_embedding = TemplateEmbedding(
                self.channel_num, self.config.template, self.global_config)

        if self.config.max_relative_idx:
            if self.config.use_chain_relative:
                rel_pos_dim = 2 * self.config.max_relative_idx + 2
                rel_chain_dim = 2 * self.config.max_relative_chain + 2
                entity_id_same_dim = 1

                self.position_activations = Linear(
                    rel_pos_dim + rel_chain_dim + entity_id_same_dim,
                    self.config.pair_channel)
            else:
                self.position_activations = Linear(
                    2 * self.config.max_relative_idx + 1,
                    self.config.pair_channel)

        if self.config.get("use_pocket_info", False):
            self.pocket_activations = Linear(
                2, self.config.msa_channel  # for is_pocket flag
            )
        # ExtraMSAEmbedder
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 14-16
        self.extra_msa_activations = Linear(
            25,  # 23 (20aa+unknown+gap+mask) + 1 (has_del) + 1 (del_val)
            self.config.extra_msa_channel)

        # Extra MSA Stack.
        # Jumper et al. (2021) Suppl. Alg. 18 "ExtraMsaStack"
        self.extra_msa_stack = nn.LayerList()
        for _ in range(self.config.extra_msa_stack_num_block):
            self.extra_msa_stack.append(
                modules.EvoformerIteration(
                    self.channel_num,
                    self.config.evoformer,
                    self.global_config,
                    is_extra_msa=True))

        # Main trunk of the network
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 17-18
        self.evoformer_iteration = nn.LayerList()
        for _ in range(self.config.evoformer_num_block):
            self.evoformer_iteration.append(
                modules.EvoformerIteration(
                    self.channel_num,
                    self.config.evoformer,
                    self.global_config,
                    is_extra_msa=False))

        self.template_single_embedding = Linear(
            # 22: aa 1-hot, 4*2: sin & cos of chi angles, 4: chi mask
            22 + 4 * 2 + 4, self.config.msa_channel)
        self.template_projection = Linear(
            self.config.msa_channel, self.config.msa_channel)

        self.single_activations = Linear(
            self.config.msa_channel, self.config.seq_channel)

    def forward(self, batch):
        # InputEmbedder
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 5
        # Jumper et al. (2021) Suppl. Alg. 3 "InputEmbedder"
        preprocess_1d = self.preprocess_1d(batch['target_feat'])
        preprocess_msa = self.preprocess_msa(batch['msa_feat'])
        msa_activations = paddle.unsqueeze(preprocess_1d, axis=1) + \
            preprocess_msa

        right_single = self.right_single(batch['target_feat'])
        left_single = self.left_single(batch['target_feat'])
        pair_activations = paddle.unsqueeze(right_single, axis=1) + \
            paddle.unsqueeze(left_single, axis=2)
        mask_2d = paddle.unsqueeze(batch['seq_mask'], axis=1) * \
            paddle.unsqueeze(batch['seq_mask'], axis=2)

        # Inject previous outputs for recycling.
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 6
        # Jumper et al. (2021) Suppl. Alg. 32 "RecyclingEmbedder"
        if self.config.recycle_pos and 'prev_pos' in batch:
            prev_pseudo_beta = pseudo_beta_fn(
                batch['aatype'], batch['prev_pos'], None)
            dgram = modules.dgram_from_positions(
                prev_pseudo_beta, **self.config.prev_pos)
            pair_activations += self.prev_pos_linear(dgram)

        if self.config.recycle_features:
            if 'prev_msa_first_row' in batch:
                prev_msa_first_row = self.prev_msa_first_row_norm(
                    batch['prev_msa_first_row'])

                # A workaround for `jax.ops.index_add`
                msa_first_row = paddle.squeeze(msa_activations[:, 0, :], axis=1)
                msa_first_row += prev_msa_first_row
                msa_first_row = paddle.unsqueeze(msa_first_row, axis=1)
                msa_activations = paddle.concat([msa_first_row, msa_activations[:, 1:, :]], axis=1)

            if 'prev_pair' in batch:
                pair_activations += self.prev_pair_norm(batch['prev_pair'])

        if self.config.max_relative_idx:
            pair_activations += self._relative_encoding(batch)

        if self.config.template.enabled:
            template_batch = {
                'template_aatype': batch['template_aatype'],
                'template_all_atom_positions': batch['template_all_atom_positions'],
                'template_all_atom_mask': batch['template_all_atom_masks'],
                'template_mask': batch['template_mask'],
                'template_pseudo_beta_mask': batch['template_pseudo_beta_mask'],
                'template_pseudo_beta': batch['template_pseudo_beta'],
            }

            # Construct a mask such that only intra-chain template
            # features are computed, since all templates are for
            # each chain individually.
            asym_id = batch['asym_id']
            multichain_mask = paddle.unsqueeze(asym_id, axis=[-1]) == \
                paddle.unsqueeze(asym_id, axis=[-2])
            template_act = self.template_embedding(
                pair_activations, template_batch, mask_2d, multichain_mask)
            pair_activations += template_act

        # ExtraMSAEmbedder
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 14-16
        extra_msa_feat = self._create_extra_msa_feature(batch)
        extra_msa_activations = self.extra_msa_activations(extra_msa_feat)

        # ==================================================
        #  Extra MSA Stack
        # Jumper et al. (2021) Suppl. Alg. 18 "ExtraMsaStack"
        # ==================================================
        extra_msa_stack_input = {
            'msa': extra_msa_activations,
            'pair': pair_activations,
        }

        for idx, extra_msa_stack_iteration in enumerate(self.extra_msa_stack):
            extra_msa_act, extra_pair_act = modules.recompute_wrapper(
                extra_msa_stack_iteration,
                extra_msa_stack_input['msa'],
                extra_msa_stack_input['pair'],
                {
                    'msa': batch['extra_msa_mask'],
                    'pair': mask_2d
                },
                is_recompute=self.training)
            extra_msa_stack_output = {
                'msa': extra_msa_act,
                'pair': extra_pair_act}
            extra_msa_stack_input = {
                'msa': extra_msa_stack_output['msa'],
                'pair': extra_msa_stack_output['pair']}

        if "res_is_pocket" in batch and self.config.get("use_pocket_info", False):
            print(f'num pocket res: {int(batch["res_is_pocket"].sum())} {batch["res_is_pocket"].shape}')
            pocket_activations = self.pocket_activations(
                nn.functional.one_hot(batch["res_is_pocket"], num_classes=2)
            ).unsqueeze(1) # [batch, 1, num_res, dim]
            print(f"pocket_activations: {pocket_activations.shape} msa_activations: {msa_activations.shape}")
            msa_activations += pocket_activations

        evoformer_input = {
            'msa': msa_activations,
            'pair': extra_msa_stack_output['pair'],
        }

        evoformer_masks = {
            'msa': batch['msa_mask'],
            'pair': mask_2d,
        }

        if self.config.template.enabled:
            template_features, template_masks = \
                self._template_embedding_1d(
                    batch, self.config.msa_channel)
            evoformer_input['msa'] = paddle.concat(
                [evoformer_input['msa'], template_features], axis=1)
            evoformer_masks['msa'] = paddle.concat(
                [evoformer_masks['msa'], template_masks], axis=1)

        if self.global_config.get('use_pair_pocket_mask', 0.) > 0 and (not 'pair_pocket_mask' in batch):
            print("No pair pocket!!")
        # ==================================================
        #  Main MSA Stack
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 17-18
        # ==================================================
        for idx, evoformer_block in enumerate(self.evoformer_iteration): 
            extra_pair_mask = paddle.ones_like(evoformer_masks["pair"])
            if self.global_config.get('use_pair_pocket_mask', 0.) > 0 \
                and (not self.global_config.get('ipa_pair_pocket_mask', False)):
                if idx <= (self.config.evoformer_num_block // 2) and ('pair_pocket_mask' in batch):
                        extra_pair_mask = batch['pair_pocket_mask']

            msa_act, pair_act = modules.recompute_wrapper(
                evoformer_block,
                evoformer_input['msa'],
                evoformer_input['pair'],
                {"msa": evoformer_masks["msa"],
                    "pair": evoformer_masks["pair"] * extra_pair_mask}, 
                is_recompute=self.training)                

            evoformer_output = {
                'msa': msa_act,
                'pair': pair_act,
            }
            evoformer_input = {
                'msa': evoformer_output['msa'],
                'pair': evoformer_output['pair'],
            }

        msa_activations = evoformer_output['msa']
        pair_activations = evoformer_output['pair']
        single_activations = self.single_activations(msa_activations[:, 0])

        num_seq = batch['msa_feat'].shape[1]
        output = {
            'single': single_activations,
            'pair': pair_activations,
            # Crop away template rows such that they are not used
            # in MaskedMsaHead.
            'msa': msa_activations[:, :num_seq],
            'msa_first_row': msa_activations[:, 0],
        }

        return output

    def _relative_encoding(self, batch):
        """Add relative position encodings.

        For position (i, j), the value is (i-j) clipped to [-k, k] and one-hotted.

        When not using 'use_chain_relative' the residue indices are used as is, e.g.
        for heteromers relative positions will be computed using the positions in
        the corresponding chains.

        When using 'use_chain_relative' we add an extra bin that denotes
        'different chain'. Furthermore we also provide the relative chain index
        (i.e. sym_id) clipped and one-hotted to the network. And an extra feature
        which denotes whether they belong to the same chain type, i.e. it's 0 if
        they are in different heteromer chains and 1 otherwise.

        Args:
            batch: batch.
        Returns:
            Feature embedding using the features as described before.
        """

        rel_feats = []

        asym_id = batch['asym_id']
        asym_id_same = paddle.unsqueeze(asym_id, axis=[-1]) == \
            paddle.unsqueeze(asym_id, axis=[-2])

        pos = batch['residue_index']
        offset = paddle.unsqueeze(pos, axis=[-1]) - \
            paddle.unsqueeze(pos, axis=[-2])
        clipped_offset = paddle.clip(
            offset + self.config.max_relative_idx,
            min=0,
            max=2 * self.config.max_relative_idx)

        if self.config.use_chain_relative:
            final_offset = paddle.where(
                asym_id_same,
                clipped_offset,
                (2 * self.config.max_relative_idx + 1) * \
                paddle.ones_like(clipped_offset))

            rel_pos = nn.functional.one_hot(
                final_offset, 2 * self.config.max_relative_idx + 2)
            rel_feats.append(rel_pos)

            entity_id = batch['entity_id']
            entity_id_same = paddle.unsqueeze(entity_id, axis=[-1]) == \
                paddle.unsqueeze(entity_id, axis=[-2])
            rel_feats.append(paddle.unsqueeze(
                entity_id_same.astype(rel_pos.dtype), axis=[-1]))

            sym_id = batch['sym_id']
            rel_sym_id = paddle.unsqueeze(sym_id, axis=[-1]) - \
                paddle.unsqueeze(sym_id, axis=[-2])

            clipped_rel_chain = paddle.clip(
                rel_sym_id + self.config.max_relative_chain,
                min=0,
                max=2 * self.config.max_relative_chain)
            final_rel_chain = paddle.where(
                entity_id_same,
                clipped_rel_chain,
                (2 * self.config.max_relative_chain + 1) * \
                paddle.ones_like(clipped_rel_chain))
            final_rel_chain = final_rel_chain.astype('int64')

            rel_chain = nn.functional.one_hot(
                final_rel_chain,
                2 * self.config.max_relative_chain + 2)
            rel_feats.append(rel_chain)

        else:
            rel_pos = nn.functional.one_hot(
                clipped_offset, 2 * self.config.max_relative_idx + 1)
            rel_feats.append(rel_pos)

        rel_feat = paddle.concat(rel_feats, axis=-1)
        return self.position_activations(rel_feat)

    def _create_extra_msa_feature(self, batch):
        # 23: 20aa + unknown + gap + bert mask
        msa_1hot = nn.functional.one_hot(batch['extra_msa'], 23)
        msa_feat = [msa_1hot,
                    paddle.unsqueeze(batch['extra_has_deletion'], axis=-1),
                    paddle.unsqueeze(batch['extra_deletion_value'], axis=-1)]
        return paddle.concat(msa_feat, axis=-1)

    def _template_embedding_1d(self, batch, num_channel):
        """Embed templates into an (num_res, num_templates, num_channels) embedding.

        Args:
            batch: A batch containing:
            template_aatype, (num_templates, num_res) aatype for the templates.
            template_all_atom_positions, (num_templates, num_residues, 37, 3) atom
                positions for the templates.
            template_all_atom_mask, (num_templates, num_residues, 37) atom mask for
                each template.
            num_channel: The number of channels in the output.

        Returns:
            An embedding of shape (num_templates, num_res, num_channels) and a mask of
            shape (num_templates, num_res).
        """
        aatype_one_hot = nn.functional.one_hot(batch['template_aatype'], 22)
        ret = all_atom.atom37_to_torsion_angles(
            aatype=batch['template_aatype'],
            all_atom_pos=batch['template_all_atom_positions'],
            all_atom_mask=batch['template_all_atom_masks'],
            # Ensure consistent behaviour during testing:
            placeholder_for_undefined=not self.global_config.zero_init)

        # All torsion angles: \omega, \phi, \psi, \chi_1, ..., \chi_4
        chi_angle_sin_cos = ret['torsion_angles_sin_cos'][..., -4:, :]
        chi_mask = ret['torsion_angles_mask'][..., -4:]

        template_features = paddle.concat(
            [aatype_one_hot,
             chi_angle_sin_cos[..., 0],
             chi_angle_sin_cos[..., 1],
             chi_mask], axis=-1)

        template_mask = chi_mask[..., 0]

        template_activations = nn.functional.relu(
            self.template_single_embedding(template_features))
        template_activations = nn.functional.relu(
            self.template_projection(template_activations))

        return template_activations, template_mask


class TemplateEmbedding(nn.Layer):
    """Embed a set of templates.
    """

    def __init__(self, channel_num, config, global_config):
        super(TemplateEmbedding, self).__init__()
        self.config = config
        self.global_config = global_config

        Linear = paddle.incubate.nn.FusedLinear if self.global_config.fuse_linear else paddle.nn.Linear

        self.single_template_embedding = SingleTemplateEmbedding(
            channel_num, config, global_config)
        self.output_linear = Linear(self.config.num_channels,
                                       channel_num['pair_channel'])

    def forward(self, query_embedding, template_batch,
                padding_mask_2d, multichain_mask_2d):
        """Generate an embedding for a set of templates.

        Args:
        query_embedding: [num_res, num_res, num_channel] a query tensor that will
            be used to attend over the templates to remove the num_templates
            dimension.
        template_batch: A dictionary containing:
            `template_aatype`: [num_templates, num_res] aatype for each template.
            `template_all_atom_positions`: [num_templates, num_res, 37, 3] atom
            positions for all templates.
            `template_all_atom_mask`: [num_templates, num_res, 37] mask for each
            template.
        padding_mask_2d: [num_res, num_res] Pair mask for attention operations.
        multichain_mask_2d: [num_res, num_res] Mask indicating which residue pairs
            are intra-chain, used to mask out residue distance based features
            between chains.

        Returns:
            An embedding of size [num_res, num_res, num_channels]
        """
        num_templates = template_batch['template_mask'].shape[1]
        _, num_res, _, query_channels = query_embedding.shape

        dtype = query_embedding.dtype
        template_mask = template_batch['template_mask']
        template_mask = template_mask.astype(dtype)

        carry = 0.
        for i in range(num_templates):
            # By default, num_templates = 4
            batch0 = {k: paddle.squeeze(v.slice([1], [i], [i+1]), axis=1)
                      for k, v in template_batch.items()}
            carry += self.single_template_embedding(
                query_embedding, batch0, padding_mask_2d,
                multichain_mask_2d)

        embedding = nn.functional.relu(carry / num_templates)
        embedding = self.output_linear(embedding)
        return embedding


class SingleTemplateEmbedding(nn.Layer):
    """Embed a single template."""

    def __init__(self, channel_num, config, global_config):
        super(SingleTemplateEmbedding, self).__init__()
        self.config = config
        self.channel_num = channel_num
        self.global_config = global_config
        num_channels = self.config.num_channels

        Linear = paddle.incubate.nn.FusedLinear if self.global_config.fuse_linear else paddle.nn.Linear

        self.query_embedding_norm = nn.LayerNorm(
            channel_num['template_pair'])

        # fc for template_dgram, [num_res, num_res, num_bins]
        self.template_pair_embedding_0 = Linear(
            self.config.dgram_features.num_bins, num_channels)

        # fc for pseudo_beta_mask_2d, [num_res, num_res]
        self.template_pair_embedding_1 = Linear(1, num_channels)

        # fc for template aatype, [1, num_res, 22]
        self.template_pair_embedding_2 = Linear(22, num_channels)

        # fc for template aatype, [num_res, 1, 22]
        self.template_pair_embedding_3 = Linear(22, num_channels)

        # fc for x, y, z of unit vector, [num_res, num_res]
        self.template_pair_embedding_4 = Linear(1, num_channels)
        self.template_pair_embedding_5 = Linear(1, num_channels)
        self.template_pair_embedding_6 = Linear(1, num_channels)

        # fc for backbone_mask_2d, [num_res, num_res]
        self.template_pair_embedding_7 = Linear(1, num_channels)

        # fc for query_embedding, [um_res, num_res, pair_channel]
        self.template_pair_embedding_8 = Linear(
            channel_num['template_pair'], num_channels)

        self.template_embedding_iteration = nn.LayerList()
        for _ in range(self.config.template_pair_stack.num_block):
            self.template_embedding_iteration.append(
                TemplateEmbeddingIteration(
                    self.channel_num,
                    self.config.template_pair_stack,
                    self.global_config))

        self.output_layer_norm = nn.LayerNorm(
            self.config.num_channels)


    def forward(self, query_embedding, batch,
                padding_mask_2d, multichain_mask_2d):
        """Build the single template embedding graph.

        Args:
        query_embedding: (num_res, num_res, num_channels) - embedding of the
            query sequence/msa.
        batch: a feature dictionary contains:
            * template_aatype: [num_res] aatype for each template.
            * template_all_atom_positions: [num_res, 37, 3] atom positions for all
                templates.
            * template_all_atom_mask: [num_res, 37] mask for each template.
        padding_mask_2d: Padding mask (Note: this doesn't care if a template
            exists, unlike the template_pseudo_beta_mask).
        multichain_mask_2d: A mask indicating intra-chain residue pairs, used
            to mask out between chain distances/features when templates are for
            single chains.

        Returns:
            A template embedding (num_res, num_res, num_channels).
        """
        assert padding_mask_2d.dtype == query_embedding.dtype
        dtype = query_embedding.dtype

        template_mask = batch['template_pseudo_beta_mask']
        # template_mask[..., None] * template_mask[..., None, :]
        template_mask_2d = template_mask.unsqueeze(axis=-1) * template_mask.unsqueeze(axis=-2)
        template_mask_2d = template_mask_2d.astype(dtype)

        # template_dgram = modules.dgram_from_positions(
        #     batch['template_pseudo_beta'],
        #     **self.config.dgram_features)
        # template_dgram = template_dgram.astype(dtype)

        act = self._construct_input(
            query_embedding, batch['template_aatype'],
            batch['template_all_atom_positions'],
            batch['template_all_atom_mask'],
            multichain_mask_2d, dtype)

        for idx, pair_encoder in enumerate(self.template_embedding_iteration):
            act = modules.recompute_wrapper(
                pair_encoder, act, padding_mask_2d,
                is_recompute=self.training)

        act = self.output_layer_norm(act)
        return act

    def _construct_input(self, query_embedding, template_aatype,
                         template_all_atom_positions,
                         template_all_atom_mask,
                         multichain_mask_2d, dtype,
                         debug=False):
        num_res = template_aatype.shape[1]
        template_positions, pseudo_beta_mask = pseudo_beta_fn(
            template_aatype,
            template_all_atom_positions,
            template_all_atom_mask)
        pseudo_beta_mask_2d = pseudo_beta_mask[..., None] * \
            pseudo_beta_mask[..., None, :]
        pseudo_beta_mask_2d *= multichain_mask_2d
        template_dgram = modules.dgram_from_positions(
            template_positions, **self.config.dgram_features)

        template_dgram *= pseudo_beta_mask_2d[..., None]
        template_dgram = template_dgram.astype(dtype)
        pseudo_beta_mask_2d = pseudo_beta_mask_2d.astype(dtype)

        to_concat = [(template_dgram, 1), (pseudo_beta_mask_2d, 0)]

        num_res = template_aatype.shape[1]
        aatype = nn.functional.one_hot(template_aatype, 22)
        aatype = aatype.astype(dtype)
        to_concat.append((aatype[..., None, :, :], 1))
        to_concat.append((aatype[..., None, :], 1))

        # Compute a feature representing the normalized vector between
        # each backbone affine - i.e. in each residues local frame,
        # what direction are each of the other residues.
        unit_vector, backbone_mask_2d = self._calc_unit_vector(
            template_all_atom_positions, template_all_atom_mask,
            multichain_mask_2d, dtype)
        to_concat.extend([(x, 1) for x in unit_vector])
        to_concat.append((backbone_mask_2d, 0))

        query_embedding = self.query_embedding_norm(query_embedding)
        to_concat.append((query_embedding, 1))

        act = 0.
        for i, (x, n_input_dims) in enumerate(to_concat):
            if n_input_dims == 0:
                act += getattr(self, f'template_pair_embedding_{i}')(
                    paddle.unsqueeze(x, axis=[-1]))
            elif n_input_dims == 1:
                act += getattr(self, f'template_pair_embedding_{i}')(x)
            else:
                raise ValueError

        if debug:
            return act, [x[0] for x in to_concat]
        else:
            return act

    def _calc_unit_vector(self, template_all_atom_positions,
                          template_all_atom_mask, multichain_mask_2d,
                          dtype, eps=1e-6):
        # n, ca, c = [residue_constants.atom_order[a]
        #             for a in ('N', 'CA', 'C')]
        # rot, trans = quat_affine.make_transform_from_reference(
        #     n_xyz=template_all_atom_positions[..., n, :],
        #     ca_xyz=template_all_atom_positions[..., ca, :],
        #     c_xyz=template_all_atom_positions[..., c, :])
        # affines = quat_affine.QuatAffine(
        #     quaternion=None,
        #     translation=trans,
        #     rotation=rot)

        # points = [paddle.unsqueeze(x, axis=-2) for x in
        #           paddle.unstack(affines.translation, axis=-1)]
        # # points = paddle.unstack(affines.translation, axis=-1)
        # affine_vec = affines.invert_point(points, extra_dims=1)
        # norm2 = sum([paddle.square(x) for x in affine_vec])
        # inv_distance_scalar = paddle.rsqrt(paddle.maximum(paddle.ones_like(norm2) * eps ** 2, norm2))
        # unit_vector = [x * inv_distance_scalar for x in affine_vec]

        # template_mask = (
        #     template_all_atom_mask[..., n] *
        #     template_all_atom_mask[..., ca] *
        #     template_all_atom_mask[..., c])

        rigid, template_mask = folding_multimer.make_backbone_affine(
            template_all_atom_positions, template_all_atom_mask)
        points = rigid.translation
        rigid_vec = rigid[..., None].inverse().apply_to_point(points)
        unit_vector = rigid_vec.normalized()
        unit_vector = [unit_vector.x, unit_vector.y, unit_vector.z]

        # template_mask[..., None] * template_mask[..., None, :]
        template_mask_2d = template_mask.unsqueeze(axis=-1) * template_mask.unsqueeze(axis=-2)
        template_mask_2d *= multichain_mask_2d
        template_mask_2d = template_mask_2d.astype(dtype)
        # inv_distance_scalar *= template_mask_2d.astype(
        #     inv_distance_scalar.dtype)

        # unit_vector = [(x * inv_distance_scalar)[..., None]
        #                for x in affine_vec]
        unit_vector = [(x * template_mask_2d)[..., None]
                       for x in unit_vector]
        unit_vector = [x.astype(dtype) for x in unit_vector]

        return unit_vector, template_mask_2d


class TemplateEmbeddingIteration(nn.Layer):
    """Single Iteration of Template Embedding."""
    def __init__(self, channel_num, config, global_config):
        super(TemplateEmbeddingIteration, self).__init__()
        self.config = config
        self.global_config = global_config

        channel_num_ = {k: v for k, v in channel_num.items()}
        channel_num_['pair_channel'] = \
            self.config.triangle_multiplication_incoming.num_intermediate_channel

        self.triangle_multiplication_outgoing = \
            modules.TriangleMultiplication(
                channel_num_,
                self.config.triangle_multiplication_outgoing,
                self.global_config)

        use_dropout_nd = self.global_config.get(
            'use_dropout_nd', False)

        p, a = self._parse_dropout_params(
            self.triangle_multiplication_outgoing)
        self.triangle_outgoing_dropout = nn.Dropout(p, axis=a) \
            if not use_dropout_nd else modules.Dropout(p, axis=a)

        self.triangle_multiplication_incoming = \
            modules.TriangleMultiplication(
                channel_num_,
                self.config.triangle_multiplication_incoming,
                self.global_config)

        p, a = self._parse_dropout_params(
            self.triangle_multiplication_incoming)
        self.triangle_incoming_dropout = nn.Dropout(p, axis=a) \
            if not use_dropout_nd else modules.Dropout(p, axis=a)

        self.triangle_attention_starting_node = \
            modules.TriangleAttention(
                channel_num_,
                self.config.triangle_attention_starting_node,
                self.global_config)

        p, a = self._parse_dropout_params(
            self.triangle_attention_starting_node)
        self.triangle_starting_dropout = nn.Dropout(p, axis=a) \
            if not use_dropout_nd else modules.Dropout(p, axis=a)

        self.triangle_attention_ending_node = \
            modules.TriangleAttention(
                channel_num_,
                self.config.triangle_attention_ending_node,
                self.global_config)

        p, a = self._parse_dropout_params(
            self.triangle_attention_ending_node)
        self.triangle_ending_dropout = nn.Dropout(p, axis=a) \
            if not use_dropout_nd else modules.Dropout(p, axis=a)

        self.pair_transition = modules.Transition(
            channel_num_,
            self.config.pair_transition,
            self.global_config,
            is_extra_msa=False,
            transition_type='pair_transition')

        p, a = self._parse_dropout_params(self.pair_transition)
        self.pair_transition_dropout = nn.Dropout(p, axis=a) \
            if not use_dropout_nd else modules.Dropout(p, axis=a)

    def forward(self, act, pair_mask):
        """Build a single iteration of the template embedder.

        Args:
            act: [num_res, num_res, num_channel] Input pairwise activations.
            pair_mask: [num_res, num_res] padding mask.
            is_training: Whether to run in training mode.
            safe_key: Safe pseudo-random generator key.

        Returns:
            [num_res, num_res, num_channel] tensor of activations.
        """
        residual = self.triangle_multiplication_outgoing(act, pair_mask)
        residual = self.triangle_outgoing_dropout(residual)
        act = act + residual

        residual = self.triangle_multiplication_incoming(act, pair_mask)
        residual = self.triangle_incoming_dropout(residual)
        act = act + residual

        residual = self.triangle_attention_starting_node(act, pair_mask)
        residual = self.triangle_starting_dropout(residual)
        act = act + residual

        residual = self.triangle_attention_ending_node(act, pair_mask)
        residual = self.triangle_ending_dropout(residual)
        act = act + residual

        residual = self.pair_transition(act, pair_mask)
        residual = self.pair_transition_dropout(residual)
        act = act + residual

        return act

    def _parse_dropout_params(self, module):
        dropout_rate = 0.0 if self.global_config.deterministic else \
            module.config.dropout_rate
        dropout_axis = None
        if module.config.shared_dropout:
            dropout_axis = {
                'per_row': [0, 2, 3],
                'per_column': [0, 1, 3],
            }[module.config.orientation]

        return dropout_rate, dropout_axis


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    gly_id = paddle.ones_like(aatype) * residue_constants.restype_order['G']
    is_gly = paddle.equal(aatype, gly_id)

    ca_idx = residue_constants.atom_order['CA']
    cb_idx = residue_constants.atom_order['CB']

    n = len(all_atom_positions.shape)
    pseudo_beta = paddle.where(
        paddle.tile(paddle.unsqueeze(is_gly, axis=-1),
                    [1] * len(is_gly.shape) + [3]),
        paddle.squeeze(
            all_atom_positions.slice([n-2], [ca_idx], [ca_idx+1]),
            axis=-2),
        paddle.squeeze(
            all_atom_positions.slice([n-2], [cb_idx], [cb_idx+1]),
            axis=-2))

    if all_atom_masks is not None:
        m = len(all_atom_masks.shape)
        pseudo_beta_mask = paddle.where(
            is_gly,
            paddle.squeeze(
                all_atom_masks.slice([m-1], [ca_idx], [ca_idx+1]),
                axis=-1),
            paddle.squeeze(
                all_atom_masks.slice([m-1], [cb_idx], [cb_idx+1]),
                axis=-1))
        pseudo_beta_mask = paddle.squeeze(pseudo_beta_mask, axis=-1)
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta
