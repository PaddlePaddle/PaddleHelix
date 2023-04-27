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

"""Modules."""

import gc
import numpy as np

import paddle
import paddle.nn as nn
from paddle.fluid.framework import _dygraph_tracer
from paddle.distributed.fleet.utils import recompute
try:
    from paddle import _legacy_C_ops as _C_ops
except:
    from paddle import _C_ops

from alphafold_paddle.common import residue_constants
from alphafold_paddle.model.utils import mask_mean, subbatch
from alphafold_paddle.model import folding, lddt, quat_affine, all_atom
from alphafold_paddle.model.utils import init_gate_linear, init_final_linear
from utils.utils import get_structure_module_bf16_op_list

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


def softmax_cross_entropy(logits, labels):
    """Computes softmax cross entropy given logits and one-hot class labels."""
    loss = -paddle.sum(labels * paddle.nn.functional.log_softmax(logits), axis=-1)
    return loss


def sigmoid_cross_entropy(logits, labels):
    """Computes sigmoid cross entropy given logits and multiple class labels."""
    log_p = paddle.nn.functional.log_sigmoid(logits)
    # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter is more numerically stable
    log_not_p = paddle.nn.functional.log_sigmoid(-logits)
    loss = -labels * log_p - (1. - labels) * log_not_p
    return loss


class Dropout(nn.Layer):
    def __init__(self, p=0.5, axis=None, mode="upscale_in_train", name=None):
        super(Dropout, self).__init__()

        if not isinstance(p, (float, int)):
            raise TypeError("p argument should be a number")
        if p < 0 or p > 1:
            raise ValueError("p argument should between 0 and 1")

        mode = 'downgrade_in_infer' if mode == 'downscale_in_infer' else mode  #semantic transfer
        if mode not in ('downscale_in_infer', 'upscale_in_train'):
            raise ValueError(
                "mode argument should be 'downscale_in_infer' or 'upscale_in_train'"
            )

        if axis and not isinstance(axis, (int, list, tuple)):
            raise TypeError("datatype of axis argument should be int or list")

        self.p = p
        self.axis = axis
        self.mode = mode
        self.name = name

    def forward(self, input):
        # fast return for p == 0
        if self.p == 0:
            return input

        if self.axis == None: 
            out = nn.functional.dropout(input,
                            p=self.p,
                            axis=self.axis,
                            training=self.training,
                            mode=self.mode,
                            name=self.name)
        else:
            seed = None
            drop_axes = [self.axis] if isinstance(self.axis, int) else list(self.axis)
            if paddle.static.default_main_program().random_seed != 0:
                seed = paddle.static.default_main_program().random_seed

            out, mask = _C_ops.dropout_nd(input, 'dropout_prob', self.p, 'is_test',
                                                    not self.training, 'fix_seed', seed
                                                    is not None, 'seed',
                                                    seed if seed is not None else 0,
                                                    'dropout_implementation', self.mode, 'axis',
                                                    drop_axes)

        return out

    def extra_repr(self):
        name_str = ', name={}'.format(self.name) if self.name else ''
        return 'p={}, axis={}, mode={}{}'.format(self.p, self.axis, self.mode,
                                                 name_str)


class AlphaFold(nn.Layer):
    """AlphaFold model with recycling.

    Jumper et al. (2021) Suppl. Alg. 2 "Inference"
    """
    def __init__(self, channel_num, config):
        super(AlphaFold, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = config.global_config

        self.alphafold_iteration = AlphaFoldIteration(
            self.channel_num, self.config, self.global_config)

    def forward(self,
                batch,
                label,
                ensemble_representations=False,
                return_representations=False,
                compute_loss=True):
        """Run the AlphaFold model.

        Arguments:
            batch: Dictionary with inputs to the AlphaFold model.
            ensemble_representations: Whether to use ensembling of representations.
            return_representations: Whether to also return the intermediate
                representations.

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
            return self.alphafold_iteration(
                ensembled_batch, label, non_ensembled_batch,
                compute_loss=compute_loss,
                ensemble_representations=ensemble_representations)

        if self.config.num_recycle:
            # aatype: (B, E, N), zeros_bn: (B, N)
            zeros_bn = paddle.zeros_like(batch['aatype'][:, 0], dtype='float32')

            emb_config = self.config.embeddings_and_evoformer

            # if not self.training: # for inference
            if not self.training and self.global_config.low_memory is True:
                prev = {
                    'prev_pos': paddle.tile(
                        zeros_bn[..., None, None],
                        [1, 1, residue_constants.atom_type_num, 3]),
                    'prev_msa_first_row': paddle.tile(
                        zeros_bn[..., None],
                        [1, 1, emb_config.msa_channel]),
                    'prev_pair': paddle.tile(
                        zeros_bn[..., None, None].cpu(),
                        [1, 1, num_residues, emb_config.pair_channel]).astype(paddle.bfloat16),
                }
            else:
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
                ret = _run_single_recycling(prev, recycle_idx, compute_loss=False)
                prev = _get_prev(ret)
                # if not self.training:
                if not self.training and self.global_config.low_memory is True:
                    del ret
                    gc.collect()

        else:
            prev = {}
            num_iter = 0

        return _run_single_recycling(prev, num_iter, compute_loss=compute_loss)


class AlphaFoldIteration(nn.Layer):
    """A single recycling iteration of AlphaFold architecture.

    Computes ensembled (averaged) representations from the provided features.
    These representations are then passed to the various heads
    that have been requested by the configuration file. Each head also returns a
    loss which is combined as a weighted sum to produce the total loss.

    Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 3-22
    """

    def __init__(self, channel_num, config, global_config):
        super(AlphaFoldIteration, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        # copy these config for later usage
        self.channel_num['extra_msa_channel'] = config.embeddings_and_evoformer.extra_msa_channel
        self.channel_num['msa_channel'] = config.embeddings_and_evoformer.msa_channel
        self.channel_num['pair_channel'] = config.embeddings_and_evoformer.pair_channel
        self.channel_num['seq_channel'] = config.embeddings_and_evoformer.seq_channel

        if self.global_config.get('dist_model', False):
            from ppfleetx.models.protein_folding.evoformer import DistEmbeddingsAndEvoformer 
            self.evoformer = DistEmbeddingsAndEvoformer(
                self.channel_num, self.config.embeddings_and_evoformer,
                self.global_config)
        else:
            self.evoformer = EmbeddingsAndEvoformer(
                self.channel_num, self.config.embeddings_and_evoformer,
                self.global_config)

        Head_modules = {
            'masked_msa': MaskedMsaHead,
            'distogram': DistogramHead,
            'structure_module': folding.StructureModule,
            'predicted_lddt': PredictedLDDTHead,
            'predicted_aligned_error': PredictedAlignedErrorHead,
            'experimentally_resolved': ExperimentallyResolvedHead,   # finetune loss
        }

        self.used_heads = []
        for head_name, head_config in sorted(self.config.heads.items()):
            if head_name not in Head_modules:
                continue

            self.used_heads.append(head_name)
            module = Head_modules[head_name](
                self.channel_num, head_config, self.global_config)

            head_name_ = Head_names.get(head_name, head_name)
            setattr(self, head_name_, module)

    def forward(self,
                ensembled_batch,
                label,
                non_ensembled_batch,
                compute_loss=False,
                ensemble_representations=False):
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
        # MaskedMSAHead is apply on batch0
        label['bert_mask'] = batch0['bert_mask']
        label['true_msa'] = batch0['true_msa']
        label['residue_index'] = batch0['residue_index']

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

        def loss(head_name_, head_config, ret, head_name, filter_ret=True):
            if filter_ret:
                value = ret[head_name]
            else:
                value = ret
            loss_output = getattr(self, head_name_).loss(value, label)
            ret[head_name].update(loss_output)
            loss = head_config.weight * ret[head_name]['loss']
            return loss

        def _forward_heads(representations, ret, batch0):
            total_loss = 0.
            for head_name, head_config in self._get_heads():
                head_name_ = Head_names.get(head_name, head_name)
                # Skip PredictedLDDTHead and PredictedAlignedErrorHead until
                # StructureModule is executed.
                if head_name in ('predicted_lddt', 'predicted_aligned_error'):
                    continue
                else:
                    ret[head_name] = getattr(self, head_name_)(representations, batch0)
                    if 'representations' in ret[head_name]:
                    # Extra representations from the head. Used by the
                    # structure module to provide activations for the PredictedLDDTHead.
                        representations.update(ret[head_name].pop('representations'))
                if compute_loss:
                    total_loss += loss(head_name_, head_config, ret, head_name)

            if self.config.heads.get('predicted_lddt.weight', 0.0):
                # Add PredictedLDDTHead after StructureModule executes.
                head_name = 'predicted_lddt'
                # Feed all previous results to give access to structure_module result.
                head_name_ = Head_names.get(head_name, head_name)
                head_config = self.config.heads[head_name]
                ret[head_name] = getattr(self, head_name_)(representations, batch0)
                if compute_loss:
                    total_loss += loss(head_name_, head_config, ret, head_name, filter_ret=False)

            if ('predicted_aligned_error' in self.config.heads
                    and self.config.heads.get('predicted_aligned_error.weight', 0.0)):
                # Add PredictedAlignedErrorHead after StructureModule executes.
                head_name = 'predicted_aligned_error'
                # Feed all previous results to give access to structure_module result.
                head_config = self.config.heads[head_name]
                head_name_ = Head_names.get(head_name, head_name)
                ret[head_name] = getattr(self, head_name_)(representations, batch0)
                if compute_loss:
                    total_loss += loss(head_name_, head_config, ret, head_name, filter_ret=False)

            return ret, total_loss

        # if not self.training:
        if not self.training and self.global_config.low_memory is True:
            black_list, white_list = get_structure_module_bf16_op_list()
            with paddle.amp.auto_cast(level='O1', custom_white_list=white_list, custom_black_list=black_list, dtype='bfloat16'):
                ret, total_loss = _forward_heads(representations, ret, batch0)
        else:
            tracer = _dygraph_tracer()
            if tracer._amp_dtype == "bfloat16":
                with paddle.amp.auto_cast(enable=False):
                    for key, value in representations.items():
                        if value.dtype in [paddle.fluid.core.VarDesc.VarType.BF16]:
                            temp_value = value.cast('float32')
                            temp_value.stop_gradient = value.stop_gradient
                            representations[key] = temp_value
                    for key, value in batch0.items():
                        if value.dtype in [paddle.fluid.core.VarDesc.VarType.BF16]:
                            temp_value = value.cast('float32')
                            temp_value.stop_gradient = value.stop_gradient
                            batch0[key] = temp_value
                    ret, total_loss = _forward_heads(representations, ret, batch0)

            else:
                ret, total_loss = _forward_heads(representations, ret, batch0)

        if compute_loss:
            return ret, total_loss
        else:
            return ret

    def _get_heads(self):
        assert 'structure_module' in self.used_heads
        head_names = [h for h in self.used_heads]

        for k in head_names:
            yield k, self.config.heads[k]


class Attention(nn.Layer):
    """Multihead attention."""

    def __init__(self, config, global_config, q_dim, kv_dim, output_dim):
        super(Attention, self).__init__()
        self.config = config
        self.global_config = global_config

        num_head = self.config.num_head
        key_dim = self.config.get('key_dim', q_dim)
        value_dim = self.config.get('value_dim', kv_dim)

        # TODO(GuoxiaWang): delete non fuse_attention related code on dcu
        self.fuse_attention = self.global_config.fuse_attention
        self.merge_qkv = (q_dim == kv_dim)

        assert key_dim % num_head == 0
        assert value_dim % num_head == 0
        key_dim = key_dim // num_head
        value_dim = value_dim // num_head

        self.key_dim = key_dim
        self.value_dim = value_dim

        self.qkv_w = None
        self.query_w = None
        self.key_w = None
        self.value_w = None
        if self.merge_qkv and self.fuse_attention:
            self.qkv_w = paddle.create_parameter(
                [3, num_head, key_dim, q_dim], 'float32',
                default_initializer=nn.initializer.XavierUniform())
        else:
            self.query_w = paddle.create_parameter(
                [q_dim, num_head, key_dim], 'float32',
                default_initializer=nn.initializer.XavierUniform())
            self.key_w = paddle.create_parameter(
                [kv_dim, num_head, key_dim], 'float32',
                default_initializer=nn.initializer.XavierUniform())
            self.value_w = paddle.create_parameter(
                [kv_dim, num_head, value_dim], 'float32',
                default_initializer=nn.initializer.XavierUniform())

        self.gating_w = None
        self.gating_b = None
        if self.config.gating:
            self.gating_w = paddle.create_parameter(
                [q_dim, num_head, value_dim], 'float32',
                default_initializer=nn.initializer.Constant(0.0))
            self.gating_b = paddle.create_parameter(
                [num_head, value_dim], 'float32',
                default_initializer=nn.initializer.Constant(1.0))

        if self.global_config.zero_init:
            init = nn.initializer.Constant(0.0)
        else:
            init = nn.initializer.XavierUniform()

        self.output_w = paddle.create_parameter(
            [num_head, value_dim, output_dim], 'float32',
            default_initializer=init)
        self.output_b = paddle.create_parameter(
            [output_dim], 'float32',
            default_initializer=nn.initializer.Constant(0.0))

    def forward(self, q_data, m_data, bias, nonbatched_bias=None):
        """Builds Attention module.
        Arguments:
            q_data: A tensor of queries, shape [batch, row_size, N_queries, q_channels].
            m_data: A tensor of memories from which the keys and values are
                projected, shape [batch, row_size, N_keys, m_channels].
            bias: A bias for the attention, shape [batch, row_size, num_head, N_queries, N_keys].
            nonbatched_bias: Shared bias, shape [N_queries, N_keys].

        Returns:
            A float32 tensor of shape [batch_size, row_size, N_queries, output_dim].
        """
        if self.fuse_attention:
            if nonbatched_bias is not None:
                nonbatched_bias = paddle.unsqueeze(nonbatched_bias, axis=1)
            _, _, _, _, _, _, _, output = _C_ops.fused_gate_attention(
                q_data, m_data, self.query_w, self.key_w, self.value_w, self.qkv_w, nonbatched_bias, bias, self.gating_w, self.gating_b,
                self.output_w, self.output_b, 'has_gating', self.config.gating, 'merge_qkv', self.merge_qkv)
        else:
            c = self.key_dim ** (-0.5)
            q = paddle.einsum('nbqa,ahc->nbqhc', q_data, self.query_w) * c
            k = paddle.einsum('nbka,ahc->nbkhc', m_data, self.key_w)
            v = paddle.einsum('nbka,ahc->nbkhc', m_data, self.value_w)
            logits = paddle.einsum('nbqhc,nbkhc->nbhqk', q, k) + bias

            if nonbatched_bias is not None:
                logits += paddle.unsqueeze(nonbatched_bias, axis=1)

            weights = nn.functional.softmax(logits)
            weighted_avg = paddle.einsum('nbhqk,nbkhc->nbqhc', weights, v)

            if self.config.gating:
                gate_values = paddle.einsum('nbqc,chv->nbqhv', q_data,
                                            self.gating_w) + self.gating_b
                gate_values = nn.functional.sigmoid(gate_values)
                weighted_avg *= gate_values

            output = paddle.einsum('nbqhc,hco->nbqo', weighted_avg,
                                self.output_w) + self.output_b 
        return output


class GlobalAttention(nn.Layer):
    """Global attention.

    Jumper et al. (2021) Suppl. Alg. 19 "MSAColumnGlobalAttention" lines 2-7
    """

    def __init__(self, config, global_config, q_dim, kv_dim, output_dim):
        super(GlobalAttention, self).__init__()
        self.config = config
        self.global_config = global_config

        num_head = self.config.num_head
        key_dim = self.config.get('key_dim', q_dim)
        value_dim = self.config.get('value_dim', kv_dim)

        assert key_dim % num_head == 0
        assert value_dim % num_head == 0
        key_dim = key_dim // num_head
        value_dim = value_dim // num_head

        self.key_dim = key_dim
        self.value_dim = value_dim

        self.query_w = paddle.create_parameter(
            [q_dim, num_head, key_dim], 'float32',
            default_initializer=nn.initializer.XavierUniform())
        self.key_w = paddle.create_parameter(
            [kv_dim, key_dim], 'float32',
            default_initializer=nn.initializer.XavierUniform())
        self.value_w = paddle.create_parameter(
            [kv_dim, value_dim], 'float32',
            default_initializer=nn.initializer.XavierUniform())

        if self.config.gating:
            self.gating_w = paddle.create_parameter(
                [q_dim, num_head, value_dim], 'float32',
                default_initializer=nn.initializer.Constant(0.0))
            self.gating_b = paddle.create_parameter(
                [num_head, value_dim], 'float32',
                default_initializer=nn.initializer.Constant(1.0))

        if self.global_config.zero_init:
            init = nn.initializer.Constant(0.0)
        else:
            init = nn.initializer.XavierUniform()

        self.output_w = paddle.create_parameter(
            [num_head, value_dim, output_dim], 'float32',
            default_initializer=init)
        self.output_b = paddle.create_parameter(
            [output_dim], 'float32',
            default_initializer=nn.initializer.Constant(0.0))

    def forward(self, q_data, m_data, q_mask):
        k = paddle.einsum('nbka,ac->nbkc', m_data, self.key_w)
        v = paddle.einsum('nbka,ac->nbkc', m_data, self.value_w)

        # NOTE: differ from non-global version using q_avg for attn
        q_avg = mask_mean(q_mask, q_data, axis=2)
        c = self.key_dim ** (-0.5)
        q = paddle.einsum('nba,ahc->nbhc', q_avg, self.query_w) * c

        q_mask_ = paddle.unsqueeze(q_mask, axis=2)[..., 0]
        bias = 1e9 * (q_mask_ - 1.)

        logits = paddle.einsum('nbhc,nbkc->nbhk', q, k) + bias
        weights = nn.functional.softmax(logits)
        weighted_avg = paddle.einsum('nbhk,nbkc->nbhc', weights, v)

        if self.config.gating:
            gate_values = paddle.einsum('nbqc,chv->nbqhv', q_data,
                                        self.gating_w) + self.gating_b
            gate_values = nn.functional.sigmoid(gate_values)
            weighted_avg = paddle.unsqueeze(weighted_avg, axis=2)
            weighted_avg *= gate_values

            output = paddle.einsum('nbqhc,hco->nbqo', weighted_avg,
                                   self.output_w) + self.output_b
        else:
            output = paddle.einsum('nbhc,hco->nbo', weighted_avg,
                                   self.output_w) + self.output_b
            output = paddle.unsqueeze(output, axis=-1)

        return output


class MSARowAttentionWithPairBias(nn.Layer):
    """MSA per-row attention biased by the pair representation.

    Jumper et al. (2021) Suppl. Alg. 7 "MSARowAttentionWithPairBias"
    """

    def __init__(self, channel_num, config, global_config, is_extra_msa):
        super(MSARowAttentionWithPairBias, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config
        self.is_extra_msa = is_extra_msa
        assert config.orientation == 'per_row'

        if is_extra_msa:
            self.query_norm = nn.LayerNorm(channel_num['extra_msa_channel'])
        else:
            self.query_norm = nn.LayerNorm(channel_num['msa_channel'])

        self.feat_2d_norm = nn.LayerNorm(channel_num['pair_channel'])
        self.feat_2d_weights = paddle.create_parameter(
            [channel_num['pair_channel'], self.config.num_head], 'float32',
            default_initializer=nn.initializer.Normal(
                std=1. / np.sqrt(channel_num['pair_channel'])))

        if is_extra_msa:
            extra_msa_channel = channel_num['extra_msa_channel']
            self.attention = Attention(
                self.config, self.global_config,
                extra_msa_channel, extra_msa_channel, extra_msa_channel)
        else:
            msa_channel = channel_num['msa_channel']
            self.attention = Attention(
                self.config, self.global_config,
                msa_channel, msa_channel, msa_channel)

    def forward(self, msa_act, msa_mask, pair_act):

        pair_act = self.feat_2d_norm(pair_act)
        
        nonbatched_bias = paddle.einsum(
            'nqkc,ch->nhqk', pair_act, self.feat_2d_weights)
        
        bias = 1e9 * (msa_mask - 1.)
        bias = paddle.unsqueeze(bias, axis=[2, 3])
        msa_act = self.query_norm(msa_act)

        if not self.training or (self.is_extra_msa and self.config.use_subbatch):
            # low memory mode using subbatch
            subbatch_size = self.config.subbatch_size
            if not self.training:
                subbatch_size = self.global_config.subbatch_size
            sb_attn = subbatch(self.attention, [0, 1, 2], [1, 1, 1],
                               subbatch_size, 1, same_arg_idx={1: 0})
            msa_act = sb_attn(msa_act, msa_act, bias, nonbatched_bias)
        else:
            msa_act = self.attention(msa_act, msa_act, bias, nonbatched_bias)

        return msa_act


class MSAColumnGlobalAttention(nn.Layer):
    """MSA per-column global attention.

    Jumper et al. (2021) Suppl. Alg. 19 "MSAColumnGlobalAttention"
    """

    def __init__(self, channel_num, config, global_config):
        super(MSAColumnGlobalAttention, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config
        assert config.orientation == 'per_column'

        extra_msa_channel = channel_num['extra_msa_channel']
        self.query_norm = nn.LayerNorm(extra_msa_channel)
        self.attention = GlobalAttention(
            self.config, self.global_config,
            extra_msa_channel, extra_msa_channel, extra_msa_channel)

    def forward(self, msa_act, msa_mask):

        msa_act = paddle.transpose(msa_act, [0, 2, 1, 3])
        msa_mask = paddle.transpose(msa_mask, [0, 2, 1])

        bias = 1e9 * (msa_mask - 1.)
        bias = paddle.unsqueeze(bias, axis=[2, 3])

        msa_mask = paddle.unsqueeze(msa_mask, axis=-1)
        msa_act = self.query_norm(msa_act)

        if not self.training:
            # low memory mode using subbatch
            sb_attn = subbatch(self.attention, [0, 1, 2], [1, 1, 1],
                               self.global_config.subbatch_size, 1, same_arg_idx={1: 0})
            msa_act = sb_attn(msa_act, msa_act, msa_mask)
        else:
            msa_act = self.attention(msa_act, msa_act, msa_mask)

        msa_act = paddle.transpose(msa_act, [0, 2, 1, 3])
        return msa_act


class MSAColumnAttention(nn.Layer):
    """MSA per-column attention.

    Jumper et al. (2021) Suppl. Alg. 8 "MSAColumnAttention"
    """

    def __init__(self, channel_num, config, global_config):
        super(MSAColumnAttention, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config
        assert config.orientation == 'per_column'

        msa_channel = channel_num['msa_channel']
        self.query_norm = nn.LayerNorm(msa_channel)
        self.attention = Attention(
            self.config, self.global_config,
            msa_channel, msa_channel, msa_channel)

    def forward(self, msa_act, msa_mask):

        msa_act = paddle.transpose(msa_act, [0, 2, 1, 3])
        msa_mask = paddle.transpose(msa_mask, [0, 2, 1])

        bias = 1e9 * (msa_mask - 1.)
        bias = paddle.unsqueeze(bias, axis=[2, 3])

        msa_act = self.query_norm(msa_act)
        if not self.training:
            # low memory mode using subbatch
            sb_attn = subbatch(self.attention, [0, 1, 2], [1, 1, 1],
                               self.global_config.subbatch_size, 1, same_arg_idx={1: 0})
            msa_act = sb_attn(msa_act, msa_act, bias)
        else:
            msa_act = self.attention(msa_act, msa_act, bias)

        msa_act = paddle.transpose(msa_act, [0, 2, 1, 3])
        return msa_act


class Transition(nn.Layer):
    """Transition layer.

    Jumper et al. (2021) Suppl. Alg. 9 "MSATransition"
    Jumper et al. (2021) Suppl. Alg. 15 "PairTransition"
    """

    def __init__(self, channel_num, config, global_config, is_extra_msa,
                 transition_type):
        super(Transition, self).__init__()
        assert transition_type in ['msa_transition', 'pair_transition']
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config
        self.is_extra_msa = is_extra_msa
        self.transition_type = transition_type

        if transition_type == 'msa_transition' and is_extra_msa:
            in_dim = channel_num['extra_msa_channel']
        elif transition_type == 'msa_transition' and not is_extra_msa:
            in_dim = channel_num['msa_channel']
        elif transition_type == 'pair_transition':
            in_dim = channel_num['pair_channel']

        self.input_layer_norm = nn.LayerNorm(in_dim)
        self.transition1 = nn.Linear(
            in_dim, int(in_dim * self.config.num_intermediate_factor),
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.KaimingNormal()))

        if self.global_config.zero_init:
            last_init = nn.initializer.Constant(0.0)
        else:
            last_init = nn.initializer.TruncatedNormal()

        self.transition2 = nn.Linear(
            int(in_dim * self.config.num_intermediate_factor), in_dim,
            weight_attr=paddle.ParamAttr(initializer=last_init))

    def forward(self, act, mask):
        act = self.input_layer_norm(act)

        def transition_module(x):
            x = self.transition1(x)
            x = nn.functional.relu(x)
            x = self.transition2(x)
            return x

        if not self.training:
            # low memory mode using subbatch
            sb_transition = subbatch(transition_module, [0], [1],
                                 self.global_config.subbatch_size, 1)
            act = sb_transition(act)
        else:
            act = transition_module(act)

        return act


class MaskedMsaHead(nn.Layer):
    """Head to predict MSA at the masked locations.

    The MaskedMsaHead employs a BERT-style objective to reconstruct a masked
    version of the full MSA, based on a linear projection of
    the MSA representation.
    Jumper et al. (2021) Suppl. Sec. 1.9.9 "Masked MSA prediction"
    """
    def __init__(self, channel_num, config, global_config, name='masked_msa_head'):
        super(MaskedMsaHead, self).__init__()
        self.config = config
        self.global_config = global_config
        self.num_output = config.num_output
        self.logits = nn.Linear(channel_num['msa_channel'], self.num_output, name='logits')

    def forward(self, representations, batch):
        """Builds MaskedMsaHead module.

        Arguments:
        representations: Dictionary of representations, must contain:
            * 'msa': MSA representation, shape [batch, N_seq, N_res, c_m].
        batch: Batch, unused.

        Returns:
        Dictionary containing:
            * 'logits': logits of shape [batch, N_seq, N_res, N_aatype] with
                (unnormalized) log probabilies of predicted aatype at position.
        """
        del batch
        logits = self.logits(representations['msa'])
        return {'logits': logits}

    def loss(self, value, batch):
        errors = softmax_cross_entropy(
            labels=paddle.nn.functional.one_hot(batch['true_msa'], num_classes=self.num_output),
            logits=value['logits'])
        loss = (paddle.sum(errors * paddle.cast(batch['bert_mask'], dtype=errors.dtype), axis=[-2, -1]) /
                (1e-8 + paddle.sum(batch['bert_mask'], axis=[-2, -1])))
        return {'loss': loss}


class PredictedLDDTHead(nn.Layer):
    """Head to predict the per-residue LDDT to be used as a confidence measure.

    Jumper et al. (2021) Suppl. Sec. 1.9.6 "Model confidence prediction (pLDDT)"
    Jumper et al. (2021) Suppl. Alg. 29 "predictPerResidueLDDT_Ca"
    """

    def __init__(self, channel_num, config, global_config, name='predicted_lddt_head'):
        super(PredictedLDDTHead, self).__init__()
        self.config = config
        self.global_config = global_config

        self.input_layer_norm = nn.LayerNorm(channel_num['seq_channel'],
                                             name='input_layer_norm')
        self.act_0 = nn.Linear(channel_num['seq_channel'],
                               self.config.num_channels, name='act_0')
        self.act_1 = nn.Linear(self.config.num_channels,
                               self.config.num_channels, name='act_1')
        self.logits = nn.Linear(self.config.num_channels,
                               self.config.num_bins, name='logits')

    def forward(self, representations, batch):
        """Builds PredictedLDDTHead module.

        Arguments:
        representations: Dictionary of representations, must contain:
            * 'structure_module': Single representation from the structure module,
                shape [n_batch, N_res, c_s].

        Returns:
        Dictionary containing :
            * 'logits': logits of shape [n_batch, N_res, N_bins] with
                (unnormalized) log probabilies of binned predicted lDDT.
        """
        act = representations['structure_module']
        act = self.input_layer_norm(act)
        act = nn.functional.relu(self.act_0(act))
        act = nn.functional.relu(self.act_1(act))
        logits = self.logits(act)

        return dict(logits=logits)

    def loss(self, value, batch):
        # Shape (n_batch, num_res, 37, 3)
        pred_all_atom_pos = value['structure_module']['final_atom_positions']
        # Shape (n_batch, num_res, 37, 3)
        true_all_atom_pos = paddle.cast(batch['all_atom_positions'], 'float32')
        # Shape (n_batch, num_res, 37)
        all_atom_mask = paddle.cast(batch['all_atom_mask'], 'float32')

        # Shape (batch_size, num_res)
        lddt_ca = lddt.lddt(
            # Shape (batch_size, num_res, 3)
            predicted_points=pred_all_atom_pos[:, :, 1, :],
            # Shape (batch_size, num_res, 3)
            true_points=true_all_atom_pos[:, :, 1, :],
            # Shape (batch_size, num_res, 1)
            true_points_mask=all_atom_mask[:, :, 1:2],
            cutoff=15.,
            per_residue=True)
        lddt_ca = lddt_ca.detach()

        # Shape (batch_size, num_res)
        num_bins = self.config.num_bins
        bin_index = paddle.floor(lddt_ca * num_bins)

        # protect against out of range for lddt_ca == 1
        bin_index = paddle.minimum(bin_index, paddle.to_tensor(num_bins - 1, dtype='float32'))
        lddt_ca_one_hot = paddle.nn.functional.one_hot(paddle.cast(bin_index, 'int64'), num_classes=num_bins)

        # Shape (n_batch, num_res, num_channel)
        logits = value['predicted_lddt']['logits']
        errors = softmax_cross_entropy(labels=lddt_ca_one_hot, logits=logits)

        # Shape (num_res,)
        mask_ca = all_atom_mask[:, :, residue_constants.atom_order['CA']]
        mask_ca = paddle.to_tensor(mask_ca, dtype='float32')
        loss = paddle.sum(errors * mask_ca, axis=-1) / (paddle.sum(mask_ca, axis=-1) + 1e-8)

        if self.config.filter_by_resolution:
            # NMR & distillation have resolution = 0
            resolution = paddle.squeeze(batch['resolution'], axis=-1)
            loss *= paddle.cast((resolution >= self.config.min_resolution)
                    & (resolution <= self.config.max_resolution), 'float32')
        output = {'loss': loss}
        return output


class PredictedAlignedErrorHead(nn.Layer):
    """Head to predict the distance errors in the backbone alignment frames.

    Can be used to compute predicted TM-Score.
    Jumper et al. (2021) Suppl. Sec. 1.9.7 "TM-score prediction"
    """
    def __init__(self, channel_num, config, global_config,
                 name='predicted_aligned_error_head'):
        super(PredictedAlignedErrorHead, self).__init__()
        self.config = config
        self.global_config = global_config

        self.logits = nn.Linear(channel_num['pair_channel'],
                                self.config.num_bins, name='logits')

    def forward(self, representations, batch):
        """Builds PredictedAlignedErrorHead module.

        Arguments:
            representations: Dictionary of representations, must contain:
                * 'pair': pair representation, shape [B, N_res, N_res, c_z].
            batch: Batch, unused.

        Returns:
            Dictionary containing:
                * logits: logits for aligned error, shape [B, N_res, N_res, N_bins].
                * bin_breaks: array containing bin breaks, shape [N_bins - 1].
        """
        logits = self.logits(representations['pair'])
        breaks = paddle.linspace(0., self.config.max_error_bin,
                                 self.config.num_bins-1)

        return dict(logits=logits, breaks=breaks)

    def loss(self, value, batch):
        # Shape (B, num_res, 7)
        predicted_affine = quat_affine.QuatAffine.from_tensor(
            value['structure_module']['final_affines'])
        # Shape (B, num_res, 7)
        true_rot = paddle.to_tensor(batch['backbone_affine_tensor_rot'], dtype='float32')
        true_trans = paddle.to_tensor(batch['backbone_affine_tensor_trans'], dtype='float32')
        true_affine = quat_affine.QuatAffine(
            quaternion=None,
            translation=true_trans,
            rotation=true_rot)
        # Shape (B, num_res)
        mask = batch['backbone_affine_mask']
        # Shape (B, num_res, num_res)
        square_mask = mask[..., None] * mask[:, None, :]
        num_bins = self.config.num_bins
        # (num_bins - 1)
        breaks = value['predicted_aligned_error']['breaks']
        # (B, num_res, num_res, num_bins)
        logits = value['predicted_aligned_error']['logits']

        # Compute the squared error for each alignment.
        def _local_frame_points(affine):
            points = [paddle.unsqueeze(x, axis=-2) for x in 
                            paddle.unstack(affine.translation, axis=-1)]
            return affine.invert_point(points, extra_dims=1)
        error_dist2_xyz = [
            paddle.square(a - b)
            for a, b in zip(_local_frame_points(predicted_affine),
                            _local_frame_points(true_affine))]
        error_dist2 = sum(error_dist2_xyz)
        # Shape (B, num_res, num_res)
        # First num_res are alignment frames, second num_res are the residues.
        error_dist2 = error_dist2.detach()

        sq_breaks = paddle.square(breaks)
        true_bins = paddle.sum(paddle.cast((error_dist2[..., None] > sq_breaks), 'int32'), axis=-1)

        errors = softmax_cross_entropy(
            labels=paddle.nn.functional.one_hot(true_bins, num_classes=num_bins), logits=logits)

        loss = (paddle.sum(errors * square_mask, axis=[-2, -1]) /
            (1e-8 + paddle.sum(square_mask, axis=[-2, -1])))

        if self.config.filter_by_resolution:
            # NMR & distillation have resolution = 0
            resolution = paddle.squeeze(batch['resolution'], axis=-1)
            loss *= paddle.cast((resolution >= self.config.min_resolution)
                    & (resolution <= self.config.max_resolution), 'float32')

        output = {'loss': loss}
        return output


class ExperimentallyResolvedHead(nn.Layer):
    """Predicts if an atom is experimentally resolved in a high-res structure.

    Only trained on high-resolution X-ray crystals & cryo-EM.
    Jumper et al. (2021) Suppl. Sec. 1.9.10 '"Experimentally resolved" prediction'
    """

    def __init__(self, channel_num, config, global_config, name='experimentally_resolved_head'):
        super(ExperimentallyResolvedHead, self).__init__()
        self.config = config
        self.global_config = global_config
        self.logits = nn.Linear(channel_num['seq_channel'], 37, name='logits')

    def forward(self, representations, batch):
        """Builds ExperimentallyResolvedHead module.

        Arguments:
        representations: Dictionary of representations, must contain:
            * 'single': Single representation, shape [B, N_res, c_s].
        batch: Batch, unused.

        Returns:
        Dictionary containing:
            * 'logits': logits of shape [B, N_res, 37],
                log probability that an atom is resolved in atom37 representation,
                can be converted to probability by applying sigmoid.
        """
        logits = self.logits(representations['single'])
        return dict(logits=logits)

    def loss(self, value, batch):
        logits = value['logits']
        assert len(logits.shape) == 3

        # Does the atom appear in the amino acid?
        atom_exists = batch['atom37_atom_exists']
        # Is the atom resolved in the experiment? Subset of atom_exists,
        # *except for OXT*
        all_atom_mask = paddle.cast(batch['all_atom_mask'], 'float32')

        xent = sigmoid_cross_entropy(labels=all_atom_mask, logits=logits)
        loss = paddle.sum(xent * atom_exists, axis=[-2, -1]) / (1e-8 + paddle.sum(atom_exists, axis=[-2, -1]))

        if self.config.filter_by_resolution:
            # NMR & distillation have resolution = 0
            resolution = paddle.squeeze(batch['resolution'], axis=-1)
            loss *= paddle.cast((resolution >= self.config.min_resolution)
                    & (resolution <= self.config.max_resolution), 'float32')

        output = {'loss': loss}
        return output


class DistogramHead(nn.Layer):
    """Head to predict a distogram.

    Jumper et al. (2021) Suppl. Sec. 1.9.8 "Distogram prediction"
    """

    def __init__(self, channel_num, config, global_config, name='distogram_head'):
        super(DistogramHead, self).__init__()
        self.config = config
        self.global_config = global_config

        self.half_logits = nn.Linear(channel_num['pair_channel'],
                                    self.config.num_bins, name='half_logits')
        init_final_linear(self.half_logits)

    def forward(self, representations, batch):
        """Builds DistogramHead module.

        Arguments:
        representations: Dictionary of representations, must contain:
            * 'pair': pair representation, shape [batch, N_res, N_res, c_z].

        Returns:
        Dictionary containing:
            * logits: logits for distogram, shape [batch, N_res, N_res, N_bins].
            * bin_breaks: array containing bin breaks, shape [batch, N_bins - 1].
        """
        half_logits = self.half_logits(representations['pair'])

        logits = half_logits + paddle.transpose(half_logits, perm=[0, 2, 1, 3])
        breaks = paddle.linspace(self.config.first_break, self.config.last_break,
                          self.config.num_bins - 1)
        breaks = paddle.tile(breaks[None, :],
                            repeat_times=[logits.shape[0], 1])

        # if not self.training:
        if not self.training and self.global_config.low_memory is True:
            logits_cpu = logits.cpu()
            del logits
        return {
            # 'logits': logits,
            'logits': logits_cpu if not self.training and self.global_config.low_memory is True else logits, 
            'bin_edges': breaks}

    def loss(self, value, batch):
        return _distogram_log_loss(value['logits'], value['bin_edges'],
                               batch, self.config.num_bins)


def _distogram_log_loss(logits, bin_edges, batch, num_bins):
    """Log loss of a distogram."""
    positions = batch['pseudo_beta']
    mask = batch['pseudo_beta_mask']

    assert positions.shape[-1] == 3

    sq_breaks = paddle.square(bin_edges).unsqueeze([1, 2])

    dist2 = paddle.sum(
        paddle.square(
            paddle.unsqueeze(positions, axis=-2) -
            paddle.unsqueeze(positions, axis=-3)),
        axis=-1,
        keepdim=True)

    true_bins = paddle.sum(dist2 > sq_breaks, axis=-1)

    errors = softmax_cross_entropy(
        labels=paddle.nn.functional.one_hot(true_bins, num_classes=num_bins), logits=logits)

    square_mask = paddle.unsqueeze(mask, axis=-2) * paddle.unsqueeze(mask, axis=-1)

    avg_error = (
        paddle.sum(errors * square_mask, axis=[-2, -1]) /
        (1e-6 + paddle.sum(square_mask, axis=[-2, -1])))
    dist2 = dist2[..., 0]
    return {
        'loss': avg_error, 
        'true_dist': paddle.sqrt(1e-6 + dist2)}


def dgram_from_positions(positions, num_bins, min_bin, max_bin):
    lower_breaks = paddle.linspace(min_bin, max_bin, num_bins)
    lower_breaks = paddle.square(lower_breaks)
    upper_breaks = paddle.concat([lower_breaks[1:],
                                    paddle.to_tensor([1e8], dtype='float32')])

    def _squared_difference(x, y):
        return paddle.square(x - y)

    dist2 = paddle.sum(
        _squared_difference(
            paddle.unsqueeze(positions, axis=-2),
            paddle.unsqueeze(positions, axis=-3)),
        axis=-1, keepdim=True)

    dgram = ((dist2 > lower_breaks.astype(dist2.dtype)).astype('float32') *
                (dist2 < upper_breaks.astype(dist2.dtype)).astype('float32'))
    return dgram

class EvoformerIteration(nn.Layer):
    """Single iteration (block) of Evoformer stack.

    Jumper et al. (2021) Suppl. Alg. 6 "EvoformerStack" lines 2-10
    """
    def __init__(self, channel_num, config, global_config, is_extra_msa=False):
        super(EvoformerIteration, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config
        self.is_extra_msa = is_extra_msa

        # Row-wise Gated Self-attention with Pair Bias
        self.msa_row_attention_with_pair_bias = MSARowAttentionWithPairBias(
            channel_num, self.config.msa_row_attention_with_pair_bias,
            self.global_config, is_extra_msa)
        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.msa_row_attention_with_pair_bias)
        self.msa_row_attn_dropout = nn.Dropout(dropout_rate, axis=dropout_axis) \
            if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        if self.is_extra_msa:
            self.msa_column_global_attention = MSAColumnGlobalAttention(
                channel_num, config.msa_column_attention, global_config)
            dropout_rate, dropout_axis = self._parse_dropout_params(
                self.msa_column_global_attention)
            self.msa_col_attn_dropout = nn.Dropout(
                dropout_rate, axis=dropout_axis) \
                    if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)
        else:
            self.msa_column_attention = MSAColumnAttention(
                channel_num, config.msa_column_attention, global_config)
            dropout_rate, dropout_axis = self._parse_dropout_params(
                self.msa_column_attention)
            self.msa_col_attn_dropout = nn.Dropout(
                dropout_rate, axis=dropout_axis) \
                    if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.msa_transition = Transition(
            channel_num, self.config.msa_transition, self.global_config,
            is_extra_msa, 'msa_transition')
        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.msa_transition)
        self.msa_transition_dropout = nn.Dropout(
            dropout_rate, axis=dropout_axis) \
                if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        # OuterProductMean
        self.outer_product_mean = OuterProductMean(channel_num,
                    self.config.outer_product_mean, self.global_config,
                    self.is_extra_msa, name='outer_product_mean')

        # Dropout
        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.outer_product_mean)
        self.outer_product_mean_dropout = nn.Dropout(
            dropout_rate, axis=dropout_axis) \
                if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        # Triangle Multiplication.
        self.triangle_multiplication_outgoing = TriangleMultiplication(channel_num,
                    self.config.triangle_multiplication_outgoing, self.global_config,
                    name='triangle_multiplication_outgoing')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_multiplication_outgoing)
        self.triangle_outgoing_dropout = nn.Dropout(dropout_rate, axis=dropout_axis) \
            if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.triangle_multiplication_incoming = TriangleMultiplication(channel_num,
                    self.config.triangle_multiplication_incoming, self.global_config,
                    name='triangle_multiplication_incoming')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_multiplication_incoming)
        self.triangle_incoming_dropout = nn.Dropout(dropout_rate, axis=dropout_axis) \
            if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        # TriangleAttention.
        self.triangle_attention_starting_node = TriangleAttention(channel_num,
                    self.config.triangle_attention_starting_node, self.global_config,
                    name='triangle_attention_starting_node')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_attention_starting_node)
        self.triangle_starting_dropout = nn.Dropout(dropout_rate, axis=dropout_axis) \
            if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.triangle_attention_ending_node = TriangleAttention(channel_num,
                    self.config.triangle_attention_ending_node, self.global_config,
                    name='triangle_attention_ending_node')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_attention_ending_node)
        self.triangle_ending_dropout = nn.Dropout(dropout_rate, axis=dropout_axis) \
            if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        # Pair transition.
        self.pair_transition = Transition(
            channel_num, self.config.pair_transition, self.global_config,
            is_extra_msa, 'pair_transition')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.pair_transition)
        self.pair_transition_dropout = nn.Dropout(dropout_rate, axis=dropout_axis) \
            if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

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

    def outer_product_mean_origin(self, msa_act, pair_act, masks):
        msa_mask, pair_mask = masks['msa'], masks['pair']
        residual = self.msa_row_attention_with_pair_bias(
            msa_act, msa_mask, pair_act)
        residual = self.msa_row_attn_dropout(residual)
        msa_act = msa_act + residual

        if self.is_extra_msa:
            residual = self.msa_column_global_attention(msa_act, msa_mask)
            residual = self.msa_col_attn_dropout(residual)
            msa_act = msa_act + residual

            residual = self.msa_transition(msa_act, msa_mask)
            residual = self.msa_transition_dropout(residual)
            msa_act = msa_act + residual

        else:
            residual = self.msa_column_attention(msa_act, msa_mask)
            residual = self.msa_col_attn_dropout(residual)
            msa_act = msa_act + residual

            residual = self.msa_transition(msa_act, msa_mask)
            residual = self.msa_transition_dropout(residual)
            msa_act = msa_act + residual

        residual = self.outer_product_mean(msa_act, msa_mask)
        residual = self.outer_product_mean_dropout(residual)
        pair_act = pair_act + residual

        residual = self.triangle_multiplication_outgoing(pair_act, pair_mask)
        residual = self.triangle_outgoing_dropout(residual)
        pair_act = pair_act + residual

        residual = self.triangle_multiplication_incoming(pair_act, pair_mask)
        residual = self.triangle_incoming_dropout(residual)
        pair_act = pair_act + residual

        residual = self.triangle_attention_starting_node(pair_act, pair_mask)
        residual = self.triangle_starting_dropout(residual)
        pair_act = pair_act + residual

        residual = self.triangle_attention_ending_node(pair_act, pair_mask)
        residual = self.triangle_ending_dropout(residual)
        pair_act = pair_act + residual

        residual = self.pair_transition(pair_act, pair_mask)
        residual = self.pair_transition_dropout(residual)
        pair_act = pair_act + residual

        return msa_act, pair_act 

    def outer_product_mean_first(self, msa_act, pair_act, masks):
        msa_mask, pair_mask = masks['msa'], masks['pair']

        residual = self.outer_product_mean(msa_act, msa_mask)
        outer_product_mean = self.outer_product_mean_dropout(residual)
        pair_act = pair_act + outer_product_mean
        
        residual = self.msa_row_attention_with_pair_bias(
            msa_act, msa_mask, pair_act)
        residual = self.msa_row_attn_dropout(residual)
        msa_act = msa_act + residual

        if self.is_extra_msa:
            residual = self.msa_column_global_attention(msa_act, msa_mask)
            residual = self.msa_col_attn_dropout(residual)
            msa_act = msa_act + residual

            residual = self.msa_transition(msa_act, msa_mask)
            residual = self.msa_transition_dropout(residual)
            msa_act = msa_act + residual

        else:
            residual = self.msa_column_attention(msa_act, msa_mask)
            residual = self.msa_col_attn_dropout(residual)
            msa_act = msa_act + residual

            residual = self.msa_transition(msa_act, msa_mask)
            residual = self.msa_transition_dropout(residual)
            msa_act = msa_act + residual

        residual = self.triangle_multiplication_outgoing(pair_act, pair_mask)
        residual = self.triangle_outgoing_dropout(residual)
        pair_act = pair_act + residual

        residual = self.triangle_multiplication_incoming(pair_act, pair_mask)
        residual = self.triangle_incoming_dropout(residual)
        pair_act = pair_act + residual

        residual = self.triangle_attention_starting_node(pair_act, pair_mask)
        residual = self.triangle_starting_dropout(residual)
        pair_act = pair_act + residual

        residual = self.triangle_attention_ending_node(pair_act, pair_mask)
        residual = self.triangle_ending_dropout(residual)
        pair_act = pair_act + residual
        
        residual = self.pair_transition(pair_act, pair_mask)
        residual = self.pair_transition_dropout(residual)
        pair_act = pair_act + residual
        return msa_act, pair_act

    def outer_product_mean_end(self, msa_act, pair_act, masks):
        msa_mask, pair_mask = masks['msa'], masks['pair']

        residual = self.msa_row_attention_with_pair_bias(
            msa_act, msa_mask, pair_act)
        residual = self.msa_row_attn_dropout(residual)
        msa_act = msa_act + residual

        if self.is_extra_msa:
            residual = self.msa_column_global_attention(msa_act, msa_mask)
            residual = self.msa_col_attn_dropout(residual)
            msa_act = msa_act + residual

            residual = self.msa_transition(msa_act, msa_mask)
            residual = self.msa_transition_dropout(residual)
            msa_act = msa_act + residual

        else:
            residual = self.msa_column_attention(msa_act, msa_mask)
            residual = self.msa_col_attn_dropout(residual)
            msa_act = msa_act + residual

            residual = self.msa_transition(msa_act, msa_mask)
            residual = self.msa_transition_dropout(residual)
            msa_act = msa_act + residual

        residual = self.outer_product_mean(msa_act, msa_mask)
        outer_product_mean = self.outer_product_mean_dropout(residual)

        residual = self.triangle_multiplication_outgoing(pair_act, pair_mask)
        residual = self.triangle_outgoing_dropout(residual)
        pair_act = pair_act + residual

        residual = self.triangle_multiplication_incoming(pair_act, pair_mask)
        residual = self.triangle_incoming_dropout(residual)
        pair_act = pair_act + residual

        residual = self.triangle_attention_starting_node(pair_act, pair_mask)
        residual = self.triangle_starting_dropout(residual)
        pair_act = pair_act + residual

        residual = self.triangle_attention_ending_node(pair_act, pair_mask)
        residual = self.triangle_ending_dropout(residual)
        pair_act = pair_act + residual

        residual = self.pair_transition(pair_act, pair_mask)
        residual = self.pair_transition_dropout(residual)
        pair_act = pair_act + residual

        pair_act = pair_act + outer_product_mean

        return msa_act, pair_act

    def forward(self, msa_act, pair_act, masks):

        if self.global_config.outer_product_mean_position in ['origin', 'middle']:
            msa_act, pair_act = self.outer_product_mean_origin(msa_act, pair_act, masks)

        elif self.global_config.outer_product_mean_position == 'first':
            msa_act, pair_act = self.outer_product_mean_first(msa_act, pair_act, masks)

        elif self.global_config.outer_product_mean_position == 'end':
            msa_act, pair_act = self.outer_product_mean_end(msa_act, pair_act, masks)

        else:
            raise Error("Only support outer_product_mean_position in ['origin', 'middle', ''first', 'end'] now!")

        return msa_act, pair_act


class EmbeddingsAndEvoformer(nn.Layer):
    """Embeds the input data and runs Evoformer.

    Produces the MSA, single and pair representations.
    Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 5-18
    """

    def __init__(self, channel_num, config, global_config):
        super(EmbeddingsAndEvoformer, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        # InputEmbedder
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 5
        # Jumper et al. (2021) Suppl. Alg. 3 "InputEmbedder"
        self.preprocess_1d = nn.Linear(channel_num['target_feat'],
                                       self.config.msa_channel, name='preprocess_1d')
        self.preprocess_msa = nn.Linear(channel_num['msa_feat'],
                                        self.config.msa_channel, name='preprocess_msa')
        self.left_single = nn.Linear(channel_num['target_feat'], self.config.pair_channel,
                                     name='left_single')
        self.right_single = nn.Linear(channel_num['target_feat'], self.config.pair_channel,
                                      name='right_single')

        # RecyclingEmbedder
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 6
        # Jumper et al. (2021) Suppl. Alg. 32 "RecyclingEmbedder"
        if self.config.recycle_pos:
            self.prev_pos_linear = nn.Linear(self.config.prev_pos.num_bins,
                                             self.config.pair_channel)

        # RelPosEmbedder
        # Jumper et al. (2021) Suppl. Alg. 4 "relpos"
        # Jumper et al. (2021) Suppl. Alg. 5 "one_hot"
        if self.config.max_relative_feature:
            self.pair_activiations = nn.Linear(
                2 * self.config.max_relative_feature + 1,
                self.config.pair_channel)

        if self.config.recycle_features:
            self.prev_msa_first_row_norm = nn.LayerNorm(
                self.config.msa_channel)
            self.prev_pair_norm = nn.LayerNorm(self.config.pair_channel)

        # Embed templates into the pair activations.
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 9-13
        if self.config.template.enabled:
            self.channel_num['template_angle'] = 57
            self.channel_num['template_pair'] = 88
            self.template_embedding = TemplateEmbedding(
                self.channel_num, self.config.template, self.global_config)

        # ExtraMSAEmbedder
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 14-16
        self.extra_msa_activations = nn.Linear(
            25,  # 23 (20aa+unknown+gap+mask) + 1 (has_del) + 1 (del_val)
            self.config.extra_msa_channel)

        # Extra MSA Stack.
        # Jumper et al. (2021) Suppl. Alg. 18 "ExtraMsaStack"
        self.extra_msa_stack = nn.LayerList()
        for _ in range(self.config.extra_msa_stack_num_block):
            self.extra_msa_stack.append(EvoformerIteration(
                self.channel_num, self.config.evoformer, self.global_config,
                is_extra_msa=True))

        # Embed templates torsion angles
        if self.config.template.enabled and self.config.template.embed_torsion_angles:
            c = self.config.msa_channel
            self.template_single_embedding = nn.Linear(
                self.channel_num['template_angle'], c)
            self.template_projection = nn.Linear(c, c)

        # Main trunk of the network
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 17-18
        self.evoformer_iteration = nn.LayerList()
        for _ in range(self.config.evoformer_num_block):
            self.evoformer_iteration.append(EvoformerIteration(
                self.channel_num, self.config.evoformer, self.global_config,
                is_extra_msa=False))

        self.single_activations = nn.Linear(
            self.config.msa_channel, self.config.seq_channel)

    def _pseudo_beta_fn(self, aatype, all_atom_positions, all_atom_masks):
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
            m = len(all_atom_masks)
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

    def _create_extra_msa_feature(self, batch):
        # 23: 20aa + unknown + gap + bert mask
        msa_1hot = nn.functional.one_hot(batch['extra_msa'], 23)
        msa_feat = [msa_1hot,
                    paddle.unsqueeze(batch['extra_has_deletion'], axis=-1),
                    paddle.unsqueeze(batch['extra_deletion_value'], axis=-1)]
        return paddle.concat(msa_feat, axis=-1)

    def forward(self, batch):
        # InputEmbedder
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 5
        # Jumper et al. (2021) Suppl. Alg. 3 "InputEmbedder"
        preprocess_1d = self.preprocess_1d(batch['target_feat'])
        # preprocess_msa = self.preprocess_msa(batch['msa_feat'])
        msa_activations = paddle.unsqueeze(preprocess_1d, axis=1) + \
                    self.preprocess_msa(batch['msa_feat'])

        right_single = self.right_single(batch['target_feat'])  # 1, n_res, 22 -> 1, n_res, 128
        right_single = paddle.unsqueeze(right_single, axis=1)   # 1, n_res, 128 -> 1, 1, n_res, 128
        left_single = self.left_single(batch['target_feat'])    # 1, n_res, 22 -> 1, n_res, 128
        left_single = paddle.unsqueeze(left_single, axis=2)     # 1, n_res, 128 -> 1, n_res, 1, 128
        pair_activations = left_single + right_single

        mask_2d = paddle.unsqueeze(batch['seq_mask'], axis=1) * paddle.unsqueeze(batch['seq_mask'], axis=2)

        # Inject previous outputs for recycling.
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 6
        # Jumper et al. (2021) Suppl. Alg. 32 "RecyclingEmbedder"
        if self.config.recycle_pos and 'prev_pos' in batch:
            prev_pseudo_beta = self._pseudo_beta_fn(
                batch['aatype'], batch['prev_pos'], None)
            dgram = dgram_from_positions(
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

        # RelPosEmbedder
        # Jumper et al. (2021) Suppl. Alg. 4 "relpos"
        # Jumper et al. (2021) Suppl. Alg. 5 "one_hot"
        if self.config.max_relative_feature:
            pos = batch['residue_index']  # [bs, N_res]
            offset = paddle.unsqueeze(pos, axis=[-1]) - \
                paddle.unsqueeze(pos, axis=[-2])
            rel_pos = nn.functional.one_hot(
                paddle.clip(
                    offset + self.config.max_relative_feature,
                    min=0,
                    max=2 * self.config.max_relative_feature),
                2 * self.config.max_relative_feature + 1)
            rel_pos_bias = self.pair_activiations(rel_pos)
            pair_activations += rel_pos_bias

        # TemplateEmbedder
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 9-13
        if self.config.template.enabled:
            template_batch = {k: batch[k] for k in batch
                              if k.startswith('template_')}
            template_pair_repr = self.template_embedding(
                pair_activations, template_batch, mask_2d)
            pair_activations += template_pair_repr

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
            extra_msa_act, extra_pair_act = recompute_wrapper(extra_msa_stack_iteration,
                    extra_msa_stack_input['msa'],
                    extra_msa_stack_input['pair'],
                    {'msa': batch['extra_msa_mask'],
                            'pair': mask_2d},
                    is_recompute=self.training and idx >= self.config.extra_msa_stack_recompute_start_block_index)
            extra_msa_stack_output = {
                'msa': extra_msa_act,
                'pair': extra_pair_act}
            extra_msa_stack_input = {
                'msa': extra_msa_stack_output['msa'],
                'pair': extra_msa_stack_output['pair']}

        evoformer_input = {
            'msa': msa_activations,
            'pair': extra_msa_stack_output['pair'],
        }

        evoformer_masks = {
            'msa': batch['msa_mask'],
            'pair': mask_2d,
        }

        # ==================================================
        #  Template angle feat
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 7-8
        # ==================================================
        if self.config.template.enabled and self.config.template.embed_torsion_angles:
            num_templ, num_res = batch['template_aatype'].shape[1:]

            aatype_one_hot = nn.functional.one_hot(batch['template_aatype'], 22)
            # Embed the templates aatype, torsion angles and masks.
            # Shape (templates, residues, msa_channels)
            ret = all_atom.atom37_to_torsion_angles(
                aatype=batch['template_aatype'],
                all_atom_pos=batch['template_all_atom_positions'],
                all_atom_mask=batch['template_all_atom_masks'],
                # Ensure consistent behaviour during testing:
                placeholder_for_undefined=not self.global_config.zero_init)

            template_features = paddle.concat([
                aatype_one_hot,
                paddle.reshape(ret['torsion_angles_sin_cos'],
                               [-1, num_templ, num_res, 14]),
                paddle.reshape(ret['alt_torsion_angles_sin_cos'],
                               [-1, num_templ, num_res, 14]),
                ret['torsion_angles_mask']], axis=-1)

            template_activations = self.template_single_embedding(
                template_features)
            template_activations = nn.functional.relu(template_activations)
            template_activations = self.template_projection(template_activations)

            # Concatenate the templates to the msa.
            evoformer_input['msa'] = paddle.concat(
                [evoformer_input['msa'], template_activations], axis=1)

            # Concatenate templates masks to the msa masks.
            # Use mask from the psi angle, as it only depends on the backbone atoms
            # from a single residue.
            torsion_angle_mask = ret['torsion_angles_mask'][..., 2]
            torsion_angle_mask = torsion_angle_mask.astype(
                evoformer_masks['msa'].dtype)
            evoformer_masks['msa'] = paddle.concat(
                [evoformer_masks['msa'], torsion_angle_mask], axis=1)


        # ==================================================
        #  Main MSA Stack
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 17-18
        # ==================================================
        for idx, evoformer_block in enumerate(self.evoformer_iteration):
            msa_act, pair_act = recompute_wrapper(evoformer_block,
                    evoformer_input['msa'],
                    evoformer_input['pair'],
                    evoformer_masks,
                    is_recompute=self.training and idx >= self.config.evoformer_recompute_start_block_index)
            evoformer_output = {
                'msa': msa_act,
                'pair': pair_act}
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


class OuterProductMean(nn.Layer):
    """Computes mean outer product.

    Jumper et al. (2021) Suppl. Alg. 10 "OuterProductMean"
    """

    def __init__(self, channel_num, config, global_config, is_extra_msa, name='outer_product_mean'):
        super(OuterProductMean, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        if is_extra_msa:
            c_m = channel_num['extra_msa_channel']
        else:
            c_m = channel_num['msa_channel']

        self.layer_norm_input = nn.LayerNorm(c_m, name='layer_norm_input')
        self.left_projection = nn.Linear(
            c_m, self.config.num_outer_channel, name='left_projection')
        self.right_projection = nn.Linear(
            c_m, self.config.num_outer_channel, name='right_projection')

        if self.global_config.zero_init:
            init_w = nn.initializer.Constant(value=0.0)
        else:
            init_w = nn.initializer.KaimingNormal()

        self.output_w = paddle.create_parameter(
            [self.config.num_outer_channel, self.config.num_outer_channel, channel_num['pair_channel']],
            'float32', default_initializer=init_w)
        self.output_b = paddle.create_parameter(
            [channel_num['pair_channel']], 'float32',
            default_initializer=nn.initializer.Constant(value=0.0))

    def forward(self, act, mask):
        """Builds OuterProductMean module.

        Arguments:
        act: MSA representation, shape [batch, N_seq, N_res, c_m].
        mask: MSA mask, shape [batch, N_seq, N_res].

        Returns:
        Update to pair representation, shape [batch, N_res, N_res, c_z].
        """
        
        act = self.layer_norm_input(act)
        right_act = self.right_projection(act)
        
        left_act = self.left_projection(act)
        mask = paddle.unsqueeze(mask, axis=-1)

        left_act = mask * left_act
        
        epsilon = 1e-3
        norm = paddle.einsum('nabc,nadc->nbdc', mask, mask) + epsilon

        def compute_chunk(left_act, right_act):
            # This is equivalent to
            #
            # act = jnp.einsum('abc,ade->dceb', left_act, right_act)
            # act = jnp.einsum('dceb,cef->bdf', act, output_w) + output_b
            #
            # but faster. maybe for subbatch inference?
            
            left_act = left_act.transpose([0, 1, 3, 2])
            act = paddle.einsum('nacb,nade->ndceb', left_act, right_act)
            act = paddle.einsum('ndceb,cef->ndbf', act, self.output_w) + self.output_b
            return act.transpose([0, 2, 1, 3])

        if not self.training:
            # low memory mode using subbatch
            sb_chunk = subbatch(compute_chunk, [0], [2],
                               self.config.chunk_size, 1)
            act = sb_chunk(left_act, right_act)
        else:
            act = compute_chunk(left_act, right_act)

        act = act / norm

        return act


class TriangleAttention(nn.Layer):
    """Triangle Attention.

    Jumper et al. (2021) Suppl. Alg. 13 "TriangleAttentionStartingNode"
    Jumper et al. (2021) Suppl. Alg. 14 "TriangleAttentionEndingNode"
    """

    def __init__(self, channel_num, config, global_config, name='triangle_attention'):
        super(TriangleAttention, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        assert config.orientation in ['per_row', 'per_column']

        self.query_norm = nn.LayerNorm(channel_num['pair_channel'],
                                    name='query_norm')
        self.feat_2d_weights = paddle.create_parameter(
            [channel_num['pair_channel'], self.config.num_head], 'float32',
            default_initializer=nn.initializer.Normal(
                std=1. / np.sqrt(channel_num['pair_channel'])))

        self.attention = Attention(self.config, self.global_config,
                        channel_num['pair_channel'], channel_num['pair_channel'],
                        channel_num['pair_channel'])


    def forward(self, pair_act, pair_mask):
        """Builds TriangleAttention module.

        Arguments:
        pair_act: [batch, N_res, N_res, c_z] pair activations tensor
        pair_mask: [batch, N_res, N_res] mask of non-padded regions in the tensor.

        Returns:
        Update to pair_act, shape [batch, N_res, N_res, c_z].
        """
        if self.config.orientation == 'per_column':
            pair_act = pair_act.transpose([0, 2, 1, 3])
            pair_mask = pair_mask.transpose([0, 2, 1])

        bias = 1e9 * (pair_mask - 1.)
        bias = paddle.unsqueeze(bias, axis=[2, 3])

        pair_act = self.query_norm(pair_act)

        nonbatched_bias = paddle.einsum('bqkc,ch->bhqk', pair_act, self.feat_2d_weights)

        if not self.training:
            # low memory mode using subbatch
            sb_attn = subbatch(self.attention, [0, 1, 2], [1, 1, 1],
                               self.global_config.subbatch_size, 1, same_arg_idx={1: 0})
            pair_act = sb_attn(pair_act, pair_act, bias, nonbatched_bias)
        else:
            pair_act = self.attention(pair_act, pair_act, bias, nonbatched_bias)

        if self.config.orientation == 'per_column':
            pair_act = pair_act.transpose([0, 2, 1, 3])

        return pair_act


class TriangleMultiplication(nn.Layer):
    """Triangle multiplication layer ("outgoing" or "incoming").

    Jumper et al. (2021) Suppl. Alg. 11 "TriangleMultiplicationOutgoing"
    Jumper et al. (2021) Suppl. Alg. 12 "TriangleMultiplicationIncoming"
    """

    def __init__(self, channel_num, config, global_config, name='triangle_multiplication'):
        super(TriangleMultiplication, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        self.layer_norm_input = nn.LayerNorm(self.channel_num['pair_channel'], name='layer_norm_input')
        self.left_projection = nn.Linear(self.channel_num['pair_channel'],
                                self.config.num_intermediate_channel, name='left_projection')
        self.right_projection = nn.Linear(self.channel_num['pair_channel'],
                                self.config.num_intermediate_channel, name='right_projection')
        self.left_gate = nn.Linear(self.channel_num['pair_channel'],
                                self.config.num_intermediate_channel, name='left_gate')
        init_gate_linear(self.left_gate)
        self.right_gate = nn.Linear(self.channel_num['pair_channel'],
                                self.config.num_intermediate_channel, name='right_gate')
        init_gate_linear(self.right_gate)

        # line 4
        self.center_layer_norm = nn.LayerNorm(self.config.num_intermediate_channel, name='center_layer_norm')
        self.output_projection = nn.Linear(self.config.num_intermediate_channel,
                                    self.channel_num['pair_channel'], name='output_projection')
        init_final_linear(self.output_projection)
        # line 3
        self.gating_linear = nn.Linear(self.channel_num['pair_channel'],
                                    self.channel_num['pair_channel'], name='output_projection')
        init_gate_linear(self.gating_linear)

    def forward(self, act, mask):
        """Builds TriangleMultiplication module.

        Arguments:
        act: Pair activations, shape [batch, N_res, N_res, c_z]
        mask: Pair mask, shape [batch, N_res, N_res].

        Returns:
        Outputs, same shape/type as act.
        """
        mask = paddle.unsqueeze(mask, axis=-1) # [batch, N_res, N_res, 1]

        act = self.layer_norm_input(act) # line 1

        left_proj_act = mask * self.left_projection(act)
        right_proj_act = mask * self.right_projection(act)
        
        left_gate_values = nn.functional.sigmoid(self.left_gate(act))
        right_gate_values = nn.functional.sigmoid(self.right_gate(act))
        
        left_proj_act = left_proj_act * left_gate_values
        right_proj_act = right_proj_act * right_gate_values

        # "Outgoing" edges equation: 'ikc,jkc->ijc'
        # "Incoming" edges equation: 'kjc,kic->ijc'
        # Note on the Suppl. Alg. 11 & 12 notation:
        # For the "outgoing" edges, a = left_proj_act and b = right_proj_act
        # For the "incoming" edges, it's swapped:
        #   b = left_proj_act and a = right_proj_act
            
        gate_values = nn.functional.sigmoid(self.gating_linear(act)) # line 3

        if self.config.equation == 'ikc,jkc->ijc':
            # Outgoing
            dim, out_idx = 1, 1
            equation = 'bikc,bjkc->bijc'
            
        elif  self.config.equation == 'kjc,kic->ijc':
            # Incoming
            dim, out_idx = 2, 2
            equation = 'bkjc,bkic->bijc'

        else:
            raise ValueError('unknown equation.')

        if not self.training:
            einsum_fn = subbatch(paddle.einsum, [1], [dim],
                                 self.global_config.subbatch_size, out_idx)
            act = einsum_fn(equation, left_proj_act, right_proj_act)
        else:
            # Outgoing equation = 'bikc,bjkc->bijc'
            # Incoming equation = 'bkjc,bkic->bijc'
            act = paddle.einsum(equation, left_proj_act, right_proj_act)

        act = self.center_layer_norm(act)
        act = self.output_projection(act)

        act = act * gate_values

        return act


class TemplatePair(nn.Layer):
    """Pair processing for the templates.

    Jumper et al. (2021) Suppl. Alg. 16 "TemplatePairStack" lines 2-6
    """
    def __init__(self, channel_num, config, global_config):
        super(TemplatePair, self).__init__()
        self.config = config
        self.global_config = global_config

        channel_num = {}
        channel_num['pair_channel'] = self.config.triangle_attention_ending_node.value_dim

        self.triangle_attention_starting_node = TriangleAttention(channel_num,
            self.config.triangle_attention_starting_node, self.global_config,
            name='triangle_attention_starting_node')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_attention_starting_node)
        self.triangle_starting_dropout = nn.Dropout(dropout_rate, axis=dropout_axis) \
            if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.triangle_attention_ending_node = TriangleAttention(channel_num,
                    self.config.triangle_attention_ending_node, self.global_config,
                    name='triangle_attention_ending_node')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_attention_ending_node)
        self.triangle_ending_dropout = nn.Dropout(dropout_rate, axis=dropout_axis) \
            if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.triangle_multiplication_outgoing = TriangleMultiplication(channel_num,
                    self.config.triangle_multiplication_outgoing, self.global_config,
                    name='triangle_multiplication_outgoing')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_multiplication_outgoing)
        self.triangle_outgoing_dropout = nn.Dropout(dropout_rate, axis=dropout_axis) \
            if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.triangle_multiplication_incoming = TriangleMultiplication(channel_num,
                    self.config.triangle_multiplication_incoming, self.global_config,
                    name='triangle_multiplication_incoming')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_multiplication_incoming)
        self.triangle_incoming_dropout = nn.Dropout(dropout_rate, axis=dropout_axis) \
            if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.pair_transition = Transition(channel_num, self.config.pair_transition,
                    self.global_config, is_extra_msa=False,
                    transition_type='pair_transition')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.pair_transition)
        self.pair_transition_dropout = nn.Dropout(dropout_rate, axis=dropout_axis) \
            if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)


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

    def forward(self, pair_act, pair_mask):
        """Builds one block of TemplatePair module.

        Arguments:
        pair_act: Pair activations for single template, shape [batch, N_res, N_res, c_t].
        pair_mask: Pair mask, shape [batch, N_res, N_res].

        Returns:
        Updated pair_act, shape [batch, N_res, N_res, c_t].
        """

        residual = self.triangle_attention_starting_node(pair_act, pair_mask)
        residual = self.triangle_starting_dropout(residual)
        pair_act = pair_act + residual

        residual = self.triangle_attention_ending_node(pair_act, pair_mask)
        residual = self.triangle_ending_dropout(residual)
        pair_act = pair_act + residual

        residual = self.triangle_multiplication_outgoing(pair_act, pair_mask)
        residual = self.triangle_outgoing_dropout(residual)
        pair_act = pair_act + residual

        residual = self.triangle_multiplication_incoming(pair_act, pair_mask)
        residual = self.triangle_incoming_dropout(residual)
        pair_act = pair_act + residual

        residual = self.pair_transition(pair_act, pair_mask)
        residual = self.pair_transition_dropout(residual)
        pair_act = pair_act + residual

        return pair_act


class SingleTemplateEmbedding(nn.Layer):
    """Embeds a single template.

    Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 9+11
    """
    def __init__(self, channel_num, config, global_config):
        super(SingleTemplateEmbedding, self).__init__()
        self.config = config
        self.channel_num = channel_num
        self.global_config = global_config

        self.embedding2d = nn.Linear(channel_num['template_pair'],
            self.config.template_pair_stack.triangle_attention_ending_node.value_dim)

        self.template_pair_stack = nn.LayerList()
        for _ in range(self.config.template_pair_stack.num_block):
            self.template_pair_stack.append(TemplatePair(
                self.channel_num, self.config.template_pair_stack, self.global_config))

        self.output_layer_norm = nn.LayerNorm(self.config.attention.key_dim)

    def forward(self, query_embedding, batch, mask_2d):
        """Build the single template embedding.

        Arguments:
            query_embedding: Query pair representation, shape [batch, N_res, N_res, c_z].
            batch: A batch of template features (note the template dimension has been
                stripped out as this module only runs over a single template).
            mask_2d: Padding mask (Note: this doesn't care if a template exists,
                unlike the template_pseudo_beta_mask).

        Returns:
            A template embedding [N_res, N_res, c_z].
        """
        assert mask_2d.dtype == query_embedding.dtype
        dtype = query_embedding.dtype
        num_res = batch['template_aatype'].shape[1]
        template_mask = batch['template_pseudo_beta_mask']
        template_mask_2d = template_mask[..., None] * template_mask[..., None, :]
        template_mask_2d = template_mask_2d.astype(dtype)

        template_dgram = dgram_from_positions(
            batch['template_pseudo_beta'],
            **self.config.dgram_features)
        template_dgram = template_dgram.astype(dtype)

        aatype = nn.functional.one_hot(batch['template_aatype'], 22)
        aatype = aatype.astype(dtype)

        to_concat = [template_dgram, template_mask_2d[..., None]]
        to_concat.append(paddle.tile(aatype[..., None, :, :],
                                     [1, num_res, 1, 1]))
        to_concat.append(paddle.tile(aatype[..., None, :],
                                     [1, 1, num_res, 1]))

        n, ca, c = [residue_constants.atom_order[a]
                    for a in ('N', 'CA', 'C')]
        rot, trans = quat_affine.make_transform_from_reference(
            n_xyz=batch['template_all_atom_positions'][..., n, :],
            ca_xyz=batch['template_all_atom_positions'][..., ca, :],
            c_xyz=batch['template_all_atom_positions'][..., c, :])
        affines = quat_affine.QuatAffine(
            quaternion=quat_affine.rot_to_quat(rot),
            translation=trans,
            rotation=rot)

        points = [paddle.unsqueeze(x, axis=-2) for x in
                  paddle.unstack(affines.translation, axis=-1)]
        affine_vec = affines.invert_point(points, extra_dims=1)
        inv_distance_scalar = paddle.rsqrt(
            1e-6 + sum([paddle.square(x) for x in affine_vec]))

        # Backbone affine mask: whether the residue has C, CA, N
        # (the template mask defined above only considers pseudo CB).
        template_mask = (
            batch['template_all_atom_masks'][..., n] *
            batch['template_all_atom_masks'][..., ca] *
            batch['template_all_atom_masks'][..., c])
        template_mask_2d = template_mask[..., None] * template_mask[..., None, :]
        inv_distance_scalar *= template_mask_2d.astype(inv_distance_scalar.dtype)

        unit_vector = [(x * inv_distance_scalar)[..., None] for x in affine_vec]
        unit_vector = [x.astype(dtype) for x in unit_vector]
        if not self.config.use_template_unit_vector:
            unit_vector = [paddle.zeros_like(x) for x in unit_vector]
        to_concat.extend(unit_vector)

        template_mask_2d = template_mask_2d.astype(dtype)
        to_concat.append(template_mask_2d[..., None])

        act = paddle.concat(to_concat, axis=-1)
        # Mask out non-template regions so we don't get arbitrary values in the
        # distogram for these regions.
        act *= template_mask_2d[..., None]

        act = self.embedding2d(act)

        for idx, pair_encoder in enumerate(self.template_pair_stack):
            act = recompute_wrapper(pair_encoder, act, mask_2d,
                is_recompute=self.training and idx >= self.config.template_pair_stack.recompute_start_block_index)

        act = self.output_layer_norm(act)
        return act


class TemplateEmbedding(nn.Layer):
    """Embeds a set of templates.

        Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 9-12
        Jumper et al. (2021) Suppl. Alg. 17 "TemplatePointwiseAttention"
    """

    def __init__(self, channel_num, config, global_config):
        super(TemplateEmbedding, self).__init__()
        self.config = config
        self.global_config = global_config

        self.single_template_embedding = SingleTemplateEmbedding(
            channel_num, config, global_config)
        self.attention = Attention(
            config.attention, global_config,
            channel_num['pair_channel'],
            config.attention.key_dim,
            channel_num['pair_channel'])

    def forward(self, query_embedding, template_batch, mask_2d):
        """Build TemplateEmbedding module.

        Arguments:
            query_embedding: Query pair representation, shape [n_batch, N_res, N_res, c_z].
            template_batch: A batch of template features.
            mask_2d: Padding mask (Note: this doesn't care if a template exists,
                unlike the template_pseudo_beta_mask).

        Returns:
            A template embedding [n_batch, N_res, N_res, c_z].
        """

        num_templates = template_batch['template_mask'].shape[1]

        num_channels = (self.config.template_pair_stack
                        .triangle_attention_ending_node.value_dim)

        num_res = query_embedding.shape[1]

        dtype = query_embedding.dtype
        template_mask = template_batch['template_mask']
        template_mask = template_mask.astype(dtype)

        query_channels = query_embedding.shape[-1]

        outs = []
        for i in range(num_templates):
            # By default, num_templates = 4
            batch0 = {k: paddle.squeeze(v.slice([1], [i], [i+1]), axis=1)
                      for k, v in template_batch.items()}
            outs.append(self.single_template_embedding(
                query_embedding, batch0, mask_2d))

        template_pair_repr = paddle.stack(outs, axis=1)

        flat_query = paddle.reshape(
            query_embedding, [-1, num_res * num_res, 1, query_channels])
        flat_templates = paddle.reshape(
            paddle.transpose(template_pair_repr, [0, 2, 3, 1, 4]),
            [-1, num_res * num_res, num_templates, num_channels])

        bias = 1e9 * (template_mask[:, None, None, None, :] - 1.)

        if not self.training:
            sb_attn = subbatch(self.attention, [0, 1], [1, 1],
                               self.config.subbatch_size, 1)
            emb = sb_attn(flat_query, flat_templates, bias)

        else:
            emb = self.attention(flat_query, flat_templates, bias)

        emb = paddle.reshape(
            emb, [-1, num_res, num_res, query_channels])

        # No gradients if no templates.
        emb *= (paddle.sum(template_mask) > 0.).astype(emb.dtype)
        return emb