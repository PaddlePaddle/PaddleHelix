#   Copyright (c) 2024 PaddleHelix Authors All Rights Reserved.
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

"""Updated modules for helixfold-3 all-atom model"""

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import gc
FLUID_DEPRECATED = not hasattr(paddle, 'fluid')
if FLUID_DEPRECATED:
    from paddle.base.framework import _dygraph_tracer
else:
    from paddle.fluid.framework import _dygraph_tracer

from helixfold.model.modules import (
    TriangleMultiplication,
    TriangleAttention,
    OuterProductMean,
    Dropout,
    recompute_wrapper,
)
from helixfold.model import diffusion
from helixfold.model.diffusion import (
    AtomAttentionEncoder,
    AtomAttentionDecoder,
    AttentionPairBias,
    RelativePositionEncoding,
)

from helixfold.model.utils import subbatch, tree_map
from helixfold.model.utils import get_all_atom_confidence_metrics
from helixfold.model import modules


class HelixFold3(nn.Layer):
    """HelixFold-3 all-atom model
    """
    def __init__(self, channel_num, config):
        super(HelixFold3, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = config.global_config

        self.input_embedder = InputEmbedder(
            self.channel_num, self.config.input_embedder, self.global_config)


        self.embeddings_and_pairformer = EmbeddingsAndPairformer(
            self.channel_num,
            self.config.embeddings_and_pairformer,
            self.global_config)

        # For the reason of batch_dim tile, we put relative_positional_encoding
        # before diffusion module
        self.diff_rel_pos_encoding = RelativePositionEncoding(
            self.channel_num,
            self.config.heads.diffusion_module.diffusion_conditioning.relative_position_encoding,
            self.global_config)
        self.diffusion_module = diffusion.DiffusionModule(
            self.channel_num,
            self.config.heads.diffusion_module,
            self.global_config)

        if self.config.heads.confidence_head.weight > 0:
            self.confidence_head = ConfidenceHead(
                self.channel_num,
                self.config.heads.confidence_head,
                self.global_config)

    def forward(self,
                batch,
                label_cropped,
                label,
                return_representations=False,
                ensemble_representations=True,
                compute_loss=True):
        single_inputs_act, single_init_act, pair_init_act = \
            self.input_embedder(batch)

        single_act = paddle.zeros_like(single_init_act)
        pair_act = paddle.zeros_like(pair_init_act)

        if 'num_iter_recycling' in batch:
            # Training trick: dynamic recycling number
            num_iter = batch['num_iter_recycling'].numpy()[0, 0]
            num_iter = min(int(num_iter), self.config.num_recycle)
        else:
            num_iter = self.config.num_recycle

        seq_mask = batch['seq_mask']
        masks = {
            'msa': batch['msa_mask'],
            'pair': seq_mask.unsqueeze(axis=1) * seq_mask.unsqueeze(axis=2)
        }

        for recycle_idx in range(1 + num_iter):
            single_act, pair_act = single_act.detach(), pair_act.detach()
            single_act, pair_act = self.embeddings_and_pairformer(
                batch, pair_init_act, pair_act,
                single_inputs_act, single_init_act, single_act,
                masks)

        representations = {
            'single_inputs': single_inputs_act,
            'single': single_act,
            'pair': pair_act
        }
        
        gc.collect()

        tracer = _dygraph_tracer()
        if tracer._amp_dtype == "bfloat16":
            with paddle.amp.auto_cast(enable=False):
                bf16 = paddle.base.core.VarDesc.VarType.BF16 if FLUID_DEPRECATED else paddle.fluid.core.VarDesc.VarType.BF16
                for key, value in representations.items():
                    if isinstance(value, paddle.Tensor) and value.dtype in [bf16]:
                        temp_value = value.cast('float32')
                        temp_value.stop_gradient = value.stop_gradient
                        representations[key] = temp_value
                for key, value in batch.items():
                    if isinstance(value, paddle.Tensor) and value.dtype in [bf16]:
                        temp_value = value.cast('float32')
                        temp_value.stop_gradient = value.stop_gradient
                        batch[key] = temp_value
                ret = self._forward_heads(representations, batch, label_cropped, label)

        else:
            ret = self._forward_heads(representations, batch, label_cropped, label)
        
        return ret
    
    def _forward_heads(self, representations, batch, label_cropped, label):
        ret = {}

        # TODO: add is_dna_aa, is_rna_aa and is_ligand_aa in data_loader
        if not 'is_dna_aa' in batch:
            for name in ['is_protein', 'is_dna', 'is_rna', 'is_ligand']:
                batch[f'{name}_aa'] = paddle.stack([x[index] for x, index 
                        in zip(batch[name], batch['ref_token2atom_idx'])])

        ## diffusion_module head
        # prepare needed keys
        diff_batch_size = self.config.heads.diffusion_module.test_diff_batch_size
        batch['all_atom_pos_mask'] = paddle.ones_like(label['all_atom_pos_mask']).cast(label['all_atom_pos'].dtype)
        # Insert diffusion dim: (B, *) -> (B, diff_batch, *)
        #   Don't tile "pair_act" and "rel_pos_encoding" until it's really needed.
        diff_repr = {
            'rel_pos_encoding': self.diff_rel_pos_encoding(batch),
            **representations}
        diff_repr = diffusion.insert_diff_batch_dim(diff_repr, diff_batch_size, 
                special_keys=['pair', 'rel_pos_encoding'])
        diff_batch = diffusion.insert_diff_batch_dim(batch, diff_batch_size)

        # iterate over batch dim
        sample_ret_list = []
        for batch_i in range(len(batch['asym_id'])):
            sample_batch = tree_map(lambda x: x[batch_i], diff_batch)   # dict of (diff_batch, *)
            sample_repr = tree_map(lambda x: x[batch_i], diff_repr)
            sample_ret = self.diffusion_module(sample_repr, sample_batch)   # dict of (diff_batch, *)
            sample_ret_list.append(sample_ret)
        diff_ret = merge_list_to_dict(sample_ret_list)    # dict of (B, diff_batch, *)
        ret['diffusion_module'] = diff_ret

        ## confidence model
        if self.config.heads.confidence_head.weight > 0:
            # prepare needed keys and rollout
            representations['rel_pos_encoding'] = self.diff_rel_pos_encoding(batch)
            names_needed = ['frame_mask', 'all_centra_token_indice',
                    'all_centra_token_indice_mask']
            batch.update({k: label[k] for k in names_needed})
            diff_rollout_ret = diff_ret      # dict of (B, diff_batch, *)
            # Insert diffusion dim: (B, *) -> (B, diff_batch, *)
            diff_batch_size = diff_rollout_ret['final_atom_positions'].shape[1]
            diff_repr = tree_map(lambda x: x.detach(), representations)
            diff_repr = diffusion.insert_diff_batch_dim(diff_repr, diff_batch_size, 
                    special_keys=['pair', 'rel_pos_encoding'])
            diff_batch = diffusion.insert_diff_batch_dim(batch, diff_batch_size)
            diff_rollout_ret = tree_map(lambda x: x.detach(), diff_rollout_ret)
            
            # iterate over batch dim and diffusion dim
            sample_conf_ret_loflist = []
            for batch_i in range(len(batch['asym_id'])):
                sample_conf_ret_loflist.append([])
                for diff_i in range(diff_batch_size):
                    _slice_sample = lambda x: x[batch_i][diff_i: diff_i + 1]
                    sample_batch = tree_map(_slice_sample, diff_batch)  # dict of (1, *)
                    sample_repr = {k: v[batch_i] if k in ['pair', 'rel_pos_encoding'] 
                            else _slice_sample(v) for k, v in diff_repr.items()}
                    sample_rollout_ret = tree_map(_slice_sample, diff_rollout_ret)
                    
                    sample_conf_ret = self.confidence_head(sample_repr, sample_batch, sample_rollout_ret)
                    # drop keys to reduce memory
                    for k in ['logits_pae', 'logits_pde', 'logits_plddt', 'logits_resolved']:
                        if k in sample_conf_ret: del sample_conf_ret[k]
                    # remove batch_dim, which =1
                    sample_conf_ret = tree_map(lambda x: x.squeeze(0), sample_conf_ret)
                    sample_conf_ret_loflist[batch_i].append(sample_conf_ret)
            conf_ret = merge_loflist_to_dict(sample_conf_ret_loflist)    # dict of (B, diff_batch, *)
            ret['confidence_head'] = conf_ret
            
        return ret


class InputEmbedder(nn.Layer):
    """InputEmbedder

    Algorithm 2: Construct an initial 1D embedding
        +
    Algorithm 1, line 1-5

    """
    def __init__(self, channel_num, config, global_config):
        super(InputEmbedder, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        self.atom_attention_encoder = AtomAttentionEncoder(
            self.channel_num,
            self.config.atom_encoder,
            self.global_config)

        self.single_project = nn.Linear(
            self.channel_num['token_channel'] + 32 + 32 + 1,
            self.channel_num['token_channel'],
            bias_attr=False)

        self.single_to_pair_project = nn.Linear(
            self.channel_num['token_channel'] + 32 + 32 + 1,
            self.channel_num['token_pair_channel'],
            bias_attr=False)

        self.relative_position_encoding = RelativePositionEncoding(
            self.channel_num,
            self.config.relative_position_encoding,
            self.global_config)

        self.token_bond_project = nn.Linear(
            1, self.channel_num['token_pair_channel'],
            bias_attr=False)

    def forward(self, batch):
        ai, _, _, _ = self.atom_attention_encoder(batch, None, None, None)

        restype = nn.functional.one_hot(batch['restype'], 32)
        single_inputs_act = paddle.concat(
            [ai, restype, batch['profile'],
             batch['deletion_mean'].unsqueeze(axis=-1)], axis=-1)

        single_init_act = self.single_project(single_inputs_act)

        single_to_pair = self.single_to_pair_project(single_inputs_act)
        pair_init_act = single_to_pair.unsqueeze(axis=-3) + \
            single_to_pair.unsqueeze(axis=-2)
        pair_init_act += self.relative_position_encoding(batch)
        pair_init_act += self.token_bond_project(
            batch['token_bonds'].unsqueeze(axis=-1)
            )

        return single_inputs_act, single_init_act, pair_init_act


class EmbeddingsAndPairformer(nn.Layer):
    """Template module, MSA module, Pairformer

    Algorithm 1, line 8-13

    """

    def __init__(self, channel_num, config, global_config):
        super(EmbeddingsAndPairformer, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        pair_channel = channel_num['token_pair_channel']
        self.pair_norm = nn.LayerNorm(pair_channel)
        self.pair_project = nn.Linear(
            pair_channel, pair_channel, bias_attr=False)

        single_channel = channel_num['token_channel']
        self.single_norm = nn.LayerNorm(single_channel)
        self.single_project = nn.Linear(
            single_channel, single_channel, bias_attr=False)

        self.template_embedder = TemplateEmbedder(
            channel_num,
            config.template_module,
            global_config)
        self.msa_module = MsaModule(
            channel_num,
            config.msa_module,
            global_config)

        self.pairformer_stack = nn.LayerList()
        for _ in range(self.config.pairformer.num_block):
            self.pairformer_stack.append(Pairformer(
                self.channel_num, self.config.pairformer,
                self.global_config))

    def forward(self, batch, pair_init_act, pair_act,
                single_inputs_act, single_init_act, single_act,
                masks):
        pair_act = pair_init_act + self.pair_project(self.pair_norm(pair_act))
        pair_act += self.template_embedder(batch, pair_act)
        pair_act += self.msa_module(batch, pair_act, single_inputs_act, masks)

        single_act = single_init_act + self.single_project(
            self.single_norm(single_act))

        single_act, pair_act = stack_forward_with_recompute(
            self.pairformer_stack,
            (single_act, pair_act),
            (masks,),
            is_recompute=self.training)

        return single_act, pair_act


class TemplateEmbedder(nn.Layer):
    """Template module

    Algorithm 16

    """
    def __init__(self, channel_num, config, global_config):
        super(TemplateEmbedder, self).__init__()

        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        self.pair_norm = nn.LayerNorm(self.channel_num['token_pair_channel'])
        self.pair_project = nn.Linear(
            self.channel_num['token_pair_channel'],
            self.config.num_channel,
            bias_attr=False)

        self.template_project = nn.Linear(
            # template_distogram (32), backbone_frame_mask (1)
            # template_unit_vector (3), pseudo_beta_mask (1)
            # template_restype_i & j (32 * 2)
            39 + 1 + 3 + 1 + 32 * 2,
            self.config.num_channel,
            bias_attr=False)

        self.pairformer_stack = nn.LayerList()
        for _ in range(self.config.pairformer_stack.num_block):
            self.pairformer_stack.append(Pairformer(
                {
                    # NOTE: `modules.*` use `pair_channel`,
                    # while `modules_all_atom.*` use `token_pair_channel`
                    'pair_channel': self.config.num_channel,
                    'token_pair_channel': self.config.num_channel,

                    # NOTE: placeholder, as pairformer in template embedder
                    # only use pair_act
                    'token_channel': self.channel_num['token_channel'],
                },
                self.config.pairformer_stack,
                self.global_config,
                pair_only=True))

        self.out_norm = nn.LayerNorm(self.config.num_channel)
        self.out_projection = nn.Linear(
            self.config.num_channel,
            self.channel_num['token_pair_channel'],
            bias_attr=False)

    def forward(self, batch, pair_act):
        backbone_frame_mask = batch['template_backbone_frame_mask']
        out_act = 0.
        num_templates = backbone_frame_mask.shape[1]
        for t in range(num_templates):
            temp_distogram = modules.dgram_from_positions(
                batch['template_pseudo_beta'][:, t],
                num_bins=39,
                min_bin=3.25,
                max_bin=50.75
            )

            backbone_frame_mask_2d = backbone_frame_mask[:, t].unsqueeze(axis=-1) * \
                backbone_frame_mask[:, t].unsqueeze(axis=-2)

            pseudo_beta_mask = batch['template_pseudo_beta_mask'][:, t]
            pseudo_beta_mask_2d = pseudo_beta_mask.unsqueeze(axis=-1) * \
                pseudo_beta_mask.unsqueeze(axis=-2)

            act = paddle.concat([
                temp_distogram,
                backbone_frame_mask_2d.unsqueeze(axis=-1),
                batch['template_unit_vector'][:, t],
                pseudo_beta_mask_2d.unsqueeze(axis=-1)], axis=-1)

            asym_id = batch['asym_id']
            multichain_mask = asym_id.unsqueeze(axis=-1) == \
                asym_id.unsqueeze(axis=-2)
            multichain_mask = paddle.cast(multichain_mask, dtype=act.dtype)
            act *= multichain_mask.unsqueeze(-1)

            if 'template_restype' in batch:
                restype = batch['template_restype'][:, t]
            else:
                restype = batch['template_aatype'][:, t]

            restype_i = paddle.expand_as(
                restype.unsqueeze(axis=-2), backbone_frame_mask_2d)
            restype_j = paddle.expand_as(
                restype.unsqueeze(axis=-1), backbone_frame_mask_2d)

            restype_i = nn.functional.one_hot(restype_i, 32)
            restype_j = nn.functional.one_hot(restype_j, 32)
            act_t = paddle.concat([act, restype_i, restype_j], axis=-1)

            pair_act_t = self.pair_project(self.pair_norm(pair_act))
            v_t = pair_act_t + self.template_project(act_t)

            residual = stack_forward_with_recompute(
                self.pairformer_stack,
                v_t,
                ({'pair': multichain_mask},),
                is_recompute=self.training)
            v_t += residual
            out_act += self.out_norm(v_t)

        out_act /= num_templates
        out_pair_act = self.out_projection(nn.functional.relu(out_act))
        return out_pair_act


class MsaModule(nn.Layer):
    """MSA module

    Algorithm 8

    """
    def __init__(self, channel_num, config, global_config):
        super(MsaModule, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        self.msa_project = nn.Linear(
            32 + 1 + 1, self.config.msa_channel, bias_attr=False)

        # 449 = 32 (restype) + 32 (profile) + 1 (deletion_mean) + 384 (token_channel)
        self.single_project = nn.Linear(
            449, self.config.msa_channel, bias_attr=False) #FIXME: 449 is hard coded

        self.evoformer_stack = nn.LayerList()
        for _ in range(self.config.num_block):
            self.evoformer_stack.append(EvoformerV3(
                self.channel_num,
                self.config,
                self.global_config))

    def forward(self, batch, pair_act, single_inputs_act, masks):
        indices = paddle.randperm(batch['msa'].shape[1])
        indices = indices[:self.config.msa_depth]

        msa_mask = paddle.index_select(masks['msa'], indices, axis=1)
        msa_feat = self._create_msa_feature(batch, indices)
        msa_act = self.msa_project(msa_feat)
        msa_act += paddle.unsqueeze(
            self.single_project(single_inputs_act), axis=1)

        msa_act, pair_act = stack_forward_with_recompute(
            self.evoformer_stack,
            (msa_act, pair_act),
            ({
                'msa': msa_mask,
                'pair': masks['pair'],
            },),
            is_recompute=self.training)
        return pair_act

    def _create_msa_feature(self, batch, indices):
        msa = paddle.index_select(batch['msa'], indices, axis=1)
        has_deletion = paddle.index_select(
            batch['has_deletion'], indices, axis=1)
        deletion_value = paddle.index_select(
            batch['deletion_value'], indices, axis=1)

        msa_1hot = nn.functional.one_hot(msa, 32)
        msa_feat = [
            msa_1hot,
            has_deletion.unsqueeze(axis=-1),
            deletion_value.unsqueeze(axis=-1)
        ]
        return paddle.concat(msa_feat, axis=-1)


class Pairformer(nn.Layer):
    """Pairformer

    Algorithm 17, line 2-8

    """

    def __init__(self, channel_num, config, global_config,
                 pair_only=False):
        super(Pairformer, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config
        self.pair_only = pair_only

        use_dropout_nd = self.global_config.get('use_dropout_nd', False)

        self.triangle_multiplication_outgoing = TriangleMultiplication(
            self.channel_num,
            self.config.triangle_multiplication_outgoing,
            self.global_config,
            name='triangle_multiplication_outgoing')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_multiplication_outgoing)
        self.triangle_outgoing_dropout = nn.Dropout(
            dropout_rate, axis=dropout_axis) \
            if not use_dropout_nd else Dropout(
                    dropout_rate, axis=dropout_axis)

        self.triangle_multiplication_incoming = TriangleMultiplication(
            self.channel_num,
            self.config.triangle_multiplication_incoming,
            self.global_config,
            name='triangle_multiplication_incoming')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_multiplication_incoming)
        self.triangle_incoming_dropout = nn.Dropout(
            dropout_rate, axis=dropout_axis) \
            if not use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.triangle_attention_starting_node = TriangleAttention(
            self.channel_num,
            self.config.triangle_attention_starting_node,
            self.global_config,
            name='triangle_attention_starting_node')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_attention_starting_node)
        self.triangle_starting_dropout = nn.Dropout(
            dropout_rate, axis=dropout_axis) \
            if not use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.triangle_attention_ending_node = TriangleAttention(
            self.channel_num,
            self.config.triangle_attention_ending_node,
            self.global_config,
            name='triangle_attention_ending_node')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_attention_ending_node)
        self.triangle_ending_dropout = nn.Dropout(
            dropout_rate, axis=dropout_axis) \
            if not use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.pair_transition = TransitionV3(
            self.channel_num,
            self.config.pair_transition,
            self.global_config,
            'pair_transition')

        if not self.pair_only:
            self.single_attention_with_pair_bias = AttentionPairBias(
                self.channel_num['token_channel'],
                self.channel_num['token_channel'],
                self.channel_num['token_pair_channel'],
                self.config.single_attention_with_pair_bias.num_head,
                False)

            self.single_transition = TransitionV3(
                self.channel_num,
                self.config.single_transition,
                self.global_config,
                'single_transition')

        if self.pair_only:
            setattr(self, 'forward', self._forward_pair)
        else:
            setattr(self, 'forward', self._forward_all)

    def _forward_pair(self, pair_act, masks):
        pair_mask = masks['pair']
        residual = self.triangle_multiplication_outgoing(pair_act, pair_mask)
        pair_act += self.triangle_outgoing_dropout(residual)
        residual = self.triangle_multiplication_incoming(pair_act, pair_mask)
        pair_act += self.triangle_incoming_dropout(residual)

        residual = self.triangle_attention_starting_node(pair_act, pair_mask)
        pair_act += self.triangle_starting_dropout(residual)

        residual = self.triangle_attention_ending_node(pair_act, pair_mask)
        pair_act += self.triangle_ending_dropout(residual)

        pair_act += self.pair_transition(pair_act)

        return pair_act

    def _forward_all(self, single_act, pair_act, masks):
        pair_act = self._forward_pair(pair_act, masks)

        pair_mask = masks['pair']
        beta = paddle.zeros_like(pair_mask)
        single_act += self.single_attention_with_pair_bias(
            single_act, None, pair_act, beta)
        single_act += self.single_transition(single_act)

        return single_act, pair_act

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


class TransitionV3(nn.Layer):
    """Transition layer v3 in HelixFold3

    Algorithm 11
    """

    def __init__(self, channel_num, config, global_config,
                 transition_type):
        super(TransitionV3, self).__init__()
        assert transition_type in ['pair_transition', 'single_transition', 'msa_transition']
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config
        self.transition_type = transition_type

        # as HF3 use pairformer at different modules
        if transition_type == 'pair_transition':
            in_dim = channel_num['token_pair_channel']
        elif transition_type == 'single_transition':
            in_dim = channel_num['token_channel']
        elif transition_type == 'msa_transition':
            in_dim = channel_num['msa_channel']

        self.input_layer_norm = nn.LayerNorm(in_dim)

        nc = int(in_dim * self.config.num_intermediate_factor)
        self.proj_a = nn.Linear(
            in_dim, nc,
            bias_attr=False,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.KaimingNormal()))
        self.proj_b = nn.Linear(
            in_dim, nc,
            bias_attr=False,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.KaimingNormal()))

        if self.global_config.zero_init:
            last_init = nn.initializer.Constant(0.0)
        else:
            last_init = nn.initializer.TruncatedNormal()

        self.proj_out = nn.Linear(
            nc, in_dim,
            bias_attr=False,
            weight_attr=paddle.ParamAttr(initializer=last_init))

        self.swish = nn.Swish()

    def forward(self, x):
        x = self.input_layer_norm(x)

        def _transition_fn(x):
            a = self.proj_a(x)
            b = self.proj_b(x)
            return self.proj_out(self.swish(a) * b)

        if not self.training:
            sb_transition = subbatch(
                _transition_fn, [0], [1],
                self.global_config.subbatch_size, 1)
            x = sb_transition(x)

        else:
            x = _transition_fn(x)

        return x


class EvoformerV3(nn.Layer):
    """Modified evoformer for HelixFold-3 in MSA module

    Algorithm 8, line 6-13
    """
    def __init__(self, channel_num, config, global_config):
        super(EvoformerV3, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        use_dropout_nd = self.global_config.get('use_dropout_nd', False)

        self.outer_product_mean = OuterProductMean(
            self.channel_num,
            self.config.outer_product_mean,
            self.global_config,
            False,
            name='outer_product_mean')

        self.msa_pair_weighted_averaging = MSAPairWeightedAveraging(
            self.channel_num,
            self.config.msa_pair_weighted_averaging,
            self.global_config)

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.msa_pair_weighted_averaging)
        self.msa_averaging_dropout = nn.Dropout(
            dropout_rate, axis=dropout_axis) \
            if not use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.msa_transition = TransitionV3(
            self.channel_num,
            self.config.msa_transition,
            self.global_config,
            'msa_transition')

        self.triangle_multiplication_outgoing = TriangleMultiplication(
            self.channel_num,
            self.config.triangle_multiplication_outgoing,
            self.global_config,
            name='triangle_multiplication_outgoing')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_multiplication_outgoing)
        self.triangle_outgoing_dropout = nn.Dropout(
            dropout_rate, axis=dropout_axis) \
            if not use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.triangle_multiplication_incoming = TriangleMultiplication(
            self.channel_num,
            self.config.triangle_multiplication_incoming,
            self.global_config,
            name='triangle_multiplication_incoming')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_multiplication_incoming)
        self.triangle_incoming_dropout = nn.Dropout(
            dropout_rate, axis=dropout_axis) \
            if not use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.triangle_attention_starting_node = TriangleAttention(
            self.channel_num,
            self.config.triangle_attention_starting_node,
            self.global_config,
            name='triangle_attention_starting_node')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_attention_starting_node)
        self.triangle_starting_dropout = nn.Dropout(
            dropout_rate, axis=dropout_axis) \
            if not use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.triangle_attention_ending_node = TriangleAttention(
            self.channel_num,
            self.config.triangle_attention_ending_node,
            self.global_config,
            name='triangle_attention_ending_node')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_attention_ending_node)
        self.triangle_ending_dropout = nn.Dropout(
            dropout_rate, axis=dropout_axis) \
            if not use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.pair_transition = TransitionV3(
            self.channel_num,
            self.config.pair_transition,
            self.global_config,
            'pair_transition')

    def forward(self, msa_act, pair_act, masks):
        msa_mask, pair_mask = masks['msa'], masks['pair']

        pair_act += self.outer_product_mean(msa_act, msa_mask)

        residual = self.msa_pair_weighted_averaging(
            msa_act, pair_act, pair_mask)
        msa_act += self.msa_averaging_dropout(residual)
        msa_act += self.msa_transition(msa_act)

        residual = self.triangle_multiplication_outgoing(pair_act, pair_mask)
        pair_act += self.triangle_outgoing_dropout(residual)

        residual = self.triangle_multiplication_incoming(pair_act, pair_mask)
        pair_act += pair_act + self.triangle_incoming_dropout(residual)

        residual = self.triangle_attention_starting_node(pair_act, pair_mask)
        pair_act += self.triangle_starting_dropout(residual)

        residual = self.triangle_attention_ending_node(pair_act, pair_mask)
        pair_act += self.triangle_ending_dropout(residual)

        pair_act += self.pair_transition(pair_act)

        return msa_act, pair_act

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


class MSAPairWeightedAveraging(nn.Layer):
    """MSA per-row attention biased by the pair representation.

    A modified `modules.MSARowAttentionWithPairBias` for HelixFold3

    Algorithm 10
    """
    def __init__(self, channel_num, config, global_config):
        super(MSAPairWeightedAveraging, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        msa_channel = channel_num['msa_channel']
        pair_channel = channel_num['pair_channel']

        self.query_norm = nn.LayerNorm(msa_channel)
        self.feat_2d_norm = nn.LayerNorm(pair_channel)

        self.v_proj_w = paddle.create_parameter(
            [msa_channel, config.num_head, config.num_channel], 'float32',
            default_initializer=nn.initializer.XavierUniform())
        self.bias_proj_w = paddle.create_parameter(
            [pair_channel, config.num_head], 'float32',
            default_initializer=nn.initializer.Normal(
                std=1. / np.sqrt(pair_channel)))
        self.gating_proj_w = paddle.create_parameter(
            [msa_channel, config.num_head, config.num_channel], 'float32',
            default_initializer=nn.initializer.Constant(0.0))
        self.out_proj_w = paddle.create_parameter(
            [config.num_head, config.num_channel, msa_channel], 'float32',
            default_initializer=nn.initializer.XavierUniform())

    def forward(self, msa_act, pair_act, pair_mask):
        mask_bias = 1e9 * (pair_mask - 1.)
        mask_bias = paddle.unsqueeze(mask_bias, axis=[1])

        pair_act = self.feat_2d_norm(pair_act)
        msa_act = self.query_norm(msa_act)

        v = paddle.einsum('nbqa,ahc->nbqhc', msa_act, self.v_proj_w)
        bias = paddle.einsum('nqkc,ch->nhqk', pair_act, self.bias_proj_w)
        gating = paddle.einsum('nbqa,ahc->nbqhc', msa_act, self.gating_proj_w)
        gating = nn.functional.sigmoid(gating)

        weights = nn.functional.softmax(bias + mask_bias)
        weighted_avg = paddle.einsum('nhqk,nbkhc->nbqhc', weights, v)
        out_act = gating * weighted_avg
        out_act = paddle.einsum('nbqhc,hco->nbqo', out_act, self.out_proj_w)
        return out_act


class ConfidenceHead(nn.Layer):
    """Confidence head 
    Algorithm 31
    """
    def __init__(self, channel_num, config, global_config):
        super(ConfidenceHead, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        self.v_bins = paddle.arange(0, 22, 0.5)
        token_channel = channel_num['token_channel']
        token_pair_channel = channel_num['token_pair_channel']

        self.atom_encoder = AtomAttentionEncoder(
                channel_num, self.config.atom_encoder, self.global_config)
        self.ln_s = nn.LayerNorm(token_channel * 2 + 32 + 32 + 1)
        self.lin_s_left = nn.Linear(
                token_channel * 2 + 32 + 32 + 1, token_pair_channel)
        self.lin_s_right = nn.Linear(
                token_channel * 2 + 32 + 32 + 1, token_pair_channel)
        self.lin_dij = nn.Linear(len(self.v_bins) + 1, token_pair_channel)

        self.pairformer_stack = nn.LayerList()
        for _ in range(self.config.pairformer.num_block):
            self.pairformer_stack.append(Pairformer(
                self.channel_num, self.config.pairformer,
                self.global_config))
        
        self.atom_decoder = AtomAttentionDecoder(
                channel_num, self.config.atom_decoder, self.global_config)
        self.ln_pae = nn.LayerNorm(token_pair_channel)
        self.lin_pae = nn.Linear(token_pair_channel, self.config.b_pae)
        self.ln_pde = nn.LayerNorm(token_pair_channel)
        self.lin_pde = nn.Linear(token_pair_channel, self.config.b_pde)
        self.ln_si = nn.LayerNorm(token_channel)
        self.lin_plddt = nn.Linear(token_pair_channel, self.config.b_plddt)
        self.lin_resolved = nn.Linear(token_pair_channel, 2)

    def _atom_value_to_token(self, atom_value, token_indice):
        token_value = paddle.stack([v[index] for v, index 
                in zip(atom_value, token_indice)])     # (B, N_token, 3)
        return token_value

    def forward(self, representations, batch, rollout_value):
        """forward"""
        s_inputs = representations['single_inputs']  # (B, N_token, d1)
        si = representations['single'] # (B, N_token, d1)
        zij = representations['pair']   # (B, N_token, N_token, d2)
        xl_pred = rollout_value['final_atom_positions'] # (B, N_atom, 3)
        xl_mask = rollout_value['final_atom_mask']
        xl_pred = diffusion.CentreRandomAugmentation(xl_pred, xl_mask)
        
        ## encode
        ai, ql_skip, cl_skip, p_lm_skip = self.atom_encoder(
                feature=batch, rl=xl_pred / self.config.sigma_data, s_trunk=si, zij=zij) 
        ai = self.ln_s(paddle.concat([ai, s_inputs], -1))
        zij += self.lin_s_left(ai)[:, :, None] + self.lin_s_right(ai)[:, None]
        rep_pos = self._atom_value_to_token(xl_pred, batch['all_centra_token_indice'])
        dij = points_self_dist(rep_pos)
        zij += self.lin_dij(one_hot(dij, self.v_bins))

        ## pairformer
        masks = {
            'msa': batch['msa_mask'],
            'pair': batch['seq_mask'].unsqueeze(axis=1) * batch['seq_mask'].unsqueeze(axis=2),
        }
        si, zij = stack_forward_with_recompute(
            self.pairformer_stack,
            (si, zij),
            (masks,),
            is_recompute=self.training)

        ## decode
        logits_pae = self.lin_pae(self.ln_pae(zij))
        logits_pde = self.lin_pde(self.ln_pde(zij + zij.transpose([0, 2, 1, 3])))
        atom_token_uid = batch['ref_token2atom_idx']
        atom_mask = paddle.ones_like(atom_token_uid)
        si = self.ln_si(si)
        al = self.atom_decoder(si, ql_skip, cl_skip, p_lm_skip,
                                     atom_token_uid, atom_mask)
        logits_plddt = self.lin_plddt(al)
        logits_resolved = self.lin_resolved(al)
        ret = {
            'logits_pae': logits_pae,   # (B, N_token, N_token, b_pae)
            'logits_pde': logits_pde,   # (B, N_token, N_token, b_pde)
            'logits_plddt': logits_plddt,   # (B, N_atom, b_plddt)
            'logits_resolved': logits_resolved, # (B, N_atom, 2)
        }
        if not self.training:
            metrics = self.get_metrics(ret, rollout_value, batch)
            ret.update(metrics)
        return ret
    

    def get_metrics(self, logit_value, structure_value, batch):
        """
        Args:
            logits_plddt: (B, N_atom, b_plddt)
            logits_pae: (B, N_token, N_token, b_pae)
        
        Returns:
            atom_plddts: (B, N_atom)
            mean_plddt: (B,)
            pae: (B, N_token, N_token)
            ptm: (B,)
            iptm: (B,)
            has_clash: (B,)
            ranking_confidence: (B,)
        """
        B = logit_value['logits_pae'].shape[0]
        breaks_pae = paddle.linspace(0., 
                self.config.stride_pae * self.config.b_pae,
                self.config.b_pae - 1)
        inputs = {
            'frame_mask': batch['frame_mask'],
            'asym_id': batch['asym_id'],
            'breaks_pae': paddle.tile(breaks_pae, [B, 1]),
            'perm_asym_id': batch['perm_asym_id'],
            'is_polymer_chain': ((batch['is_protein_aa'] + 
                    batch['is_dna_aa'] + batch['is_rna_aa']) > 0),
            **logit_value,
            **structure_value,
        }

        ret_list = []
        for i in range(B):
            cur_input = tree_map(lambda x: x[i].numpy(), inputs)
            ret = get_all_atom_confidence_metrics(cur_input)
            ret_list.append(ret)
            
        metrics = {}
        for k, v in ret_list[0].items():
            metrics[k] = paddle.to_tensor(np.stack([r[k] for r in ret_list]))
        return metrics


def one_hot(value, v_bins):
    """
    Args:
        value: (*)
        v_bins: (M)
    Returns:
        (*, M)
    """
    num_bins = v_bins.shape[0] + 1
    value = value[..., None]
    counts = paddle.sum(value > v_bins, -1)
    emb = F.one_hot(counts, num_classes=num_bins)
    return emb


def points_self_dist(x):
    """x: (..., n, 3)"""
    dist2 = paddle.sum((x.unsqueeze(-2) - x.unsqueeze(-3)) ** 2, -1)
    return paddle.sqrt(dist2 + 1e-8) # (..., n, n)


def stack_forward_with_recompute(stack, iter_args, common_args, is_recompute=True):
    ret = iter_args
    for iteration in stack:
        if isinstance(ret, paddle.Tensor):
            ret = (ret,)
        ret = recompute_wrapper(
            iteration,
            *ret,
            *common_args,
            is_recompute=is_recompute)

    return ret


def merge_list_to_dict(sample_list):
    """
    merge sample list into a batch dict
    """
    batch = {}
    for k in sample_list[0]:
        if isinstance(sample_list[0][k], paddle.Tensor):
            batch[k] = paddle.stack([sr[k] for sr in sample_list])
        else:
            batch[k] = [sr[k] for sr in sample_list]
    return batch


def merge_loflist_to_dict(sample_loflist):
    """
    merge sample list of list into a batch dict
    """
    batch = {}
    for k in sample_loflist[0][0]:
        if isinstance(sample_loflist[0][0][k], paddle.Tensor):
            batch[k] = paddle.stack([paddle.stack([s[k] for s in slist])
                    for slist in sample_loflist])
        else:
            batch[k] = [[s[k] for s in slist] 
                    for slist in sample_loflist]
    return batch