"""
dap version of modules_all_atom.py
"""
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import gc
from copy import deepcopy
from helixfold.model import chain_align_np_aa, lddt, modules
from ppfleetx.distributed.protein_folding import dap
from ppfleetx.models.protein_folding.outer_product_mean import (OuterProductMean, )
from helixfold.model.dist.attentions import (
    TriangleMultiplication,
    TriangleAttention, )
from ppfleetx.models.protein_folding.common import (
    Dropout,
    recompute_wrapper,
    dgram_from_positions, 
    subbatch, )

from helixfold.model.diffusion import (
    AttentionPairBias, CentreRandomAugmentation
)
from helixfold.model.modules_all_atom import ConfidenceHead, points_self_dist, one_hot
from helixfold.model.utils import fused_act_bias_wrapper


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

        pair_act = self.feat_2d_norm(pair_act) # [b, n_tok, n_tok, 128]
        msa_act = self.query_norm(msa_act)  # [b, n_seq, n_tok, 64]
        eisum_sp2_gt2 = subbatch(paddle.einsum, [1], [2], self.global_config.subbatch_size, 2) 
        v = paddle.einsum('nbqa,ahc->nbqhc', msa_act, self.v_proj_w)   # [b, n_seq, n_tok, nh, 32]
        bias = paddle.einsum('nqkc,ch->nhqk', pair_act, self.bias_proj_w)  # [b, nh, n_tok(q), n_tok(k)]

        gating = paddle.einsum('nbqa,ahc->nbqhc', msa_act, self.gating_proj_w)  # [b, nh, n_tok(q), n_tok(k)]
        gating = nn.functional.sigmoid(gating)
        weights = nn.functional.softmax(bias + mask_bias)  # [b, nh, n_tok(q), n_tok(k)]
        weighted_avg = eisum_sp2_gt2('nhqk,nbkhc->nbqhc', weights, v)  # [b, n_seq, n_tok(q), nh, 32]
        out_act = gating * weighted_avg  # [b, n_seq, n_tok(q), 64]
        out_act = eisum_sp2_gt2('nbqhc,hco->nbqo', out_act, self.out_proj_w) # [b, n_seq, n_tok(q), 64]
        return out_act


class DistEmbeddingsAndPairformer(nn.Layer):
    """Template module, MSA module, Pairformer

    Algorithm 1, line 8-13

    """

    def __init__(self, channel_num, config, global_config):
        super(DistEmbeddingsAndPairformer, self).__init__()
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
        """ tbd. """
        pair_act = dap.scatter(pair_act, axis=1)  # [b, n_tok, n_tok, d] -> [b, n_tok/D, n_tok, d]
        pair_init_act = dap.scatter(pair_init_act, axis=1)
        if not self.training and self.global_config.low_memory is True:
            # pair_act = pair_init_act
            # pair_act.add_(self.pair_project(self.pair_norm(pair_act)))

            # pair_init_act.add_(self.pair_project(self.pair_norm(pair_act)) )
            # pair_act = pair_init_act

            pair_act = self.pair_project(self.pair_norm(pair_act))
            pair_act.add_(pair_init_act)
            del pair_init_act
            gc.collect()
        else:
            pair_act = pair_init_act + self.pair_project(self.pair_norm(pair_act))
        if not self.training and self.global_config.low_memory is True:
            pair_act.add_(self.template_embedder(batch, pair_act))
        else:
            pair_act += self.template_embedder(batch, pair_act)
        if not self.training and self.global_config.low_memory is True:
            pair_act.add_(self.msa_module(batch, pair_act, single_inputs_act, masks))
        else:
            pair_act += self.msa_module(batch, pair_act, single_inputs_act, masks)

        single_act = single_init_act + self.single_project(
            self.single_norm(single_act))

        # pair_act = dap.scatter(pair_act, axis=1) # [b, n_tok, n_tok, d] -> [b, n_tok/D, n_tok, d]
        single_act, pair_act = stack_forward_with_recompute(
            self.pairformer_stack,
            (single_act, pair_act),
            (masks,),
            is_recompute=self.training)
        pair_act = dap.gather(pair_act, axis=1) # [b, n_tok/D, n_tok, d] -> [b, n_tok, n_tok, d]
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

        # # TODO(zhukunrui) FIXME
        config_pairformer_stack_ = deepcopy(self.config.pairformer_stack)
        # config_pairformer_stack_.triangle_multiplication_outgoing.num_intermediate_channel = 64 
        # config_pairformer_stack_.triangle_multiplication_incoming.num_intermediate_channel = 64 

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
                config_pairformer_stack_,
                self.global_config,
                pair_only=True))

        self.out_norm = nn.LayerNorm(self.config.num_channel)
        self.out_projection = nn.Linear(
            self.config.num_channel,
            self.channel_num['token_pair_channel'],
            bias_attr=False)

    def forward(self, batch, pair_act):
        """ tbd. """
        # scatter moved to DistEmbeddingsAndPairformer
        # pair_act = dap.scatter(pair_act, axis=1)  # [b, n_tok, n_tok, d] -> [b, n_tok/D, n_tok, d]
        backbone_frame_mask = batch['template_backbone_frame_mask']
        out_act = 0.
        total_num_templates = backbone_frame_mask.shape[1]
        if self.training:
            p0 = 1.0 / (total_num_templates + 1) #TODO: move default value to config
            if hasattr(self.config, 'zero_prob'):
                p0 = self.config.zero_prob
            assert 0 < p0 < 1.0
            p_sample = [p0] + [(1.0 - p0) / total_num_templates] * total_num_templates
            rand_num_temp = np.random.choice(total_num_templates + 1, 1, p=p_sample)
            num_templates = min(rand_num_temp, self.config.max_templates)
            indices = np.random.choice(total_num_templates, 
                    num_templates, replace=False)
        else:
            num_templates = min(total_num_templates, self.config.max_templates)
            indices = list(range(num_templates))

        asym_id = batch['asym_id']
        multichain_mask = asym_id.unsqueeze(axis=-1) == \
            asym_id.unsqueeze(axis=-2)
        
        if num_templates == 0:
            # no template feature, return 0 tensor
            out_act = paddle.zeros(multichain_mask.shape + [self.config.num_channel])
            out_pair_act = self.out_projection(nn.functional.relu(out_act))
            return out_pair_act

        for t in indices:
            temp_distogram = modules.dgram_from_positions(
                batch['template_pseudo_beta'][:, t],
                num_bins=39,
                min_bin=3.25,
                max_bin=50.75
            ) # [b, n_tok, n_tok, 39]

            backbone_frame_mask_2d = backbone_frame_mask[:, t].unsqueeze(axis=-1) * \
                backbone_frame_mask[:, t].unsqueeze(axis=-2)

            pseudo_beta_mask = batch['template_pseudo_beta_mask'][:, t]
            pseudo_beta_mask_2d = pseudo_beta_mask.unsqueeze(axis=-1) * \
                pseudo_beta_mask.unsqueeze(axis=-2)

            # [1, num_tok, num_tok, 44]
            act = paddle.concat([
                temp_distogram,
                backbone_frame_mask_2d.unsqueeze(axis=-1),
                batch['template_unit_vector'][:, t],
                pseudo_beta_mask_2d.unsqueeze(axis=-1)], axis=-1)

            multichain_mask.cast_(act.dtype)
            act *= multichain_mask.unsqueeze(-1) # .unsqueeze([1, -1])

            act = dap.scatter(act, axis=1)  # [b, n_tok, n_tok, d] -> [b, n_tok/D, n_tok, d]
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
            restype_i = dap.scatter(restype_i, axis=1)
            restype_j = dap.scatter(restype_j, axis=1)
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
            del residual, v_t, pair_act_t
            gc.collect()

        out_act /= num_templates
        out_pair_act = self.out_projection(nn.functional.relu(out_act))
        return out_pair_act


class MsaModule(nn.Layer):
    """MSA module

    Algorithm 8

    """
    def __init__(self, channel_num, config, global_config):
        """ tbd. """
        super(MsaModule, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        self.msa_project = nn.Linear(
            32 + 1 + 1, self.config.msa_channel, bias_attr=False)

        # 449 = 32 (restype) + 32 (profile) + 1 (deletion_mean) + 384 (token_channel)
        self.single_project = nn.Linear(
            449, self.config.msa_channel, bias_attr=False)

        self.evoformer_stack = nn.LayerList()
        for _ in range(self.config.num_block):
            self.evoformer_stack.append(EvoformerV3(
                self.channel_num,
                self.config,
                self.global_config))

    def forward(self, batch, pair_act, single_inputs_act, masks):
        """ tbd. """
        indices = paddle.randperm(batch['msa'].shape[1])
        indices = indices[:self.config.msa_depth]

        msa_mask = paddle.index_select(masks['msa'], indices, axis=1)
        msa_feat = self._create_msa_feature(batch, indices)
        msa_act = self.msa_project(msa_feat)
        if not self.training and self.global_config.low_memory is True:
            msa_act.add_(paddle.unsqueeze(
                self.single_project(single_inputs_act), axis=1))
        else:
            msa_act += paddle.unsqueeze(
            self.single_project(single_inputs_act), axis=1)

        msa_act = dap.scatter(msa_act, axis=2) # [b, n_seq, n_tok, d] -> [b, n_tok, n_tok/D, d]
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
        # pair_act [b, n_tok/D, n_tok, d]
        pair_mask = masks['pair']  # [b, n_tok, n_tok, d]
        pair_mask_row = dap.scatter(pair_mask, axis=1)
        pair_mask_col = dap.scatter(pair_mask, axis=2)
        residual = self.triangle_multiplication_outgoing(pair_act, pair_mask_row)
        if not self.training and self.global_config.low_memory is True:
            residual = self.triangle_outgoing_dropout(residual)
            pair_act.add_(residual)
            del residual
            gc.collect()
        else:      
            pair_act += self.triangle_outgoing_dropout(residual)
        pair_act = dap.row_to_col(pair_act)  # [b, n_tok/D, n_tok, d] ->  [b, n_tok, n_tok/D, d]
        residual = self.triangle_multiplication_incoming(pair_act, pair_mask_col)
        if not self.training and self.global_config.low_memory is True:
            residual = self.triangle_incoming_dropout(residual)
            pair_act.add_(residual)
            del residual
            gc.collect()
        else:
            pair_act += self.triangle_incoming_dropout(residual)
        pair_act = dap.col_to_row(pair_act) # [b, n_tok, n_tok/D, d] ->  [b, n_tok/D, n_tok, d]
        residual = self.triangle_attention_starting_node(pair_act, pair_mask_row)
        if not self.training and self.global_config.low_memory is True:
            residual = self.triangle_starting_dropout(residual)
            pair_act.add_(residual)
            del residual
            gc.collect()
        else:
            pair_act += self.triangle_starting_dropout(residual)
        pair_act = dap.row_to_col(pair_act)  # [b, n_tok/D, n_tok, d] ->  [b, n_tok, n_tok/D, d]
        residual = self.triangle_attention_ending_node(pair_act, pair_mask_col)
        if not self.training and self.global_config.low_memory is True:
            residual = self.triangle_ending_dropout(residual)
            pair_act.add_(residual)
            pair_act.add_(self.pair_transition(pair_act))
            del residual
            gc.collect()          
        else:
            pair_act += self.triangle_ending_dropout(residual)
            pair_act += self.pair_transition(pair_act)
        pair_act = dap.col_to_row(pair_act)  # [b, n_tok, n_tok/D, d] ->  [b, n_tok/D, n_tok, d]
        return pair_act

    def _forward_all(self, single_act, pair_act, masks):
        pair_act = self._forward_pair(pair_act, masks)
        pair_act = dap.gather(pair_act, axis=1)  # [b, n_tok/D, n_tok, d] -> [b, n_tok, n_tok, d]
        pair_mask = masks['pair']
        beta = paddle.zeros_like(pair_mask)
        single_act += self.single_attention_with_pair_bias(
            single_act, None, pair_act, beta)
        single_act += self.single_transition(single_act)

        pair_act = dap.scatter(pair_act, axis=1)  # [b, n_tok, n_tok, d] -> [b, n_tok/D, n_tok, d]
        return single_act, pair_act

    def _parse_dropout_params(self, module):
        """ tbd. """
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

        # FIXME: @xueyang02, may have different channel size
        # as af3 use pairformer at different modules
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
        self.first_run = True
        self.proj_ab_weight = None

    def forward(self, x):
        """ tbd. """
        x = self.input_layer_norm(x)

        def _transition_fn(x):
          try:
            from paddle.base.layer_helper import LayerHelper
            if self.first_run:
                self.proj_ab_weight = paddle.concat([self.proj_a.weight, self.proj_b.weight], axis=-1)
                self.first_run = False
                del self.proj_a
                del self.proj_b
                
            ffn1_out = paddle.matmul(x, self.proj_ab_weight)
            ffn1_out = fused_act_bias_wrapper(ffn1_out, None, act_method="swiglu")
            return self.proj_out(ffn1_out)
          except:
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
        """ tbd. """
        # msa_act: [b, n_seq, n_tok/D, d]
        # pair_act: [b, n_tok/D, n_tok, d]
        msa_mask, pair_mask = masks['msa'], masks['pair']

        # pair_act = dap.scatter(pair_act, axis=1) # [b, n_tok, n_tok, d] -> [b, n_tok/D, n_tok, d]
        product_mean = self.outer_product_mean(msa_act, msa_mask)

        if not self.training and self.global_config.low_memory is True:
            pair_act.add_(product_mean)
            del product_mean
            gc.collect()
        else:
            pair_act += product_mean
        pair_act = dap.gather(pair_act, axis=1) # [b, n_tok/D, n_tok, d] -> [b, n_tok, n_tok, d]

        msa_act = dap.col_to_row(msa_act)
        residual = self.msa_pair_weighted_averaging(
            msa_act, pair_act, pair_mask)
        if not self.training and self.global_config.low_memory is True:
            residual = self.msa_averaging_dropout(residual)
            msa_act.add_(residual)
            msa_act.add_(self.msa_transition(msa_act))
            del residual
            gc.collect()
        else:
            msa_act += self.msa_averaging_dropout(residual)
            msa_act += self.msa_transition(msa_act)
        msa_act = dap.gather(msa_act, axis=1) # [b, n_seq/D, n_tok, d] -> [b, n_seq, n_tok, d]

        pair_mask_row = dap.scatter(pair_mask, axis=1)
        pair_mask_col = dap.scatter(pair_mask, axis=2)
        pair_act = dap.scatter(pair_act, axis=1)  # [b, n_tok, n_tok, d] ->  [b, n_tok/D, n_tok, d]
        residual = self.triangle_multiplication_outgoing(pair_act, pair_mask_row)
        if not self.training and self.global_config.low_memory is True:
            residual = self.triangle_outgoing_dropout(residual)
            pair_act.add_(residual)
            del residual
            gc.collect()
        else:
            pair_act += self.triangle_outgoing_dropout(residual)

        pair_act = dap.row_to_col(pair_act)  # [b, n_tok/D, n_tok, d] ->  [b, n_tok, n_tok/D, d]
        residual = self.triangle_multiplication_incoming(pair_act, pair_mask_col)
        if not self.training and self.global_config.low_memory is True:
            residual = pair_act + self.triangle_incoming_dropout(residual)
            pair_act.add_(residual)
            del residual
            gc.collect()
        else:
            pair_act += pair_act + self.triangle_incoming_dropout(residual)

        pair_act = dap.col_to_row(pair_act) # [b, n_tok, n_tok/D, d] ->  [b, n_tok/D, n_tok, d]
        residual = self.triangle_attention_starting_node(pair_act, pair_mask_row)
        if not self.training and self.global_config.low_memory is True:
            residual = self.triangle_starting_dropout(residual)
            pair_act.add_(residual)
            del residual
            gc.collect()    
        else:
            pair_act += self.triangle_starting_dropout(residual)

        pair_act = dap.row_to_col(pair_act)  # [b, n_tok/D, n_tok, d] ->  [b, n_tok, n_tok/D, d]
        residual = self.triangle_attention_ending_node(pair_act, pair_mask_col)
        if not self.training and self.global_config.low_memory is True:
            residual = self.triangle_ending_dropout(residual)
            pair_act.add_(residual)
            pair_act.add_(self.pair_transition(pair_act))
            del residual
            gc.collect()          
        else:
            pair_act += self.triangle_ending_dropout(residual)

            pair_act += self.pair_transition(pair_act)
        pair_act = dap.col_to_row(pair_act)  # [b, n_tok, n_tok/D, d] ->  [b, n_tok/D, n_tok, d]
        msa_act = dap.scatter(msa_act, axis=2)  # [b, n_seq, n_tok, d] -> [b, n_seq, n_tok/D, d]
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


class DistConfidenceHead(ConfidenceHead):
    """Confidence head with distributed PairFormer
    Algorithm 31
    """
    def __init__(self, channel_num, config, global_config):
        super(DistConfidenceHead, self).__init__(
            channel_num=channel_num, config=config, global_config=global_config)

        # re-init self.pairformer_stack with distributed PairFormer
        self.pairformer_stack = nn.LayerList()
        for _ in range(self.config.pairformer.num_block):
            self.pairformer_stack.append(Pairformer(
                self.channel_num, self.config.pairformer,
                self.global_config))

    def forward(self, representations, batch, rollout_value):
        """forward"""
        s_inputs = representations['single_inputs']  # (B, N_token, d1)
        si = representations['single'] # (B, N_token, d1)
        zij = representations['pair']   # (B, N_token, N_token, d2)
        xl_pred = rollout_value['final_atom_positions'] # (B, N_atom, 3)
        xl_mask = rollout_value['final_atom_mask']
        xl_pred = CentreRandomAugmentation(xl_pred, xl_mask)
        
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
        zij = dap.scatter(zij, axis=1) # [b, n_tok, n_tok, d] -> [b, n_tok/D, n_tok, d]
        si, zij = stack_forward_with_recompute(
            self.pairformer_stack,
            (si, zij),
            (masks,),
            is_recompute=self.training)
        zij = dap.gather(zij, axis=1) # [b, n_tok/D, n_tok, d] -> [b, n_tok, n_tok, d]
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


def stack_forward_with_recompute(stack, iter_args, common_args, is_recompute=True):
    """ tbd. """
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