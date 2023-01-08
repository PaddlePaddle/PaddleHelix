import paddle
import paddle.nn as nn
import dap
from paddle.distributed.fleet.utils import recompute
from tools import all_atom, residue_constants
from layers.basics import (
    MSAColumnAttention, 
    MSARowAttentionWithPairBias, 
    MSAColumnGlobalAttention, 
    Transition,
    OuterProductMean,
    TriangleAttention,
    TriangleMultiplication,
    dgram_from_positions
)
from layers.embeddings import TemplateEmbedding


def recompute_wrapper(func, *args, is_recompute=True):
    """Function wrapper for recompute"""
    if is_recompute:
        return recompute(func, *args)
    else:
        return func(*args)


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
        self.msa_row_attn_dropout = nn.Dropout(dropout_rate, axis=dropout_axis)

        if self.is_extra_msa:
            self.msa_column_global_attention = MSAColumnGlobalAttention(
                channel_num, config.msa_column_attention, global_config)
            dropout_rate, dropout_axis = self._parse_dropout_params(
                self.msa_column_global_attention)
            self.msa_col_attn_dropout = nn.Dropout(
                dropout_rate, axis=dropout_axis)
        else:
            self.msa_column_attention = MSAColumnAttention(
                channel_num, config.msa_column_attention, global_config)
            dropout_rate, dropout_axis = self._parse_dropout_params(
                self.msa_column_attention)
            self.msa_col_attn_dropout = nn.Dropout(
                dropout_rate, axis=dropout_axis)

        self.msa_transition = Transition(
            channel_num, self.config.msa_transition, self.global_config,
            is_extra_msa, 'msa_transition')
        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.msa_transition)
        self.msa_transition_dropout = nn.Dropout(
            dropout_rate, axis=dropout_axis)

        # OuterProductMean
        self.outer_product_mean = OuterProductMean(channel_num,
                    self.config.outer_product_mean, self.global_config,
                    self.is_extra_msa, name='outer_product_mean')

        # Dropout
        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.outer_product_mean)
        self.outer_product_mean_dropout = nn.Dropout(
            dropout_rate, axis=dropout_axis)

        # Triangle Multiplication.
        self.triangle_multiplication_outgoing = TriangleMultiplication(channel_num,
                    self.config.triangle_multiplication_outgoing, self.global_config,
                    name='triangle_multiplication_outgoing')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_multiplication_outgoing)
        self.triangle_outgoing_dropout = nn.Dropout(dropout_rate, axis=dropout_axis)

        self.triangle_multiplication_incoming = TriangleMultiplication(channel_num,
                    self.config.triangle_multiplication_incoming, self.global_config,
                    name='triangle_multiplication_incoming')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_multiplication_incoming)
        self.triangle_incoming_dropout = nn.Dropout(dropout_rate, axis=dropout_axis)

        # TriangleAttention.
        self.triangle_attention_starting_node = TriangleAttention(channel_num,
                    self.config.triangle_attention_starting_node, self.global_config,
                    name='triangle_attention_starting_node')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_attention_starting_node)
        self.triangle_starting_dropout = nn.Dropout(dropout_rate, axis=dropout_axis)

        self.triangle_attention_ending_node = TriangleAttention(channel_num,
                    self.config.triangle_attention_ending_node, self.global_config,
                    name='triangle_attention_ending_node')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_attention_ending_node)
        self.triangle_ending_dropout = nn.Dropout(dropout_rate, axis=dropout_axis)

        # Pair transition.
        self.pair_transition = Transition(
            channel_num, self.config.pair_transition, self.global_config,
            is_extra_msa, 'pair_transition')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.pair_transition)
        self.pair_transition_dropout = nn.Dropout(dropout_rate, axis=dropout_axis)

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

    def forward(self, 
        msa_act, # [1, 512, len_dim, 256], dtype='float32'
        pair_act, # [1, len_dim, len_dim, 128], dtype='float32'
        msa_mask, # [1, 512, len_dim], dtype='float32'
        pair_mask # [1, len_dim, len_dim], dtype='float32'
    ):
        # [B, N_seq//dap_size, N_res, c_m]
        residual = self.msa_row_attention_with_pair_bias(
            msa_act, msa_mask, pair_act)
        residual = self.msa_row_attn_dropout(residual)
        msa_act = msa_act + residual

        # [B, N_seq//dap_size, N_res, c_m] => [B, N_seq, N_res//dap_size, c_m]
        msa_act = dap.row_to_col(msa_act)

        if self.is_extra_msa:
            # [B, N_seq, N_res//dap_size, c_m]
            residual = self.msa_column_global_attention(msa_act, msa_mask)
            residual = self.msa_col_attn_dropout(residual)
            msa_act = msa_act + residual

            # [B, N_seq, N_res//dap_size, c_m]
            residual = self.msa_transition(msa_act)
            residual = self.msa_transition_dropout(residual)
            msa_act = msa_act + residual

        else:
            # [B, N_seq, N_res//dap_size, c_m]
            residual = self.msa_column_attention(msa_act, msa_mask)
            residual = self.msa_col_attn_dropout(residual)
            msa_act = msa_act + residual

            # [B, N_seq, N_res//dap_size, c_m]
            residual = self.msa_transition(msa_act)
            residual = self.msa_transition_dropout(residual)
            msa_act = msa_act + residual

        # return msa_act, pair_act, pair_mask # 128GB
        
        # [B, N_res//dap_size, N_res, c_z]
        residual = self.outer_product_mean(msa_act, msa_mask)
        residual = self.outer_product_mean_dropout(residual)
        pair_act = pair_act + residual

        # return msa_act, pair_act, pair_mask # single-thread computation 129 GB
        # [B, N_seq, N_res//dap_size, c_m] => [B, N_seq//dap_size, N_res, c_m]
        msa_act = dap.all_to_all(msa_act, in_axis=1, out_axis=2)

        # scatter if using dap, otherwise do nothing
        pair_mask_row = dap.scatter(pair_mask, axis=1)
        pair_mask_col = dap.scatter(pair_mask, axis=2)

        # [B, N_res//dap_size, N_res, c_z]
        residual = self.triangle_multiplication_outgoing(pair_act, pair_mask_row)
        residual = self.triangle_outgoing_dropout(residual)
        pair_act = pair_act + residual

        # return msa_act, pair_act, pair_mask # 141 GB
    
        # [B, N_res//dap_size, N_res, c_z] => [B, N_res, N_res//dap_size, c_z]
        pair_act = dap.row_to_col(pair_act)
        # [B, N_res, N_res//dap_size, c_z]
        residual = self.triangle_multiplication_incoming(pair_act, pair_mask_col)
        residual = self.triangle_incoming_dropout(residual)
        pair_act = pair_act + residual

        # return msa_act, pair_act, pair_mask # 141 GB
        
        # [B, N_res, N_res//dap_size, c_z] => [B, N_res//dap_size, N_res, c_z]
        pair_act = dap.col_to_row(pair_act)
        # [B, N_res//dap_size, N_res, c_z]
        residual = self.triangle_attention_starting_node(pair_act, pair_mask_row)
        residual = self.triangle_starting_dropout(residual)
        pair_act = pair_act + residual
        
        # return msa_act, pair_act, pair_mask # 149 GB
    
        # [B, N_res//dap_size, N_res, c_z] => [B, N_res, N_res//dap_size, c_z]
        pair_act = dap.row_to_col(pair_act)
        # [B, N_res, N_res//dap_size, c_z]
        residual = self.triangle_attention_ending_node(pair_act, pair_mask_col)
        residual = self.triangle_ending_dropout(residual)
        pair_act = pair_act + residual

        residual = self.pair_transition(pair_act)
        residual = self.pair_transition_dropout(residual)
        pair_act = pair_act + residual

        # return msa_act, pair_act, pair_mask # 303 GB
        
        # [B, N_res, N_res//dap_size, c_z] => [B, N_res//dap_size, N_res, c_z]
        pair_act = dap.col_to_row(pair_act)

        # wait if using async communication and dap, otherwise do nothing
        # [B, N_seq//dap_size, N_res, c_m]
        msa_act = dap.all_to_all_opp(msa_act, in_axis=1, out_axis=2)

        return msa_act, pair_act


class ExtraEvoformerIterations(nn.Layer):
    def __init__(self, channel_num, config, global_config):
        super(ExtraEvoformerIterations, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config
        self.is_extra_msa = True
        self.n_layers = self.config['extra_msa_stack_num_block']
        self.extra_msa_stack = nn.LayerList([EvoformerIteration(
            channel_num,
            self.config['evoformer'],
            self.global_config,
            self.is_extra_msa
        ) for _ in range(self.n_layers)])

    def forward(self, extra_msa_act, extra_pair_act, extra_msa_mask, mask_2d):
        for extra_msa_stack_iteration in self.extra_msa_stack:
            extra_msa_act, extra_pair_act = extra_msa_stack_iteration(
                extra_msa_act, extra_pair_act, extra_msa_mask, mask_2d)
        return extra_msa_act, extra_pair_act


class Embeddings(nn.Layer):
    """Embeds the input data and runs Evoformer.

    Produces the MSA, single and pair representations.
    Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 5-18
    """

    def __init__(self, channel_num, config, global_config):
        super(Embeddings, self).__init__()
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

    def _pseudo_beta_fn(self, aatype, all_atom_positions):
        gly_id = paddle.ones_like(aatype) * residue_constants.restype_order['G'] # gly_id = (1, len_dim)
        is_gly = paddle.equal(aatype, gly_id) # is_gly = (1, len_dim)
        is_gly_dim = len(is_gly.shape)
        new_is_gly = paddle.unsqueeze(is_gly, axis=-1)
        new_is_gly.stop_gradient = True
        
        ca_idx = residue_constants.atom_order['CA'] # 1
        cb_idx = residue_constants.atom_order['CB'] # 3
        n = len(all_atom_positions.shape)
        pseudo_beta = paddle.where(
            paddle.tile(new_is_gly, [1] * is_gly_dim + [3]), # 1, len_dim, 3
            paddle.squeeze(all_atom_positions.slice([n-2], [ca_idx], [ca_idx+1]),axis=-2), # 1, len_dim
            paddle.squeeze(all_atom_positions.slice([n-2], [cb_idx], [cb_idx+1]),axis=-2)  # 1, len_dim
        )
        return pseudo_beta # = (1, len_dim, 3)

    def _create_extra_msa_feature(self,
        extra_msa,
        extra_has_deletion,
        extra_deletion_value):
        # 23: 20aa + unknown + gap + bert mask
        extra_msa = extra_msa.astype(paddle.int32)
        msa_1hot = nn.functional.one_hot(extra_msa, 23)
        msa_feat = [msa_1hot,
                    paddle.unsqueeze(extra_has_deletion, axis=-1),
                    paddle.unsqueeze(extra_deletion_value, axis=-1)]
        return paddle.concat(msa_feat, axis=-1)

    def forward(self,
        target_feat,
        msa_feat,
        seq_mask,
        aatype,
        residue_index,
        template_mask,
        template_aatype,
        template_pseudo_beta_mask,
        template_pseudo_beta,
        template_all_atom_positions,
        template_all_atom_masks,
        extra_msa,
        extra_has_deletion,
        extra_deletion_value,
        prev_pos=None,
        prev_msa_first_row=None,
        prev_pair=None):
        # InputEmbedder
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 5
        # Jumper et al. (2021) Suppl. Alg. 3 "InputEmbedder"
        preprocess_1d = self.preprocess_1d(target_feat)
        # preprocess_msa = self.preprocess_msa(batch['msa_feat'])
        msa_activations = paddle.unsqueeze(preprocess_1d, axis=1) + \
                    self.preprocess_msa(msa_feat)

        right_single = self.right_single(target_feat)  # 1, n_res, 22 -> 1, n_res, 128
        right_single = paddle.unsqueeze(right_single, axis=1)   # 1, n_res, 128 -> 1, 1, n_res, 128
        left_single = self.left_single(target_feat)    # 1, n_res, 22 -> 1, n_res, 128
        left_single = paddle.unsqueeze(left_single, axis=2)     # 1, n_res, 128 -> 1, n_res, 1, 128
        pair_activations = left_single + right_single

        mask_2d = paddle.unsqueeze(seq_mask, axis=1) * paddle.unsqueeze(seq_mask, axis=2)
        # Inject previous outputs for recycling.
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 6
        # Jumper et al. (2021) Suppl. Alg. 32 "RecyclingEmbedder"

        if self.config.recycle_pos: # and prev_pos is not None:
            prev_pseudo_beta = self._pseudo_beta_fn(aatype, prev_pos)
            dgram = dgram_from_positions(
                prev_pseudo_beta, **self.config.prev_pos)
            pair_activations += self.prev_pos_linear(dgram)

        if self.config.recycle_features:
            if prev_msa_first_row is not None:
                prev_msa_first_row = self.prev_msa_first_row_norm(
                    prev_msa_first_row)

                # A workaround for `jax.ops.index_add`
                msa_first_row = paddle.squeeze(msa_activations[:, 0, :], axis=1)
                msa_first_row += prev_msa_first_row
                msa_first_row = paddle.unsqueeze(msa_first_row, axis=1)
                msa_activations_raw = paddle.concat([msa_first_row, msa_activations[:, 1:, :]], axis=1)

            if 'prev_pair' is not None:
                pair_activations += self.prev_pair_norm(prev_pair)

        # RelPosEmbedder
        # Jumper et al. (2021) Suppl. Alg. 4 "relpos"
        # Jumper et al. (2021) Suppl. Alg. 5 "one_hot"
        if self.config.max_relative_feature:
            pos = residue_index  # [bs, N_res]
            offset = paddle.unsqueeze(pos, axis=[-1]) - \
                paddle.unsqueeze(pos, axis=[-2])
            offset = offset.astype(dtype=paddle.int32)
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
        if self.config.template.enabled: # [TODO] check if valid
            #template_batch = {k: batch[k] for k in batch if k.startswith('template_')}
            # pdb.set_trace()
            template_pair_repr = self.template_embedding(
                pair_activations, # 1xlxlx128
                template_mask, # 1x4
                template_aatype, # 1xl
                template_pseudo_beta_mask, # 1xl 
                template_pseudo_beta, # 1xlx3
                template_all_atom_positions, # 1xlx37x3
                template_all_atom_masks, # 1xlx37
                mask_2d # 1xlxl
            )
            pair_activations += template_pair_repr
        
        # ExtraMSAEmbedder
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 14-16
        extra_msa_feat = self._create_extra_msa_feature( # [INFO] done
            extra_msa, extra_has_deletion, extra_deletion_value
        )
        extra_msa_activations = self.extra_msa_activations(extra_msa_feat)
        # ==================================================
        #  Extra MSA Stack
        # Jumper et al. (2021) Suppl. Alg. 18 "ExtraMsaStack"
        # ==================================================
        # extra_msa_stack_input = {
        #     'msa': extra_msa_activations,
        #     'pair': pair_activations,
        # }

        # scatter if using dap, otherwise do nothing
        # [B, N_seq, N_res, c_m] => [B, N_seq//dap_size, N_res, c_m]
        extra_msa_act = dap.scatter(extra_msa_activations, axis=1)
        # [B, N_res, N_res, c_z] => [B, N_res//dap_size, N_res, c_z]
        extra_pair_act = dap.scatter(pair_activations, axis=1)

        return (
            msa_activations_raw, #  (1, 508, len_dim, 256)
            extra_msa_act, # (1, 5120, len_dim, 64)
            extra_pair_act, # (1, len_dim, len_dim, 128)
            mask_2d # (1, len_dim, len_dim)
        )


class ExtraMsa(nn.Layer):
    def __init__(self, channel_num, config, global_config):
        super(ExtraMsa, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        # Extra MSA Stack.
        # Jumper et al. (2021) Suppl. Alg. 18 "ExtraMsaStack"
        self.extra_msa_stack = nn.LayerList()
        for _ in range(self.config.extra_msa_stack_num_block):
            self.extra_msa_stack.append(EvoformerIteration(
                self.channel_num, self.config.evoformer, self.global_config,
                is_extra_msa=True))

    def _create_extra_msa_feature(self,
        extra_msa,
        extra_has_deletion,
        extra_deletion_value):
        # 23: 20aa + unknown + gap + bert mask
        extra_msa = extra_msa.astype(paddle.int32)
        msa_1hot = nn.functional.one_hot(extra_msa, 23)
        msa_feat = [msa_1hot,
                    paddle.unsqueeze(extra_has_deletion, axis=-1),
                    paddle.unsqueeze(extra_deletion_value, axis=-1)]
        return paddle.concat(msa_feat, axis=-1)

    def forward(self,
        extra_msa_act,
        extra_pair_act,
        extra_msa_mask,
        mask_2d
    ):
        for extra_msa_stack_iteration in self.extra_msa_stack:
            extra_msa_act_new, extra_pair_act_new = recompute_wrapper( # [TODO] check if valid
                extra_msa_stack_iteration,
                extra_msa_act,
                extra_pair_act,
                extra_msa_mask,
                mask_2d,
                is_recompute=self.training)
            extra_msa_act = extra_msa_act_new
            extra_pair_act = extra_pair_act_new

        # gather if using dap, otherwise do nothing
        # [B, N_res//dap_size, N_res, c_z] => [B, N_res, N_res, c_z]
        extra_pair_act= dap.gather(extra_pair_act, axis=1) # 1xlxlx128
        # msa_activations_raw = 1x508xlx256

        return extra_msa_act, extra_pair_act


class SingleTemplateEmbedding(nn.Layer):
    def __init__(self, 
        channel_num, 
        config, # model_config['model']['embeddings_and_evoformer']
        global_config
    ):
        super(SingleTemplateEmbedding, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        # Embed templates torsion angles
        if self.config.template.enabled and self.config.template.embed_torsion_angles:
            c = self.config.msa_channel
            self.template_single_embedding = nn.Linear(
                self.channel_num['template_angle'], c)
            self.template_projection = nn.Linear(c, c)

    def forward(self, 
        msa_mask, 
        torsion_angles_mask, 
        msa_activations_raw, 
        template_features
    ):
        template_activations = self.template_single_embedding(
            template_features)
        template_activations = nn.functional.relu(template_activations)
        template_activations = self.template_projection(template_activations)

        # Concatenate the templates to the msa.
        msa_activations = paddle.concat(
            [msa_activations_raw, template_activations], axis=1)

        # Concatenate templates masks to the msa masks.
        # Use mask from the psi angle, as it only depends on the backbone atoms
        # from a single residue.
        torsion_angle_mask = torsion_angles_mask[..., 2]
        torsion_angle_mask = torsion_angle_mask.astype(msa_mask.dtype)
        msa_mask = paddle.concat([msa_mask, torsion_angle_mask], axis=1)
        return msa_activations, msa_mask


class SingleActivations(nn.Layer):
    def __init__(self, channel_num, config, global_config):
        super(SingleActivations, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config
        
        self.single_activations = nn.Linear(
            self.config['msa_channel'], self.config['seq_channel'])
    
    def forward(self, msa_activation):
        return self.single_activations(msa_activation)
