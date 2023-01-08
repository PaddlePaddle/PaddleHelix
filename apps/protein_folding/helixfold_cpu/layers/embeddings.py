import pdb
from paddle import nn
import paddle
from layers.basics import (
  Attention,
  TriangleAttention, 
  TriangleMultiplication, 
  Transition,
  dgram_from_positions
)
from tools import residue_constants, quat_affine
import numpy as np

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
        self.triangle_starting_dropout = nn.Dropout(dropout_rate, axis=dropout_axis)

        self.triangle_attention_ending_node = TriangleAttention(channel_num,
                    self.config.triangle_attention_ending_node, self.global_config,
                    name='triangle_attention_ending_node')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_attention_ending_node)
        self.triangle_ending_dropout = nn.Dropout(dropout_rate, axis=dropout_axis)

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

        self.pair_transition = Transition(channel_num, self.config.pair_transition,
                    self.global_config, is_extra_msa=False,
                    transition_type='pair_transition')

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

        residual = self.pair_transition(pair_act)
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
        # {'target_feat': 22, 
        # 'msa_feat': 49, 
        # 'extra_msa_channel': 64, 
        # 'msa_channel': 256, 
        # 'pair_channel': 128, 
        # 'seq_channel': 384, 
        # 'template_pair': 85}
        self.global_config = global_config
        # self.dtype = query_embedding_dtype
        self.embedding2d = nn.Linear(channel_num['template_pair'],
            self.config.template_pair_stack.triangle_attention_ending_node.value_dim)

        self.template_pair_stack = nn.LayerList()
        for _ in range(self.config.template_pair_stack.num_block):
            self.template_pair_stack.append(TemplatePair(
                self.channel_num, self.config.template_pair_stack, self.global_config))

        self.output_layer_norm = nn.LayerNorm(self.config.attention.key_dim)

    def forward(self, 
                template_aatype, 
                template_pseudo_beta_mask, 
                template_pseudo_beta, 
                template_all_atom_positions, 
                template_all_atom_masks, 
                mask_2d):
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
        dtype = mask_2d.dtype
        num_res = template_aatype.shape[1]
        template_mask = template_pseudo_beta_mask
        template_mask_2d = template_mask[..., None] * template_mask[..., None, :]
        template_mask_2d = template_mask_2d.astype(dtype)

        template_dgram = dgram_from_positions(
            template_pseudo_beta,
            **self.config.dgram_features)
        template_dgram = template_dgram.astype(dtype)

        aatype = nn.functional.one_hot(template_aatype, 22)
        aatype = aatype.astype(dtype)

        to_concat = [template_dgram, template_mask_2d[..., None]]
        to_concat.append(paddle.tile(aatype[..., None, :, :],
                                     [1, num_res, 1, 1]))
        to_concat.append(paddle.tile(aatype[..., None, :],
                                     [1, 1, num_res, 1]))

        #if self.config.use_template_unit_vector:
        n, ca, c = [residue_constants.atom_order[a]
                    for a in ('N', 'CA', 'C')]
        rot, trans = quat_affine.make_transform_from_reference(
            n_xyz=template_all_atom_positions[..., n, :], # reference shape [1, len, 37, 3]
            ca_xyz=template_all_atom_positions[..., ca, :],
            c_xyz=template_all_atom_positions[..., c, :])
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
            template_all_atom_masks[..., n] *
            template_all_atom_masks[..., ca] *
            template_all_atom_masks[..., c])
        template_mask_2d = template_mask[..., None] * template_mask[..., None, :]
        inv_distance_scalar *= template_mask_2d.astype(inv_distance_scalar.dtype)

        unit_vector = [(x * inv_distance_scalar)[..., None] for x in affine_vec]
        unit_vector = [x.astype(dtype) for x in unit_vector]

        ### [UnboundLocalError] local variable 'x' ....
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
        for pair_encoder in self.template_pair_stack: # InvalidArgumentError
            act = pair_encoder(act, mask_2d)

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
    
    def subbatch_attention(self, 
        msa_act1:paddle.Tensor, 
        msa_act2:paddle.Tensor, 
        msa_mask:paddle.Tensor):
        arg_idx = [0,1]
        dim = [1,1]
        out_idx = 1
        self.bs = self.config.subbatch_size
        #inps = [args[i] for i in arg_idx]
        inps = [msa_act1, msa_act2, msa_mask]
        dim_width = [inp.shape[d] for inp, d in zip(inps, dim)]
        inp_dim = {inp: d for inp, d in zip(inps, dim)}
        dim_width = dim_width[0]
        if dim_width < self.bs:
            return self.attention(msa_act1, msa_act2, msa_mask)

        outs = []
        for slice_at in np.arange(0, dim_width, self.bs): # use np.arange to escape the warning: for-range when cvt to static graph
            _args = []
            for i, inp in enumerate(inps):
                if i in arg_idx:
                    inp = inp.slice([inp_dim[inp]], [slice_at], [slice_at + self.bs])
                _args.append(inp)
            outs.append(self.attention(_args[0], _args[1], _args[2]))

        return paddle.concat(outs, out_idx)

    def forward(self, 
        query_embedding, 
        template_mask,
        template_aatype, 
        template_pseudo_beta_mask, 
        template_pseudo_beta, 
        template_all_atom_positions, 
        template_all_atom_masks, 
        mask_2d):
        """Build TemplateEmbedding module.

        Arguments:
            query_embedding: Query pair representation, shape [n_batch, N_res, N_res, c_z].
            template_batch: A batch of template features.
            mask_2d: Padding mask (Note: this doesn't care if a template exists,
                unlike the template_pseudo_beta_mask).

        Returns:
            A template embedding [n_batch, N_res, N_res, c_z].
        """
        num_templates = template_mask.shape[0]
        num_channels = (self.config.template_pair_stack
                        .triangle_attention_ending_node.value_dim)
        num_res = query_embedding.shape[1]
        dtype = query_embedding.dtype
        template_mask = template_mask.astype(dtype)

        query_channels = query_embedding.shape[-1]
        template_batch = {'template_mask': template_mask}
 
        outs = []
        for i in range(num_templates):
            # By default, num_templates = 4
            template_aatype = paddle.squeeze(template_aatype.slice([1], [i], [i+1]), axis=1)
            template_pseudo_beta_mask = paddle.squeeze(template_pseudo_beta_mask.slice([1], [i], [i+1]), axis=1)
            template_pseudo_beta = paddle.squeeze(template_pseudo_beta.slice([1], [i], [i+1]), axis=1)
            template_all_atom_positions = paddle.squeeze(template_all_atom_positions.slice([1], [i], [i+1]), axis=1)
            template_all_atom_masks = paddle.squeeze(template_all_atom_masks.slice([1], [i], [i+1]), axis=1)
            outs.append(self.single_template_embedding(
                template_aatype, # [1,len_dim]
                template_pseudo_beta_mask, # [1,len_dim]
                template_pseudo_beta, # [1,len_dim, 3]
                template_all_atom_positions, # [1,len_dim, 37, 3]
                template_all_atom_masks,  # [1,len_dim, 37]
                mask_2d)) # [1,len_dim, len_dim]

        template_pair_repr = paddle.stack(outs, axis=1)

        flat_query = paddle.reshape(
            query_embedding, [-1, num_res * num_res, 1, query_channels])
        flat_templates = paddle.reshape(
            paddle.transpose(template_pair_repr, [0, 2, 3, 1, 4]),
            [-1, num_res * num_res, num_templates, num_channels])

        bias = 1e9 * (template_mask[:, None, None, None, :] - 1.)
        # OK until here

        # if not self.training:
            # sb_attn = subbatch(self.attention, [0, 1], [1, 1],
            #                    self.config.subbatch_size, 1)
            #emb = self.subbatch_attention(flat_query, flat_templates, bias) # will comes out with a huge graph
            # emb = self.attention(flat_query, flat_templates, bias)
        # else:
            # emb = self.attention(flat_query, flat_templates, bias)
        emb = self.attention(flat_query, flat_templates, bias)


        emb = paddle.reshape(
            emb, [-1, num_res, num_res, query_channels])

        # No gradients if no templates.
        emb *= (paddle.sum(template_mask) > 0.).astype(emb.dtype)
        return emb

