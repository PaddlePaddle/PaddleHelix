
import pdb
from paddle.distributed.fleet.utils import recompute
import paddle
from paddle import nn
from tools import dap, all_atom, residue_constants
from layers.basics import dgram_from_positions
from layers.backbones import EvoformerIteration
from layers.embeddings import TemplateEmbedding
import numpy as np

def recompute_wrapper(func, *args, is_recompute=True):
    """Function wrapper for recompute"""
    if is_recompute:
        return recompute(func, *args)
    else:
        return func(*args)


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

    #def forward(self, batch):
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
        extra_msa_mask,
        msa_mask,
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
        
        # [INFO] --- extra_msa start ---
        for extra_msa_stack_iteration in self.extra_msa_stack:
            print('# [INFO] inference one MSA stack iteration')
            extra_msa_act, extra_pair_act = extra_msa_stack_iteration(
                extra_msa_act,
                extra_pair_act,
                extra_msa_mask,
                mask_2d)

        # gather if using dap, otherwise do nothing
        # [B, N_res//dap_size, N_res, c_z] => [B, N_res, N_res, c_z]
        extra_pair_act= dap.gather(extra_pair_act, axis=1)

        # [INFO] --- extra_msa end ---
        # ==================================================
        #  Template angle feat
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 7-8
        # ==================================================
        if self.config.template.enabled and self.config.template.embed_torsion_angles:
            num_templ, num_res = template_aatype.shape[1:]

            aatype_one_hot = nn.functional.one_hot(template_aatype, 22)
            # Embed the templates aatype, torsion angles and masks.
            # Shape (templates, residues, msa_channels)
            ret = all_atom.atom37_to_torsion_angles(
                aatype=template_aatype,
                all_atom_pos=template_all_atom_positions,
                all_atom_mask=template_all_atom_masks,
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
            msa_activations = paddle.concat(
                [msa_activations_raw, template_activations], axis=1)

            # Concatenate templates masks to the msa masks.
            # Use mask from the psi angle, as it only depends on the backbone atoms
            # from a single residue.
            torsion_angle_mask = ret['torsion_angles_mask'][..., 2]
            torsion_angle_mask = torsion_angle_mask.astype(msa_mask.dtype)
            msa_mask = paddle.concat([msa_mask, torsion_angle_mask], axis=1)

        # scatter if using dap, otherwise do nothing
        # [B, N_seq, N_res, c_m] => [B, N_seq//dap_size, N_res, c_m]
        msa_activations = dap.scatter(msa_activations, axis=1) # [TODO] check if valid
        # [B, N_res, N_res, c_z] => [B, N_res//dap_size, N_res, c_z]
        extra_pair_act = dap.scatter(extra_pair_act, axis=1)

        # ==================================================
        #  Main MSA Stack
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 17-18
        # ==================================================
        
        # [INFO] --- evoformer start ---
        i_block=0
        for evoformer_block in self.evoformer_iteration:
            print('# [INFO] evoformer iteration %d' % i_block)
            i_block += 1
            msa_act, pair_act = recompute_wrapper(
                evoformer_block,
                msa_activations,
                extra_pair_act,
                msa_mask,
                mask_2d,
                is_recompute=self.training)
            msa_activations = msa_act
            extra_pair_act = pair_act

        # gather if using dap, otherwise do nothing
        # [B, N_seq//dap_size, N_res, c_m] => [B, N_seq, N_res, c_m]
        msa_act = dap.gather(msa_act, axis=1)
        # [B, N_res//dap_size, N_res, c_z] => [B, N_res, N_res, c_z]
        pair_act = dap.gather(pair_act, axis=1)

        msa_activations = msa_act
        pair_activations = pair_act
        single_activations = self.single_activations(msa_activations[:, 0])

        # [INFO] --- evoformer end ---
        num_seq = msa_feat.shape[1]
        # output = {
        #     'single': single_activations,
        #     'pair': pair_activations,
        #     # Crop away template rows such that they are not used
        #     # in MaskedMsaHead.
        #     'msa': msa_activations[:, :num_seq],
        #     'msa_first_row': msa_activations[:, 0],
        # }

        return single_activations, pair_activations, msa_activations[:, :num_seq], msa_activations[:, 0]
