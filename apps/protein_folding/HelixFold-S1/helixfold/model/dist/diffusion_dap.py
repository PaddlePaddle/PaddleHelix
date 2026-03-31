"""
dap version of diffusion.py
"""
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import gc
from copy import deepcopy
from helixfold.model.diffusion import DiffusionModule, FourierEmbedding, AdaLN, ConditionedTransitionBlock, AtomPairUtil
from helixfold.model.diffusion import AttentionIndex, tile_batch_dim, aggregate_atom_feat_to_token
from helixfold.model.diffusion import AtomTransformer
from ppfleetx.distributed.protein_folding import dap
from helixfold.model.utils import fused_act_bias_wrapper
from ppfleetx.models.protein_folding.common import (
    recompute_wrapper,
    subbatch, )

class DistDiffusionModule(DiffusionModule):
    """
    Distirbuted Diffusion
    """
    def __init__(self, channel_num, config, global_config):
        super(DistDiffusionModule, self).__init__(
            channel_num=channel_num, config=config, global_config=global_config
        )

        # re-init self.diffusion_conditioning with distributed PairFormer
        self.diffusion_conditioning = DiffusionConditioning(
                channel_num, self.config.diffusion_conditioning, self.global_config)

        self.dist_infer = True
        if self.dist_infer:
            self.atom_encoder = AtomAttentionEncoder(
                channel_num, self.config.atom_encoder, self.global_config)
            
            self.diffusion_transformer = DiffusionTransformer(
                        channel_num, self.config.diffusion_transformer, self.global_config)
                        
            self.atom_decoder = AtomAttentionDecoder(
                    channel_num, self.config.atom_decoder, self.global_config)

    def _forward_model(self, x_noisy, t_hat, batch, representations, return_r=False):
        """
        x_noisy: (B, N_atom, 3)
        t_hat: (B)
        """
        s_inputs = representations['single_inputs']  # (B, N_token, d1)
        s_trunk = representations['single'] # (B, N_token, d1)
        z_trunk = representations['pair']   # (raw_batch, N_token, N_token, d2)
        rel_pos_encoding = representations['rel_pos_encoding']   # (raw_batch, N_token, N_token, d2)
        atom_mask = batch['ref_mask']  # (B, N_atom)
        seq_mask = batch['seq_mask']        # (B, N_token)
        x_noisy = dap.scatter(x_noisy, axis=1)
        s_inputs = dap.scatter(s_inputs, axis=1)
        s_trunk = dap.scatter(s_trunk, axis=1)
        z_trunk = dap.scatter(z_trunk, axis=1)
        rel_pos_encoding = dap.scatter(rel_pos_encoding, axis=1)
        si, zij = recompute_wrapper(self.diffusion_conditioning,
                t_hat, rel_pos_encoding, s_inputs, s_trunk, z_trunk, self.sigma_data,
                is_recompute=self.training)

        t_hat = t_hat.unsqueeze([1, 2])
        r_noisy = x_noisy / paddle.sqrt(t_hat ** 2 + self.sigma_data ** 2)

        atom_token_uid = batch['ref_token2atom_idx']
        ai, ql_skip, cl_skip, p_lm_skip = self.atom_encoder(feature=batch, rl=r_noisy, 
                                                            s_trunk=s_trunk, zij=zij) 

        ai += self.lin1(self.ln1(si))
        beta = (1 - seq_mask[:, :, None] * seq_mask[:, None]) * (-1e8)
        beta = dap.scatter(beta, axis=1)
        atom_mask = dap.scatter(atom_mask, axis=1)

        ai = self.diffusion_transformer(ai, si, zij, beta=beta,
            )

        ai = self.ln2(ai)
        r_update = self.atom_decoder(ai, ql_skip, cl_skip, p_lm_skip,
                                     atom_token_uid, atom_mask)

        x_out = self._c_ckip(t_hat) * x_noisy + self._c_out(t_hat) * r_update
        atom_mask = paddle.cast(atom_mask, dtype=x_out.dtype)
        x_out = x_out * atom_mask[..., None]
        
        if self.dist_infer:
            x_out = dap.gather(x_out, axis=1)
            if return_r:
                r_update, atom_mask = dap.gather(r_update, axis=1), dap.gather(atom_mask, axis=1)
        if return_r:
            return x_out, r_update * atom_mask[..., None]
        return x_out


class AtomAttentionEncoder(nn.Layer):
    """
    AtomAttentionEncoder: only support multimer-monomer
    """
    def __init__(self, channel_num, config, global_config):
        super(AtomAttentionEncoder, self).__init__()
        self.config = config
        in_token_channel = channel_num[self.config.in_token_channel_name]
        out_token_channel = channel_num[self.config.out_token_channel_name]
        token_pair_channel = channel_num['token_pair_channel']
        atom_channel = channel_num['atom_channel']
        atom_pair_channel = channel_num['atom_pair_channel']

        self.ap_util = AtomPairUtil()
        self.dense = config.use_dense_mode

        f_dim = 3 + 1 + 1 + 128 + 4 * 64
        self.lin_atom_meta_to_cond_feat = \
            nn.Linear(f_dim, atom_channel, bias_attr=False)
        self.lin_pos_offset_to_apair = \
            nn.Linear(3, atom_pair_channel, bias_attr=False)
        self.lin_inv_sq_dist_to_apair = \
            nn.Linear(3, atom_pair_channel, bias_attr=False)
        self.lin_valid_mask_to_apair = \
            nn.Linear(1, atom_pair_channel, bias_attr=False)

        # embed trunk single embedding to cond atom feat
        self.ln_trunk_single_to_cond_atom_feat = nn.LayerNorm(in_token_channel)
        self.lin_trunk_single_to_cond_atom_feat = \
            nn.Linear(in_token_channel, atom_channel, bias_attr=False)
        
        
        # embed cond pair embedding to pair representation
        self.ln_cond_pair_feat_to_pair_repr = nn.LayerNorm(token_pair_channel)
        self.lin_cond_pair_feat_to_pair_repr = \
            nn.Linear(token_pair_channel, atom_pair_channel, bias_attr=False)

        # embed noise position
        self.lin_noise_pos_to_single_repr = \
            nn.Linear(3, atom_channel, bias_attr=False)
        
        # embed single cond to pair representation
        self.act_single_cond_to_pair_repr = nn.ReLU(atom_channel)
        self.lin_single_cond_to_pair_repr = nn.Linear(
            atom_channel, atom_pair_channel, bias_attr=False)

        # pair activation MLP
        self.mlp_pair_active = paddle.nn.Sequential(
            nn.ReLU(atom_pair_channel),
            nn.Linear(atom_pair_channel, atom_pair_channel, bias_attr=False),
            nn.ReLU(atom_pair_channel),
            nn.Linear(atom_pair_channel, atom_pair_channel, bias_attr=False),
            nn.ReLU(atom_pair_channel),
            nn.Linear(atom_pair_channel, atom_pair_channel, bias_attr=False),
        )

        self.atom_transformer = AtomTransformer(
            channel_num=channel_num, config=config.atom_transformer, global_config=global_config)
    
        # aggregate atom representation to token representation
        self.act_atom_to_token = nn.ReLU()
        self.lin_atom_to_token = \
            nn.Linear(atom_channel, out_token_channel, bias_attr=False)

    def forward(self, feature, rl, s_trunk, zij):
        """
        Args:
        - rl:         [B, M, 3]
        - s_trunk:    [B, N, c_s=384]
        - zij:        [b, N, N, c_z=128]

        Use features:
        - f_ref_pos:              [B, M, 3]
        - f_ref_charge:           [B, M]
        - f_ref_mask:             [B, M]
        - f_ref_element:          [B, M, 128]
        - f_ref_atom_name_chars:  [B, M, 4, 64]
        - f_ref_space_uid:        [B, M]

        Returns:
        - ai:     [B, N, C_t=768]
        - ql:     [B, M, C_a=128]
        - cl:     [B, M, C_a=128]
        - plm:    [B, M, M, C_ap=16]
        """
        DIFFUSION = rl is not None
        if DIFFUSION: # late tile zij for avoiding OOM
            diff_batch_size = rl.shape[0] / zij.shape[0] # B/b
        
        atom_token_uid = feature['ref_token2atom_idx'] # [B,M]
        atom_mask = feature['ref_mask'] # [B,M]

        # create teh atom single conditioning: embed per-atom meta data
        f_ref_element = F.one_hot(feature['ref_element'], num_classes=128) # (B, M, 128)
        f_ref_pos = feature['ref_pos']   # (B, M, 3)
        f_ref_space_uid = feature['ref_space_uid'] # (B, M)
        f_ref_charge = feature['ref_charge'].unsqueeze(-1).cast(f_ref_pos.dtype) # (B, M, 1)
        f_ref_mask = feature['ref_mask'].unsqueeze(-1).cast(f_ref_pos.dtype) # (B, M, 1)
        f_atom_name_chars = F.one_hot(
            feature['ref_atom_name_chars'], num_classes=64) # (B, M, 4, 64)
        f_atom_name_chars = f_atom_name_chars.reshape(
            f_atom_name_chars.shape[:2] + [4 * 64]) # (B, M, 4*64)
        
        atom_mask = dap.scatter(atom_mask, axis=1)
        f_ref_element = dap.scatter(f_ref_element, axis=1)
        f_ref_pos = dap.scatter(f_ref_pos, axis=1)
        f_ref_space_uid = dap.scatter(f_ref_space_uid, axis=1)
        f_ref_charge = dap.scatter(f_ref_charge, axis=1)
        f_ref_mask = dap.scatter(f_ref_mask, axis=1)
        f_atom_name_chars = dap.scatter(f_atom_name_chars, axis=1)
        atom_feat_concat = paddle.concat(
                [f_ref_pos, f_ref_charge, f_ref_mask, f_ref_element, f_atom_name_chars], 
                axis=-1) * f_ref_mask
        
        cl = self.lin_atom_meta_to_cond_feat(atom_feat_concat) # (B, M, c_a) #TODO: mask

        # embed offsets between atom reference positions
        if DIFFUSION and diff_batch_size > 1:
            f_ref_pos = f_ref_pos[:zij.shape[0]] # (b, M, 3)
            f_ref_space_uid = f_ref_space_uid[:zij.shape[0]] # (b, M)

        dlm = self.ap_util.add_2_seqs(f_ref_pos, - f_ref_pos, dense=self.dense) # [b,C,nq,nk,3]
        vlm = self.ap_util.cmp_2_seqs(f_ref_space_uid, f_ref_space_uid, dense=self.dense) # [b,C,nq,nk,1]
        vlm = vlm.cast(dtype='float32')

        plm = self.lin_pos_offset_to_apair(dlm) * vlm # (b, C, nq, nk, c_atompair)

        # embed pairwise inverse squared distancs, and the valid mask
        plm += self.lin_inv_sq_dist_to_apair(1 / (1 + dlm**2)) * vlm
        plm += self.lin_valid_mask_to_apair(vlm) * vlm

        # initialize the atom single representation as the single conditioning.
        ql = cl # (B, M, c_a)

        # if provided, add trunk embedding and noisy positions
        atom_mask = atom_mask.cast(ql.dtype)
        if rl is not None:
            # convert s_trunk_tok_i to s_trunk_atom_l
            s_trunk_atom = seq_to_atom_feat(s_trunk, atom_token_uid, atom_mask) # (B, M, c_s)

            # broadcast the single and pair embedding from the trunk
            cl += self.lin_trunk_single_to_cond_atom_feat(
                    self.ln_trunk_single_to_cond_atom_feat(s_trunk_atom)) \
                    * atom_mask.unsqueeze(-1) # (B, M, c_a)
            
            zij = self.lin_cond_pair_feat_to_pair_repr(
                    self.ln_cond_pair_feat_to_pair_repr(zij)) 
                    # TODO: zij mask # (batch_zij, N, N, C_atompair)
            zij = dap.gather(zij, axis=1)
            plm = dap.gather(plm, axis=1)
            plm += self.ap_util.to_atompair(zij=zij, atom_token_uid=atom_token_uid, 
                                            atom_mask=atom_mask, dense=self.dense)
            # assert plm.shape[1] == cl.shape[1]
            plm = dap.scatter(plm, axis=1)

            # Add the noisy positions
            ql += self.lin_noise_pos_to_single_repr(rl) \
                  * atom_mask.unsqueeze(-1) # (B, M, c_a)

        # add the combined single conditioning to the pair representation
        if DIFFUSION and diff_batch_size > 1:
            single_cond = cl[:zij.shape[0]] # (b, M, C_a)
        else:
            single_cond = cl # (B, M, c_a)
        single_cond = self.lin_single_cond_to_pair_repr(
                        self.act_single_cond_to_pair_repr(single_cond))# (b, M, c_atompair)

        single_cond = self.ap_util.add_2_seqs(single_cond, single_cond, dense=self.dense) # (b, M, M, c_atompair)
        plm += single_cond 

        # run MLP on the pair activation
        plm += self.mlp_pair_active(plm) #TODO: mask
        
        # cross attention transformer
        ql = self.atom_transformer(ql, cl, plm) #TODO: mask

        # aggregate per-atom representation to per-token representation
        N_token =  feature['residue_index'].shape[1]
        al = self.lin_atom_to_token(self.act_atom_to_token(ql)) # (B, M, c_t)
        al = dap.gather(al, axis=1)
        atom_mask = dap.gather(atom_mask, axis=1)
        ai = aggregate_atom_feat_to_token(
            al, atom_token_uid, atom_mask, N_token) # (B, N_res, c_t)
        ai = dap.scatter(ai, axis=1)
        return ai, ql, cl, plm


class AtomAttentionDecoder(nn.Layer):
    " Atom Attention Decoder for AF3. "

    def __init__(self, channel_num, config, global_config):
        super(AtomAttentionDecoder, self).__init__()
        self.config = config
        token_channel = channel_num[self.config.in_token_channel_name]
        if 'out_channel_name' in self.config:
            out_channel = channel_num[self.config.out_channel_name]
        else:
            out_channel = 3
        atom_channel = channel_num['atom_channel']
        self.lin0 = nn.Linear(token_channel, atom_channel, bias_attr=False)
        self.atom_transformer = AtomTransformer(
            channel_num, config.atom_transformer, global_config)
        self.ln1 = nn.LayerNorm(atom_channel)
        if config.get('final_zero_init', True):
            weight_init = nn.initializer.Constant(value=0.0)
        else:
            weight_init = None
        self.lin1 = nn.Linear(atom_channel, out_channel, bias_attr=False,
                weight_attr=weight_init)

    def forward(self, ai, ql_skip, cl_skip, plm_skip, atom_token_uid, atom_mask):
        """
        Args:
        ai: [B,N_res,C_s]
        ql: [B,M,C_a]
        cl: [B,M,C_a]
        plm: [B,M,M,C_atompair]
        atom_token_uid: [B,M]
        atom_mask: [B,M]

        Returns:
        r_udpate: [B, M, 3]
        """ 

        # Broadcast per-token activiations to per-atom activations and add the skip connection
        al = seq_to_atom_feat(ai, atom_token_uid, atom_mask) # (B, M, C_t)
        atom_mask = paddle.cast(atom_mask, dtype=al.dtype)
        al = al * atom_mask.unsqueeze(-1) # (B, M, C_s)
        ql_skip += self.lin0(al) * atom_mask.unsqueeze(-1) # (B, M, C_a)
        # cross attention transformer
        ql_skip = self.atom_transformer(ql_skip, cl_skip, plm_skip, ) \
                    * atom_mask.unsqueeze(-1)
        
        # Map to position update
        r_update = self.lin1(self.ln1(ql_skip)) * atom_mask.unsqueeze(-1) # (B, M, 3)

        return r_update


def seq_to_atom_feat(f_token, atom_token_uid, atom_mask):
    """
    Convert per-token sequence features to per-atom features.
    Args:
        f_token:              [B,N,D]
        atom_token_uid:       [B,M]
        atom_mask:            [B,M]

    Returns:
        f_atom:         [B,M,D]
    """
    f_token = dap.gather(f_token, axis=1)
    atom_mask = dap.gather(atom_mask, axis=1)
    B, M = atom_token_uid.shape[:2]
    # f_atom = f_token.flatten(0,1)[atom_token_uid.flatten()]
    # f_atom = f_atom * atom_token_uid_mask.reshape([-1, 1])
    # return f_atom.reshape([B, M, -1])
    # atom_token_uid -= atom_token_uid[:, 0].unsqueeze(-1) # substract val from pos_0 for dap indicies
    # for loop TODO: tmp work around
    f_atom = []
    atom_mask = atom_mask.cast(f_token.dtype)
    for i in range(B):
        idx = atom_token_uid[i]  # M
        mask = atom_mask[i]  # M
        atom = f_token[i][idx] * mask[..., None]
        f_atom.append(atom.unsqueeze(0))
    f_atom = paddle.concat(f_atom, axis=0)
    f_atom = dap.scatter(f_atom, axis=1)
    atom_mask = dap.scatter(atom_mask, axis=1)
    return f_atom


class DiffusionTransformer(nn.Layer):
    """
    DiffusionTransformer
    """
    def __init__(self, channel_num, config, global_config):
        super(DiffusionTransformer, self).__init__()
        self.config = config
        a_channel = channel_num[self.config.a_channel_name]
        s_channel = channel_num[self.config.s_channel_name]
        z_channel = channel_num[self.config.z_channel_name]

        self.attention_list = nn.LayerList()
        self.transition_list = nn.LayerList()
        for n in range(self.config.n_block):
            self.attention_list.append(AttentionPairBias(
                    a_channel, s_channel, z_channel, 
                    self.config.n_head, has_si=True))
            self.transition_list.append(ConditionedTransitionBlock(
                    a_channel, s_channel))

    def forward(self, ai, si, zij, beta):
        """forward"""
        for attention, transition in zip(self.attention_list, self.transition_list):
            ai += recompute_wrapper(attention, 
                    ai, si, zij, beta, is_recompute=self.training)
            ai += recompute_wrapper(transition, 
                    ai, si, is_recompute=self.training)
        return ai


class AttentionPairBias(nn.Layer):
    """AttentionPairBias"""
    def __init__(self, a_channel, s_channel, z_channel, 
            n_head, has_si, dropout_rate=0.1):
        super(AttentionPairBias, self).__init__()

        self.has_si = has_si
        self.n_head = n_head
        self.head_dim = a_channel // n_head

        if has_si:
            self.ln = AdaLN(a_channel, s_channel)
        else:
            self.ln = nn.LayerNorm(a_channel)
        self.q_lin = nn.Linear(a_channel, a_channel)
        self.k_lin = nn.Linear(a_channel, a_channel, bias_attr=False)
        self.v_lin = nn.Linear(a_channel, a_channel, bias_attr=False)
        self.b_ln = nn.LayerNorm(z_channel)
        self.b_lin = nn.Linear(z_channel, n_head, bias_attr=False)
        self.g_lin = nn.Linear(a_channel, a_channel, bias_attr=False)
        self.alpha_dropout = nn.Dropout(dropout_rate)
        self.out_lin1 = nn.Linear(a_channel, a_channel, bias_attr=False)
        self.out_dropout = nn.Dropout(dropout_rate)
        if has_si:
            self.out_lin2 = nn.Linear(s_channel, a_channel,
                    bias_attr=nn.initializer.Constant(value=-2.0))

        default_M = 10000
        self._AttenIndex = AttentionIndex(max_atom_num=default_M)
  
    def forward(self, ai, si, zij, beta):
        """
        ai: (d, Nq/D, d1)
        si: (d, Nq/D, d1)
        zij: (B, Nq/D, Nk, d2) or (b, 98, 32, 128, head)
        beta: (B, Nq/D, Nk) or (b, 98, 32, 128)
        attention_idx
        """
        is_atom_level = len(zij.shape) == 5
        assert self.has_si == (not si is None)

        B, N, D = paddle.shape(ai)
        H, d = self.n_head, self.head_dim
        if not is_atom_level:
            _, Nq, Nk, _ = paddle.shape(zij) 
            assert Nq == N, f"zij {zij.shape} shape missmatch with ai {ai.shape}"

        if self.has_si:
            ai = self.ln(ai, si)
        else:
            ai = self.ln(ai)

        q = self.q_lin(ai).reshape([B, N, H, d]).transpose([0, 2, 1, 3]) # (B, H, Nq/D, d)

        if is_atom_level or Nq == Nk:
            k = self.k_lin(ai).reshape([B, N, H, d]).transpose([0, 2, 1, 3]) # (B, H, Nk, d)
            v = self.v_lin(ai).reshape([B, N, H, d]).transpose([0, 2, 1, 3]) # (B, H, Nk, d)
        else:
            ai_full = dap.gather(ai, axis=1)
            k = self.k_lin(ai_full).reshape([B, Nk, H , d]).transpose([0, 2, 1, 3]) # (B, H, Nk, d)
            v = self.v_lin(ai_full).reshape([B, Nk, H , d]).transpose([0, 2, 1, 3]) # (B, H, Nk, d)   
        # zij is not tiled by diff_batch_size so far
        b = self.b_lin(self.b_ln(zij)) # (B, Nq, Nk, H) or (B, C, N, N, H)
        g = nn.functional.sigmoid(self.g_lin(ai))\
                         .reshape([B, N, H, d]).transpose([0, 2, 1, 3]) # (B, H, Nq/D, d)
        diff_batch_size = ai.shape[0] // b.shape[0]

        if is_atom_level:
            # local attention
            M = ai.shape[1]
            atten_idx = self._AttenIndex.get_atten_idx(M)

            query_idx = atten_idx['query_idx'].flatten() # [C,32]
            query_mask = atten_idx['query_mask'][..., None, None, None] # [C,32,1,1,1]
            key_idx = atten_idx['key_idx'].flatten() # [C,128,1,1,1]
            key_mask = atten_idx['key_mask'][..., None, None, None]   # [C,128,1,1,1]
            alpha_mask = atten_idx['alpha_mask'] # [C,32,128]
            C, n_query, n_key = alpha_mask.shape

            q = q.transpose([2, 0, 1, 3]) # (B, H, Nq/D, d) -> (Nq/D, B, H, d)
            k = k.transpose([2, 0, 1, 3]) # (B, H, Nk, d) -> (Nk, B, H, d)
            v = v.transpose([2, 0, 1, 3]) # (B, H, Nk, d) -> (Nk, B, H, d)
            g = g.transpose([2, 0, 1, 3]) # (B, H, Nq/D, d) -> (Nq/D, B, H, d)

            query_like_shape = [C, n_query, B, H, d]
            key_like_shape = [C, n_key, B, H, d]

            query_mask = query_mask.cast(q.dtype)
            key_mask = key_mask.cast(k.dtype)
            q = q[query_idx].reshape(query_like_shape) * query_mask # (C, 32, B, H, d)
            k = k[key_idx].reshape(key_like_shape) * key_mask  # (C, 128, B, H, d)
            v = v[key_idx].reshape(key_like_shape) * key_mask # (C, 128, B, H, d)
            g = g[query_idx].reshape(query_like_shape) * query_mask # (C, 32, B, H, d)

            q = q.transpose([2, 3, 0, 1, 4]) # (C, 32, B, H, d) -> (B, H, C, 32, d)
            k = k.transpose([2, 3, 0, 1, 4]) # (C, 128, B, H, d) -> (B, H, C, 128, d)
            v = v.transpose([2, 3, 0, 1, 4]) # (C, 128, B, H, d) -> (B, H, C, 128, d)
            g = g.transpose([2, 3, 0, 1, 4]) # (C, 32, B, H, d) -> (B, H, C, 32, d)
            b = b.transpose([0, 4, 1, 2, 3]) # (b, C, 32, 128, H) -> (b, H, C, 32, 128)

            if diff_batch_size > 1:
                b = tile_batch_dim(b, diff_batch_size)
            beta = tile_batch_dim(beta, ai.shape[0])
            b = (b + beta.unsqueeze(1)) # (B, H, C, 32, 128)
            alpha = paddle.matmul(q / np.sqrt(d), k, transpose_y=True) + b # (B, H, C, 32, 128)
            alpha = paddle.nn.functional.softmax(alpha)
            alpha = self.alpha_dropout(alpha)

            ai = paddle.matmul(alpha, v) * g # (B, H, C, 32, d)
            ai = ai.reshape([B, H, C * n_query, d])
            ai = ai[:, :, :si.shape[1], :]

        else:
            # global attention
            if diff_batch_size > 1:
                b = tile_batch_dim(b, diff_batch_size)
            if beta.shape[0] == 1:
                beta = beta.tile([ai.shape[0], 1, 1])
            b = (b + beta[..., None]).transpose([0, 3, 1, 2]) # (B, Nq/D, Nk, H) -> (B, H, Nq/D, Nk)

            alpha = paddle.matmul(q / np.sqrt(d), k, transpose_y=True) + b    # (B, H, Nq/D, Nk)
            alpha = F.softmax(alpha) # (B, H, Nq/D, Nk)
            alpha = self.alpha_dropout(alpha)

            ai = paddle.matmul(alpha, v) * g # (B, H, Nq/D, d)
        
        ai = ai.transpose([0, 2, 1, 3]).reshape([B, N, D])
        ai = self.out_lin1(ai)
        ai = self.out_dropout(ai)
        if self.has_si:
            ai = nn.functional.sigmoid(self.out_lin2(si)) * ai

        return ai


class DiffusionConditioning(nn.Layer):
    """
    DiffusionConditioning with dap
    """
    def __init__(self, channel_num, config, global_config):
        super(DiffusionConditioning, self).__init__()
        self.config = config
        self.global_config = global_config
        token_channel = channel_num['token_channel']
        pair_channel = channel_num['token_pair_channel']
        
        self.pair_ln = nn.LayerNorm(pair_channel * 2)
        self.pair_lin = nn.Linear(pair_channel * 2, pair_channel, bias_attr=False)
        self.pair_trans1 = Transition(pair_channel, n=2, global_config=global_config)
        self.pair_trans2 = Transition(pair_channel, n=2, global_config=global_config)

        self.single_ln1 = nn.LayerNorm(token_channel * 2 + 32 + 32 + 1)
        self.single_lin1 = nn.Linear(
                token_channel * 2 + 32 + 32 + 1, token_channel, bias_attr=False)
        self.fourier_embedding = FourierEmbedding(256)
        self.single_ln2 = nn.LayerNorm(256)
        self.single_lin2 = nn.Linear(256, token_channel, bias_attr=False)
        self.single_trans1 = Transition(token_channel, n=2, global_config=global_config)
        self.single_trans2 = Transition(token_channel, n=2, global_config=global_config)

    def forward(self, t_hat, rel_pos_encoding, s_inputs, s_trunk, z_trunk, sigma_data):
        """forward"""
        # z_trunk = dap.scatter(z_trunk, axis=1)
        # rel_pos_encoding = dap.scatter(rel_pos_encoding, axis=1)
        zij = paddle.concat([z_trunk, rel_pos_encoding], -1)
        zij = self.pair_lin(self.pair_ln(zij))

        if not self.training and self.global_config.low_memory is True:
            zij.add_(self.pair_trans1(zij))
            zij.add_(self.pair_trans2(zij))
        else:
            zij += self.pair_trans1(zij)
            zij += self.pair_trans2(zij)
        # zij = dap.gather(zij, axis=1)

        si = paddle.concat([s_trunk, s_inputs], -1)
        si = self.single_lin1(self.single_ln1(si))
        n = self.fourier_embedding(0.25 * paddle.log(t_hat / sigma_data))   # (B, c)
        si += self.single_lin2(self.single_ln2(n[:, None]))
        si += self.single_trans1(si)
        si += self.single_trans2(si)
        return si, zij


class Transition(nn.Layer):
    """
    Transition
    """
    def __init__(self, in_channel, n=4, global_config=None):
        super(Transition, self).__init__()
        self.ln = nn.LayerNorm(in_channel)
        self.lin1 = nn.Linear(in_channel, in_channel * n, bias_attr=False)
        self.lin2 = nn.Linear(in_channel, in_channel * n, bias_attr=False)
        self.lin3 = nn.Linear(in_channel * n, in_channel, bias_attr=False)
        self.first_run = True
        self.proj_ab_weight = None
        self.global_config=global_config

    def forward(self, x):
        """forward"""
        x = self.ln(x)
        def _transition_fn(x):
          try:
            from paddle.base.layer_helper import LayerHelper
            if self.first_run:
                self.proj_ab_weight = paddle.concat([self.lin1.weight, self.lin2.weight], axis=-1)
                self.first_run = False
                del self.lin1
                del self.lin2
                gc.collect()
            ffn1_out = paddle.matmul(x, self.proj_ab_weight)
            ffn1_out = fused_act_bias_wrapper(ffn1_out, None, act_method="swiglu")
            return self.lin3(ffn1_out)
          except:
            a = self.lin1(x)
            b = self.lin2(x)
            x = self.lin3(nn.functional.swish(a) * b)
            return x

        if not self.training:
            sb_transition = subbatch(
                _transition_fn, [0], [1],
                self.global_config.subbatch_size, 1)
            x = sb_transition(x)

        else:
            x = _transition_fn(x)

        return x