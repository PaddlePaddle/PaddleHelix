#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""Modules and utilities for the diffusion module."""

import os
import copy
import numpy as np
import paddle
from paddle import nn
import paddle.nn.functional as F
from scipy.spatial.transform import Rotation

from helixfold.common import residue_constants
from helixfold.common import all_atom_pdb_save
from helixfold.model import quat_affine, geometry, all_atom, r3, lddt
from helixfold.model.utils import recompute_wrapper
from helixfold.model import chain_align_np_aa
from helixfold.model import rotary

def tile_batch_dim(batch, repeat_time):
    """
    Tile tensor along the batch dimension.
    Args:
        batch: a dict of paddle tensor, or a pure paddle tensor
        repeat_time: int
    """
    def _tile(x, repeat_time):
        if isinstance(x, paddle.Tensor):
            shape = [repeat_time] + [1] * (len(x.shape) - 1)
            return paddle.tile(x, shape)
        elif isinstance(x, list):
            return x * repeat_time
        else:
            raise ValueError(f'Unsupported type {type(x)}')

    if isinstance(batch, dict):
        new_batch = {}
        for name, x in batch.items():
            new_batch[name] = _tile(x, repeat_time)
        return new_batch
    else:
        return _tile(batch, repeat_time)


def insert_diff_batch_dim(batch, repeat_time, special_keys=None):
    """
    Insert a diff_batch_dim after the batch (0-th) dim.
    Args:
        batch: a dict of paddle tensor, or a pure paddle tensor
        repeat_time: int
        special_keys: list of keys whose repeat_time = 1
    """
    def _insert(x, repeat_time):
        if isinstance(x, paddle.Tensor):
            return paddle.repeat_interleave(x[:, None], 
                repeat_time, axis=1)
        elif isinstance(x, list):
            return [[y] * repeat_time for y in x]
        else:
            raise ValueError(f'Unsupported type {type(x)}')

    if isinstance(batch, dict):
        new_batch = {}
        for name, x in batch.items():
            if not special_keys is None and name in special_keys:
                new_batch[name] = _insert(x, 1)
            else:
                new_batch[name] = _insert(x, repeat_time)
        return new_batch
    else:
        return _insert(batch, repeat_time)


def get_noise_schedule(sigma_data, s_max, s_min, p, step_size):
    """
    get_noise_schedule
    """
    t = paddle.arange(0, 1 + step_size, step_size)
    t_tau = sigma_data * (s_max ** (1 / p) + \
            t * (s_min ** (1 / p) - s_max ** (1 / p))) ** p
    return t_tau


def CentreRandomAugmentation(x, mask, s_trans=1, uniform=False):
    """
    x: (B, N_atom, 3)
    mask: (B, N_atom)
    """
    mask = mask[..., None]
    mean_x = (x * mask).sum([1]) / (mask.sum([1]) + 1e-6)
    x = x - mean_x[:, None]
    B = x.shape[0]
    R = [Rotation.random().as_matrix() for _ in range(B)]
    R = paddle.to_tensor(R, x.dtype)    # (B, 3, 3)
    if uniform:
        t = s_trans * paddle.uniform(shape=[B, 1, 3])
    else:
        t = s_trans * paddle.normal(shape=[B, 1, 3])
    x = x @ R + t
    x = x * mask
    return x


def weighted_rigid_align_per_sample(x_gt, x, w):
    """
    weighted_rigid_align_per_sample with outer product
    Args:
        x_gt: [N, 3]
        x: [N, 3]
        w: [N]
    """
    assert x_gt.shape[0] >= 3
    x_gt_mean = np.sum(x_gt * w[:, None], axis=0) / (np.sum(w) + 1e-8)
    x_mean = np.sum(x * w[:, None], axis=0) / (np.sum(w) + 1e-8)

    P = x_gt - x_gt_mean
    Q = x - x_mean

    C = np.sum(w[:, None, None] * P[:, :, None] * Q[:, None], axis=0)
    U, S, V = np.linalg.svd(C)
    d = (np.linalg.det(U) * np.linalg.det(V)) < 0.0
    if d:
        U[:, -1] = -U[:, -1]

    R = np.matmul(U, V)
    x_gt_align = P @ R + x_mean
    return x_gt_align


def weighted_rigid_align(x_gt, x, w, mask):
    """
    x_gt: (B, N_atom, 3)
    x: (B, N_atom, 3)
    w: (B, N_atom)
    mask: (B, N_atom)
    """
    B = x_gt.shape[0]
    x_gt = x_gt.numpy()
    x = x.numpy()
    w = w.numpy()
    mask = mask.numpy()
    
    new_gt_list = []
    for i in range(B):
        c_mask = mask[i] == 1
        gt_pos = x_gt[i][c_mask]
        pred_pos = x[i][c_mask]
        c_w = w[i][c_mask]

        new_gt = np.zeros_like(x[i])
        x_gt_align = weighted_rigid_align_per_sample(
                gt_pos, pred_pos, c_w)
        new_gt[c_mask] = x_gt_align
        new_gt_list.append(new_gt)
    new_gt_list = paddle.to_tensor(np.stack(new_gt_list))
    return new_gt_list


def get_x_gt_aligned(x_gt1, x_gt2, x, w, mask):
    """
    Choose a better align from x_gt1 and x_gt2
    Args:
        x_gt1: (B, N_atom, 3)
        x_gt2: (B, N_atom, 3), another gt option
        x: (B, N_atom, 3), prediction
        w: (B, N_atom)
        mask: (B, N_atom)
    """
    aligned1 = weighted_rigid_align(x_gt1, x, w, mask)
    rmsd1 = paddle.sum((aligned1 - x) ** 2 * mask[:, :, None], 
            axis=[1, 2], keepdim=True)
    aligned2 = weighted_rigid_align(x_gt2, x, w, mask)
    rmsd2 = paddle.sum((aligned2 - x) ** 2 * mask[:, :, None], 
            axis=[1, 2], keepdim=True)
    aligned = paddle.where(rmsd1 < rmsd2, aligned1, aligned2)
    return aligned


def get_typewise_rmsd(x_gt, x, mask, batch):
    """
    x_gt: (B, N_atom, 3)
    x: (B, N_atom, 3)
    mask: (B, N_atom)
    """
    ret = {}
    sq_error = paddle.sum((x - x_gt) ** 2, -1) # (B, N_atom)
    ret['all'] = paddle.sqrt((sq_error * mask).sum([1]) / (mask.sum([1]) + 1e-8))
    for name in ['protein', 'dna', 'rna', 'ligand']:
        typemask = mask * batch[f'is_{name}_aa']    # (B, N_atom)
        rmsd = paddle.sqrt((sq_error * typemask).sum([1]) / (typemask.sum([1]) + 1e-8))
        ret[f'{name}'] = rmsd
        ret[f'{name}_exist'] = (typemask.sum([1]) > 0).cast(rmsd.dtype)
        ret[f'{name}_ratio'] = typemask.sum([1]) / (mask.sum([1]) + 1e-8)
    return ret


def get_typewise_pair_error(pair_error_flat, idx1, idx2, asym_id, batch, batch_i):
    """
    pair_error_flat: (m,) error from a pair matrix indexed by idx1 and idx2
    idx1: (m,)
    idx2: (m,)
    """
    ret = {}
    type_names = ['protein', 'dna', 'rna', 'ligand']

    intra_chain_mask = (asym_id[:, None] \
            == asym_id[None]).cast(pair_error_flat.dtype)    # (N_atom, N_atom)
    inter_chain_mask = 1 - intra_chain_mask
    is_protein = batch['is_protein_aa'][batch_i].cast(pair_error_flat.dtype)
    for name in type_names:
        # get intra error
        type_mask = batch[f'is_{name}_aa'][batch_i, :, None].cast(is_protein.dtype) \
                * batch[f'is_{name}_aa'][batch_i, None].cast(is_protein.dtype) \
                * intra_chain_mask  # (N_atom, N_atom)
        type_mask_flat = type_mask[idx1, idx2]     # (m,)
        error = pair_error_flat[type_mask_flat == 1]
        mean_error = paddle.to_tensor(0.0) if len(error) == 0 else error.mean()
        ret[f'intra_{name}'] = mean_error
        ret[f'intra_{name}_exist'] = (type_mask_flat.sum() > 0).cast(error.dtype)
        ret[f'intra_{name}_ratio'] = type_mask_flat.mean()
        # get inter error
        type_mask = is_protein[:, None] \
                * batch[f'is_{name}_aa'][batch_i, None].cast(is_protein.dtype) \
                * inter_chain_mask
        type_mask_flat = type_mask[idx1, idx2]     # (m,)
        error = pair_error_flat[type_mask_flat == 1]
        mean_error = paddle.to_tensor(0.0) if len(error) == 0 else error.mean()
        ret[f'protein_{name}'] = mean_error
        ret[f'protein_{name}_exist'] = (type_mask_flat.sum() > 0).cast(error.dtype)
        ret[f'protein_{name}_ratio'] = type_mask_flat.mean()
    return ret


class DiffusionModule(nn.Layer):
    """
    Diffusion Module
    """
    def __init__(self, channel_num, config, global_config):
        super(DiffusionModule, self).__init__()
        self.config = config
        self.global_config = global_config

        ## diffusion coefficient
        self.sigma_data = 16
        self.s_max = 160
        self.s_min = 4e-4
        self.p = 7
        self.step_num = self.config.step_num
        self.gamma0 = self.config.gamma0
        self.gamma_min = self.config.gamma_min
        self.lambda_ = self.config.get('lambda')
        self.eta = self.config.eta
        self.P_mean = -1.2
        self.P_std = 1.5
        
        ## and easy way to change diffusion args, not recommended
        DIFFUSION_ARGS = os.environ.get('DIFFUSION_ARGS', '')
        if len(DIFFUSION_ARGS) > 0:
            new_args = {item.split(':')[0]: float(item.split(':')[1]) 
                    for item in DIFFUSION_ARGS.split(',')}
            print('[DIFFUSION_ARGS]', new_args)
            for k, v in new_args.items():
                assert hasattr(self, k)
                setattr(self, k, v)

        token_channel = channel_num['token_channel']
        diffusion_token_channel = channel_num['diffusion_token_channel']

        ## network
        self.diffusion_conditioning = DiffusionConditioning(
                channel_num, self.config.diffusion_conditioning, self.global_config)
        self.atom_encoder = AtomAttentionEncoder(
                channel_num, self.config.atom_encoder, self.global_config)
        
        self.ln1 = nn.LayerNorm(token_channel)
        self.lin1 = nn.Linear(token_channel, diffusion_token_channel, bias_attr=False)
        self.diffusion_transformer = DiffusionTransformer(
                channel_num, self.config.diffusion_transformer, self.global_config)
        
        self.ln2 = nn.LayerNorm(diffusion_token_channel)
        self.atom_decoder = AtomAttentionDecoder(
                channel_num, self.config.atom_decoder, self.global_config)

        ## loss
        self.loss_type = self.config.get('loss_type', 'mse')
        self.force_centering = self.config.get('force_centering', False)
        self.fape_z = self.config.get("fape_z", 15.0)

        print(f'[DiffusionModule] loss_type: {self.loss_type}')
        print(f'[DiffusionModule] force_centering: {self.force_centering}')
        print(f'[DiffusionModule] step_num: {self.step_num}')
        print(f'[DiffusionModule] gamma0: {self.gamma0}')
        print(f'[DiffusionModule] lambda_: {self.lambda_}')
        print(f'[DiffusionModule] eta: {self.eta}')

    def _noise_schedule(self, step_num):
        assert step_num > 0
        step_size = 1.0 / step_num
        return get_noise_schedule(self.sigma_data, 
                self.s_max, self.s_min, self.p, step_size)
    
    def _c_ckip(self, t_hat):
        return self.sigma_data ** 2 / (self.sigma_data ** 2 + t_hat ** 2)
    
    def _c_out(self, t_hat):
        return self.sigma_data * t_hat / paddle.sqrt(self.sigma_data ** 2 + t_hat ** 2)
    
    def _forward_model(self, x_noisy, t_hat, batch, representations, return_r=False):
        """
        x_noisy: (B, N_atom, 3)
        t_hat: (B)
        """
        s_inputs = representations['single_inputs']  # (B, N_token, d1)
        s_trunk = representations['single'] # (B, N_token, d1)
        z_trunk = representations['pair']   # (raw_batch, N_token, N_token, d2)
        rel_pos_encoding = representations['rel_pos_encoding']   # (raw_batch, N_token, N_token, d2)
        z_interface = representations['interface_info_encoding'] # (raw_batch, N_token, N_token, d2)
        atom_mask = batch['ref_mask']  # (B, N_atom)
        seq_mask = batch['seq_mask']        # (B, N_token)

        si, zij = recompute_wrapper(self.diffusion_conditioning,
                t_hat, rel_pos_encoding, s_inputs, s_trunk, z_trunk, z_interface, self.sigma_data,
                is_recompute=self.training)

        t_hat = t_hat.unsqueeze([1, 2])
        r_noisy = x_noisy / paddle.sqrt(t_hat ** 2 + self.sigma_data ** 2)

        atom_token_uid = batch['ref_token2atom_idx']
        ai, ql_skip, cl_skip, p_lm_skip = self.atom_encoder(feature=batch, rl=r_noisy, 
                                                            s_trunk=s_trunk, zij=zij) 

        ai += self.lin1(self.ln1(si))
        beta = (1 - seq_mask[:, :, None] * seq_mask[:, None]) * (-1e8)
        ai = self.diffusion_transformer(ai, si, zij, beta=beta)

        ai = self.ln2(ai)
        r_update = self.atom_decoder(ai, ql_skip, cl_skip, p_lm_skip,
                                     atom_token_uid, atom_mask)
        
        x_out = self._c_ckip(t_hat) * x_noisy + self._c_out(t_hat) * r_update
        atom_mask = paddle.cast(atom_mask, dtype=x_out.dtype)
        x_out = x_out * atom_mask[..., None]

        if return_r:
            return x_out, r_update * atom_mask[..., None]
        return x_out
    
    def forward(self, representations, batch):
        """forward"""
        if self.training:
            return self.train(representations, batch)
        else:
            return self.sample_diffusion(representations, batch)
    
    ### for train

    def train(self, representations, batch):
        """
        train
        """
        y = batch['all_atom_pos']       # (B, N_atom, 3)
        atom_mask = batch['all_atom_pos_mask']    # (B, N_atom)
        s_trans = self.config.get('s_trans', 1)
        uniform_aug = self.config.get('uniform_aug', False)
        y = CentreRandomAugmentation(y, atom_mask, s_trans=s_trans, uniform=uniform_aug)
        B, N = y.shape[:2]

        rnd_normal = paddle.normal(shape=[B])
        # sample sigma: ln(sigma / sigma_data) ~ N(P_mean, P_std^2)
        sigma = self.sigma_data * (rnd_normal * self.P_std + self.P_mean).exp()
        n = paddle.normal(shape=y.shape) * sigma.unsqueeze([1, 2])
        x_denoised, r_update = self._forward_model(
                y + n, sigma, batch, representations, return_r=True)

        ret = {
            'all_atom_pos_mask': atom_mask,     # (B, N_atom)
            't_hat': sigma,     # (B,)
            'y': y,     # (B, N_atom, 3)
            'x_noisy': y + n,   # (B, N_atom, 3)
            'x_denoised': x_denoised,   # (B, N_atom, 3)
            'r_update': r_update,   # (B, N_atom, 3)
            # used for multi_chain_permutation_align
            'final_atom_positions': x_denoised,
            'final_atom_mask': atom_mask,
        }
        return ret
        
    def _force_center(self, y, aligned, mask):
        """
        force the aligned label to be at center of y with tolerance of 5A
        """
        tolerance = 5.0
        mask = mask[..., None]
        y_center = (y * mask).sum([1]) / (mask.sum([1]) + 1e-6)
        align_center = (aligned * mask).sum([1]) / (mask.sum([1]) + 1e-6)
        offset = (align_center - y_center)[:, None]
        new_aligned = aligned - offset + paddle.clip(offset, -tolerance, tolerance)
        return new_aligned * mask

    def loss(self, value, batch, label):
        """
        train diffusion loss
        """
        loss = 0.0
        ret = {}

        t_hat = value['t_hat']
        x = value['x_denoised']
        y = value['y']
        x_gt = label['all_atom_pos']
        mask = label['all_atom_pos_mask']
        w_type = 1.0 + batch['is_dna_aa'] * 5.0 + batch['is_rna_aa'] * 5.0 + batch['is_ligand_aa'] * 10.0

        ## align gt
        x_gt_aligned = get_x_gt_aligned(x_gt, y, x, w_type, mask)
        x_gt_aligned.stop_gradient = True
        ## permutation align within each ligand
        x_gt_aligned = chain_align_np_aa.ligand_permutation_align(
                batch, x, x_gt_aligned, mask)
        ## force center
        if self.force_centering:
            x_gt_aligned = self._force_center(y, x_gt_aligned, mask)

        ## loss_mse
        if self.loss_type == 'mse':
            loss_mse = (x - x_gt_aligned) ** 2
            w = w_type * mask
            loss_mse = (loss_mse * w[..., None]).sum([1, 2]) / (w.sum([1]) + 1e-6) / 3.0 # (B,)
            # The formula of weight in af3 is different from EDM which seems to be a typo.
            weight = (t_hat ** 2 + self.sigma_data ** 2) / (t_hat * self.sigma_data) ** 2
            loss += weight * loss_mse
            ret['loss_mse'] = loss_mse
            ret['loss_mse_w'] = weight * loss_mse
        elif self.loss_type == 'huber':
            t_hat = t_hat.unsqueeze([1, 2])
            r_gt = (x_gt_aligned - self._c_ckip(t_hat) * value['x_noisy']) / self._c_out(t_hat)
            r_gt.stop_gradient = True
            loss_huber = F.smooth_l1_loss(value['r_update'], r_gt, reduction='none')
            w = w_type * mask
            loss_huber = (loss_huber * w[..., None]).sum([1, 2]) / (w.sum([1]) + 1e-6) # (B,)
            loss += loss_huber
            ret['loss_huber'] = loss_huber
        else:
            raise ValueError(f'Unknown loss_type: {self.loss_type}')

        ## loss_rmsd, used for visualization, don't add to total_loss for training
        ret.update({f'loss_rmsd_frz-{k}': v for k, v 
                in get_typewise_rmsd(x_gt_aligned, x, mask, batch).items()})

        ## smooth lddt loss
        # token_len = batch['seq_mask'][0].sum()
        # if self.config.loss_smooth_lddt_weight > 0 and token_len < 500:
        if self.config.loss_smooth_lddt_weight > 0:
            ret.update(self._loss_smooth_lddt(x, x_gt_aligned, mask, batch))
            loss += ret['loss_smooth_lddt'] * self.config.loss_smooth_lddt_weight

        ## bond loss
        if self.config.loss_bond_weight > 0:
            ret.update(self._loss_bond(x, x_gt_aligned, mask, batch))
            loss += ret['loss_bond'] * self.config.loss_bond_weight

        ## Frame Aligned Point Error
        if self.config.loss_fape_weight > 0:
            ret.update(self._loss_fape(x, x_gt, batch))
            loss += ret['loss_fape'] * self.config.loss_fape_weight
        
        ret.update({
            'x_gt': x_gt,   # (B, N_atom, 3)
            'x_gt_aligned': x_gt_aligned,   # (B, N_atom, 3)
            'loss': loss,           # (B,)
        })
        return ret
    
    def _loss_smooth_lddt(self, x, x_gt, mask, batch):
        """
        `loss_smooth_lddt` is used for training.
        `loss_smooth_lddt_frz-intra*` and `loss_smooth_lddt_frz-protein*` are
        used for visualization only, not training.
        Args:
            x: (B, N_atom, 3)
            x_gt: (B, N_atom, 3)
            mask: (B, N_atom)
        """
        def _dist(x):
            """x: (..., n, 3)"""
            dist2 = paddle.sum((x.unsqueeze(-2) - x.unsqueeze(-3)) ** 2, -1)
            return paddle.sqrt(dist2 + 1e-8, -1) # (..., n, n)
        def _dist_v2(x1, x2):
            """x1: (..., m, 3), x2: (..., m, 3)"""
            dist2 = paddle.sum((x1 - x2) ** 2, -1)
            return paddle.sqrt(dist2 + 1e-8, -1) # (..., m)
        
        ret = {}
        ret['loss_smooth_lddt'] = []

        B = x.shape[0]
        lddt_list = []
        for bi in range(B):
            cur_mask = mask[bi]     # (N_atom)
            cur_x = x[bi]           # (N_atom, 3)
            cur_x_gt = x_gt[bi]     # (N_atom, 3)

            ## get matrix index
            cur_dist_gt = _dist(cur_x_gt)               # (N_atom, N_atom) 
            flag = cur_mask[None] * cur_mask[:, None]   # (N_atom, N_atom) 
            flag *= (cur_dist_gt < 15).cast(mask.dtype) # gt dist < 15
            flag *= (1 - paddle.eye(flag.shape[0], dtype=mask.dtype))   # not self mask
            idx = paddle.nonzero(flag)  # (m, 2)
            idx1, idx2 = idx[:, 0], idx[:, 1]

            ## cal smooth lddt
            delta_x = _dist_v2(cur_x[idx1], cur_x[idx2])    # (m)
            delta_x_gt = _dist_v2(cur_x_gt[idx1], cur_x_gt[idx2])
            delta = paddle.abs(delta_x - delta_x_gt)
            lddt = 0.0
            for thres in [0.5, 1, 2, 4]:
                lddt += F.sigmoid(thres - delta)   # (m)
            lddt *= 0.25
            error = 1.0 - lddt     # convert to 1 - lddt
            ret['loss_smooth_lddt'].append(paddle.mean(error))

            ## type-wise smooth lddt
            asym_id = batch['perm_asym_id'][bi]
            typewise_dict = get_typewise_pair_error(
                    error, idx1, idx2, asym_id, batch, bi)
            for k, v in typewise_dict.items():
                if k not in ret:
                    ret[f'loss_smooth_lddt_frz-{k}'] = []
                ret[f'loss_smooth_lddt_frz-{k}'].append(v)

        for k, v in ret.items():
            ret[k] = paddle.stack(v)  # (B)
        return ret

    def _loss_bond(self, x, x_gt, mask, batch):
        
        def _l1_dist(x1, x2):
            """x1: (..., m, 3), x2: (..., m, 3)"""
            dist2 = paddle.sum((x1 - x2) ** 2, -1)
            return paddle.sqrt(dist2 + 1e-8, -1) # (..., m)
        
        ret = {}
        B = x.shape[0]
        zero_loss = paddle.zeros([B,], dtype=x.dtype)
        ret['loss_bond'] = []
        if 'covalent_bonds' not in batch.keys() or batch['covalent_bonds'].sum() == 0:
            ret['loss_bond'] = zero_loss
            return ret
        
        N = batch['asym_id'].shape[1]
        token_order_id = paddle.arange(N)
            
        for bi in range(B):
            cur_mask = mask[bi]     # (N_atom)
            cur_x = x[bi]           # (N_atom, 3)
            cur_x_gt = x_gt[bi]     # (N_atom, 3)

            ## get atom indices that covalent bonds connect
            # The indices of tokens that covalent bonds connect
            token_id_pair = batch['covalent_bonds'][bi].nonzero().reshape([-1, 2]) # (i,j)      
            atom_token_uid = batch['ref_token2atom_idx'][bi] # atoms' token_uid
            atom_token_ord_id = token_order_id[atom_token_uid] # atoms' token_order_id
            is_acsending = paddle.all(atom_token_ord_id[:-1] <= atom_token_ord_id[1:]).numpy()
            assert is_acsending == True, "atom_token_ord_id isn't ascending"
            
            # The indices of atoms that covalent bonds connect
            atom_id_pair = paddle.searchsorted(atom_token_ord_id, token_id_pair) # (l,m)

            # check atom validation with atom mask
            valid_bond_id_pair = []
            for i in range(atom_id_pair.shape[0]):
                idx, idy = atom_id_pair[i]
                if cur_mask[idx] != 0 and cur_mask[idy] != 0:
                    valid_bond_id_pair.append(atom_id_pair[i])
            if len(valid_bond_id_pair) == 0:
                ret['loss_bond'].append(paddle.to_tensor(0.0, dtype=x.dtype))
                continue
            valid_bond_id_pair = paddle.stack(valid_bond_id_pair)

            idx1, idx2 = valid_bond_id_pair[:, 0], valid_bond_id_pair[:, 1] 

            ## compute loss bond, the mean of bond distance differences
            delta_x = _l1_dist(cur_x[idx1], cur_x[idx2]) * 0.1   # (m,3)
            delta_x_gt = _l1_dist(cur_x_gt[idx1], cur_x_gt[idx2]) * 0.1
            cur_loss = F.smooth_l1_loss(delta_x, delta_x_gt)
            ret['loss_bond'].append(cur_loss)

        for k, v in ret.items():
            ret[k] = paddle.stack(v)  # (B)
        print(f"[Diffusion Loss] [INFO] Found covalent bond")
        return ret

    def _loss_fape(self, x, x_gt, batch):
        #NOTE: assume all frame atom indices are valid
        Z = self.fape_z
        ret = {}
        ret['loss_fape'] = []

        prot_dna_rna = batch['is_protein'].logical_or(batch['is_dna']).logical_or(batch['is_rna'].cast(paddle.bool))
        frame_mask = batch['frame_mask'].cast(paddle.bool).logical_and(prot_dna_rna)
        pair_mask = frame_mask[..., None] * frame_mask[:, None] # [B,N,N]
        ae = lddt.alignment_error(
            x, 
            x_gt,
            frame_ai_indice=batch['frame_ai_indice'],
            frame_bi_indice=batch['frame_bi_indice'],
            frame_ci_indice=batch['frame_ci_indice'],
            frame_mask=frame_mask)
        ae *= pair_mask
        fape = ae.clip(max=Z).sum([1, 2]) / (pair_mask.sum([1, 2]) * Z + 1e-8)

        ret['loss_fape'] = fape
        return ret
    
    
    ### for sampling

    def sample_diffusion(self, representations, batch, step_num=None):
        """
        sample_diffusion
        """
        if step_num is None:
            step_num = self.step_num
        
        single_act = representations['single']  # (B, N, d1)
        atom_mask = batch['all_atom_pos_mask']
        B, N_atom = atom_mask.shape[:2]
        with paddle.amp.auto_cast(enable=False):
            c_list = self._noise_schedule(step_num)
            x = c_list[0] * paddle.normal(shape=[B, N_atom, 3])
        x0_list = []
        xt_list = []
        for i in range(1, len(c_list)):
            with paddle.amp.auto_cast(enable=False):
                c_tau = c_list[i]
                c_tau_1 = c_list[i - 1]
                x = CentreRandomAugmentation(x, atom_mask)
                gamma = self.gamma0 if c_tau > self.gamma_min else 0
                t_hat = c_tau_1 * (gamma + 1)
                xi = self.lambda_ * paddle.sqrt(t_hat ** 2 - c_tau_1 ** 2) * paddle.normal(shape=[B, N_atom, 3])
                x_noisy = x + xi
            x_denoised = self._forward_model(x_noisy, paddle.tile(t_hat, [B]), batch, representations)
            with paddle.amp.auto_cast(enable=False):
                delta = (x_noisy - x_denoised) / t_hat
                dt = c_tau - t_hat
                x = x_noisy + self.eta * dt * delta

                x0_list.append(x_denoised)
                xt_list.append(x)
        with paddle.amp.auto_cast(enable=False):
            x0_list = paddle.stack(x0_list, 1) * atom_mask.unsqueeze([1, 3])  # (B, T, N_atom, 3)
            xt_list = paddle.stack(xt_list, 1) * atom_mask.unsqueeze([1, 3])  # (B, T, N_atom, 3)

            x *= atom_mask[..., None]
        ret = {
            'x0_list': x0_list,   # (B, T, N_atom, 3)
            'xt_list': xt_list,   # (B, T, N_atom, 3)
            'final_atom_positions': x,   # (B, N_atom, 3)
            'final_atom_mask': atom_mask, # (B, N_atom)
        }
        return ret


class DiffusionConditioning(nn.Layer):
    """
    DiffusionConditioning
    """
    def __init__(self, channel_num, config, global_config):
        super(DiffusionConditioning, self).__init__()
        self.config = config
        self.global_config = global_config
        token_channel = channel_num['token_channel']
        pair_channel = channel_num['token_pair_channel']
        
        self.pair_ln = nn.LayerNorm(pair_channel * 2)
        self.pair_lin = nn.Linear(pair_channel * 2, pair_channel, bias_attr=False)
        self.pair_trans1 = Transition(pair_channel, n=2)
        self.pair_trans2 = Transition(pair_channel, n=2)

        self.single_ln1 = nn.LayerNorm(token_channel * 2 + 32 + 32 + 1)
        self.single_lin1 = nn.Linear(
                token_channel * 2 + 32 + 32 + 1, token_channel, bias_attr=False)
        self.fourier_embedding = FourierEmbedding(256)
        self.single_ln2 = nn.LayerNorm(256)
        self.single_lin2 = nn.Linear(256, token_channel, bias_attr=False)
        self.single_trans1 = Transition(token_channel, n=2)
        self.single_trans2 = Transition(token_channel, n=2)

    def forward(self, t_hat, rel_pos_encoding, s_inputs, s_trunk, z_trunk, z_interface, sigma_data):
        """forward"""

        zij = paddle.concat([z_trunk, rel_pos_encoding], -1)
        zij = self.pair_lin(self.pair_ln(zij))
        # fusion zij with interface info
        zij += z_interface
        zij += self.pair_trans1(zij)
        zij += self.pair_trans2(zij)

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
    def __init__(self, in_channel, n=4):
        super(Transition, self).__init__()
        self.ln = nn.LayerNorm(in_channel)
        self.lin1 = nn.Linear(in_channel, in_channel * n, bias_attr=False)
        self.lin2 = nn.Linear(in_channel, in_channel * n, bias_attr=False)
        self.lin3 = nn.Linear(in_channel * n, in_channel, bias_attr=False)

    def forward(self, x):
        """forward"""
        x = self.ln(x)
        a = self.lin1(x)
        b = self.lin2(x)
        x = self.lin3(nn.functional.swish(a) * b)
        return x


class FourierEmbedding(nn.Layer):
    """
    FourierEmbedding
    """
    def __init__(self, c):
        super(FourierEmbedding, self).__init__()
        self.w = paddle.create_parameter(
                shape=[c], 
                dtype='float32', 
                default_initializer=nn.initializer.Normal())
        self.w.stop_gradient = True
        self.b = paddle.create_parameter(
                shape=[c], 
                dtype='float32', 
                default_initializer=nn.initializer.Normal())
        self.b.stop_gradient = True

    def forward(self, t_hat):
        """
        t_hat: (B,)
        return:
            (B, c)
        """
        y = paddle.cos(2 * np.pi * (t_hat[:, None] * self.w[None] + self.b[None]))
        return y


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

        use_rotary = self.config.get('use_rotary', False)

        self.attention_list = nn.LayerList()
        self.transition_list = nn.LayerList()
        for n in range(self.config.n_block):
            self.attention_list.append(AttentionPairBias(
                    a_channel, s_channel, z_channel, 
                    self.config.n_head, has_si=True,
                    use_rotary=use_rotary))
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
            n_head, has_si, use_rotary=False, dropout_rate=0.1):
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
        
        # TODO: should consider input_position if having multiple chains
        self.use_rotary = use_rotary
        if self.use_rotary:
            self.rope = rotary.RotaryPositionalEmbeddings(dim=self.head_dim)

        default_M = 10000
        self._AttenIndex = AttentionIndex(max_atom_num=default_M)
  
    def forward(self, ai, si, zij, beta):
        """
        ai: (B, N, d1)
        si: (B, N, d1)
        zij: (B, N, N, d2)
        beta: (B, N, N) or (1, N, N)
        attention_idx
        """
        assert self.has_si == (not si is None)

        B, N, D = paddle.shape(ai)
        H, d = self.n_head, self.head_dim

        if self.has_si:
            ai = self.ln(ai, si)
        else:
            ai = self.ln(ai)

        # zij is not tiled by diff_batch_size so far
        b = self.b_lin(self.b_ln(zij)) # (B, N, N, H) or (B, C, nq, nk, H)
        g = nn.functional.sigmoid(self.g_lin(ai))\
                         .reshape([B, N, H, d]).transpose([0, 2, 1, 3]) # (B, H, N, d)
        diff_batch_size = ai.shape[0] // b.shape[0]

        if len(zij.shape) == 5:
            # local attention
            M = ai.shape[1]
            atten_idx = self._AttenIndex.get_atten_idx(M)

            query_idx = atten_idx['query_idx'].flatten() # [C,32]
            query_mask = atten_idx['query_mask'][..., None, None] # [C,32,1,1]
            key_idx = atten_idx['key_idx'].flatten() # [C,128,1,1,1]
            key_mask = atten_idx['key_mask'][..., None, None]   # [C,128,1,1]
            alpha_mask = atten_idx['alpha_mask'] # [C,32,128]
            C, n_query, n_key = alpha_mask.shape

            query_like_shape = [C, n_query, B, D]
            key_like_shape = [C, n_key, B, D]

            query_mask = query_mask.cast(b.dtype)
            key_mask = key_mask.cast(b.dtype)
            q = ai.transpose([1, 0, 2])  # [N, B, d]
            k = ai.transpose([1, 0, 2])  # [N, B, d]
            v = ai.transpose([1, 0, 2])  # [N, B, d]
            g = g.transpose([2, 0, 1, 3])  # [N, B, H, d]

            q = q[query_idx].reshape(query_like_shape) * query_mask # (C, 32, B, d)
            k = k[key_idx].reshape(key_like_shape) * key_mask # (C, 128, B, d)
            v = v[key_idx].reshape(key_like_shape) * key_mask # (C, 128, B, d)
            g = g[query_idx].reshape([C, n_query, B, H, d]) * query_mask[..., None] # (C, 32, B, H, d)

            q = q.transpose([2, 0, 1, 3]) # (C, 32, B, d) -> (B, C, 32, d)
            k = k.transpose([2, 0, 1, 3]) # (C, 128, B, d) -> (B, C, 128, d)
            v = v.transpose([2, 0, 1, 3]) # (C, 128, B, d) -> (B, C, 128, d)
            g = g.transpose([2, 3, 0, 1, 4]) # (C, 32, B, H, d) -> (B, H, C, 32, d)
            b = b.transpose([0, 4, 1, 2, 3]) # (b, C, 32, 128, H) -> (b, H, C, 32, 128)

            if diff_batch_size > 1:
                b = tile_batch_dim(b, diff_batch_size)
            beta = tile_batch_dim(beta, ai.shape[0])
            b = (b + beta.unsqueeze(1)) # (B, H, C, 32, 128)

            # (B, H, C, n_query, d)
            q = self.q_lin(q).reshape([B, C, n_query, H, d]).transpose([0, 3, 1, 2, 4])
            # (B, H, C, n_key, d)
            k = self.k_lin(k).reshape([B, C, n_key, H, d]).transpose([0, 3, 1, 2, 4])
            if self.use_rotary:
                q = self.rope(q)
                k = self.rope(k)
            # (B, H, C, n_key, d)
            v = self.v_lin(v).reshape([B, C, n_key, H, d]).transpose([0, 3, 1, 2, 4])

            # (B, H, C, n_query, n_key)
            alpha = paddle.matmul(q / np.sqrt(d), k, transpose_y=True) + b
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
            b = (b + beta[..., None]).transpose([0, 3, 1, 2]) # (B, N, N, H) -> (B, H, N, N)

            q = self.q_lin(ai).reshape([B, N, H, d]).transpose([0, 2, 1, 3]) # (B, H, N, d)
            k = self.k_lin(ai).reshape([B, N, H, d]).transpose([0, 2, 1, 3]) # (B, H, N, d)
            if self.use_rotary:
                q = self.rope(q)
                k = self.rope(k)
            v = self.v_lin(ai).reshape([B, N, H, d]).transpose([0, 2, 1, 3]) # (B, H, N, d)

            alpha = paddle.matmul(q / np.sqrt(d), k, transpose_y=True) + b    # (B, H, N, N)
            alpha = F.softmax(alpha) # (B, H, N, N)
            alpha = self.alpha_dropout(alpha)

            ai = paddle.matmul(alpha, v) * g # (B, H, N, d)
        
        ai = ai.transpose([0, 2, 1, 3]).reshape([B, N, D])
        ai = self.out_lin1(ai)
        ai = self.out_dropout(ai)
        if self.has_si:
            ai = nn.functional.sigmoid(self.out_lin2(si)) * ai
        return ai


class ConditionedTransitionBlock(nn.Layer):
    """
    ConditionedTransitionBlock
    """
    def __init__(self, a_channel, s_channel, n=2):
        super(ConditionedTransitionBlock, self).__init__()
        self.ln = AdaLN(a_channel, s_channel)
        self.lin1 = nn.Linear(a_channel, a_channel * n, bias_attr=False)
        self.lin2 = nn.Linear(a_channel, a_channel * n, bias_attr=False)
        self.lin3 = nn.Linear(s_channel, a_channel, 
                bias_attr=nn.initializer.Constant(value=-2.0))
        self.lin4 = nn.Linear(a_channel * n, a_channel, bias_attr=False)

    def forward(self, a, s):
        """forward"""
        a = self.ln(a, s)
        b = nn.functional.swish(self.lin1(a)) * self.lin2(a)
        a = nn.functional.sigmoid(self.lin3(s)) * self.lin4(b)
        return a


class AdaLN(nn.Layer):
    """
    AdaLN
    """
    def __init__(self, a_channel, s_channel):
        super(AdaLN, self).__init__()
        self.a_ln = nn.LayerNorm(a_channel, weight_attr=False, bias_attr=False)
        self.s_ln = nn.LayerNorm(s_channel, bias_attr=False)
        self.lin1 = nn.Linear(s_channel, a_channel)
        self.lin2 = nn.Linear(s_channel, a_channel, bias_attr=False)

    def forward(self, a, s):
        """forward"""
        a = self.a_ln(a)
        s = self.s_ln(s)
        a = nn.functional.sigmoid(self.lin1(s)) * a + self.lin2(s)
        return a


""" Atom Attention """

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
        plm = self.lin_pos_offset_to_apair(dlm) * vlm # (b, M, M, c_atompair)

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
            
            plm += self.ap_util.to_atompair(zij=zij, atom_token_uid=atom_token_uid, 
                                            atom_mask=atom_mask, dense=self.dense)
            # assert plm.shape[1] == cl.shape[1]
            
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
        ai = aggregate_atom_feat_to_token(
            al, atom_token_uid, atom_mask, N_token) # (B, N_res, c_t)

        return ai, ql, cl, plm


class AtomTransformer(nn.Layer):
    " Atom Transformer for AF3. "
    
    def __init__(self, channel_num, config, global_config):
        super(AtomTransformer, self).__init__()
        self.config = config
        self.n_query = config.n_query
        self.n_key = config.n_key
        self.default_size = 10000
        self.diff_transformer = DiffusionTransformer(
            channel_num, config.diffusion_transformer, global_config)
        self._AttenIndex = AttentionIndex(self.default_size, self.n_query, self.n_key)

    def forward(self, ql, cl, plm):
        """
        ql: (B, M, d1)
        cl: (B, M, d1)
        plm: (B, M, M, d2)
        atten_idx: q,k,v,b,g indices for local seq attention
        """

        M = ql.shape[1]
        atten_idx = self._AttenIndex.get_atten_idx(M)
        if len(plm.shape) == 4:
            # sparse plm
            beta = self._get_beta_mask(M) # [M,M]
        else:
            # dense plm
            assert len(plm.shape) == 5
            alpha_mask = atten_idx['alpha_mask']
            beta = paddle.full_like(alpha_mask, fill_value=-10.0**10, dtype='bfloat16') #TODO: type check
            beta[alpha_mask == 1] = 0

        beta = beta[None] # [1, M, M] or [1, C, 32, 128] 
        ql = self.diff_transformer(ql, cl, plm, beta=beta)
        return ql
    
    def _get_beta_mask(self, M):
        if self.beta is None:
            self.beta = self._gen_beta_mask(self.default_size)
        
        if M > self.default_size:
            # gen a larger beta mask
            return self._gen_beta_mask(M)
        
        return self.beta[:M, :M]
    
    def _gen_beta_mask(self, M):
        subset_centers = self._get_subset_centers(M)
        beta = np.full([M, M], -10.0**10, dtype=np.float32)
        half_width = self.n_query // 2
        half_height = self.n_key // 2
        
        for c in subset_centers:
            left = np.max([c - half_width, 0])
            right = np.min([c + half_width, M])
            top = np.max([c - half_height, 0])
            bottom = np.min([c + half_height, M])
            beta[left:right, top:bottom] = 0.0

        return paddle.to_tensor(beta)

    def _get_subset_centers(self, M):
        half = (self.n_query - 1.0) * 0.5
        centers = np.arange(half, M + half, self.n_query, dtype=np.float32)
        return np.round(centers).astype(np.int64)


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


class RelativePositionEncoding(nn.Layer):
    """
    Algorithm 3: RelativePositionEncoding
    """
    def __init__(self, channel_num, config, global_config):
        super(RelativePositionEncoding, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        self.rel_position_project = nn.Linear(
            # rel_pos & rel_token: R_max * 2 + 2
            # same_entity: 1
            # rel_chain: S_max * 2 + 2
            2 * (self.config.relative_token_max * 2 + 2) + 1 + \
            2 * self.config.relative_chain_max + 2,
            self.channel_num['token_pair_channel'],
            bias_attr=False)

    def forward(self, batch):
        asym_id = batch['asym_id']
        same_chain = asym_id.unsqueeze(axis=-2) == asym_id.unsqueeze(axis=-1)

        pos = batch['residue_index']
        same_residue = pos.unsqueeze(axis=-2) == pos.unsqueeze(axis=-1)

        entity_id = batch['entity_id']
        same_entity = entity_id.unsqueeze(axis=-2) == entity_id.unsqueeze(axis=-1)

        def _calc_clipped_offset(fi, r_max, is_same):
            offset = fi.unsqueeze(axis=-1) - fi.unsqueeze(axis=-2)
            clipped_offset = paddle.clip(offset + r_max, min=0, max=2 * r_max)
            final_offset = paddle.where(
                is_same,
                clipped_offset,
                (2 * r_max + 1) * paddle.ones_like(clipped_offset))
            return nn.functional.one_hot(final_offset, 2 * r_max + 2)

        rel_pos = _calc_clipped_offset(
            pos, self.config.relative_token_max, same_residue)

        token_id = batch['token_index']
        rel_token = _calc_clipped_offset(
            token_id, self.config.relative_token_max,
            paddle.logical_and(same_residue, same_chain))

        sym_id = batch['sym_id']
        rel_chain = _calc_clipped_offset(
            sym_id, self.config.relative_chain_max,
            paddle.logical_not(same_chain))

        same_entity_ = paddle.cast(same_entity.unsqueeze(axis=-1),
                                   rel_pos.dtype)
        rel_act = paddle.concat(
            [rel_pos, rel_token, same_entity_, rel_chain], axis=-1)
        return self.rel_position_project(rel_act)


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


@singleton
class AttentionIndex():
    
    " Attention Index for Local Sequence Attention. "

    def __init__(self, max_atom_num=10000, n_query=32, n_key=128):
        # TODO: optimize construction of this singleton class
        self.n_query = n_query
        self.n_key = n_key
        self._key_list = ['query_idx', 'query_mask', 'key_idx', 'key_mask', 
                    'alpha_mask', 'pair_idx']
        self._upd_M_N_create_index(max_atom_num)
        self.attn_cache = dict()
        self.xy_idx_cache = dict()
        
    def _upd_M_N_create_index(self, M):
        """ Update M and create a larger arange of indices. 
        
        [IMPORTANT] call order matters
        """
        self.max_atom_num = M # always update M first
        self._create_subset_centers()
        self._create_local_pair_idx()
        self._create_attention_idx()
        self._prepare_api()

    def _create_subset_centers(self):
        """ Find all subset centers for local sequence attention.
         
        Create:
        - self._centers: ndarray, shape [C]
        - self.n_centers: int, the number of centers.
        """
        M = self.max_atom_num
        half = (self.n_query - 1.0) * 0.5
        centers = np.arange(half, M + half, self.n_query, dtype=np.float32)
        centers = np.round(centers).astype(np.int32)
        self._centers = centers
        self.n_centers = len(centers)


    def _create_local_pair_idx(self):
        """ Find involved atom-level features' indices for local sequence attention.

        For local sequence attention of a pair of atom-level query ql and key qm, 
        compute the indices of corresponding features for every local attention.

        Create:
        - xid: ndarray, shape: [C*(B-T)*(R-L)]
            The dense indices of query feat in local sequence attention.
        - yid: ndarray, shape: [C*(B-T)*(R-L)]
            The dense indices of key feat in local sequence attention.
        """
        nq, nk = self.n_query, self.n_key
        M = self.max_atom_num
        centers, C = self.get_current_centers()

        xid, yid = [], []
        for c in centers:
            T = np.max([c - nq // 2, 0])
            B = np.min([c + nq // 2, M])
            L = np.max([c - nk // 2, 0])
            R = np.min([c + nk // 2, M])
            x, y = np.meshgrid(np.arange(T, B, dtype=np.int32), 
                            np.arange(L, R, dtype=np.int32))
            # shape: x:[R-L, B-T], y:[R-L, B-T]
            xid.append(x.T.flatten()) # [(B-T)*(R-L)]
            yid.append(y.T.flatten()) # [(B-T)*(R-L)]
        
        xid = np.concatenate(xid, axis=0)
        yid = np.concatenate(yid, axis=0)
        self._xid, self._yid = xid, yid

    def _create_attention_idx(self):
        """ Find all relevent indices required by attention.
        
        Create Indices:
        - query_idx: ndarray [C, n_query]
        - query_mask: ndarray [C, n_query]
        - key_idx: ndarray [C, n_key]
        - key_mask: ndarray [C, n_key]
        - alpha_mask: ndarray [C, n_query, n_key]
        - pair_idx: ndarray [C, n_query, n_key]
        """
        self.attention_idx = None # clear old index for saving memory
        centers, C = self.get_current_centers()
        nq, nk = self.n_query, self.n_key
        M = self.max_atom_num
        
        # query indices
        query_idx = np.arange(M, dtype=np.int32)
        query_pad_len = C * nq - M
        query_pad = np.full([query_pad_len], -1, dtype=np.int32)
        query_idx = np.concatenate([query_idx, query_pad]).reshape(C, nq)
        query_mask = (query_idx != -1).astype(np.int32)
        query_idx *= query_mask

        # key indices
        key_blks = []
        for c in centers:
            start = np.max([c - nk // 2, 0])
            end = np.min([c + nk // 2, M])
            window_len = end - start
            key_id_c = np.full(nk, -1, dtype=np.int32)
            key_id_c[:window_len] = np.arange(start, end, dtype=np.int32)
            key_blks.append(key_id_c)
        key_idx = np.concatenate(key_blks).reshape([C, nk])
        key_mask = (key_idx != -1).astype(np.int32)
        key_idx *= key_mask

        # alpha mask
        alpha_mask = query_mask[..., np.newaxis] * key_mask[:, np.newaxis] #[C,nq,nk]
        valid_alpha_idx = alpha_mask.flatten().nonzero()[0]

        # bias indices and beta indices
        valid_pair_id = self._xid * M + self._yid
        pair_idx = np.zeros_like(alpha_mask, dtype=np.int32).flatten()
        np.put_along_axis(pair_idx, valid_alpha_idx, valid_pair_id, axis=0)
        pair_idx = pair_idx.reshape([C, nq, nk])

        # save
        self._query_idx, self._query_mask = query_idx, query_mask
        self._key_idx, self._key_mask = key_idx, key_mask
        self._alpha_mask = alpha_mask
        self._pair_idx = pair_idx


    def _prepare_api(self):
        """ Convert indices from ndarray to paddle tensor. """
        self.xy_idx = (paddle.to_tensor(self._xid), paddle.to_tensor(self._yid))
        self.attention_idx = {'query_idx': paddle.to_tensor(self._query_idx), 
                              'query_mask': paddle.to_tensor(self._query_mask), 
                              'key_idx': paddle.to_tensor(self._key_idx), 
                              'key_mask': paddle.to_tensor(self._key_mask),
                              'alpha_mask': paddle.to_tensor(self._alpha_mask), 
                              'pair_idx': paddle.to_tensor(self._pair_idx), 
                              'centers': paddle.to_tensor(self._centers)}

    
    def _slice_subset_index(self, atten_idx, C):
        return copy.deepcopy({k: atten_idx[k][:C] for k in self._key_list})

    def _padding(self, index, M):
        """ Replace out-of-range indices by 0s.
        Args:
        - index: dict[tensor]
        - M: int, new maximum number of atoms.
        """
        for k, m in [('query_idx', 'query_mask'), ('key_idx', 'key_mask')]:
            idx = index[k]
            mask = index[m]
            mask[idx >= M] = 0
            idx[idx >= M] = 0
            index[k], index[m] = idx, mask

        index['alpha_mask'] = \
            index['query_mask'][..., None] * index['key_mask'][:, None]
        
        C, nq, nk = index['pair_idx'].shape
        xid, yid = self.get_xy_idx(M)
        valid_pair_id = xid * M + yid
        valid_alpha_idx = index['alpha_mask'].flatten().nonzero()
        index['pair_idx'] = paddle.scatter_nd(index=valid_alpha_idx, 
                                              updates=valid_pair_id, 
                                              shape=[C * nq * nk])
        index['pair_idx'] = index['pair_idx'].reshape([C, nq, nk])
    
    def get_current_centers(self):
        """ Get currently-using subset center. No update. """
        return self._centers, self.n_centers

    def get_centers(self, M, n_query=32):
        """ Compute the subset centers for any given M. """
        half = (n_query - 1.0) * 0.5
        centers = np.arange(half, M + half, n_query, dtype=np.float32)
        centers = np.round(centers).astype(np.int32)
        return centers, len(centers)
    
    def get_xy_idx(self, M):
        """ Get query and key indices for any given max number of atoms.

        Returns:
        - x_idx: tensor
        - y_idx: tensor
        """
        assert self.xy_idx is not None
        if M == self.max_atom_num:
            return self.xy_idx
        if M > self.max_atom_num:
            # create_larger index for larger M
            self._upd_M_N_create_index(M)
            self.xy_idx_cache[M] = self.xy_idx
            return self.xy_idx_cache[M][0], self.xy_idx_cache[M][1]
        if M not in self.xy_idx_cache:
            # slice subset indices
            xid, yid = self.xy_idx
            subset_mask = (xid < M) * (yid < M)
            self.xy_idx_cache[M] = (xid[subset_mask],yid[subset_mask])
        return self.xy_idx_cache[M][0], self.xy_idx_cache[M][1]
    
    def get_atten_idx(self, M):
        """ Get attention indices for any given max number of atoms.

        [Important] Assume Idx's content won't be changed. Otherwise, return a deepcopy version. 
        
        Returns:
        attention index: dict[tensor]
        - query_idx: tensor, shape: [C, n_query]
        - query_mask: tensor, shape: [C, n_query]
        - key_idx: tensor, shape: [C, n_key]
        - key_mask: tensor, shape: [C, n_key]
        - alpha_mask: tensor, shape: [C, n_query, n_key]
        - pair_idx: tensor, shape: [C, n_query, n_key]
        - centers: tensor, shape: [C]
        """
        if M == self.max_atom_num:
            return self.attention_idx
        
        if M > self.max_atom_num:
            # create larger index for larger M
            self._upd_M_N_create_index(M)
            self.attn_cache[M] = self.attention_idx
            return self.attention_idx
        
        if M not in self.attn_cache:
            # slice subset indices
            centers, C = self.get_centers(M)

            sub_atten_idx = self._slice_subset_index(self.attention_idx, C)
            # replace out-of-range indices by zero padding
            self._padding(sub_atten_idx, M)
            for k in self._key_list:
                sub_atten_idx[k] = paddle.to_tensor(sub_atten_idx[k])
            sub_atten_idx['centers'] = paddle.to_tensor(centers)
            self.attn_cache[M] = sub_atten_idx
        return self.attn_cache[M]

@singleton
class AttentionIndexPd():
    
    " paddle version of AttentionIndex, Attention Index for Local Sequence Attention. "

    def __init__(self, max_atom_num=10000, n_query=32, n_key=128):
        # TODO: optimize construction of this singleton class
        self.n_query = n_query
        self.n_key = n_key
        self._key_list = ['query_idx', 'query_mask', 'key_idx', 'key_mask', 
                    'alpha_mask', 'pair_idx']
        self._upd_M_N_create_index(max_atom_num)

        
    def _upd_M_N_create_index(self, M):
        """ Update M and create a larger arange of indices. 
        
        [IMPORTANT] call order matters
        """
        self.max_atom_num = M # always update M first
        self._create_subset_centers()
        self._create_local_pair_idx()
        self._create_attention_idx()
        self._prepare_api()

    def _create_subset_centers(self):
        """ Find all subset centers for local sequence attention.
         
        Create:
        - self._centers: ndarray, shape [C]
        - self.n_centers: int, the number of centers.
        """
        M = self.max_atom_num
        half = (self.n_query - 1.0) * 0.5
        centers = paddle.arange(half, M + half, self.n_query, dtype=paddle.float32)
        centers = paddle.round(centers).astype(paddle.int32)
        self._centers = centers
        self.n_centers = len(centers)


    def _create_local_pair_idx(self):
        """ Find involved atom-level features' indices for local sequence attention.

        For local sequence attention of a pair of atom-level query ql and key qm, 
        compute the indices of corresponding features for every local attention.

        Create:
        - xid: ndarray, shape: [C*(B-T)*(R-L)]
            The dense indices of query feat in local sequence attention.
        - yid: ndarray, shape: [C*(B-T)*(R-L)]
            The dense indices of key feat in local sequence attention.
        """
        nq, nk = self.n_query, self.n_key
        M = self.max_atom_num
        centers, C = self.get_current_centers()

        xid, yid = [], []
        for c in centers:
            T = np.max([c - nq // 2, 0])
            B = np.min([c + nq // 2, M])
            L = np.max([c - nk // 2, 0])
            R = np.min([c + nk // 2, M])
            x, y = paddle.meshgrid(paddle.arange(T, B, dtype=paddle.int32), 
                            paddle.arange(L, R, dtype=paddle.int32))
            # shape: x:[R-L, B-T], y:[R-L, B-T]
            xid.append(x.flatten()) # [(B-T)*(R-L)]
            yid.append(y.flatten()) # [(B-T)*(R-L)]
        
        xid = paddle.concat(xid, axis=0)
        yid = paddle.concat(yid, axis=0)
        self._xid, self._yid = xid, yid

    def _create_attention_idx(self):
        """ Find all relevent indices required by attention.
        
        Create Indices:
        - query_idx: ndarray [C, n_query]
        - query_mask: ndarray [C, n_query]
        - key_idx: ndarray [C, n_key]
        - key_mask: ndarray [C, n_key]
        - alpha_mask: ndarray [C, n_query, n_key]
        - pair_idx: ndarray [C, n_query, n_key]
        """
        self.attention_idx = None # clear old index for saving memory
        centers, C = self.get_current_centers()
        nq, nk = self.n_query, self.n_key
        M = self.max_atom_num
        
        # query indices
        query_idx = paddle.arange(M, dtype=paddle.int32)
        query_pad_len = C * nq - M
        query_pad = paddle.full([query_pad_len], -1, dtype=paddle.int32)
        query_idx = paddle.concat([query_idx, query_pad]).reshape([C, nq])
        query_mask = (query_idx != -1).astype(paddle.int32)
        query_idx *= query_mask

        # key indices
        key_blks = []
        for c in centers:
            start = np.max([c - nk // 2, 0])
            end = np.min([c + nk // 2, M])
            window_len = end - start
            key_id_c = paddle.full(nk, -1, dtype=paddle.int32)
            key_id_c[:window_len] = paddle.arange(start, end, dtype=paddle.int32)
            key_blks.append(key_id_c)
        key_idx = paddle.concat(key_blks).reshape([C, nk])
        key_mask = (key_idx != -1).astype(paddle.int32)
        key_idx *= key_mask

        # alpha mask
        alpha_mask = query_mask[..., None] * key_mask[:, None] #[C,nq,nk]
        valid_alpha_idx = alpha_mask.flatten().nonzero().squeeze(1)
        # bias indices and beta indices
        valid_pair_id = self._xid * M + self._yid
        pair_idx = paddle.zeros_like(alpha_mask, dtype=paddle.int32).flatten()
        pair_idx = paddle.put_along_axis(pair_idx, valid_alpha_idx, valid_pair_id, axis=0)
        pair_idx = pair_idx.reshape([C, nq, nk])

        # save
        self._query_idx, self._query_mask = query_idx, query_mask
        self._key_idx, self._key_mask = key_idx, key_mask
        self._alpha_mask = alpha_mask
        self._pair_idx = pair_idx


    def _prepare_api(self):
        """ Convert indices from ndarray to paddle tensor. """
        self.xy_idx = (paddle.to_tensor(self._xid), paddle.to_tensor(self._yid))
        self.attention_idx = {'query_idx': paddle.to_tensor(self._query_idx), 
                              'query_mask': paddle.to_tensor(self._query_mask), 
                              'key_idx': paddle.to_tensor(self._key_idx), 
                              'key_mask': paddle.to_tensor(self._key_mask),
                              'alpha_mask': paddle.to_tensor(self._alpha_mask), 
                              'pair_idx': paddle.to_tensor(self._pair_idx), 
                              'centers': paddle.to_tensor(self._centers)}

    
    def _slice_subset_index(self, atten_idx, C):
        return copy.deepcopy({k: atten_idx[k][:C] for k in self._key_list})

    def _padding(self, index, M):
        """ Replace out-of-range indices by 0s.
        Args:
        - index: dict[tensor]
        - M: int, new maximum number of atoms.
        """
        for k, m in [('query_idx', 'query_mask'), ('key_idx', 'key_mask')]:
            idx = index[k]
            mask = index[m]
            mask[idx >= M] = 0
            idx[idx >= M] = 0
            index[k], index[m] = idx, mask

        index['alpha_mask'] = \
            index['query_mask'][..., None] * index['key_mask'][:, None]
        
        C, nq, nk = index['pair_idx'].shape
        xid, yid = self.get_xy_idx(M)
        valid_pair_id = xid * M + yid
        valid_alpha_idx = index['alpha_mask'].flatten().nonzero()
        index['pair_idx'] = paddle.scatter_nd(index=valid_alpha_idx, 
                                              updates=valid_pair_id, 
                                              shape=[C * nq * nk])
        index['pair_idx'] = index['pair_idx'].reshape([C, nq, nk])
    
    def get_current_centers(self):
        """ Get currently-using subset center. No update. """
        return self._centers, self.n_centers

    def get_centers(self, M, n_query=32):
        """ Compute the subset centers for any given M. """
        half = (n_query - 1.0) * 0.5
        centers = paddle.arange(half, M + half, n_query, dtype=paddle.float32)
        centers = paddle.round(centers).astype(paddle.int32)
        return centers, len(centers)
    
    def get_xy_idx(self, M):
        """ Get query and key indices for any given max number of atoms.

        Returns:
        - x_idx: tensor
        - y_idx: tensor
        """
        assert self.xy_idx is not None
        if M == self.max_atom_num:
            return self.xy_idx
        if M > self.max_atom_num:
            # create_larger index for larger M
            self._upd_M_N_create_index(M)
            self.xy_idx_cache[M] = self.xy_idx
            return self.xy_idx_cache[M][0], self.xy_idx_cache[M][1]
        if M not in self.xy_idx_cache:
            # slice subset indices
            xid, yid = self.xy_idx
            subset_mask = (xid < M) * (yid < M)
            self.xy_idx_cache[M] = (xid[subset_mask],yid[subset_mask])
        return self.xy_idx_cache[M][0], self.xy_idx_cache[M][1]
    
    def get_atten_idx(self, M):
        """ Get attention indices for any given max number of atoms.

        [Important] Assume Idx's content won't be changed. Otherwise, return a deepcopy version. 
        
        Returns:
        attention index: dict[tensor]
        - query_idx: tensor, shape: [C, n_query]
        - query_mask: tensor, shape: [C, n_query]
        - key_idx: tensor, shape: [C, n_key]
        - key_mask: tensor, shape: [C, n_key]
        - alpha_mask: tensor, shape: [C, n_query, n_key]
        - pair_idx: tensor, shape: [C, n_query, n_key]
        - centers: tensor, shape: [C]
        """
        if M == self.max_atom_num:
            return self.attention_idx
        
        if M > self.max_atom_num:
            # create larger index for larger M
            self._upd_M_N_create_index(M)
            self.attn_cache[M] = self.attention_idx
            return self.attention_idx
        
        if M not in self.attn_cache:
            # slice subset indices
            centers, C = self.get_centers(M)

            sub_atten_idx = self._slice_subset_index(self.attention_idx, C)
            # replace out-of-range indices by zero padding
            self._padding(sub_atten_idx, M)
            for k in self._key_list:
                sub_atten_idx[k] = paddle.to_tensor(sub_atten_idx[k])
            sub_atten_idx['centers'] = paddle.to_tensor(centers)
            self.attn_cache[M] = sub_atten_idx
        return self.attn_cache[M]


class AtomPairUtil():
    """ Compute atompair features according to atom attention indices. """
    def __init__(self, M=10000, n_query=32, n_key=128):
        self.M = M
        self.nq = n_query
        self.nk = n_key
        self._AttenIdx = AttentionIndex(M, n_query=n_query, n_key=n_key)
            
    def to_atompair(self, zij, atom_token_uid, atom_mask, dense):
        """ Convert token-level pair feature to atom-pair features. 
        
        Args:
        - zij: Tensor, [B,N,N,D]
            token-level pair feature
        - atom_token_uid: Tensor, [B,M]
            token to atom unique id.
        - atom_mask: Tensor, [B,M]
        - local: bool, use local attention (return dense feature) 
                        or global attention (return sparse feature)
        """
        if dense:
            return self._pair_to_atompair_dense(zij, atom_token_uid)
        else:
            return self._pair_to_atompair_sparse(zij, atom_token_uid, atom_mask)
        

    def add_2_seqs(self, ql, qm, dense):
        """ Compute atompair by adding two atom-level sequences feature. 
        
        Args:
        - ql, qm: Tensor [B,M,D]
        - local: bool, use local attention (return dense feature) 
                        or global attention (return sparse feature)
        """
        if len(ql.shape) == 2:
            ql, qm = ql.unsqueeze(-1), qm.unsqueeze(-1)
        result = self._add_2_seqs_dense(ql, qm) if dense else self._add_2_seqs_sparse(ql, qm)
        return result if len(ql.shape) != 2 else result.squeeze(-1)
    

    def cmp_2_seqs(self, ql, qm, dense):
        """ Compute atompair by comparing two atom-level sequences feature. 
        
        Args:
        - ql, qm: Tensor [B,M,D]
        - local: bool, use local attention (return dense feature) 
                        or global attention (return sparse feature)
        """
        if len(ql.shape) == 2:
            ql, qm = ql.unsqueeze(-1), qm.unsqueeze(-1)
        result = self._cmp_2_seqs_dense(ql, qm) if dense else self._cmp_2_seqs_sparse(ql, qm)
        return result
    
    
    def _pair_to_atompair_sparse(self, f_pair, atom_token_uid, atom_mask):
        """ Convert per-token pair feature to sparse atom-pair features.

        Args:
        - f_pair:         Tensor: [B,N,N,D]
        - atom_token_uid: Tensor: [B,M]
        - atom_mask:      Tensor: [B,M]

        Returns:
        - f_pair:         Tensor: [B,M,M,D]
        """
        B, N, _, D = f_pair.shape
        M = atom_token_uid.shape[1]
        diff_batch_size = atom_token_uid.shape[0] // B
        if diff_batch_size > 1:
            atom_token_uid = atom_token_uid[:B]
            atom_mask = atom_mask[:B]
        
        atom_token_uid = atom_token_uid.flatten(0, 1) # [B*M]
        f_pair = f_pair.flatten(0, 1)   # [B*N,N,D]
        f_pair = f_pair[atom_token_uid] * atom_mask.reshape([-1, 1, 1]) #[B*M,N,D]
        f_pair = f_pair.reshape([B, M, N, D]).transpose([0, 2, 1, 3]).flatten(0, 1) #[B*N,M,D]
        f_pair = f_pair[atom_token_uid] * atom_mask.reshape([-1, 1, 1]) #[B*M,M,D]
        
        return f_pair.reshape([B, M, M, D]).transpose([0, 2, 1, 3]) # [B,M,M,D]
    
    
    def _pair_to_atompair_dense(self, f_pair, atom_token_uid):
        """ Convert per-token pair feature to dense atom-pair features.

        Args:
        - f_pair:         Tensor: [B,N,N,D]
        - atom_token_uid: Tensor: [B,M]

        Returns:
        - f_pair:         Tensor: [B,C,nq,nk,D]
        """
        b, _, _, D = f_pair.shape
        M = atom_token_uid.shape[1]
        pair_mask = self._AttenIdx.get_atten_idx(M)['alpha_mask'] # [C,nq,nk]
        C, nq, nk = pair_mask.shape
        # local indices (x,y) in atompair, dense tensor:[num_valid_ap] ≈ C*nq*nk
        ap_id_x, ap_id_y = self._AttenIdx.get_xy_idx(M)
        valid_pair_idx = pair_mask.flatten().nonzero() #[num_valid_ap]

        ap = []
        for i in range(b):
            uid = atom_token_uid[i]
            # query and key feats
            x_seq_id, y_seq_id = uid[ap_id_x], uid[ap_id_y]
            ap_feat_dense = f_pair[i][x_seq_id, y_seq_id].reshape([
                                 x_seq_id.size, D]) # [num_valid_ap, D]
            atompair = paddle.scatter_nd(index=valid_pair_idx, 
                                        updates=ap_feat_dense,
                                        shape=[C * nq * nk, D])
            atompair = atompair.reshape([C, nq, nk, D]) # [B,C,nq,nk,D]
            ap.append(atompair)

        ap = paddle.stack(ap)
        return ap
    

    def _operate_2_seqs(self, ql, qm, func):
        """ Compute atompair from two atom-level sequences feature thru give function.
        Args:
        - ql: Tensor: [B,M,D]
        - qm: Tensor: [B,M,D]
        - func: binary function take (ql, qm) to operate
        
        Returns:
        - ap: Tensor:[B,C,nq,nk,D]
        """
        B, M, D = ql.shape
        xid, yid = self._AttenIdx.get_xy_idx(M) # dense:[num_valid_ap]->C*nq*nk
        pair_mask = self._AttenIdx.get_atten_idx(M)['alpha_mask'] # [C,nq,nk]
        C, nq, nk = pair_mask.shape
        valid_pair_idx = pair_mask.flatten().nonzero() #[num_valid_ap]

        ql = ql.transpose([1, 0, 2]) # [M, B, D]
        qm = qm.transpose([1, 0, 2]) # [M, B, D]
        ap_dense = func(ql[xid].reshape([xid.size, B, D]),
                         qm[yid].reshape([yid.size, B, D])) # [num_valid_ap, B, D]
        
        ap_sparse = paddle.scatter_nd(index=valid_pair_idx, 
                                      updates=ap_dense, 
                                      shape=[C * nq * nk, B, D])
        return ap_sparse.reshape([C, nq, nk, B, D]).transpose([3, 0, 1, 2, 4]) # [B,C,nq,nk,D]
    

    def _add_2_seqs_sparse(self, ql, qm):
        """ Create sparse feature by adding two atom-level sequences feature.
        Args:
        - ql: Tensor:[B,M,D]
        - qm: Tensor:[B,M,D]
        
        Returns:
        - ap: Tensor:[B,M,M,D]
        """
        return ql.unsqueeze(2) + qm.unsqueeze(1)
    

    def _add_2_seqs_dense(self, ql, qm):
        """ Create dense feature by adding two atom-level sequences feature.
        Args:
        - ql: Tensor: [B,M,D]
        - qm: Tensor: [B,M,D]
        
        Returns:
        - ap: Tensor:[B,C,nq,nk,D]
        """
        return self._operate_2_seqs(ql, qm, lambda x, y: x + y)


    def _cmp_2_seqs_sparse(self, ql, qm):
        """ Create sparse feature by comparing two atom-level sequences feature.
        
        Args:
        - ql: Tensor:[B,M,D]
        - qm: Tensor:[B,M,D]
        
        Returns:
        - ap: Tensor:[B,M,M,D]
        """
        ap = ql.unsqueeze(1) == qm.unsqueeze(2)
        return ap.cast("float32").unsqueeze(-1)
    

    def _cmp_2_seqs_dense(self, ql, qm):
        """ Create dense feature by comparing two atom-level sequences feature.
        
        Args:
        - ql: Tensor: [B,M,D]
        - qm: Tensor: [B,M,D]
        
        Returns:
        - ap: Tensor:[B,C,nq,nk,D]
        """
        return self._operate_2_seqs(ql, qm, lambda x, y: (x == y).cast(x.dtype))



""" Per-token features to per-atom features """


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
    B, M = atom_token_uid.shape[:2]
    # f_atom = f_token.flatten(0,1)[atom_token_uid.flatten()]
    # f_atom = f_atom * atom_token_uid_mask.reshape([-1, 1])
    # return f_atom.reshape([B, M, -1])

    # for loop TODO: tmp work around
    f_atom = []
    atom_mask = atom_mask.cast(f_token.dtype)
    for i in range(B):
        idx = atom_token_uid[i]
        mask = atom_mask[i]
        atom = f_token[i][idx] * mask[..., None]
        f_atom.append(atom.unsqueeze(0))
    f_atom = paddle.concat(f_atom, axis=0)
    return f_atom


def pair_to_atompair(f_pair, atom_token_uid, atom_mask):
    """
    Convert per-token pair feature to atom-pair features.

    Args:
    f_pair:         [B,N,N,D]
    atom_token_uid: [B,M]
    atom_mask:      [B,M]

    Returns:
    f_pair:         [B,M,M,D]
    """
    B, N, _, D = f_pair.shape
    M = atom_token_uid.shape[1]
    diff_batch_size = atom_token_uid.shape[0] // B
    if diff_batch_size > 1:
        atom_token_uid = atom_token_uid[:B]
        atom_mask = atom_mask[:B]
    
    atom_token_uid = atom_token_uid.flatten(0, 1) # [B*M]
    f_pair = f_pair.flatten(0, 1)   # [B*N,N,D]
    f_pair = f_pair[atom_token_uid] * atom_mask.reshape([-1, 1, 1]) #[B*M,N,D]
    f_pair = f_pair.reshape([B, M, N, D]).transpose([0, 2, 1, 3]).flatten(0, 1) #[B*N,M,D]
    f_pair = f_pair[atom_token_uid] * atom_mask.reshape([-1, 1, 1]) #[B*M,M,D]
    
    return f_pair.reshape([B, M, M, D]).transpose([0, 2, 1, 3])


""" Per-atom features to per-token features """


def aggregate_atom_feat_to_token(f_atom, atom_token_uid, atom_mask, n_token):
    """ Aggregate per-atom features to per-token features.
    
    Args:
    f_atom: [B,M,D]
    atom_token_uid: [B,M]
    atom_mask: [B,M]
    n_token: the number of token

    Returns:
    f_token: [B,N,D]
    """
    B, _, D = f_atom.shape[:3]
    # for loop TODO: tmp work around
    f_atom_mean = []
    for i in range(B):
        idx = atom_token_uid[i].cast('int64')
        mask = atom_mask[i]

        with paddle.amp.auto_cast(enable=False):
            # FIXME: paddle.geometric.segment_sum is not support bfloat16 now (3.0.0dev20241231)
            fa = f_atom[i].astype('float32')
            m = mask.astype('float32')

            atom_sum = paddle.geometric.segment_sum(
                    data=fa * m[:, None], segment_ids=idx)
            atom_sum = atom_sum.astype('bfloat16')
            atom_count = paddle.geometric.segment_sum(
                    data=m, segment_ids=idx)
            atom_count = atom_count.astype('bfloat16')

            atom_mean = atom_sum / (atom_count[:, None] + 1e-8)
        atom_mean = paddle.concat([atom_mean, 
                paddle.zeros([n_token - len(atom_mean), D], dtype=f_atom.dtype)], 0)
        f_atom_mean.append(atom_mean)
    f_token = paddle.stack(f_atom_mean)
    return f_token
