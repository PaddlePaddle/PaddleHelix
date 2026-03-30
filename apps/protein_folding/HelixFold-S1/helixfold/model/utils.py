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

"""Utils."""

import os
import time
import numbers
import functools
import collections
import paddle
import numpy as np
from typing import Any, Mapping
from collections import defaultdict

from paddle.distributed.fleet.utils import recompute

from helixfold.common import protein
from helixfold.common import confidence
from helixfold.common import confidence_pd
from sklearn.metrics.pairwise import pairwise_distances
import logging

try:
    from paddle.framework import in_dynamic_mode
    from paddle.base.layer_helper import LayerHelper
except:
    in_dynamic_mode = None
    LayerHelper = None

def jax_params_to_paddle(params):
    """
    Rule 1: helixfold/helixfold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/* ==>
        '...template_pair_stack.0.*'
        '...template_pair_stack.1.*'
        ...

    Rule 2: helixfold/helixfold_iteration/evoformer/extra_msa_stack/* ==>
        'helixfold_iteration.evoformer.extra_msa_stack.0.*',
        'helixfold_iteration.evoformer.extra_msa_stack.1.*',
        ...

    Rule 3: helixfold/helixfold_iteration/evoformer/evoformer_iteration/* ==>
        'helixfold.helixfold_iteration.evoformer.evoformer_iteration.0.*',
        'helixfold.helixfold_iteration.evoformer.evoformer_iteration.1.*',
        ...

    Rule 4: */__layer_stack_no_state/* ==> '*.*'

    Rule 5: *//weights ==> '*.weight'

    Rule 6: *//bias ==> '*.bias'

    Rule 7: *//scale ==> '*.weight'

    Rule 8: *//offset ==> '*.bias'

    From Rule 9, extra rules are for multimer

    Rule 9: helixfold/helixfold_iteration/evoformer/template_embedding/single_template_embedding/template_embedding_iteration/* ==>
        '...template_embedding_iteration.0.*'
        '...template_embedding_iteration.1.*'
        ...

    Rule 10: helixfold/helixfold_iteration/evoformer/~_relative_encoding/* ==>
        'helixfold.helixfold_iteration.evoformer.*'

    Rule 11: .../point_projection//* ==> '...*'

    Rule 12: .../fold_iteration/quat_rigid/rigid//bias ==>
        '...fold_iteration.affine_update.bias'
    """
    rule_1_prefix = 'helixfold/helixfold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/'
    rule_2_prefix = 'helixfold/helixfold_iteration/evoformer/extra_msa_stack/'
    rule_3_prefix = 'helixfold/helixfold_iteration/evoformer/evoformer_iteration/'
    rule_4_prefix = '__layer_stack_no_state/'

    rule_9_prefix = 'helixfold/helixfold_iteration/evoformer/template_embedding/single_template_embedding/template_embedding_iteration/'

    rule_10_infix = '~_relative_encoding'

    rule_11_infix = 'point_projection'

    rule_12_infix = ('quat_rigid.rigid', 'affine_update')

    pd_params = dict()

    def _parse_stack_or_iteration(rule_prefix, k):
        n = params[k].shape[0]
        suffix = k[len(rule_prefix):]

        # rule 4
        if suffix.startswith(rule_4_prefix):
            suffix = suffix[len(rule_4_prefix):]

        # rule 5
        suffix = suffix.replace('//weights', '.weight')
        # rule 6
        suffix = suffix.replace('//bias', '.bias')
        # rule 7
        suffix = suffix.replace('//scale', '.weight')
        # rule 8
        suffix = suffix.replace('//offset', '.bias')

        suffix = suffix.replace('//', '.')
        suffix = suffix.replace('/', '.')

        prefix = rule_prefix.replace('/', '.')
        for i in range(n):
            k_ = f'{prefix}{i}.{suffix}'
            pd_params[k_] = np.copy(params[k][i])

    def _auto_pad_1_weight(k_, k, params):
        # [N] => [1, N]
        for_template = 'template_pair_embedding_' in k_
        if for_template and len(params[k].shape) == 1:
            w = np.copy(params[k])
            w = np.reshape(w, [1, w.shape[0]])
        else:
            w = np.copy(params[k])

        return w

    for k in params.keys():
        if k.startswith(rule_1_prefix):
            _parse_stack_or_iteration(rule_1_prefix, k)

        elif k.startswith(rule_2_prefix):
            _parse_stack_or_iteration(rule_2_prefix, k)

        elif k.startswith(rule_3_prefix):
            _parse_stack_or_iteration(rule_3_prefix, k)

        elif k.startswith(rule_9_prefix):
            _parse_stack_or_iteration(rule_9_prefix, k)

        else:
            k_ = k.replace('//weights', '.weight')
            k_ = k_.replace('//scale', '.weight')
            k_ = k_.replace('//offset', '.bias')
            k_ = k_.replace('//', '.')
            k_ = k_.replace('/', '.')

            if rule_10_infix in k_:
                k_ = k_.replace(f'.{rule_10_infix}.', '.')

            if rule_11_infix in k_:
                k_ = k_.replace(f'.{rule_11_infix}.', '.')

            if rule_12_infix[0] in k_:
                k_ = k_.replace(f'.{rule_12_infix[0]}.',
                                f'.{rule_12_infix[1]}.')

            if k_.endswith('.weight'):
                pd_params[k_] = _auto_pad_1_weight(k_, k, params)
            else:
                pd_params[k_] = np.copy(params[k])

    return pd_params


def pd_params_merge_qkvw(pd_params):
    # FIXME: not work for multimer mode
    qkv_dicts = defaultdict(dict)
    for key in pd_params:
        if 'msa_column_global_attention' not in key and 'attention' in key and (
                'query_w' in key or 'key_w' in key
                or 'value_w' in key):
            prefix = key[:key.rfind('.')]
            if 'extra_msa_stack' in key:
                qkv_dicts[prefix][key] = pd_params[key]
                # print(key)
            elif 'evoformer_iteration' in key:
                qkv_dicts[prefix][key] = pd_params[key]
                # print(key)
            elif 'template_pair_stack' in key:
                qkv_dicts[prefix][key] = pd_params[key]
                # print(key)
            elif 'template_embedding_iteration' in key:
                qkv_dicts[prefix][key] = pd_params[key]
                # print('========>', key)

    for prefix in qkv_dicts:
        query_w = qkv_dicts[prefix][prefix + '.query_w']
        key_w = qkv_dicts[prefix][prefix + '.key_w']
        value_w = qkv_dicts[prefix][prefix + '.value_w']
        if query_w.shape[0] == key_w.shape[0] and key_w.shape[
                0] == value_w.shape[0]:
            # 1. merge to [3, num_head, key_dim, q_dim]
            qkv_w = np.stack([query_w, key_w, value_w],
                                axis=0).transpose((0, 2, 3, 1))

            # 2. remove seperated param
            del pd_params[prefix + '.query_w']
            del pd_params[prefix + '.key_w']
            del pd_params[prefix + '.value_w']

            # 3. add merged param to pd_params
            pd_params[prefix + '.qkv_w'] = qkv_w


def slice_batch(batch, i, keys='all'):
    if keys == 'all':
        b = {k: v[i] for k, v in batch.items()}
    else:
        b = dict()
        for k, v in batch.items():
            b[k] = v[i] if k in keys else v

    return b

def add_batch_dim(batch):
    b = {k: v[None,] for k, v in batch.items()}
    return b

def map_to_tensor(batch, add_batch=False):
    if add_batch:
        batch = add_batch_dim(batch)

    b = {k: paddle.to_tensor(v) for k, v in batch.items()}
    return b


def mask_mean(mask, value, axis=None, drop_mask_channel=False, eps=1e-10):
    if drop_mask_channel:
        mask = mask[:, 0]

    mask_shape = mask.shape
    value_shape = value.shape
    assert len(mask_shape) == len(value_shape)

    if isinstance(axis, numbers.Integral):
        axis = [axis]
    elif axis is None:
        axis = list(range(len(mask_shape)))

    assert isinstance(axis, collections.abc.Iterable), \
        'axis needs to be either an iterable, integer or "None"'

    broadcast_factor = 1.
    for axis_ in axis:
        value_size = value_shape[axis_]
        mask_size = mask_shape[axis_]
        if mask_size == 1:
            broadcast_factor *= value_size
        else:
            assert mask_size == value_size

    return (paddle.sum(mask * value, axis=axis) /
            (paddle.sum(mask, axis=axis) * broadcast_factor + eps))


def batched_gather(params, indices, axis=0, batch_dims=0):
    # Implement gather with batching, like tensorflow:
    # https://www.tensorflow.org/api_docs/python/tf/gather#batching
    # print(params.shape, indices.shape, axis)
    p, i = params, indices
    rank = len(p.shape)
    axis = (rank + axis) % rank
    # The stride of axis
    stride = p.shape[batch_dims + axis]

    if batch_dims == 0 and len(i.shape) == 1:
        return paddle.gather(p, i, axis=axis)

    elif batch_dims == 0:
        flat_i = i.reshape([-1])
        gathered = paddle.gather(p, flat_i, axis=axis)
        shape = p.shape[:axis] + i.shape
        if axis < rank - 1:
            shape += params.shape[axis + 1:]
        return gathered.reshape(shape)

    b = batch_dims
    a = axis
    assert p.shape[:b] == i.shape[:b]
    bn = np.prod(p.shape[:b])

    # Shift batch dimensions right to bundle with axis
    if a > 0:
        perm = list(range(rank))
        perm = perm[b:(b + a)] + perm[:b] + perm[(b + a):]
        p = p.transpose(perm)

    # Merge params' batch+axis
    p = p.reshape(p.shape[:a] + [-1] + p.shape[(b + a + 1):])

    # indices = [Batch..., Index...]
    # Expand the index values across batch elements
    strides = paddle.arange(bn, dtype="int64").unsqueeze(-1) * stride
    i = i.reshape([bn, -1])
    flat_i = paddle.flatten(i + strides)

    # Do gather
    gathered = paddle.gather(p, flat_i, axis=axis)

    # Unbundle batch and index dimensions
    unbundled_shape = p.shape[:a] + indices.shape + p.shape[a + 1:]
    gathered = gathered.reshape(unbundled_shape)

    # Shift batch dimensions back to the left
    if a > 0:
        perm = list(range(len(unbundled_shape)))
        perm = perm[a:(a + b)] + perm[:a] + perm[(a + b):]
        gathered = gathered.transpose(perm)

    return gathered


def subbatch(f, arg_idx, dim, bs, out_idx, same_arg_idx={}):
    """ Converts a function to one that applies to subbatch of an input
    dimension.

    Args:
        f(Callable): original function.
        arg_idx([int]): indices of the inputs to be subbatched.
        dim([int]): index of the dimension to be subbatched.
        bs(int): subbatch size.
        out_idx(int): index of the output dimension that needs stacking
        same_arg_idx(dict), optional: index of same arg mapping. e.g {1: 0} means arg[1] == arg[0],
                            we assign _args[1] = _args[0] avoiding slice repeatly.

    Returns:
        converted function.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):

        assert len(arg_idx) == len(dim), f'Number of batching args and number of batching dims should match.'

        inps = [args[i] for i in arg_idx]
        dim_width = [inp.shape[d] for inp, d in zip(inps, dim)]
        assert len(set(dim_width)) == 1, f'Batch sizes should be kept equal.'

        inp_dim = {inp: d for inp, d in zip(inps, dim)}

        dim_width = dim_width[0]
        if dim_width < bs:
            return f(*args, **kwargs)

        outs = []
        for slice_at in np.arange(0, dim_width, bs):
            _args = []
            for i, inp in enumerate(args):
                if i in same_arg_idx:
                    assert i > same_arg_idx[i], f"expect i > same_arg_idx[i], but got i: {i} and same_arg_idx[i]: {same_arg_idx[i]}"
                    _args.append(_args[same_arg_idx[i]])
                elif i in arg_idx:
                    inp = inp.slice([inp_dim[inp]], [slice_at], [slice_at + bs])
                    _args.append(inp)
                else:
                    _args.append(inp)
            outs.append(f(*_args, **kwargs))

        return paddle.concat(outs, out_idx)

    return wrapper


def get_confidence_metrics(
        prediction_result: Mapping[str, Any], 
        multimer_mode: bool) -> Mapping[str, Any]:
    """Post processes prediction_result to get confidence metrics."""
    confidence_metrics = {}
    confidence_metrics['plddt'] = confidence.compute_plddt(
        prediction_result['predicted_lddt']['logits'])
    if 'predicted_aligned_error' in prediction_result:
        confidence_metrics.update(confidence.compute_predicted_aligned_error(
            logits=prediction_result['predicted_aligned_error']['logits'],
            breaks=prediction_result['predicted_aligned_error']['breaks']))
        confidence_metrics['ptm'] = confidence.predicted_tm_score(
            logits=prediction_result['predicted_aligned_error']['logits'],
            breaks=prediction_result['predicted_aligned_error']['breaks'],
            asym_id=None)
    if multimer_mode:
      # Compute the ipTM only for the multimer model.
      confidence_metrics['iptm'] = confidence.predicted_tm_score(
          logits=prediction_result['predicted_aligned_error']['logits'],
          breaks=prediction_result['predicted_aligned_error']['breaks'],
          asym_id=prediction_result['predicted_aligned_error']['asym_id'],
          interface=True)
      confidence_metrics['ranking_confidence'] = (
          0.8 * confidence_metrics['iptm'] + 0.2 * confidence_metrics['ptm'])

    if not multimer_mode:
        # Monomer models use mean pLDDT for model ranking.
        confidence_metrics['ranking_confidence'] = np.mean(
            confidence_metrics['plddt'])

    return confidence_metrics

def get_all_atom_confidence_metrics_pd(
        prediction_result: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = {}
    metrics['atom_plddts'] = confidence_pd.compute_plddt(
            prediction_result['logits_plddt'])
    metrics['mean_plddt'] = metrics['atom_plddts'].mean()
    metrics['chain_plddt_asym_id'], metrics['chain_plddt'] = confidence_pd.compute_chain_plddt(
            metrics['atom_plddts'], prediction_result['perm_asym_id'])

    metrics['chain_inter_pde_asym_id'], metrics['chain_inter_pde'] = \
            confidence_pd.compute_chain_inter_pde(
                logits=prediction_result['logits_pde'],
                breaks=prediction_result['breaks_pde'],
                asym_id=prediction_result['asym_id'])

    metrics['pae'] = confidence_pd.compute_predicted_aligned_error(
            logits=prediction_result['logits_pae'],
            breaks=prediction_result['breaks_pae'])['predicted_aligned_error']
    metrics['ptm'] = confidence_pd.predicted_tm_score(
            logits=prediction_result['logits_pae'],
            breaks=prediction_result['breaks_pae'],
            residue_weights=prediction_result['frame_mask'],
            asym_id=None)['score']
    metrics['iptm'] = confidence_pd.predicted_tm_score(
            logits=prediction_result['logits_pae'],
            breaks=prediction_result['breaks_pae'],
            residue_weights=prediction_result['frame_mask'],
            asym_id=prediction_result['asym_id'],
            interface=True)['score']
    metrics['token_ptm'] = confidence_pd.predicted_tm_score(
            logits=prediction_result['logits_pae'],
            breaks=prediction_result['breaks_pae'],
            residue_weights=prediction_result['frame_mask'],
            asym_id=None)['token_score']
    metrics['token_iptm'] = confidence_pd.predicted_tm_score(
            logits=prediction_result['logits_pae'],
            breaks=prediction_result['breaks_pae'],
            residue_weights=prediction_result['frame_mask'],
            asym_id=prediction_result['asym_id'],
            interface=True)['token_score']
    metrics['has_clash'] = confidence_pd.get_has_clash(
            prediction_result['final_atom_positions'],
            prediction_result['final_atom_mask'],
            prediction_result['perm_asym_id'],
            prediction_result['is_polymer_chain'])
    metrics['ranking_confidence'] = (
            0.8 * metrics['iptm'] + 0.2 * metrics['ptm'] 
            - 1.0 * metrics['has_clash'])
    ## get chain_pair_asym_ids, chain_pair_iptm and chain_pair_mask
    metrics.update(confidence_pd.predicted_chain_pair_iptm(
        logits=prediction_result['logits_pae'],
        breaks=prediction_result['breaks_pae'],
        residue_weights=prediction_result['frame_mask'],
        asym_id=prediction_result['asym_id'],
    ))
    
    ## get chain/token level_has_clash
    metrics.update(confidence_pd.get_has_clash_token_level(
        atom_pos=prediction_result['final_atom_positions'],
        atom_mask=prediction_result['final_atom_mask'],
        asym_id=prediction_result['perm_asym_id'], 
        token2atom=prediction_result['token2atom_idx'], 
        is_polymer_chain=prediction_result['is_polymer_chain'])
    )

    return metrics


def get_all_atom_confidence_metrics(
        prediction_result: Mapping[str, Any]) -> Mapping[str, Any]:
    """get_all_atom_confidence_metrics."""
    metrics = {}
    metrics['atom_plddts'] = confidence.compute_plddt(
            prediction_result['logits_plddt'])
    metrics['mean_plddt'] = metrics['atom_plddts'].mean()
    metrics['chain_plddt_asym_id'], metrics['chain_plddt'] = confidence.compute_chain_plddt(
            metrics['atom_plddts'], prediction_result['perm_asym_id'])

    metrics['chain_inter_pde_asym_id'], metrics['chain_inter_pde'] = \
            confidence.compute_chain_inter_pde(
                logits=prediction_result['logits_pde'],
                breaks=prediction_result['breaks_pde'],
                asym_id=prediction_result['asym_id'])

    metrics['pae'] = confidence.compute_predicted_aligned_error(
            logits=prediction_result['logits_pae'],
            breaks=prediction_result['breaks_pae'])['predicted_aligned_error']
    metrics['ptm'] = confidence.predicted_tm_score(
            logits=prediction_result['logits_pae'],
            breaks=prediction_result['breaks_pae'],
            residue_weights=prediction_result['frame_mask'],
            asym_id=None)
    metrics['iptm'] = confidence.predicted_tm_score(
            logits=prediction_result['logits_pae'],
            breaks=prediction_result['breaks_pae'],
            residue_weights=prediction_result['frame_mask'],
            asym_id=prediction_result['asym_id'],
            interface=True)

    if 'is_ligand' in prediction_result and prediction_result['is_ligand'].sum()>0:
        is_ligand = prediction_result['is_ligand']
        # masking the second dim using is_ligand
        pairwise_residue_weights = is_ligand[None, :] * np.ones((is_ligand.shape[0], is_ligand.shape[0]))
        metrics['ligand_iptm'] = confidence.predicted_tm_score(
            logits=prediction_result['logits_pae'],
            breaks=prediction_result['breaks_pae'],
            residue_weights=prediction_result['frame_mask'],
            asym_id=prediction_result['asym_id'],
            pairwise_residue_weights=pairwise_residue_weights,
            interface=True)
        metrics['ligand_iptm_mean'] = confidence.predicted_tm_score(
            logits=prediction_result['logits_pae'],
            breaks=prediction_result['breaks_pae'],
            residue_weights=prediction_result['frame_mask'],
            asym_id=prediction_result['asym_id'],
            pairwise_residue_weights=pairwise_residue_weights,
            interface=True,
            aggregation_type='mean')

    if 'is_ligand_aa' in prediction_result and prediction_result['is_ligand_aa'].sum()>0:
        is_ligand_aa = prediction_result['is_ligand_aa'].astype('bool')
        metrics['ligand_mean_plddt'] = metrics['atom_plddts'][is_ligand_aa].sum() / is_ligand_aa.sum()

    metrics['has_clash'] = get_has_clash(
            prediction_result['final_atom_positions'],
            prediction_result['final_atom_mask'],
            prediction_result['perm_asym_id'],
            prediction_result['is_polymer_chain'])
    metrics['ranking_confidence'] = (
            0.8 * metrics['iptm'] + 0.2 * metrics['ptm'] 
            - 1.0 * metrics['has_clash'])
    ## get chain_pair_asym_ids, chain_pair_iptm and chain_pair_mask
    metrics.update(confidence.predicted_chain_pair_iptm(
        logits=prediction_result['logits_pae'],
        breaks=prediction_result['breaks_pae'],
        residue_weights=prediction_result['frame_mask'],
        asym_id=prediction_result['asym_id'],
    ))

    ## get actifpTM with contact map probability
    metrics['actifpTM'] = confidence.get_actifPTM(
        result=prediction_result,
        use_contact_map_prob=True,
        contact_dist=8.0
    )['actifpTM']
    ## get actifpTM with interface mask
    metrics['actifpTM_interfaceMask'] = confidence.get_actifPTM(
        result=prediction_result,
        use_contact_map_prob=False,
        interface_dist=5.0,
    )['actifpTM']
    return metrics


def get_has_clash(atom_pos, atom_mask, asym_id, is_polymer_chain):
    """
    A structure is marked as having a clash (has_clash) if for any two
    polymer chains A,B in the prediction clashes(A,B) > 100 or 
    clashes(A,B) / min(NA,NB) > 0.5 where NA is the number of atoms in 
    chain A.
    Args:
        atom_pos: [N_atom, 3]
        atom_mask: [N_atom]
        asym_id: [N_atom]
        is_polymer_chain: [N_atom]
    """
    flag = np.logical_and(atom_mask == 1, is_polymer_chain == 1)
    atom_pos = atom_pos[flag]
    asym_id = asym_id[flag]
    uniq_asym_ids = np.unique(asym_id)
    n = len(uniq_asym_ids)
    if n == 1:
        return 0
    for idx, aid1 in enumerate(uniq_asym_ids[:-1]):
        for aid2 in uniq_asym_ids[idx + 1:]:
            pos1 = atom_pos[asym_id == aid1]
            pos2 = atom_pos[asym_id == aid2]
            dist = np.sqrt(np.sum((pos1[None] - pos2[:, None]) ** 2, -1))
            n_clash = np.sum(dist < 1.1).astype('float32')
            min_len = min(len(pos1), len(pos2))
            if n_clash > 100 or n_clash / min_len > 0.5:
                #TODO: more informatic way to display clashes
                print(f"[WARNING]: clashes: {n_clash} out of {min_len} atoms for asym id {aid1} and {aid2}")
                return 1
    return 0


def generate_unrelaxed_pdb(aatype, residue_index, asym_id, model_output, pdb_path,
                           b_factors=None):
    fold_output = model_output['structure_module']
    if b_factors is None:
        b_factors = np.zeros_like(fold_output['final_atom_mask'])

    if asym_id is None:
        chain_index = np.zeros(aatype.shape)
    else:
        chain_index = asym_id - 1

    # NOTE: for single protein, chain_index is always 'A' (idx:0)
    prot = protein.Protein(
        aatype=aatype,
        atom_positions=fold_output['final_atom_positions'],
        atom_mask=fold_output['final_atom_mask'],
        residue_index=residue_index + 1,
        chain_index=chain_index,
        b_factors=b_factors)

    with open(pdb_path, 'w') as f:
        f.write(protein.to_pdb(prot))

    return prot


def set_tensor_constant(tensor, constant):
    tensor.set_value(paddle.full_like(tensor, constant))


def init_gate_linear(linear):
    set_tensor_constant(linear.weight, 0)
    set_tensor_constant(linear.bias, 1)


def init_final_linear(linear):
    set_tensor_constant(linear.weight, 0)


def recompute_wrapper(func, *args, is_recompute=True):
    """Function wrapper for recompute"""
    if is_recompute:
        return recompute(func, *args)
    else:
        return func(*args)


def tree_map(f, d):
    new_d = {}
    for k in d:
        if type(d[k]) is dict:
            new_d[k] = tree_map(f, d[k])
        else:
            new_d[k] = f(d[k])
    return new_d


def tree_flatten(d):
    new_d = {}
    for k in d:
        if type(d[k]) is dict:
            cur_d = tree_flatten(d[k])
            for sub_k, sub_v in cur_d.items():
                new_d[f'{k}.{sub_k}'] = sub_v
        else:
            new_d[k] = d[k]
    return new_d


tik_tok_time_list = []
def tik():
    tik_tok_time_list.append(time.time())


def tok(message=''):
    assert len(tik_tok_time_list) >= 1
    print(f"{message} {time.time() - tik_tok_time_list[-1]} s")
    del tik_tok_time_list[-1]


def fused_act_bias_wrapper(
    x,
    bias=None,
    dequant_scales=None,
    shift=None,
    smooth=None,
    act_method="gelu",
    compute_dtype="default",
    quant_scale=-1,
    quant_round_type=0,
    quant_max_bound=0,
    quant_min_bound=0,
):
    if in_dynamic_mode():
        return paddle._C_ops.fused_bias_act(
            x,
            bias,
            dequant_scales,
            shift,
            smooth,
            act_method,
            compute_dtype,
            quant_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
        )
    helper = LayerHelper("fused_bias_act")
    if x.dtype == "int32":
        if compute_dtype == "bf16":
            dtype = "uint16"
        elif compute_dtype == "fp16":
            dtype = "float16"
        elif compute_dtype == "fp32":
            dtype = "float32"
        out = helper.create_variable_for_type_inference(dtype=dtype)
    else:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

    inputs = {}
    inputs["x"] = x
    if bias is not None:
        inputs["bias"] = bias
    if dequant_scales is not None:
        inputs["bias"] = dequant_scales

    if shift is not None:
        inputs["shift"] = shift

    if smooth is not None:
        inputs["smooth"] = smooth

    attrs = {
        "act_method": act_method,
        "compute_dtype": compute_dtype,
        "quant_scale": quant_scale,
        "quant_round_type": quant_round_type,
        "quant_max_bound": quant_max_bound,
        "quant_min_bound": quant_min_bound,
    }

    helper.append_op(
        type="fused_bias_act",
        inputs=inputs,
        outputs={"out": out},
        attrs=attrs,
    )
    return out


def get_batch_constrain_metrics(constrain_info, constrain_pairs,
                                pd_atom_pos, gt_atom_pos, atom_mask, 
                                atom_to_token_mapping,
                                logging_marker="[batch_constrain_metrics]"):
    """
    constrain_info: [N, N, 20]
    constrain_pairs: [n_pairs]
    pd_atom_pos: # [M, 3]
    gt_atom_pos: [M, 3]
    atom_mask: [M]
    """
    pair_recall = []
    pair_gt_dist = []
    atom_mask = atom_mask.astype('bool')
    pd_atom_pos = pd_atom_pos[atom_mask]  # [M, 3]
    gt_atom_pos = gt_atom_pos[atom_mask] if not gt_atom_pos is None else None # [M, 3]
    atom_to_token_mapping = atom_to_token_mapping[atom_mask] # [M]
    for cons_id_pair in constrain_pairs: # constrain_id_pairs:
        if not cons_id_pair.mean() == -1:  # skip paddings
            logging.info(f'{logging_marker} cons_id_pair: {cons_id_pair.numpy()}')
            # try:
            if True:
                src_pd_token_pos = pd_atom_pos[atom_to_token_mapping == \
                                            cons_id_pair[0]].astype('float32') # [src_atom, 3]
                tgt_pd_token_pos = pd_atom_pos[atom_to_token_mapping == \
                                               cons_id_pair[1]].astype('float32') # [tgt_atom, 3]
                src_gt_token_pos = gt_atom_pos[atom_to_token_mapping == cons_id_pair[0]].astype('float32')
                tgt_gt_token_pos = gt_atom_pos[atom_to_token_mapping == cons_id_pair[1]].astype('float32')
                logging.info(f'{logging_marker} src_gt_token_pos: {src_gt_token_pos.shape}" \
                             + f" tgt_gt_token_pos: {tgt_gt_token_pos.shape}')
                targeted_dist_vec = constrain_info[int(cons_id_pair[0])][int(cons_id_pair[1])].astype('float32')
                logging.info(f"targeted_dist_vec: {targeted_dist_vec}")

                pd_dist = pairwise_distances(src_pd_token_pos, tgt_pd_token_pos) # [src_atom, tgt_atom]
                pd_min_dist = np.min(pd_dist)

                gt_dist = pairwise_distances(src_gt_token_pos, tgt_gt_token_pos) # [src_atom, tgt_atom]
                gt_min_dist = np.min(gt_dist)
                logging.info(f'{logging_marker} gt_min_dist: {np.min(gt_min_dist)}')
                logging.info(f'{logging_marker} pd_min_dist: {pd_min_dist}')

                targeted_dist = (20 - targeted_dist_vec.astype('float32').sum() + 1)
                logging.info(f'{logging_marker} targeted_dist: {targeted_dist.astype(paddle.int32).numpy()}')
                # target_dist.append(min(pd_min_dist - targeted_dist, 0))
                pair_recall.append(float(pd_min_dist <= targeted_dist))

                pair_gt_dist.append(np.abs(float(pd_dist.min() - gt_dist.min())))
                logging.info(f'{logging_marker} pair_gt_dist: {pair_gt_dist}')
                logging.info(f'{logging_marker} pair_recall: {pair_recall}')
            # except Exception as e:
            else:
                logging.info(f'{logging_marker} {e}')
    return pair_recall, pair_gt_dist
