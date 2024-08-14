#   Copyright (c) 2024 PaddleHelix Authors. All Rights Reserved.
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

"""Utils."""

import numbers
import functools
import collections
import paddle
import numpy as np
from typing import Any, Mapping
from collections import defaultdict

from paddle.distributed.fleet.utils import recompute

from helixfold.common import confidence


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



def add_batch_dim(batch):
    b = {k: v[None,] for k, v in batch.items()}
    return b

def map_to_tensor(batch, add_batch=False):
    if add_batch:
        batch = add_batch_dim(batch)

    b = {k: paddle.to_tensor(v) for k, v in batch.items()}
    return b

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


def get_all_atom_confidence_metrics(
        prediction_result: Mapping[str, Any]) -> Mapping[str, Any]:
    """get_all_atom_confidence_metrics."""
    metrics = {}
    metrics['atom_plddts'] = confidence.compute_plddt(
            prediction_result['logits_plddt'])
    metrics['mean_plddt'] = metrics['atom_plddts'].mean()
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
    metrics['has_clash'] = get_has_clash(
            prediction_result['final_atom_positions'],
            prediction_result['final_atom_mask'],
            prediction_result['perm_asym_id'],
            prediction_result['is_polymer_chain'])
    metrics['ranking_confidence'] = (
            0.8 * metrics['iptm'] + 0.2 * metrics['ptm'] 
            - 1.0 * metrics['has_clash'])
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
    for aid1 in uniq_asym_ids[:-1]:
        for aid2 in uniq_asym_ids[1:]:
            pos1 = atom_pos[asym_id == aid1]
            pos2 = atom_pos[asym_id == aid2]
            dist = np.sqrt(np.sum((pos1[None] - pos2[:, None]) ** 2, -1))
            n_clash = np.sum(dist < 1.1).astype('float32')
            if n_clash > 100 or n_clash / min(len(pos1), len(pos2)) > 0.5:
                return 1
    return 0


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