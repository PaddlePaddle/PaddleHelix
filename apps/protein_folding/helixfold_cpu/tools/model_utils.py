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

import numbers
import functools
import collections
import paddle
import numpy as np
from typing import Any, Mapping

import protein
import confidence


def jax_params_to_paddle(params):
    """
    Rule 1: alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/* ==>
        '...template_pair_stack.0.*'
        '...template_pair_stack.1.*'
        ...

    Rule 2: alphafold/alphafold_iteration/evoformer/extra_msa_stack/* ==>
        'alphafold_iteration.evoformer.extra_msa_stack.0.*',
        'alphafold_iteration.evoformer.extra_msa_stack.1.*',
        ...

    Rule 3: alphafold/alphafold_iteration/evoformer/evoformer_iteration/* ==>
        'alphafold.alphafold_iteration.evoformer.evoformer_iteration.0.*',
        'alphafold.alphafold_iteration.evoformer.evoformer_iteration.1.*',
        ...

    Rule 4: */__layer_stack_no_state/* ==> '*.*'

    Rule 5: *//weights ==> '*.weight'

    Rule 6: *//bias ==> '*.bias'

    Rule 7: *//scale ==> '*.weight'

    Rule 8: *//offset ==> '*.bias'
    """
    rule_1_prefix = 'alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/'
    rule_2_prefix = 'alphafold/alphafold_iteration/evoformer/extra_msa_stack/'
    rule_3_prefix = 'alphafold/alphafold_iteration/evoformer/evoformer_iteration/'
    rule_4_prefix = '__layer_stack_no_state/'

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

    for k in params.keys():
        if k.startswith(rule_1_prefix):
            _parse_stack_or_iteration(rule_1_prefix, k)

        elif k.startswith(rule_2_prefix):
            _parse_stack_or_iteration(rule_2_prefix, k)

        elif k.startswith(rule_3_prefix):
            _parse_stack_or_iteration(rule_3_prefix, k)

        else:
            k_ = k.replace('//weights', '.weight')
            k_ = k_.replace('//scale', '.weight')
            k_ = k_.replace('//offset', '.bias')
            k_ = k_.replace('//', '.')
            k_ = k_.replace('/', '.')
            pd_params[k_] = np.copy(params[k])

    return pd_params


def slice_batch(batch, i):
    b = {k: v[i] for k, v in batch.items()}
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

    assert isinstance(axis, collections.Iterable), \
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
    strides = paddle.arange(bn).unsqueeze(-1) * stride
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


def subbatch(f, arg_idx, dim, bs, out_idx):
    """ Converts a function to one that applies to subbatch of an input
    dimension.

    Args:
        f(Callable): original function.
        arg_idx([int]): indices of the inputs to be subbatched.
        dim([int]): index of the dimension to be subbatched.
        bs(int): subbatch size.
        out_idx(int): index of the output dimension that needs stacking

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
                if i in arg_idx:
                    inp = inp.slice([inp_dim[inp]], [slice_at], [slice_at + bs])
                _args.append(inp)
            outs.append(f(*_args, **kwargs))

        return paddle.concat(outs, out_idx)

    return wrapper


def get_confidence_metrics(
        prediction_result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Post processes prediction_result to get confidence metrics."""

    confidence_metrics = {}
    confidence_metrics['plddt'] = confidence.compute_plddt(
        prediction_result['predicted_lddt']['logits'])

    if 'predicted_aligned_error' in prediction_result:
        confidence_metrics.update(confidence.compute_predicted_aligned_error(
            prediction_result['predicted_aligned_error']['logits'],
            prediction_result['predicted_aligned_error']['breaks']))

        confidence_metrics['ptm'] = confidence.predicted_tm_score(
            prediction_result['predicted_aligned_error']['logits'],
            prediction_result['predicted_aligned_error']['breaks'])

    return confidence_metrics


def generate_unrelaxed_pdb(aatype, residue_index, model_output, pdb_path,
                           b_factors=None):
    fold_output = model_output['structure_module']
    if b_factors is None:
        b_factors = np.zeros_like(fold_output['final_atom_mask'])

    # NOTE: for single protein, chain_index is always 'A' (idx:0)
    prot = protein.Protein(
        aatype=aatype,
        atom_positions=fold_output['final_atom_positions'],
        atom_mask=fold_output['final_atom_mask'],
        residue_index=residue_index + 1,
        chain_index=np.zeros(aatype.shape),
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
