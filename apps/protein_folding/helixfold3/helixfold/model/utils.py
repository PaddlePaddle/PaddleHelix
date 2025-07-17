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

import paddle
import functools
import numpy as np
from typing import Any, Mapping

from paddle.distributed.fleet.utils import recompute
from helixfold.common import confidence


def add_batch_dim(batch):
    b = {k: v[None,] for k, v in batch.items()}
    return b


def map_to_tensor(batch, add_batch=False):
    if add_batch:
        batch = add_batch_dim(batch)

    b = {k: paddle.to_tensor(v) for k, v in batch.items()}
    return b


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
                print(f"[WARNING]: clashes: {n_clash} out of {min_len} atoms for asym id {aid1} and {aid2}")
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

