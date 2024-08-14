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
from collections.abc import Mapping, Sequence
import numpy as np
import paddle
from paddle.framework import core


def get_custom_amp_list():
    """tbd."""

    black_list = {"reduce_sum"}
    white_list = {
        "concat",
        "dropout_nd",
        "einsum",
        "elementwise_add",
        "elementwise_div",
        "elementwise_mul",
        "elementwise_sub",
        "fill_any_like",
        "fill_constant",
        "fused_gate_attention",
        "fused_gemm_epilogue",
        "gather",
        "gaussian_random",
        "layer_norm",
        "log_softmax",
        "matmul_v2",
        "p_norm",
        "py_layer",
        "relu",
        "scale",
        "sigmoid",
        "slice",
        "softmax",
        "softplus",
        "split",
        "split_with_num",
        "sqrt",
        "square",
        "stack",
        "sum",
        "transpose2",
        "unsqueeze2",
        "unstack",
        "where"
    }
    return black_list, white_list


def all_atom_collate_fn(batch):
    """
    token-level features are of the same length,
    but atom-level features are of different length.

    Args:
        batch(list of sample data): batch should be a list of sample data.

    Returns:
        Batched data: batched each number, numpy array and paddle.Tensor
                      in input data.
    """
    def _pad_to_same_0_dim(array_list):
        if array_list[0].ndim == 0:
            return array_list
        max_0dim = np.max([len(i) for i in array_list])
        res = []
        for i, array in enumerate(array_list):
            pad_shape = [max_0dim - len(array)] + list(array.shape[1:])
            res.append(np.concatenate([array, np.zeros(pad_shape, dtype=array.dtype)], 0))
        return res

    sample = batch[0]
    if isinstance(sample, np.ndarray):
        batch = _pad_to_same_0_dim(batch)
        batch = np.stack(batch, axis=0)
        return batch
    elif isinstance(sample, (paddle.Tensor, core.eager.Tensor)):
        return paddle.stack(batch, axis=0)
    elif isinstance(sample, numbers.Number):
        batch = np.array(batch)
        return batch
    elif isinstance(sample, (str, bytes)):
        return batch
    elif isinstance(sample, Mapping):
        return {
            key: all_atom_collate_fn([d[key] for d in batch]) for key in sample
        }
    elif isinstance(sample, Sequence):
        sample_fields_num = len(sample)
        if not all(len(sample) == sample_fields_num for sample in iter(batch)):
            raise RuntimeError(
                "fileds number not same among samples in a batch"
            )
        return [all_atom_collate_fn(fields) for fields in zip(*batch)]

    raise TypeError(
        "batch data con only contains: tensor, numpy.ndarray, "
        "dict, list, number, but got {}".format(type(sample))
    )
