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
"""
fuse parameters helper functions
"""

from collections import OrderedDict, defaultdict

import copy
import numpy as np
import paddle
from paddle.fluid import core, unique_name
from paddle.fluid.framework import ParamBase
from paddle.distributed.fleet.utils.internal_storage import ParamStorage, GradStorage

alignment = {"gpu": 256, }
align = {
    paddle.float16: 2,
    paddle.float32: 4,
}


def assign_group_by_size(parameters, group_size=256 * 1024 * 1024):
    """
    assign group by size
    """
    is_sparse_gradient = [False] * len(parameters)
    group_indices = core.assign_group_by_size(parameters, is_sparse_gradient,
                                              [group_size, group_size])
    var_groups = OrderedDict()
    for group_idx, indices in enumerate(group_indices):
        for index in indices:
            var_groups.setdefault(group_idx, []).append(parameters[index])
    return var_groups


def flatten_dense_tensors(parameters):
    """
    flatten dense tensors
    """

    _buffer_size = 0
    _param2align = {}
    dtype = parameters[0].dtype
    stop_gradient = parameters[0].stop_gradient
    state = copy.deepcopy(parameters[0].__dict__)

    for param in parameters:
        assert str(state) == str(param.__dict__)
        size = np.prod(param.shape) * align[dtype]
        remaining = size % alignment["gpu"]
        ali = 0 if remaining == 0 else alignment["gpu"] - remaining
        align_ = ali // align[dtype]
        _buffer_size += np.prod(param.shape) + align_
        _param2align[param.name] = align_

    param_storage = ParamStorage(size=_buffer_size, dtype=dtype, device="gpu")

    param_storage.add_rank_params(parameters, _param2align)

    # process gradient 
    # grad_storage = None
    grad_storage = GradStorage(
        size=_buffer_size,
        dtype=dtype,
        device="gpu",
        destination="0",
        parm2align=_param2align)

    for param in parameters:
        grad_storage.add_grad(param, _param2align[param.name])

    fused_param = ParamBase(
        shape=param_storage.buffer.shape,
        dtype=dtype,
        name=unique_name.generate('fused_param'))
    param_storage.buffer._share_buffer_to(fused_param)
    fused_param._copy_gradient_from(grad_storage.buffer)
    fused_param.__dict__.update(state)
    fused_param.stop_gradient = stop_gradient

    return fused_param


def get_fused_params(params):
    """
    get fused tensor from parameters
    """
    if len(params) < 1:
        return []

    var_groups = assign_group_by_size(params)
    fused_params = []
    for group_idx, parameters in var_groups.items():
        fused_param = flatten_dense_tensors(parameters)
        fused_params.append(fused_param)
    return fused_params
