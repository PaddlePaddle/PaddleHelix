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

import os
import copy
import contextlib
from enum import Enum
import numpy as np
import paddle
import paddle.fluid
from paddle.fluid import core, unique_name
from paddle.fluid.framework import in_dygraph_mode

if in_dygraph_mode():
    from paddle.fluid.framework import EagerParamBase
else:
    from paddle.fluid.framework import ParamBase

alignment = {"gpu": 256, }
align = {
    paddle.bfloat16: 2,
    paddle.float16: 2,
    paddle.float32: 4,
}


def check_dtype(dtype):
    assert dtype in [paddle.float32, paddle.float16, paddle.bfloat16
                     ], f"It does not supported dtype {dtype}"


@contextlib.contextmanager
def device_guard(dev_id=0, device="cpu"):
    origin_device = paddle.device.get_device()
    if device == "cpu":
        paddle.set_device(device)
    elif device == "gpu":
        paddle.set_device("gpu:{}".format(dev_id))
    try:
        yield
    finally:
        paddle.set_device(origin_device)


class ParamStorage(object):
    """
    This is a basic class to simplify the handling of parameter InternalStorages.
    """

    def __init__(self, size, dtype, device, convert_cpu=True):
        self._params = []
        self._param_ids = []
        self._fill = 0
        self._device = device
        self._dtype = dtype

        check_dtype(dtype)

        # The actual flat tensor
        size = [size] if isinstance(size, int) else size
        if convert_cpu:
            if dtype == paddle.float16:
                value = np.zeros(size, dtype=np.float16)
            elif dtype == paddle.bfloat16:
                value = np.zeros(size, dtype=np.uint16)
            else:
                value = np.zeros(size, dtype=np.float32)

            if in_dygraph_mode():
                self.buffer = core.eager.Tensor(value=value, place=core.CPUPlace())
            else:
                self.buffer = core.VarBase(value=value, place=core.CPUPlace())
        else:
            self.buffer = paddle.zeros(size, dtype=dtype)

        self.param2align = None

    def to(self, device, dtype=None, keep_alignment=True):
        """
        Move the underlying buffer
        """
        assert self.buffer is not None, "Cannot move a collapsed bucket, please rebuild it"
        check_dtype(dtype)

        dev_id = 0 if paddle.get_device() == "cpu" else int(paddle.get_device()
                                                            .split(":")[1])

        if self._device != device:
            tmp_buffer = self.buffer.cuda(
                dev_id) if device == "gpu" else self.buffer.cpu()
            for param in self._params:
                param.clear_gradient(False)

            del self.buffer
            self.buffer = tmp_buffer
            self._device = device

        if dtype is not None:
            self.buffer = self.buffer.cast(dtype=dtype)
            self._dtype = dtype

        if keep_alignment:
            self._array_params()

    @paddle.no_grad()
    def add_rank_params(self, trainable_params, param2align, convert_gpu=True):
        """
        Add new parameters to the InternalStorage. Params becomes a view of this InternalStorage buffer.
        """

        assert all([
            id(param) not in self._param_ids for param in trainable_params
        ]), "The same param cannot be checked in twice"
        assert self.buffer is not None

        self.param2align = param2align

        cpu_param_shape = list()
        for param in trainable_params:
            p_shape = self._add_param_as_view(param, param2align[param.name],
                                              convert_gpu)
            cpu_param_shape.append(p_shape)

        if convert_gpu:
            # buffer convert from cpu to cuda
            dev_id = int(paddle.get_device().split(":")[1])
            self.buffer = self.buffer.cuda(dev_id)

        self._fill = 0

        for idx, param in enumerate(trainable_params):
            self._convert_buffer(param, cpu_param_shape[idx],
                                 param2align[param.name])
            self._params.append(param)
            self._param_ids.append(id(param))

    @paddle.no_grad()
    def _add_param_as_view(self, param, align, convert_gpu=True):

        assert (
            param.dtype == self.buffer.dtype
        ), "Different types for the InternalStorage and the param, cannot proceed: {} - {}".format(
            param.dtype, self.buffer.dtype)

        var_end = self._fill + np.prod(param.shape)
        offset = var_end + align
        assert offset <= np.prod(self.buffer.shape)

        p_shape = param.shape

        origin_state = param.stop_gradient
        param.stop_gradient = True
        param.flatten_()
        param.stop_gradient = origin_state

        # Copy the current param value
        dev_id = 0 if paddle.get_device() == "cpu" else int(paddle.get_device()
                                                            .split(":")[1])
        with device_guard(dev_id, "cpu"):
            if in_dygraph_mode():
                tmp_var = self.buffer._slice(self._fill, var_end)

                if convert_gpu:
                    param_cpu = param.cpu()
                    param._clear_data()
                    tmp_var.set_value(param_cpu)
                else:
                    tmp_var.set_value(param)
                del tmp_var

            else:
                tmp_var = core.VarBase(
                    tensor=self.buffer._slice(self._fill, var_end))
                tmp_var.value().get_tensor()._set_dims(param.shape)
                if convert_gpu:
                    param_cpu = param.cpu()
                    param.value().get_tensor()._clear()
                    tmp_var.set_value(param_cpu)
                else:
                    tmp_var.set_value(param)

        self._fill = offset
        return p_shape

    @paddle.no_grad()
    def _convert_buffer(self, param, p_shape, align):

        var_end = self._fill + np.prod(p_shape)
        offset = var_end + align
        assert offset <= np.prod(self.buffer.shape)

        # Convert the param value
        if in_dygraph_mode():
            dev_id = 0 if paddle.get_device() == "cpu" else int(paddle.get_device().split(":")[1])
            with device_guard(dev_id, self._device):
                tmp_tensor = self.buffer._slice(self._fill, var_end)
                tmp_tensor._share_buffer_to(param)
                param.get_tensor()._set_dims(p_shape)
        else:
            tmp_tensor = self.buffer._slice(self._fill, var_end)
            param.value().get_tensor()._share_data_with(tmp_tensor)
            param.value().get_tensor()._set_dims(p_shape)

        self._fill = offset

    @paddle.no_grad()
    def _array_params(self):
        """
        Given the parameters which have been registered previously, rebuild the whole InternalStorage.
        """
        assert len(self._params) > 0
        assert self.param2align is not None

        self._fill = 0
        for p in self._params:
            self._convert_buffer(p, p.shape, self.param2align[p.name])  # modify


class GradStorage(object):
    """
    This is a basic class to simplify the handling of gradient InternalStorages
    """

    def __init__(self,
                 size,
                 dtype,
                 device,
                 destination,
                 parm2align,
                 convert_cpu=False):
        if isinstance(size, np.int64):
            size = size.tolist()

        check_dtype(dtype)

        self._params = []
        self._param_ids = []
        self._fill = 0
        self._device = device
        self._dtype = dtype

        # The actual flat tensor
        size = [size] if isinstance(size, int) else size
        if convert_cpu:
            if dtype == paddle.float16:
                value = np.zeros(size, dtype=np.float16)
            elif dtype == paddle.bfloat16:
                value = np.zeros(size, dtype=np.uint16)
            else:
                value = np.zeros(size, dtype=np.float32)
            if in_dygraph_mode():
                self.buffer = core.eager.Tensor(value=value, place=core.CPUPlace())
            else:
                self.buffer = core.VarBase(value=value, place=core.CPUPlace())
        else:
            self.buffer = paddle.zeros(size, dtype=dtype)

        self._max_size = size
        self._release = False

        self.params_checked_in = 0
        self.destination = destination
        self._parm2align = parm2align
        self.sent = False

    def reset_checked_in(self):
        """ Reset the counter of the parameter grads which have been checked in
        """
        self.params_checked_in = 0
        self.sent = False

    @property
    def all_checked_in(self):
        """ Judge all the expected gradient check-in happened """
        return len(self._params) == self.params_checked_in

    def can_add_grad_view(self, param, align):
        """ Is there enough InternalStorage to add this parameter gradient, and whether this param have already checked in.
        """
        return self._fill + np.prod(
            param.shape) + align <= self._max_size and id(
                param) not in self._param_ids

    def to(self, device, dtype=None, keep_alignment=True):
        """
        Move the underlying buffer
        """
        check_dtype(dtype)

        if self._release:
            self.rebuild()

        assert self.buffer is not None, "Cannot move a collapsed bucket, please rebuild it"

        dev_id = 0 if paddle.get_device() == "cpu" else int(paddle.get_device()
                                                            .split(":")[1])

        if self._device != device:
            tmp_buffer = self.buffer.cuda(
                dev_id) if device == "gpu" else self.buffer.cpu()
            for param in self._params:
                param.clear_gradient(False)
            del self.buffer
            self.buffer = tmp_buffer
            self._device = device

        if dtype is not None:
            self.buffer = self.buffer.cast(dtype=dtype)
            self._dtype = dtype

        if keep_alignment:
            self._array_grads()

    @paddle.no_grad()
    def add_grad(self, param, align):
        """
        Add a new parameter gradient to the InternalStorage. Param.grad becomes a view of this InternalStorage buffer.
        """

        assert id(
            param
        ) not in self._param_ids, "The same gradients cannot be checked in twice"

        self._add_grad_as_view(param, align)
        self._params.append(param)
        self._param_ids.append(id(param))

    @paddle.no_grad()
    def manumal_relase(self):
        """
        Release the buffer from InternalStorage. The InternalStorage will need to be rebuilt before use.
        """
        if not self._release:
            for p in self._params:
                if p.grad is not None:
                    p.clear_gradient(False)

            self.buffer = None
            self._fill = 0
            self.params_checked_in = 0
            self._release = True

    @paddle.no_grad()
    def rebuild(self):
        """
        Given the parameter gradients which have been registered previously, rebuild the whole InternalStorage.
        """

        if self._release:
            self.buffer = paddle.zeros([self._max_size], dtype=self._dtype)

            for p in self._params:
                self._add_grad_as_view(p, self._parm2align[p.name])

            self._release = False

    @paddle.no_grad()
    def _array_grads(self):
        """
        Given the parameters gradients which have been registered previously, rebuild the whole InternalStorage.
        """
        if len(self._params) > 0:
            self._fill = 0
            for p in self._params:
                self._add_grad_as_view(p, self._parm2align[p.name])

    @paddle.no_grad()
    def _add_grad_as_view(self, param, align):
        assert np.prod(
            self.buffer.shape
        ) > 0, "Cannot add a gradient to a released InternalStorage, please rebuild"
        assert param.dtype == self.buffer.dtype

        grad_end = self._fill + np.prod(param.shape)
        offset = grad_end + align
        assert offset <= np.prod(self.buffer.shape)

        # Copy the current grad value to InternalStorage
        dev_id = 0 if paddle.get_device() == "cpu" else int(paddle.get_device()
                                                            .split(":")[1])

        if in_dygraph_mode():
            with device_guard(dev_id, self._device):
                tmp_var = self.buffer._slice(self._fill, grad_end)
                tmp_var.get_tensor()._set_dims(param.shape)
                param._copy_gradient_from(tmp_var)
                del tmp_var
        else:
            if self._device == "cpu":
                with device_guard(dev_id, self._device):
                    if in_dygraph_mode():
                        tmp_var = core.eager.Tensor(
                            self.buffer._slice(self._fill, grad_end))
                        tmp_var.get_tensor()._set_dims(param.shape)
                    else:
                        tmp_var = core.VarBase(self.buffer._slice(self._fill, grad_end))
                        tmp_var.value().get_tensor()._set_dims(param.shape)
                    param._copy_gradient_from(tmp_var)

            elif self._device == "gpu":
                if in_dygraph_mode():
                    tmp_var = core.eager.Tensor(
                        self.buffer._slice(self._fill, grad_end))
                    tmp_var.get_tensor()._set_dims(param.shape)
                else:
                    tmp_var = core.VarBase(self.buffer._slice(self._fill, grad_end))
                    tmp_var.value().get_tensor()._set_dims(param.shape)
                param._copy_gradient_from(tmp_var)

        self._fill = offset


def assign_group_by_size(parameters, group_size=256 * 1024 * 1024):
    """
    assign group by size
    """
    is_sparse_gradient = [False] * len(parameters)
    if in_dygraph_mode():
        group_indices = core.eager_assign_group_by_size(
            parameters, is_sparse_gradient, [group_size, group_size])
    else:
        group_indices = core.assign_group_by_size(
            parameters, is_sparse_gradient, [group_size, group_size])
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

    if in_dygraph_mode():
        fused_param = EagerParamBase(
            shape=param_storage.buffer.shape,
            dtype=dtype,
            name=unique_name.generate('fused_param'))
    else:
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