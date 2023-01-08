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
Dynamic Axial Parallelism and Duality Async Operation helper functions
paper ref: FastFold: Reducing AlphaFold Training Time from 11 Days to 67 Hours, https://arxiv.org/abs/2203.00854
code ref: https://github.com/hpcaitech/FastFold.git
"""

import warnings
import paddle
from paddle import distributed as dist
from paddle.autograd import PyLayer

__all__ = [
    'init_dap',
    'dap_is_initialized',
    'get_tensor_model_parallel_group',
    'get_data_parallel_group',
    'get_tensor_model_parallel_world_size',
    'get_tensor_model_parallel_rank',
    'get_data_parallel_world_size',
    'get_data_parallel_rank',
    'get_tensor_model_parallel_src_rank',
    'scatter',
    'gather',
    'all_gather',
    'all_gather_opp',
    'all_to_all',
    'all_to_all_opp',
    'row_to_col',
    'col_to_row'
    ]

# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None

# These values enable us to change the mpu sizes on the fly.
_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_TENSOR_MODEL_PARALLEL_RANK = None

# communication whether use_calc_stream (sync) or not (async). Default True
_COMM_SYNC = None


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(numerator, denominator)


def divide(numerator, denominator):
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def init_dap(tensor_model_parallel_size_=1, sync=True):

    global _COMM_SYNC
    assert _COMM_SYNC is None, \
        'communication manner `sync` is already initialized'
    _COMM_SYNC = sync

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # check dist config
    ensure_divisibility(world_size, tensor_model_parallel_size_)
    data_parallel_size_ = world_size // tensor_model_parallel_size_

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, \
        'data parallel group is already initialized'
    for i in range(tensor_model_parallel_size_):
        ranks = list(range(i, world_size, tensor_model_parallel_size_))
        group = dist.new_group(ranks)
        print('> dp ranks:', ranks, 'dp group:', group)
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group

    global _TENSOR_MODEL_PARALLEL_GROUP
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, \
        'tensor model parallel group is already initialized'
    # Build the model-parallel groups.
    for i in range(data_parallel_size_):
        ranks = list(range(i * tensor_model_parallel_size_, (i + 1) * tensor_model_parallel_size_))
        group = dist.new_group(ranks)
        print('> mp ranks:', ranks, 'mp group', group)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group

    if dist.get_rank() == 0:
        print('> initialize tensor model parallel with size {}'.format(tensor_model_parallel_size_))
        print('> initialize data parallel with size {}'.format(data_parallel_size_))
        
def dap_is_initialized():
    """Check if model and data parallel groups are initialized."""
    global _DATA_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GROUP
    if _TENSOR_MODEL_PARALLEL_GROUP is None or \
        _DATA_PARALLEL_GROUP is None:
        return False
    return True

def is_sync():
    return _COMM_SYNC

def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, \
        'intra_layer_model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    if not dap_is_initialized():
        warnings.warn("DAP comminication group is not initialized.")
        return 1
    global _TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return get_tensor_model_parallel_group().nranks


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    if not dap_is_initialized():
        warnings.warn("DAP comminication group is not initialized.")
        return 0
    global _TENSOR_MODEL_PARALLEL_RANK
    if _TENSOR_MODEL_PARALLEL_RANK is not None:
        return _TENSOR_MODEL_PARALLEL_RANK
    return get_tensor_model_parallel_group().rank


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    if not dap_is_initialized():
        warnings.warn("DAP comminication group is not initialized.")
        return 1
    return get_data_parallel_group().nranks


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    if not dap_is_initialized():
        warnings.warn("DAP comminication group is not initialized.")
        return 0
    return get_data_parallel_group().rank


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = dist.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


@paddle.no_grad()
def _gather(tensor, axis=-1):    
    tensor_list = []
    dist.all_gather(tensor_list,
                    tensor,
                    group=get_tensor_model_parallel_group(),
                    use_calc_stream=True)
    output = paddle.concat(tensor_list, axis=axis)
    return output


@paddle.no_grad()
def _split(tensor, axis=-1):
    ensure_divisibility(tensor.shape[axis], get_tensor_model_parallel_world_size())
    tensor_list = paddle.split(tensor, get_tensor_model_parallel_world_size(), axis=axis)

    output = tensor_list[get_tensor_model_parallel_rank()]

    return output


class Scatter(PyLayer):
    """ Scatter PyLayer Op"""
    @staticmethod
    def forward(ctx, input, axis:-1):
        ctx.axis = axis
        return _split(input, axis=axis)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output, axis=ctx.axis)
    
    
def scatter(input, axis=-1):
    """ split a tensor according axis by dap size """
    if get_tensor_model_parallel_world_size() == 1:
        return input
    
    if not input.stop_gradient:
        output = Scatter.apply(input, axis=axis)
    else:
        output = _split(input, axis=axis)
    return output


class Gather(PyLayer):
    """ Gather PyLayer Op """
    @staticmethod
    def forward(ctx, input, axis=-1):
        ctx.axis = axis
        return _gather(input, axis=axis)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, axis=ctx.axis)

def gather(input, axis=-1):
    """ gather tensor form all rank in dap group in axis """
    if get_tensor_model_parallel_world_size() == 1:
        return input
    
    if not input.stop_gradient:
        output = Gather.apply(input, axis=axis)
    else:
        output = _gather(input, axis=axis)
    return output

@paddle.no_grad()
def _all_gather(tensor, axis=-1, sync=True):
    if not sync:
        dist.wait(tensor, group=get_tensor_model_parallel_group(), use_calc_stream=True)

    group = get_tensor_model_parallel_group()
    ring_id = group.id
    nranks = group.nranks
    output = paddle._C_ops.c_allgather(tensor, 'use_calc_stream', sync, 'ring_id', ring_id, 'nranks', nranks)

    return output

@paddle.no_grad()
def _reduce_scatter(tensor, sync=True):
    if not sync:
        dist.wait(tensor, group=get_tensor_model_parallel_group(), use_calc_stream=True)

    group = get_tensor_model_parallel_group()
    ring_id = group.id
    nranks = group.nranks
    output = paddle._C_ops.c_reducescatter(tensor, 'use_calc_stream', sync, 'ring_id', ring_id, 'nranks', nranks)
    paddle.device.cuda.synchronize()
    return output

class AllGather(PyLayer):
    """ AllGather PyLayer Op """
    @staticmethod
    def forward(ctx, input, axis=-1, sync=True):
        ctx.axis = axis
        ctx.sync = sync
        output = _all_gather(input, axis=axis, sync=sync)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.sync:
            dist.wait(grad_output, group=get_tensor_model_parallel_group(), use_calc_stream=ctx.sync)
        return grad_output

class AllGather_Opp(PyLayer):
    """ Duality Async Operation for AllGather """
    @staticmethod
    def forward(ctx, input, axis=-1, sync=True):
        ctx.axis = axis
        ctx.sync = sync
        return input

    @staticmethod
    def backward(ctx, grad_output):
        output = _reduce_scatter(grad_output, sync=ctx.sync)
        return output


def all_gather(input, axis=-1, sync=None):
    """ gather tensors from all rank in dap group and all get the result.
        if sync=None, sync will be assign according init_dap setting.

        when using async communication, sync=False, do not use the output as same as input.
        E.g. do not use `a = all_gather(a, ...)`, recommend to use `b = all_gather(a, ...)`
    """
    if get_tensor_model_parallel_world_size() == 1:
        return input

    if sync is None:
        sync = is_sync()
    
    if not input.stop_gradient:
        output = AllGather.apply(input, axis, sync=sync)
    else:
        output = _all_gather(input, axis, sync=sync)
    return output


def all_gather_opp(output, axis=-1, sync=None):
    """ Duality Async Operation for all_gather.
        if sync=None, sync will be assign according init_dap setting.
    """
    nranks = get_tensor_model_parallel_world_size()
    if nranks == 1:
        return output

    if sync is None:
        sync = is_sync()
    
    if not sync:
        dist.wait(output, group=get_tensor_model_parallel_group(), use_calc_stream=sync)

    if not output.stop_gradient:
        output = AllGather_Opp.apply(output, axis, sync=sync)

    if axis != 0:
        output = paddle.concat(paddle.split(output, nranks, 0), axis=axis)

    return output


@paddle.no_grad()
def _all_to_all(tensor, in_axis=-1, out_axis=-1, sync=True):
    if not sync:
        dist.wait(tensor, group=get_tensor_model_parallel_group(), use_calc_stream=True)

    group = get_tensor_model_parallel_group()
    ring_id = group.id
    
    output = paddle._C_ops.alltoall(tensor, 'use_calc_stream', sync, 'ring_id', ring_id)
    
    return output


class All_to_All(PyLayer):
    """ All_to_All PyLayer Op"""
    @staticmethod
    def forward(ctx,
                input,
                in_axis=-1,
                out_axis=-1,
                sync=True):
        ctx.in_axis = in_axis
        ctx.out_axis = out_axis
        ctx.sync = sync
        return _all_to_all(input, in_axis=in_axis, out_axis=out_axis, sync=sync)

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.sync:
            dist.wait(grad_output, group=get_tensor_model_parallel_group(), use_calc_stream=ctx.sync)
        return grad_output


class All_to_All_Opp(PyLayer):
    """ Duality Async Operation for All_to_All """
    @staticmethod
    def forward(ctx, output, in_axis=-1, out_axis=-1, sync=True):
        ctx.in_axis = in_axis
        ctx.out_axis = out_axis
        ctx.sync = sync
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return _all_to_all(grad_output, in_axis=ctx.out_axis, out_axis=ctx.in_axis, sync=ctx.sync)
    

class All2All(PyLayer):
    @staticmethod
    def forward(ctx,
                input,
                in_axis=-1,
                out_axis=-1):
        ctx.in_axis = in_axis
        ctx.out_axis = out_axis
        return _all_to_all(input, in_axis=in_axis, out_axis=out_axis, sync=True)

    @staticmethod
    def backward(ctx, grad_output):
        return _all_to_all(grad_output, in_axis=ctx.out_axis, out_axis=ctx.in_axis, sync=True)


def all_to_all(input, in_axis, out_axis, sync=True):
    """ all to all according in_axis and out_axis.
        if sync=None, sync will be assign according init_dap setting.
    """
    if get_tensor_model_parallel_world_size() == 1:
        return input

    if sync is None:
        sync = is_sync()

    if in_axis != 0:
        ensure_divisibility(input.shape[in_axis], get_tensor_model_parallel_world_size())
        input = paddle.concat(paddle.split(input, get_tensor_model_parallel_world_size(), axis=in_axis), axis=0)

    if not input.stop_gradient:
        output = All_to_All.apply(input, in_axis=in_axis, out_axis=out_axis, sync=sync)
    else:
        output = _all_to_all(input, in_axis=in_axis, out_axis=out_axis, sync=sync)

    return output


def all_to_all_opp(output, in_axis, out_axis, sync=True):
    """ Duality Async Operation for all_to_all.
        if sync=None, sync will be assign according init_dap setting.
    """
    if get_tensor_model_parallel_world_size() == 1:
        return output

    if sync is None:
        sync = is_sync()

    if not sync:
        dist.wait(output, group=get_tensor_model_parallel_group(), use_calc_stream=sync)

    if not output.stop_gradient:
        output = All_to_All_Opp.apply(output, in_axis=in_axis, out_axis=out_axis, sync=sync)

    if out_axis != 0:
        ensure_divisibility(output.shape[0], get_tensor_model_parallel_world_size())
        output = paddle.concat(paddle.split(output, get_tensor_model_parallel_world_size(), axis=0), axis=out_axis)

    return output


def row_to_col(input):
    """ N, S, R, C => N, R, S, C using sync all_to_all """
    if get_tensor_model_parallel_world_size() == 1:
        return input

    ensure_divisibility(input.shape[2], get_tensor_model_parallel_world_size())
    input = paddle.concat(paddle.split(input, get_tensor_model_parallel_world_size(), axis=2), axis=0)

    if not input.stop_gradient:
        output = All2All.apply(input, in_axis=2, out_axis=1)
    else:
        output = _all_to_all(input, in_axis=2, out_axis=1)

    output = paddle.concat(paddle.split(output, get_tensor_model_parallel_world_size(), axis=0), axis=1)
    return output


def col_to_row(input):
    """ N, R, S, C => N, S, R, C using sync all_to_all """
    if get_tensor_model_parallel_world_size() == 1:
        return input

    ensure_divisibility(input.shape[1], get_tensor_model_parallel_world_size())
    input = paddle.concat(paddle.split(input, get_tensor_model_parallel_world_size(), axis=1), axis=0)

    if not input.stop_gradient:
        output = All2All.apply(input, in_axis=1, out_axis=2)
    else:
        output = _all_to_all(input, in_axis=1, out_axis=2)

    output = paddle.concat(paddle.split(output, get_tensor_model_parallel_world_size(), axis=0), axis=2)
    return output


@paddle.no_grad()
def grad_sync(param_groups, comm_group):
    """
        sync the gradients of params
    """

    nranks = comm_group.nranks

    if nranks < 2:
        return

    for group in param_groups:
        if group.get("dap", False):
            for p in group['params']:
                if p.is_distributed:
                    continue

                grad = p.grad
                if grad is None:
                    continue

                paddle.distributed.all_reduce(
                    grad, use_calc_stream=True, group=comm_group)

    return None
