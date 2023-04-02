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
""" Branch Parallel helper function"""

import paddle
from paddle.autograd import PyLayer
from alphafold_paddle.distributed.comm_group import scg

def broadcast(tensor, src):
    """ broadcast tensor from src rank in bp group """
    if scg.get_bp_world_size() == 1:
        return tensor

    assert src in [0, 1], "Branch Parallel is only support bp_degree=2 now!"
    
    group = scg.get_bp_group()
    ring_id = group.id
    paddle._C_ops.c_broadcast(tensor, tensor, 'root', src, 'use_calc_stream', True, 'ring_id', ring_id)

class BroadcastGrad(PyLayer):
    """ A PyLayer Op broadcast gradient in backward stage """
    @staticmethod
    def forward(ctx, input, src):
        """ return input directly """
        ctx.src = src
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        """ broadcast grad form src """
        broadcast(grad_output, ctx.src)
        return grad_output.clone()

def broadcast_grad_for_backward(input, src):
    """ a warpper for boradcast gradient in backward stage """
    if scg.get_bp_world_size() == 1:
        return input

    if not input.stop_gradient:
        output = BroadcastGrad.apply(input, src)
    else:
        output = input.clone()
    return output

@paddle.no_grad()
def all_reduce(tensor):
    """ allreduce a tensor in bp group """
    if scg.get_bp_world_size() == 1:
        return tensor

    group = scg.get_bp_group()
    paddle.distributed.all_reduce(
        tensor, use_calc_stream=True, group=group)

    return tensor

@paddle.no_grad()
def grad_sync(param_groups, comm_group):
    """
        sync the gradients of params
    """

    nranks = comm_group.nranks

    if nranks < 2:
        return

    for group in param_groups:
        if group.get("bp", False):
            for p in group['params']:
                if p.is_distributed:
                    continue

                grad = p.grad
                if grad is None:
                    continue

                paddle.distributed.all_reduce(
                    grad, use_calc_stream=True, group=comm_group)

    return None
