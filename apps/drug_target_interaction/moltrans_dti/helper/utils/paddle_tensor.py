#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
Paddle Tensor
"""

import paddle
import copy
from functools import partial


def add_tensor_function(func):
    """
    Add tensor function
    """
    setattr(paddle.Tensor, func.__name__, func)


@add_tensor_function
def item(self):
    """
    Item function
    """
    return float(self.numpy())


@add_tensor_function
def permute(self, *dims):
    """
    Permute function
    """
    return self.transpose(dims)


@add_tensor_function
def contiguous(self):
    """
    Contiguous function
    """
    return self


@add_tensor_function
def view(self, *shape):
    """
    View function
    """
    return self.reshape(*shape)


@add_tensor_function
def repeat(self, *sizes):
    """
    Repeat function
    """
    return self.tile(sizes)


@add_tensor_function
def dim(self):
    """
    Dim function
    """
    return self.ndim


@add_tensor_function
def long(self, memory_format=None):
    """
    Long function
    """
    return paddle.cast(self, dtype="int64")


@add_tensor_function
def float(self, memory_format=None):
    """
    Float function
    """
    return paddle.cast(self, dtype="float32")


@add_tensor_function
def cuda(self):
    """
    Cuda function
    """
    return self


@add_tensor_function
def size(self, dim=None):
    """
    Size function
    """
    if dim is not None:
        return self.shape[dim]
    else:
        return self.shape
            

@add_tensor_function
def to(self, *args, **kwargs):
    """
    To function
    """
    if len(args) == 1 and "dtype" not in kwargs:
        try:
            return paddle.cast(self, dtype=args[0])
        except Exception:
            return self
    else:
        if len(kwargs) > 0:
            if "dtype" in kwargs:
                return paddle.cast(self, dtype=kwargs["dtype"])
            else:
                return self
        else:
            return self


@add_tensor_function
def index_fill_(self, dim, index, val):
    """
    Index fill function
    """
    x_shape = self.shape
    index_shape = index.shape
    if dim != 0:
        perm_list = list(range(len(x_shape)))
        while dim < 0:
            dim += len(x_shape)
        perm_list.pop(dim)
        perm_list = [dim] + perm_list
        self = paddle.transpose(self, perm=perm_list)
        s = x_shape.pop(dim)
        x_shape = [s] + x_shape
    updates_shape = index_shape + x_shape[1:]
    updates = paddle.full(updates_shape, fill_value=val, dtype=self.dtype)
    out = paddle.scatter(self, index, updates)
    if dim != 0:
        perm_list = list(range(len(x_shape)))
        perm_list.pop(0)
        perm_list.insert(dim, 0)
        out = paddle.transpose(out, perm=perm_list)
    paddle.assign(out, output=self)


sum_tmp = partial(paddle.Tensor.sum)
@add_tensor_function
def sum(self, dim, keepdim=False, dtype=None):
    """
    Sum function
    """
    return sum_tmp(self, axis=dim, dtype=dtype, keepdim=keepdim)


sort_tmp = partial(paddle.Tensor.sort)
@add_tensor_function
def sort(self, dim=-1, descending=False, out=None):
    """
    Sort function
    """
    return sort_tmp(self, axis=dim, descending=descending), paddle.argsort(self, axis=dim, descending=descending)


reshape_tmp = partial(paddle.Tensor.reshape)
@add_tensor_function
def reshape(self, *shape):
    """
    Reshape function
    """
    return reshape_tmp(self, shape)


transpose_tmp = partial(paddle.Tensor.transpose)
@add_tensor_function
def transpose(self, dim0, dim1=None):
    """
    Transpose function
    """
    if dim1 is None:
        return transpose_tmp(self, dim0)
    else:
        shape = self.shape
        perm = list(range(len(shape)))
        dim0 = (dim0 + len(shape)) if dim0 < 0 else dim0
        dim1 = (dim1 + len(shape)) if dim1 < 0 else dim1
        perm[dim0] = dim1
        perm[dim1] = dim0
        return transpose_tmp(self, perm)