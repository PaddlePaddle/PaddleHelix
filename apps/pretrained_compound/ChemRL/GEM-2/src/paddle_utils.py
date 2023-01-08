
#!/usr/bin/python                                                                                                
#-*-coding:utf-8-*- 
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
paddle utils
"""

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed.fleet.utils import recompute


def dist_all_reduce(x, *args, **argv):
    """
    make dist.all_reduce returnable.
    x: tensor
    """
    dist.all_reduce(x, *args, **argv)
    return x


def dist_mean(array, distributed=False):
    n = len(array)
    x_sum = 0 if n == 0 else np.sum(array)
    if distributed:
        n = int(dist_all_reduce(paddle.to_tensor(n, dtype='int64')))
        x_sum = float(dist_all_reduce(paddle.to_tensor(x_sum, dtype='float32')))
    x_mean = 0 if n == 0 else x_sum / n
    return x_mean


def dist_sum(array, distributed=False):
    n = len(array)
    x_sum = 0 if n == 0 else np.sum(array)
    if distributed:
        x_sum = float(dist_all_reduce(paddle.to_tensor(x_sum, dtype='float32')))
    return x_sum


def dist_length(array, distributed=False):
    n = len(array)
    if distributed:
        n = int(dist_all_reduce(paddle.to_tensor(n, dtype='int64')))
    return n


def recompute_wrapper(func, *args, is_recompute=True):
    if is_recompute:
        return recompute(func, *args)
    else:
        return func(*args)
        