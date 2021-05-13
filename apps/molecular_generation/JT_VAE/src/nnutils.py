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
"""neural network utils"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def index_select_ND(source, dim, index):
    """Return nodes for index"""
    index_size = index.shape
    suffix_dim = source.shape[1:]
    final_size = index_size + suffix_dim
    target = paddle.index_select(x=source, axis=dim, index=paddle.reshape(index, shape=[-1]))
    return target.reshape(final_size)


def avg_pool(all_vecs, scope, dim):
    """Average pooling"""
    size = paddle.to_tensor([le for _, le in scope])
    return paddle.sum(all_vecs, axis=dim) / paddle.unsqueeze(size, axis=-1)


def stack_pad_tensor(tensor_list):
    """Stack tensor with padding"""
    max_len = max([t.shape[0] for t in tensor_list])
    for i, tensor in enumerate(tensor_list):
        pad_len = max_len - tensor.shape[0]
        tensor_list[i] = F.pad(tensor, (0, 0, 0, pad_len))
    return paddle.stack(tensor_list, axis=0)


def flatten_tensor(tensor, scope):
    """Flat tensor"""
    assert tensor.shape[0] == len(scope)
    tlist = []
    for i, tup in enumerate(scope):
        le = tup[1]
        tlist.append(tensor[i, 0:le])
    return paddle.concat(tlist, axis=0)


def inflate_tensor(tensor, scope):
    """Inflate tensor"""
    max_len = max([le for _, le in scope])
    batch_vecs = []
    for st, le in scope:
        cur_vecs = tensor[st: st + le]
        cur_vecs = F.pad(cur_vecs, (0, 0, 0, max_len - le))
        batch_vecs.append(cur_vecs)

    return paddle.stack(batch_vecs, axis=0)




