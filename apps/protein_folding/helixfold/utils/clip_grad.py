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
Clips gradient by global norm
"""

import paddle

@paddle.no_grad()
def clip_grad_norm_(parameters, clip_norm, auto_skip_clip=False):
    """ This function clip grad by global norm.
        Args:
          parameters: A list of parameters
          clip_norm: clip norm value
          auto_skip_clip: only when calculated global_norm_value
                          greater than clip_norm do grad clip. Default Fasle
    """

    if clip_norm == 0:
        return None

    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]

    grads = [p.grad for p in parameters if p.grad is not None]

    params_and_grads = []
    sum_square_list = []
    sum_square_list_fp16 = []
    sum_square_list_fp32 = []
    for g in grads:
        sum_square = paddle._C_ops.squared_l2_norm(g)
        if sum_square.dtype == paddle.float16:
            sum_square_list_fp16.append(sum_square)
        elif sum_square.dtype == paddle.float32:
            sum_square_list_fp32.append(sum_square)
        else:
            sum_square_list.append(sum_square)

    # all parameters have been filterd out
    if len(sum_square_list) + len(sum_square_list_fp16) + len(
            sum_square_list_fp32) == 0:
        return None

    sum_dtype = 'float64' if len(sum_square_list) > 0 else "float32"
    global_norm_var = []
    if len(sum_square_list_fp16) > 0:
        global_norm_var_fp16 = paddle.add_n(sum_square_list_fp16)
        global_norm_var.append(global_norm_var_fp16.astype(sum_dtype))
    if len(sum_square_list_fp32) > 0:
        global_norm_var_fp32 = paddle.add_n(sum_square_list_fp32)
        if sum_dtype == 'float32':
            global_norm_var.append(global_norm_var_fp32)
        else:
            global_norm_var.append(global_norm_var_fp32.astype(sum_dtype))
    if len(sum_square_list) > 0:
        global_norm_var_fp64 = paddle.add_n(sum_square_list)
        global_norm_var.append(global_norm_var_fp64)
    global_norm_var = paddle.add_n(global_norm_var)
    global_norm_var = paddle.sqrt(global_norm_var)
    max_global_norm = paddle.fluid.layers.fill_constant(
        shape=[1], dtype=global_norm_var.dtype, value=clip_norm)

    need_clip = False
    if not auto_skip_clip:  # always apply clip
        need_clip = True
        clip_var = paddle.fluid.layers.elementwise_div(
            x=max_global_norm,
            y=paddle.fluid.layers.elementwise_max(
                x=global_norm_var, y=max_global_norm))
    elif global_norm_var > max_global_norm:
        # only when global_norm_var > max_global_norm, grad need clip
        need_clip = True
        clip_var = paddle.fluid.layers.elementwise_div(
            x=max_global_norm, y=global_norm_var)

    if need_clip:
        for g in grads:
            clip_input = (clip_var.astype('float16')
                          if g.dtype == paddle.float16 else
                          clip_var)
            g.detach().scale_(clip_input)
