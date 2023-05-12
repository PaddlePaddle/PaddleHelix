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

"""Utils."""

import os
from os.path import basename
import numpy as np
import paddle


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


def get_structure_module_bf16_op_list():
    black_list = {
        "clip",
        "dropout_nd",
        "elementwise_add",
        "elementwise_div",
        "elementwise_mul",
        "elementwise_sub",
        "fill_any_like",
        "fill_constant",
        "fused_gate_attention",
        "gaussian_random",
        "linspace",
        "log_softmax",
        "p_norm",
        "py_layer",
        "reduce_mean",
        "reduce_min",
        "reduce_prod",
        "reduce_sum",
        "scale",
        "sigmoid",
        "softmax",
        "softplus",
        "sqrt",
        "square",
        "squared_l2_norm",
        "sum",
        "uniform_random",
    }
    white_list = {
        "abs",
        "bitwise_or",
        "concat",
        "elementwise_max",
        "elementwise_min",
        "equal", 
        "eye",
        "gather",
        "greater_than",
        "layer_norm",
        "less_than",
        "matmul_v2",
        "one_hot_v2",
        "reduce_max",
        "relu",
        "reshape2",
        "slice", 
        "split",
        "squeeze2",
        "stack",
        "transpose2",
        "unsqueeze2",
        "tile",
    }
    return black_list, white_list 


def get_model_parameter_size(model):
    """tbd"""
    size = 0
    for param in model.parameters():
        size += np.product(param.shape)
    return size


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


def tree_filter(key_cond, value_cond, d):
    new_d = {}
    for k in d:
        if not key_cond is None and not key_cond(k):
            continue
        if not value_cond is None and not value_cond(d[k]):
            continue

        if type(d[k]) is dict:
            cur_d = tree_filter(key_cond, value_cond, d[k])
            if len(cur_d) != 0:
                new_d[k] = cur_d
        else:
            new_d[k] = d[k]
    return new_d


def add_to_data_writer(data_writer, step, results, prefix=''):
    """tbd"""
    print("step:%d %s:%s" % (step, prefix, str(results)))
    if data_writer is None:
        return
    for k, v in results.items():
        data_writer.add_scalar("%s/%s" % (prefix, k), v, step)


def upload_to_hadoop(args, cur_step):
    def _upload_file(local_file, hadoop_dir):
        assert len(hadoop_dir) > 10, \
            f"hadoop_dir ({hadoop_dir}) is too short"
        file_name = basename(local_file)
        os.system(f"{hadoop_fs} -mkdir {hadoop_dir}")
        os.system(f"{hadoop_fs} -rmr {hadoop_dir}/{file_name}")
        os.system(f"{hadoop_fs} -put {local_file} {hadoop_dir}/{file_name}")    

    hadoop_fs = os.environ["HADOOP_FS"]
    output_path = os.environ["OUTPUT_PATH"]

    # upload models
    _upload_file(
            f'{args.model_dir}/step_{cur_step}.pdparams', 
            f'{output_path}/models')

    # upload tensorboard log
    files = os.listdir(f'{args.log_dir}/tensorboard_log_dir')
    for file in files:
        _upload_file(
                f'{args.log_dir}/tensorboard_log_dir/{file}', 
                f'{output_path}/log/tensorboard_log_dir')


def csv_print(d):
    keys = sorted(list(d.keys()))
    values = [str(d[k]) for k in keys]
    print(' '.join([str(x) for x in keys]))
    print(' '.join([str(x) for x in values]))
