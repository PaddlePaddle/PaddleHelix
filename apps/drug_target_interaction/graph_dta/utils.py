#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
utils
"""
import os
import logging
import numpy as np

from paddle import fluid
from paddle.fluid.incubate.fleet.base import role_maker
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy

def save_data_list_to_npz(data_list, npz_file):
    """tbd"""
    keys = data_list[0].keys()
    merged_data = {}
    for key in keys:
        lens = np.array([len(data[key]) for data in data_list])
        values = np.concatenate([data[key] for data in data_list], 0)
        merged_data[key] = values
        merged_data[key + '.seq_len'] = lens
    np.savez_compressed(npz_file, **merged_data)


def load_npz_to_data_list(npz_file):
    """tbd"""
    def split_data(values, seq_lens):
        """tbd"""
        res = []
        s = 0
        for l in seq_lens:
            res.append(values[s: s + l])
            s += l
        return res

    merged_data = np.load(npz_file)
    names = [name for name in merged_data.keys() if not name.endswith('.seq_len')]
    data_dict = {}
    for name in names:
        data_dict[name] = split_data(merged_data[name], merged_data[name + '.seq_len'])

    data_list = []
    n = len(data_dict[names[0]])
    for i in range(n):
        data = {name:data_dict[name][i] for name in names}
        data_list.append(data)
    return data_list


def load_partial_vars(exe, init_model, main_program):
    """tbd"""
    assert os.path.exists(init_model), "[%s] cann't be found." % init_model

    def existed_params(var):
        """tbd"""
        if not isinstance(var, fluid.framework.Parameter):
            logging.info("%s not existed" % var.name)
            return False
        if os.path.exists(os.path.join(init_model, var.name)):
            logging.info("load %s successful" % var.name)
            return True
        else:
            logging.info("%s not existed" % var.name)
            return False

    fluid.io.load_vars(
            exe,
            init_model,
            main_program=main_program,
            predicate=existed_params)


def get_exe_params(is_distributed, use_cuda, thread_num):
    """
    Set the default execute parameters.
    """
    gpu_id = 0
    trainer_num = 1
    trainer_id = 0
    dist_strategy = None
    if is_distributed:
        if use_cuda:
            role = role_maker.PaddleCloudRoleMaker(is_collective=True)
            fleet.init(role)

            gpu_id = int(os.getenv("FLAGS_selected_gpus"))
            trainer_num = fleet.worker_num()
            trainer_id = fleet.worker_index()

            exec_strategy = fluid.ExecutionStrategy()
            exec_strategy.use_experimental_executor = True
            exec_strategy.num_threads = 4
            exec_strategy.num_iteration_per_drop_scope = 1

            dist_strategy = DistributedStrategy()
            dist_strategy.exec_strategy = exec_strategy
            dist_strategy.nccl_comm_num = 2
            dist_strategy.fuse_all_reduce_ops = True

            dist_strategy.use_amp = True
            dist_strategy.amp_loss_scaling = 12800.0
        else:
            print('Only gpu is supported for distributed mode at present.')
            exit(-1)
    else:
        if not use_cuda:
            os.environ['CPU_NUM'] = str(thread_num)

    if use_cuda:
        print('gpu_id', gpu_id)
        exe = fluid.Executor(fluid.CUDAPlace(gpu_id))
    else:
        exe = fluid.Executor(fluid.CPUPlace())

    exe_params = {
            'trainer_num': trainer_num,
            'trainer_id': trainer_id,
            'gpu_id': gpu_id,
            'dist_strategy': dist_strategy}

    return exe, exe_params


def concordance_index(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci
