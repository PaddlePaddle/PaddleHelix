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
Evaluation for the sequence-based models for protein.
"""

import argparse 
import json
import numpy as np
import os
import time
import paddle
import paddle.fluid as fluid
from pahelix.utils.paddle_utils import load_partial_params
from data_gen import setup_data_loader
from tape_model import TAPEModel
from utils import *

def main(args):
    """main"""
    paddle.enable_static()

    model_config = json.load(open(args.model_config, 'r'))

    exe_params = default_exe_params(False, args.use_cuda, args.thread_num)
    exe = exe_params['exe']
    trainer_num = exe_params['trainer_num']
    trainer_id = exe_params['trainer_id']
    places = exe_params['places']

    task = model_config['task']

    model = TAPEModel(model_config=model_config, name=task)

    test_program = fluid.Program()
    test_startup = fluid.Program()
    with fluid.program_guard(test_program, test_startup):
        with fluid.unique_name.guard():
            model.forward(True)
            model.cal_loss()
            test_data_loader = setup_data_loader(
                    model.input_list,
                    model_config,
                    args.test_data,
                    trainer_id,
                    trainer_num,
                    places,
                    args.batch_size)
            exe.run(test_startup)
    test_metric = get_metric(task)

    if not args.init_model is None and args.init_model != "":
        load_partial_params(exe, args.init_model, test_program)
    else:
        raise RuntimeError('Please set init_model.')

    test_fetch_list = model.get_fetch_list(),
    for data in test_data_loader():
        results = exe.run(
                program=test_program,
                feed=data,
                fetch_list=test_fetch_list,
                return_numpy=False)
        update_metric(task, test_metric, results)
    test_metric.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true') 

    parser.add_argument('--batch_size', type=int, default=128) 
    parser.add_argument('--thread_num', type=int, default=8, help='thread for cpu') 

    parser.add_argument('--test_data', default='./test_data') 

    parser.add_argument('--model_config', default='', help='the file of model configuration')
    parser.add_argument('--init_model', default='./init_model')
    args = parser.parse_args()

    main(args)

