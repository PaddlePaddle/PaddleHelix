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
Train sequence-based models for protein.
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
    paddle.enable_static()

    model_config = json.load(open(args.model_config, 'r'))

    exe_params = default_exe_params(args.is_distributed, args.use_cuda, args.thread_num)
    exe = exe_params['exe']
    trainer_num = exe_params['trainer_num']
    trainer_id = exe_params['trainer_id']
    dist_strategy = exe_params['dist_strategy']
    places = exe_params['places']

    task = model_config['task']

    model = TAPEModel(model_config=model_config, name=task)

    train_program = fluid.Program()
    train_startup = fluid.Program()
    with fluid.program_guard(train_program, train_startup):
        with fluid.unique_name.guard():
            model.forward(False)
            model.cal_loss()

            optimizer = default_optimizer(args.lr, args.warmup_steps, args.max_grad_norm)
            setup_optimizer(optimizer, model, args.use_cuda, args.is_distributed, dist_strategy)

            optimizer.minimize(model.loss)

            train_data_loader = setup_data_loader(
                    model.input_list,
                    model_config,
                    args.train_data,
                    trainer_id,
                    trainer_num,
                    places,
                    args.batch_size)
            exe.run(train_startup)

    train_metric = get_metric(task)
    train_fetch_list = model.get_fetch_list()

    if args.test_data is not None:
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
        test_fetch_list = model.get_fetch_list()

    if args.init_model is not None and args.init_model != "":
        load_partial_params(exe, args.init_model, train_program)
        load_partial_params(exe, args.init_model, test_program)

    if not args.is_distributed:
        train_program = fluid.compiler.CompiledProgram(train_program).with_data_parallel(
                loss_name=model.loss.name)
        if args.test_data is not None and args.test_data != "":
            test_program = fluid.compiler.CompiledProgram(test_program).with_data_parallel(
                    loss_name=model.loss.name)

    for epoch_id in range(args.max_epoch):
        print(time.time(), 'Start epoch %d' % epoch_id)
        print('Train:')
        train_metric.clear()
        for data in train_data_loader():
            results = exe.run(
                    program=train_program,
                    feed=data,
                    fetch_list=train_fetch_list,
                    return_numpy=False)
            update_metric(task, train_metric, results)
            train_metric.show()
        print()
        
        if args.test_data is not None and args.test_data != "":
            print('Test:')
            test_metric.clear()
            for data in test_data_loader():
                results = exe.run(
                        program=test_program,
                        feed=data,
                        fetch_list=test_fetch_list,
                        return_numpy=False)
                update_metric(task, test_metric, results)
            test_metric.show()
            print()
        
        if trainer_id == 0:
            print(time.time(), "Save model epoch%d." % epoch_id)

            is_exist = os.path.exists(args.model_dir)
            if not is_exist:
                os.makedirs(args.model_dir)
            fluid.io.save_params(exe, '%s/epoch%d' % (args.model_dir, epoch_id), train_program)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cuda", action='store_true', default=False)
    parser.add_argument('--distributed', dest='is_distributed', action='store_true')

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument('--thread_num', type=int, default=8, help='thread for cpu') 

    parser.add_argument("--train_data", type=str)
    parser.add_argument("--test_data", type=str)

    parser.add_argument("--model_config", type=str)
    parser.add_argument("--init_model", type=str)
    parser.add_argument("--model_dir", type=str)
    args = parser.parse_args()

    main(args)
