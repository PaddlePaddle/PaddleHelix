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
import paddle
import paddle.fluid as fluid
import sys
import time
from pahelix.utils.paddle_utils import load_partial_params
from pahelix.utils.protein_tools import ProteinTokenizer
from utils import *
from data_gen import gen_batch_data
from tape_model import TAPEModel


def show_results(examples, pred, task):
    """
    Show the results.
    """
    if task == 'classification':
        pred_label = pred.argmax(axis=1)
        for (i, example) in enumerate(examples):
            print('%s: %.3f' % (example, pred_label[i]))
    elif task == 'seq_classification':
        pred_label = pred.argmax(axis=1)
        offset = 0
        for example in examples:
            print('%s: ' % example)
            cur_pred_label = pred_label[offset + 1: offset + len(example) + 1]
            print('%s' % ' '.join([str(val) for val in cur_pred_label]))
            offset += len(example) + 2
    elif task == 'regression':
        for (i, example) in enumerate(examples):
            print('%s: %.3f' % (example, pred[i]))

def main(args):
    """main"""
    paddle.enable_static()

    model_config = json.load(open(args.model_config, 'r'))

    exe_params = default_exe_params(False, args.use_cuda, args.thread_num)
    exe = exe_params['exe']
    gpu_id = exe_params['gpu_id']
    if args.use_cuda:
        place = fluid.CUDAPlace(gpu_id)
    else:
        place = fluid.CPUPlace()

    task = model_config['task']

    model = TAPEModel(model_config=model_config, task=task)

    test_program = fluid.Program()
    test_startup = fluid.Program()
    with fluid.program_guard(test_program, test_startup):
        with fluid.unique_name.guard():
            model.forward(True)
            exe.run(test_startup)

    if not args.init_model is None and args.init_model != "":
        load_partial_params(exe, args.init_model, test_program)
    else:
        raise RuntimeError('Please set init_model.')

    tokenizer = ProteinTokenizer()
    test_fetch_list = model.get_fetch_list(is_inference=True)

    examples = []
    for line in sys.stdin:
        if len(line.strip()) == 0:
            continue
        examples.append(line.strip())

    for i in range(0, len(examples), args.batch_size):
        inputs = gen_batch_data(examples[i: min(len(examples), i + args.batch_size)], tokenizer, place)
        results = exe.run(
                program=test_program,
                feed=inputs,
                fetch_list=test_fetch_list,
                return_numpy=False)
        pred = np.array(results[0])
        print(pred)
        show_results(examples, pred, task)
    

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
