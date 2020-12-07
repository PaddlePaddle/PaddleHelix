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
pretrain attribute mask
"""

import os
from os.path import join, exists
import sys
import json
import argparse
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.collective import fleet

paddle.enable_static()

from pahelix.model_zoo import PreGNNAttrmaskModel
from pahelix.datasets import load_zinc_dataset
from pahelix.featurizer import PreGNNAttrMaskFeaturizer
from pahelix.utils.paddle_utils import load_partial_params, get_distributed_optimizer
from pahelix.utils.splitters import RandomSplitter
from pahelix.utils.compound_tools import CompoundConstants


def train(args, exe, train_prog, model, train_dataset, featurizer):
    """tbd"""
    data_gen = train_dataset.iter_batch(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=True,
            collate_fn=featurizer.collate_fn)
    list_loss = []
    for batch_id, feed_dict in enumerate(data_gen):
        train_loss, = exe.run(train_prog, 
                feed=feed_dict, fetch_list=[model.loss], return_numpy=False)
        list_loss.append(np.array(train_loss).mean())
    return np.mean(list_loss)


def evaluate(args, exe, test_prog, model, test_dataset, featurizer):
    """tbd"""
    data_gen = test_dataset.iter_batch(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=True,
            collate_fn=featurizer.collate_fn)
    list_loss = []
    for batch_id, feed_dict in enumerate(data_gen):
        test_loss, = exe.run(test_prog, 
                feed=feed_dict, fetch_list=[model.loss], return_numpy=False)
        list_loss.append(np.array(test_loss).mean())
    return np.mean(list_loss)


def main(args):
    """tbd"""
    model_config = json.load(open(args.model_config, 'r'))

    ### build model
    train_prog = fluid.Program()
    test_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            model = PreGNNAttrmaskModel(model_config)
            model.forward()
            opt = fluid.optimizer.Adam(learning_rate=args.lr)
            if args.distributed:
                opt = get_distributed_optimizer(opt)
            opt.minimize(model.loss)
    with fluid.program_guard(test_prog, fluid.Program()):
        with fluid.unique_name.guard():
            model = PreGNNAttrmaskModel(model_config)
            model.forward(is_test=True)

    place = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0))) \
            if args.use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if not args.init_model is None and not args.init_model == "":
        load_partial_params(exe, args.init_model, train_prog)

    ### load data
    featurizer = PreGNNAttrMaskFeaturizer(
            model.graph_wrapper, 
            atom_type_num=len(CompoundConstants.atom_num_list),
            mask_ratio=args.mask_ratio)
    dataset = load_zinc_dataset(args.data_path, featurizer=featurizer)

    splitter = RandomSplitter()
    train_dataset, _, test_dataset = splitter.split(
            dataset, frac_train=0.9, frac_valid=0, frac_test=0.1)
    if args.distributed:
        indices = list(range(fleet.worker_num(), len(train_dataset), fleet.worker_index()))
        train_dataset = train_dataset[indices]
    print("Train/Test num: %s/%s" % (len(train_dataset), len(test_dataset)))

    ### start train
    list_test_loss = []
    for epoch_id in range(args.max_epoch):
        train_loss = train(args, exe, train_prog, model, train_dataset, featurizer)
        test_loss = evaluate(args, exe, test_prog, model, test_dataset, featurizer)
        if not args.distributed or fleet.worker_index() == 0:
            fluid.io.save_params(exe, '%s/epoch%s' % (args.model_dir, epoch_id), train_prog)
            list_test_loss.append(test_loss)
            print("epoch:%d train/loss:%s" % (epoch_id, train_loss))
            print("epoch:%d test/loss:%s" % (epoch_id, test_loss))

    if not args.distributed or fleet.worker_index() == 0:
        best_epoch_id = np.argmax(list_test_loss)
        fluid.io.load_params(exe, '%s/epoch%d' % (args.model_dir, best_epoch_id), train_prog)
        fluid.io.save_params(exe, '%s/epoch_best' % (args.model_dir), train_prog)
        return list_test_loss[best_epoch_id]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cuda", action='store_true', default=False)
    parser.add_argument("--distributed", action='store_true', default=False)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data_path", type=str, default=None)

    parser.add_argument("--model_config", type=str)
    parser.add_argument("--dropout_rate", type=float)
    parser.add_argument("--init_model", type=str)
    parser.add_argument("--model_dir", type=str)

    parser.add_argument("--mask_ratio", type=float, default=0.15)
    args = parser.parse_args()
    
    main(args)

