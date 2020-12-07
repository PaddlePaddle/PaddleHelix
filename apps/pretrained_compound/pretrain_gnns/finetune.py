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
finetune
"""

import os
from os.path import join, exists
import json
import argparse
import numpy as np

import paddle
import paddle.fluid as fluid

paddle.enable_static()

from pahelix.utils.paddle_utils import load_partial_params

from model import DownstreamModel
from featurizer import DownstreamFeaturizer
from utils import get_dataset, create_splitter, get_downstream_task_names, calc_rocauc_score


def train(args, exe, train_prog, model, train_dataset, featurizer):
    """tbd"""
    data_gen = train_dataset.iter_batch(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=True,
            collate_fn=featurizer.collate_fn)
    list_loss = []
    for batch_id, feed_dict in enumerate(data_gen):
        train_loss, = exe.run(train_prog, feed=feed_dict, fetch_list=[model.loss], return_numpy=False)
        list_loss.append(np.array(train_loss).mean())
    return np.mean(list_loss)


def evaluate(args, exe, test_prog, model, test_dataset, featurizer):
    """tbd"""
    data_gen = test_dataset.iter_batch(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=True,
            collate_fn=featurizer.collate_fn)
    total_pred = []
    total_label = []
    total_valid = []
    for batch_id, feed_dict in enumerate(data_gen):
        pred, = exe.run(test_prog, feed=feed_dict, fetch_list=[model.pred], return_numpy=False)
        total_pred.append(np.array(pred))
        total_label.append(feed_dict['finetune_label'])
        total_valid.append(feed_dict['valid'])
    total_pred = np.concatenate(total_pred, 0)
    total_label = np.concatenate(total_label, 0)
    total_valid = np.concatenate(total_valid, 0)
    return calc_rocauc_score(total_label, total_pred, total_valid)


def main(args):
    """tbd"""
    model_config = json.load(open(args.model_config, 'r'))
    task_names = get_downstream_task_names(args.dataset_name, args.data_path)
    model_config['num_tasks'] = len(task_names)

    ### build model
    train_prog = fluid.Program()
    test_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            model = DownstreamModel(model_config)
            model.forward()
            opt = fluid.optimizer.Adam(learning_rate=args.lr)
            opt.minimize(model.loss)
    with fluid.program_guard(test_prog, fluid.Program()):
        with fluid.unique_name.guard():
            model = DownstreamModel(model_config)
            model.forward(is_test=True)

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if not args.init_model is None and not args.init_model == "":
        load_partial_params(exe, args.init_model, train_prog)

    ### load data
    featurizer = DownstreamFeaturizer(model.graph_wrapper)
    dataset = get_dataset(
            args.dataset_name, args.data_path, task_names, featurizer)
    splitter = create_splitter(args.split_type)
    train_dataset, valid_dataset, test_dataset = splitter.split(
            dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    print("Train/Valid/Test num: %s/%s/%s" % (
            len(train_dataset), len(valid_dataset), len(test_dataset)))

    ### start train
    list_val_auc, list_test_auc = [], []
    for epoch_id in range(args.max_epoch):
        train_loss = train(args, exe, train_prog, model, train_dataset, featurizer)
        val_auc = evaluate(args, exe, test_prog, model, valid_dataset, featurizer)
        test_auc = evaluate(args, exe, test_prog, model, test_dataset, featurizer)

        list_val_auc.append(val_auc)
        list_test_auc.append(test_auc)
        test_auc_by_eval = list_test_auc[np.argmax(list_val_auc)]
        print("epoch:%s train/loss:%s" % (epoch_id, train_loss))
        print("epoch:%s val/auc:%s" % (epoch_id, val_auc))
        print("epoch:%s test/auc:%s" % (epoch_id, test_auc))
        print("epoch:%s test/auc_by_eval:%s" % (epoch_id, test_auc_by_eval))
        fluid.io.save_params(exe, '%s/epoch%d' % (args.model_dir, epoch_id), train_prog)

    best_epoch_id = np.argmax(list_val_auc)
    fluid.io.load_params(exe, '%s/epoch%d' % (args.model_dir, best_epoch_id), train_prog)
    fluid.io.save_params(exe, '%s/epoch_best' % (args.model_dir), train_prog)
    return list_test_auc[best_epoch_id]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cuda", action='store_true', default=False)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset_name", 
            choices=['bace', 'bbbp', 'clintox', 'hiv', 
                'muv', 'sider', 'tox21', 'toxcast'])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--split_type", 
            choices=['random', 'scaffold', 'random_scaffold', 'index'])

    parser.add_argument("--model_config", type=str)
    parser.add_argument("--dropout_rate", type=float)
    parser.add_argument("--init_model", type=str)
    parser.add_argument("--model_dir", type=str)
    args = parser.parse_args()
    
    main(args)
