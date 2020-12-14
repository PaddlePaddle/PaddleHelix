#!/usr/bin/python                                                                                                                                                                                             
#-*-coding:utf-8-*- 
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
pretrain context pred
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

# Enable static graph mode.
paddle.enable_static()

from pahelix.model_zoo import PreGNNContextpredModel
from pahelix.datasets import load_zinc_dataset
from pahelix.featurizers import PreGNNContextPredFeaturizer
from pahelix.utils.paddle_utils import load_partial_params, get_distributed_optimizer
from pahelix.utils.splitters import RandomSplitter
from pahelix.utils.compound_tools import CompoundConstants


def train(args, exe, train_prog, model, train_dataset, featurizer):
    """
    Define the training function according to the given settings, calculate the training loss.

    Args:
        args,exe,train_prog,model,train_dataset,featurizer;
    Returns:
        the average of the list loss
    
    """
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
    """
    Define the evaluate function

    In the dataset, a proportion of labels are blank. So we use a `valid` tensor 
    to help eliminate these blank labels in both training and evaluation phase.

    """
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
    """
    Call the configuration function of the model, build the model and load data, then start training.

    model_config:
        a json file  with the  model configurations,such as dropout rate ,learning rate,num tasks and so on;

    context_pooling:
        it means the pooling type of context prediction;
    
    PreGNNContextpredModel:
        It is an unsupervised pretraining model which use subgraphs to predict their surrounding graph structures. Our goal is to pre-train a GNN so that it maps nodes appearing in similar structural contexts to nearby embeddings.

    """
    model_config = json.load(open(args.model_config, 'r'))
    if not args.dropout_rate is None:
        model_config['dropout_rate'] = args.dropout_rate
    model_config['context_pooling'] = args.context_pooling

    ### build model
    train_prog = fluid.Program()
    test_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            model = PreGNNContextpredModel(model_config)
            model.forward()
            opt = fluid.optimizer.Adam(learning_rate=args.lr)
            if args.distributed:
                opt = get_distributed_optimizer(opt)
            opt.minimize(model.loss)
    with fluid.program_guard(test_prog, fluid.Program()):
        with fluid.unique_name.guard():
            model = PreGNNContextpredModel(model_config)
            model.forward(is_test=True)

    # Use CUDAPlace for GPU training, or use CPUPlace for CPU training.
    place = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0))) \
            if args.use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if not args.init_model is None and not args.init_model == "":
        load_partial_params(exe, args.init_model, train_prog)

    ### load data
    # PreGNNContextPredFeaturizer:
    #     It is used along with `PreGNNContextPredModel`. It inherits from the super class `Featurizer` which is used for feature extractions. The `Featurizer` has two functions: `gen_features` for converting from a single raw smiles to a single graph data, `collate_fn` for aggregating a sublist of graph data into a big batch.
    # k is the number of layer,l1 and l2 are the different size of context,usually l1 < l2.
    # splitter:
    #     split type of the dataset:random,scaffold,random with scaffold. Here is randomsplit.
    #     `ScaffoldSplitter` will firstly order the compounds according to Bemis-Murcko scaffold, 
    #     then take the first `frac_train` proportion as the train set, the next `frac_valid` proportion as the valid set 
    #     and the rest as the test set. `ScaffoldSplitter` can better evaluate the generalization ability of the model on 
    #     out-of-distribution samples. Note that other splitters like `RandomSplitter`, `RandomScaffoldSplitter` 
    #     and `IndexSplitter` is also available."
    k = model_config['layer_num']
    l1 = k - 1
    l2 = l1 + args.context_size
    featurizer = PreGNNContextPredFeaturizer(
            model.substruct_graph_wrapper, 
            model.context_graph_wrapper, 
            k, l1, l2)
    dataset = load_zinc_dataset(args.data_path, featurizer=featurizer)

    splitter = RandomSplitter()
    train_dataset, _, test_dataset = splitter.split(
            dataset, frac_train=0.9, frac_valid=0, frac_test=0.1)
    if args.distributed:
        indices = list(range(fleet.worker_index(), len(train_dataset), fleet.worker_num()))
        train_dataset = train_dataset[indices]
    print("Train/Test num: %s/%s" % (len(train_dataset), len(test_dataset)))

    ### start train
    # Load the train function and calculate the train loss and test loss in each epoch.
    # Here we set the epoch is in range of max epoch,you can change it if you want. 

    # Then we will calculate the train loss ,test loss and print them.
    # Finally we save the best epoch to the model according to the dataset.
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
        best_epoch_id = np.argmin(list_test_loss)
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
    parser.add_argument("--data_path", type=str)

    parser.add_argument("--model_config", type=str)
    parser.add_argument("--dropout_rate", type=float)
    parser.add_argument("--init_model", type=str)
    parser.add_argument("--model_dir", type=str)

    parser.add_argument("--context_size", type=int, default=3)
    parser.add_argument("--context_pooling", type=str, default='average')
    args = parser.parse_args()
    
    main(args)

