#!/usr/bin/python
#-*-coding:utf-8-*-
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
"""
Reproduction of paper Pretrain GNNs
"""

import os
from os.path import join, exists
import json
import argparse
import numpy as np

import paddle
import paddle.nn as nn
import paddle.distributed as dist
import pgl

from pahelix.model_zoo.pretrain_gnns_model import PretrainGNNModel, AttrmaskModel
from pahelix.datasets.zinc_dataset import load_zinc_dataset
from pahelix.utils.splitters import RandomSplitter
from pahelix.featurizers.pretrain_gnn_featurizer import AttrmaskTransformFn, AttrmaskCollateFn
from pahelix.utils import load_json_config


def train(args, model, train_dataset, collate_fn, opt):
    """
    Define the training function according to the given settings, calculate the training loss.
    Args:
        args,model,train_dataset,collate_fn,opt;
    Returns:
        the average of the list loss.
    """
    data_gen = train_dataset.get_data_loader(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            collate_fn=collate_fn)
    list_loss = []
    model.train()
    for graphs, masked_node_indice, masked_node_label in data_gen:
        graphs = graphs.tensor()
        masked_node_indice = paddle.to_tensor(masked_node_indice, 'int64')
        masked_node_label = paddle.to_tensor(masked_node_label, 'int64')
        loss = model(graphs, masked_node_indice, masked_node_label)
        loss.backward()
        opt.step()
        opt.clear_grad()
        list_loss.append(loss.numpy())
    return np.mean(list_loss)


def evaluate(args, model, test_dataset, collate_fn):
    """
    Define the evaluate function
    In the dataset, a proportion of labels are blank. So we use a `valid` tensor 
    to help eliminate these blank labels in both training and evaluation phase.
    """
    data_gen = test_dataset.get_data_loader(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=collate_fn)
    list_loss = []
    model.eval()
    for graphs, masked_node_indice, masked_node_label in data_gen:
        graphs = graphs.tensor()
        masked_node_indice = paddle.to_tensor(masked_node_indice, 'int64')
        masked_node_label = paddle.to_tensor(masked_node_label, 'int64')
        loss = model(graphs, masked_node_indice, masked_node_label)
        list_loss.append(loss.numpy())
    return np.mean(list_loss)


def main(args):
    """
    Call the configuration function of the compound encoder and model, 
    build the model and load data, then start training.

    compound_encoder_config: a json file with the compound encoder configurations,
    such as dropout rate ,learning rate,num tasks and so on;

    model_config: a json file with the pretrain_gnn model configurations,such as dropout rate ,
    learning rate,num tasks and so on;

    lr: It means the learning rate of different optimizer;
    
    AttrmaskModel: It is an unsupervised pretraining model which randomly masks the atom type 
    of some node and then use the masked atom type as the prediction targets.
    """
    if args.dist:
        dist.init_parallel_env()

    compound_encoder_config = load_json_config(args.compound_encoder_config)
    model_config = load_json_config(args.model_config)
    if not args.dropout_rate is None:
        compound_encoder_config['dropout_rate'] = args.dropout_rate
        model_config['dropout_rate'] = args.dropout_rate

    ### build model
    compound_encoder = PretrainGNNModel(compound_encoder_config)
    model = AttrmaskModel(model_config, compound_encoder)
    if args.dist:
        model = paddle.DataParallel(model)
    opt = paddle.optimizer.Adam(args.lr, parameters=model.parameters())

    if not args.init_model is None and not args.init_model == "":
        compound_encoder.set_state_dict(paddle.load(args.init_model))
        print('Load state_dict from %s' % args.init_model)

    ### load data
    dataset = load_zinc_dataset(args.data_path)
    splitter = RandomSplitter()
    train_dataset, _, test_dataset = splitter.split(
            dataset, frac_train=0.9, frac_valid=0.0, frac_test=0.1, seed=32)
    if args.dist:
        train_dataset = train_dataset[dist.get_rank()::dist.get_world_size()]
    transform_fn = AttrmaskTransformFn()
    train_dataset.transform(transform_fn, num_workers=args.num_workers)
    test_dataset.transform(transform_fn, num_workers=args.num_workers)
    print("Train/Test num: %s/%s" % (len(train_dataset), len(test_dataset)))

    ### start train
    collate_fn = AttrmaskCollateFn(
            atom_names=compound_encoder_config['atom_names'],
            bond_names=compound_encoder_config['bond_names'],
            mask_ratio=model_config['mask_ratio'])
    for epoch_id in range(args.max_epoch):
        train_loss = train(args, model, train_dataset, collate_fn, opt)
        test_loss = evaluate(args, model, test_dataset, collate_fn)
        if not args.dist or dist.get_rank() == 0:
            print("epoch:%d train/loss:%s" % (epoch_id, train_loss))
            print("epoch:%d test/loss:%s" % (epoch_id, test_loss))
            paddle.save(compound_encoder.state_dict(),
                    '%s/epoch%d/compound_encoder.pdparams' % (args.model_dir, epoch_id))
            paddle.save(model.state_dict(),
                    '%s/epoch%d/model.pdparams' % (args.model_dir, epoch_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", action='store_true', default=False)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--data_path", type=str, default=None)

    parser.add_argument("--compound_encoder_config", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--init_model", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout_rate", type=float)
    args = parser.parse_args()
    
    main(args)

