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
GEM pretrain
"""

import os
from os.path import join, exists, basename
import sys
import argparse
import time
import numpy as np
from glob import glob
import logging

import paddle
import paddle.distributed as dist

from pahelix.datasets.inmemory_dataset import InMemoryDataset
from pahelix.utils import load_json_config
from pahelix.featurizers.gem_featurizer import GeoPredTransformFn, GeoPredCollateFn
from pahelix.model_zoo.gem_model import GeoGNNModel, GeoPredModel
from src.utils import exempt_parameters


def train(args, model, optimizer, data_gen):
    """tbd"""
    model.train()
    
    steps = get_steps_per_epoch(args)
    step = 0
    list_loss = []
    for graph_dict, feed_dict in data_gen:
        print('rank:%s step:%s' % (dist.get_rank(), step))
        # if dist.get_rank() == 1:
        #     time.sleep(100000)
        for k in graph_dict:
            graph_dict[k] = graph_dict[k].tensor()
        for k in feed_dict:
            feed_dict[k] = paddle.to_tensor(feed_dict[k])
        train_loss = model(graph_dict, feed_dict)
        train_loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        list_loss.append(train_loss.numpy().mean())
        step += 1
        if step > steps:
            print("jumpping out")
            break         
    return np.mean(list_loss)


@paddle.no_grad()
def evaluate(args, model, test_dataset, collate_fn):
    """tbd"""
    model.eval()
    data_gen = test_dataset.get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=True, 
            collate_fn=collate_fn)

    dict_loss = {'loss': []}
    for graph_dict, feed_dict in data_gen:
        for k in graph_dict:
            graph_dict[k] = graph_dict[k].tensor()
        for k in feed_dict:
            feed_dict[k] = paddle.to_tensor(feed_dict[k])
        loss, sub_losses = model(graph_dict, feed_dict, return_subloss=True)

        for name in sub_losses:
            if not name in dict_loss:
                dict_loss[name] = []
            v_np = sub_losses[name].numpy()
            dict_loss[name].append(v_np)
        dict_loss['loss'] = loss.numpy()
    dict_loss = {name: np.mean(dict_loss[name]) for name in dict_loss}
    return dict_loss


def get_steps_per_epoch(args):
    """tbd"""
    # add as argument
    if args.dataset == 'zinc':
        train_num = int(20000000 * (1 - args.test_ratio))
    else:
        raise ValueError(args.dataset)
    if args.DEBUG:
        train_num = 100
    steps_per_epoch = int(train_num / args.batch_size)
    if args.distributed:
        steps_per_epoch = int(steps_per_epoch / dist.get_world_size())
    return steps_per_epoch


def load_smiles_to_dataset(data_path):
    """tbd"""
    files = sorted(glob('%s/*' % data_path))
    data_list = []
    for file in files:
        with open(file, 'r') as f:
            tmp_data_list = [line.strip() for line in f.readlines()]
        data_list.extend(tmp_data_list)
    dataset = InMemoryDataset(data_list=data_list)
    return dataset


def main(args):
    """tbd"""
    compound_encoder_config = load_json_config(args.compound_encoder_config)
    model_config = load_json_config(args.model_config)
    if not args.dropout_rate is None:
        compound_encoder_config['dropout_rate'] = args.dropout_rate
        model_config['dropout_rate'] = args.dropout_rate

    compound_encoder = GeoGNNModel(compound_encoder_config)
    model = GeoPredModel(model_config, compound_encoder)
    if args.distributed:
        model = paddle.DataParallel(model)
    opt = paddle.optimizer.Adam(learning_rate=args.lr, parameters=model.parameters())
    print('Total param num: %s' % (len(model.parameters())))
    for i, param in enumerate(model.named_parameters()):
        print(i, param[0], param[1].name)

    if not args.init_model is None and not args.init_model == "":
        compound_encoder.set_state_dict(paddle.load(args.init_model))
        print('Load state_dict from %s' % args.init_model)
    
    # get dataset
    dataset = load_smiles_to_dataset(args.data_path)
    if args.DEBUG:
        dataset = dataset[100:180]
    dataset = dataset[dist.get_rank()::dist.get_world_size()]
    smiles_lens = [len(smiles) for smiles in dataset]
    print('Total size:%s' % (len(dataset)))
    print('Dataset smiles min/max/avg length: %s/%s/%s' % (
            np.min(smiles_lens), np.max(smiles_lens), np.mean(smiles_lens)))
    transform_fn = GeoPredTransformFn(model_config['pretrain_tasks'], model_config['mask_ratio'])
    # this step will be time consuming due to rdkit 3d calculation
    dataset.transform(transform_fn, num_workers=args.num_workers)
    test_index = int(len(dataset) * (1 - args.test_ratio))
    train_dataset = dataset[:test_index]
    test_dataset = dataset[test_index:]
    print("Train/Test num: %s/%s" % (len(train_dataset), len(test_dataset)))

    collate_fn = GeoPredCollateFn(
            atom_names=compound_encoder_config['atom_names'],
            bond_names=compound_encoder_config['bond_names'], 
            bond_float_names=compound_encoder_config['bond_float_names'],
            bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
            pretrain_tasks=model_config['pretrain_tasks'],
            mask_ratio=model_config['mask_ratio'],
            Cm_vocab=model_config['Cm_vocab'])
    train_data_gen = train_dataset.get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=True, 
            collate_fn=collate_fn)
    
    list_test_loss = []
    for epoch_id in range(args.max_epoch):
        s = time.time()
        train_loss = train(args, model, opt, train_data_gen)
        test_loss = evaluate(args, model, test_dataset, collate_fn)
        if not args.distributed or dist.get_rank() == 0:
            paddle.save(compound_encoder.state_dict(), 
                '%s/epoch%d.pdparams' % (args.model_dir, epoch_id))
            list_test_loss.append(test_loss['loss'])
            print("epoch:%d train/loss:%s" % (epoch_id, train_loss))
            print("epoch:%d test/loss:%s" % (epoch_id, test_loss))
            print("Time used:%ss" % (time.time() - s))
    
    if not args.distributed or dist.get_rank() == 0:
        print('Best epoch id:%s' % np.argmin(list_test_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", action='store_true', default=False)
    parser.add_argument("--distributed", action='store_true', default=False)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--dataset", type=str, default='zinc')
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--compound_encoder_config", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--init_model", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    args = parser.parse_args()

    if args.distributed:
        dist.init_parallel_env()

    main(args)