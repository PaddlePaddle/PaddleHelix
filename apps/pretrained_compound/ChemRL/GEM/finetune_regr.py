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
Finetune:to do some downstream task
"""

import os
from os.path import join, exists, basename
import argparse
import numpy as np

import paddle
import paddle.nn as nn
import pgl

from pahelix.model_zoo.gem_model import GeoGNNModel
from pahelix.utils import load_json_config
from pahelix.datasets.inmemory_dataset import InMemoryDataset

from src.model import DownstreamModel
from src.featurizer import DownstreamTransformFn, DownstreamCollateFn
from src.utils import get_dataset, create_splitter, get_downstream_task_names, get_dataset_stat, \
        calc_rocauc_score, calc_rmse, calc_mae, exempt_parameters


def train(
        args,
        model, label_mean, label_std,
        train_dataset, collate_fn,
        criterion, encoder_opt, head_opt):
    """
    Define the train function 
    Args:
        args,model,train_dataset,collate_fn,criterion,encoder_opt,head_opt;
    Returns:
        the average of the list loss
    """
    data_gen = train_dataset.get_data_loader(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            collate_fn=collate_fn)
    list_loss = []
    model.train()
    for atom_bond_graphs, bond_angle_graphs, labels in data_gen:
        if len(labels) < args.batch_size * 0.5:
            continue
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        scaled_labels = (labels - label_mean) / (label_std + 1e-5)
        scaled_labels = paddle.to_tensor(scaled_labels, 'float32')
        preds = model(atom_bond_graphs, bond_angle_graphs)
        loss = criterion(preds, scaled_labels)
        loss.backward()
        encoder_opt.step()
        head_opt.step()
        encoder_opt.clear_grad()
        head_opt.clear_grad()
        list_loss.append(loss.numpy())
    return np.mean(list_loss)


def evaluate(
        args,
        model, label_mean, label_std,
        test_dataset, collate_fn, metric):
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
    total_pred = []
    total_label = []
    model.eval()
    for atom_bond_graphs, bond_angle_graphs, labels in data_gen:
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        labels = paddle.to_tensor(labels, 'float32')
        scaled_preds = model(atom_bond_graphs, bond_angle_graphs)
        preds = scaled_preds.numpy() * label_std + label_mean
        total_pred.append(preds)
        total_label.append(labels.numpy())
    total_pred = np.concatenate(total_pred, 0)
    total_label = np.concatenate(total_label, 0)
    if metric == 'rmse':
        return calc_rmse(total_label, total_pred)
    else:
        return calc_mae(total_label, total_pred)


def get_label_stat(dataset):
    """tbd"""
    labels = np.array([data['label'] for data in dataset])
    return np.min(labels), np.max(labels), np.mean(labels)


def get_metric(dataset_name):
    """tbd"""
    if dataset_name in ['esol', 'freesolv', 'lipophilicity']:
        return 'rmse'
    elif dataset_name in ['qm7', 'qm8', 'qm9', 'qm9_gdb']:
        return 'mae'
    else:
        raise ValueError(dataset_name)


def main(args):
    """
    Call the configuration function of the model, build the model and load data, then start training.
    model_config:
        a json file  with the hyperparameters,such as dropout rate ,learning rate,num tasks and so on;
    num_tasks:
        it means the number of task that each dataset contains, it's related to the dataset;
    """
    ### config for the body
    compound_encoder_config = load_json_config(args.compound_encoder_config)
    if not args.dropout_rate is None:
        compound_encoder_config['dropout_rate'] = args.dropout_rate

    ### config for the downstream task
    task_type = 'regr'
    metric = get_metric(args.dataset_name)
    task_names = get_downstream_task_names(args.dataset_name, args.data_path)
    dataset_stat = get_dataset_stat(args.dataset_name, args.data_path, task_names)
    label_mean = np.reshape(dataset_stat['mean'], [1, -1])
    label_std = np.reshape(dataset_stat['std'], [1, -1])

    model_config = load_json_config(args.model_config)
    if not args.dropout_rate is None:
        model_config['dropout_rate'] = args.dropout_rate
    model_config['task_type'] = task_type
    model_config['num_tasks'] = len(task_names)
    print('model_config:')
    print(model_config)

    ### build model
    compound_encoder = GeoGNNModel(compound_encoder_config)
    model = DownstreamModel(model_config, compound_encoder)
    if metric == 'square':
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()
    encoder_params = compound_encoder.parameters()
    head_params = exempt_parameters(model.parameters(), encoder_params)
    encoder_opt = paddle.optimizer.Adam(args.encoder_lr, parameters=encoder_params)
    head_opt = paddle.optimizer.Adam(args.head_lr, parameters=head_params)
    print('Total param num: %s' % (len(model.parameters())))
    print('Encoder param num: %s' % (len(encoder_params)))
    print('Head param num: %s' % (len(head_params)))
    for i, param in enumerate(model.named_parameters()):
        print(i, param[0], param[1].name)

    if not args.init_model is None and not args.init_model == "":
        compound_encoder.set_state_dict(paddle.load(args.init_model))
        print('Load state_dict from %s' % args.init_model)

    ### load data
    if args.task == 'data':
        print('Preprocessing data...')
        dataset = get_dataset(args.dataset_name, args.data_path, task_names)
        transform_fn = DownstreamTransformFn()
        dataset.transform(transform_fn, num_workers=args.num_workers)
        dataset.save_data(args.cached_data_path)
        return
    else:
        if args.cached_data_path is None or args.cached_data_path == "":
            print('Processing data...')
            dataset = get_dataset(args.dataset_name, args.data_path, task_names)
            transform_fn = DownstreamTransformFn()
            dataset.transform(transform_fn, num_workers=args.num_workers)
        else:
            print('Read preprocessing data...')
            dataset = InMemoryDataset(npz_data_path=args.cached_data_path)

    splitter = create_splitter(args.split_type)
    train_dataset, valid_dataset, test_dataset = splitter.split(
            dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    print("Train/Valid/Test num: %s/%s/%s" % (
            len(train_dataset), len(valid_dataset), len(test_dataset)))
    print('Train min/max/mean %s/%s/%s' % get_label_stat(train_dataset))
    print('Valid min/max/mean %s/%s/%s' % get_label_stat(valid_dataset))
    print('Test min/max/mean %s/%s/%s' % get_label_stat(test_dataset))

    ### start train
    list_val_metric, list_test_metric = [], []
    collate_fn = DownstreamCollateFn(
            atom_names=compound_encoder_config['atom_names'],
            bond_names=compound_encoder_config['bond_names'],
            bond_float_names=compound_encoder_config['bond_float_names'],
            bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
            task_type=task_type)
    for epoch_id in range(args.max_epoch):
        train_loss = train(
                args, model, label_mean, label_std,
                train_dataset, collate_fn,
                criterion, encoder_opt, head_opt)
        val_metric = evaluate(
                args, model, label_mean, label_std,
                valid_dataset, collate_fn, metric)
        test_metric = evaluate(
                args, model, label_mean, label_std,
                test_dataset, collate_fn, metric)

        list_val_metric.append(val_metric)
        list_test_metric.append(test_metric)
        test_metric_by_eval = list_test_metric[np.argmin(list_val_metric)]
        print("epoch:%s train/loss:%s" % (epoch_id, train_loss))
        print("epoch:%s val/%s:%s" % (epoch_id, metric, val_metric))
        print("epoch:%s test/%s:%s" % (epoch_id, metric, test_metric))
        print("epoch:%s test/%s_by_eval:%s" % (epoch_id, metric, test_metric_by_eval))
        paddle.save(compound_encoder.state_dict(),
                '%s/epoch%d/compound_encoder.pdparams' % (args.model_dir, epoch_id))
        paddle.save(model.state_dict(),
                '%s/epoch%d/model.pdparams' % (args.model_dir, epoch_id))

    outs = {
        'model_config': basename(args.model_config).replace('.json', ''),
        'metric': '',
        'dataset': args.dataset_name,
        'split_type': args.split_type,
        'batch_size': args.batch_size,
        'dropout_rate': args.dropout_rate,
        'encoder_lr': args.encoder_lr,
        'head_lr': args.head_lr,
    }
    best_epoch_id = np.argmin(list_val_metric)
    for metric, value in [
            ('test_%s' % metric, list_test_metric[best_epoch_id]),
            ('max_valid_%s' % metric, np.min(list_val_metric)),
            ('max_test_%s' % metric, np.min(list_test_metric))]:
        outs['metric'] = metric
        print('\t'.join(['FINAL'] + ["%s:%s" % (k, outs[k]) for k in outs] + [str(value)]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=['train', 'data'], default='train')

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--dataset_name",
            choices=['esol', 'freesolv', 'lipophilicity',
                'qm7', 'qm8', 'qm9', 'qm9_gdb'])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--cached_data_path", type=str, default=None)
    parser.add_argument("--split_type",
            choices=['random', 'scaffold', 'random_scaffold', 'index'])

    parser.add_argument("--compound_encoder_config", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--init_model", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--encoder_lr", type=float, default=0.001)
    parser.add_argument("--head_lr", type=float, default=0.001)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    args = parser.parse_args()
    
    main(args)
