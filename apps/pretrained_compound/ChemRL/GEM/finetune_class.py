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
from src.utils import get_dataset, create_splitter, get_downstream_task_names, \
        calc_rocauc_score, exempt_parameters


def train(args, model, train_dataset, collate_fn, criterion, encoder_opt, head_opt):
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
    for atom_bond_graphs, bond_angle_graphs, valids, labels in data_gen:
        if len(labels) < args.batch_size * 0.5:
            continue
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        labels = paddle.to_tensor(labels, 'float32')
        valids = paddle.to_tensor(valids, 'float32')
        preds = model(atom_bond_graphs, bond_angle_graphs)
        loss = criterion(preds, labels)
        loss = paddle.sum(loss * valids) / paddle.sum(valids)
        loss.backward()
        encoder_opt.step()
        head_opt.step()
        encoder_opt.clear_grad()
        head_opt.clear_grad()
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
    total_pred = []
    total_label = []
    total_valid = []
    model.eval()
    for atom_bond_graphs, bond_angle_graphs, valids, labels in data_gen:
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        labels = paddle.to_tensor(labels, 'float32')
        valids = paddle.to_tensor(valids, 'float32')
        preds = model(atom_bond_graphs, bond_angle_graphs)
        total_pred.append(preds.numpy())
        total_valid.append(valids.numpy())
        total_label.append(labels.numpy())
    total_pred = np.concatenate(total_pred, 0)
    total_label = np.concatenate(total_label, 0)
    total_valid = np.concatenate(total_valid, 0)
    return calc_rocauc_score(total_label, total_pred, total_valid)


def get_pos_neg_ratio(dataset):
    """tbd"""
    labels = np.array([data['label'] for data in dataset])
    return np.mean(labels == 1), np.mean(labels == -1)


def main(args):
    """
    Call the configuration function of the model, build the model and load data, then start training.
    model_config:
        a json file  with the hyperparameters,such as dropout rate ,learning rate,num tasks and so on;
    num_tasks:
        it means the number of task that each dataset contains, it's related to the dataset;
    """
    compound_encoder_config = load_json_config(args.compound_encoder_config)
    if not args.dropout_rate is None:
        compound_encoder_config['dropout_rate'] = args.dropout_rate

    model_config = load_json_config(args.model_config)
    if not args.dropout_rate is None:
        model_config['dropout_rate'] = args.dropout_rate
    task_names = get_downstream_task_names(args.dataset_name, args.data_path)
    model_config['task_type'] = 'class'
    model_config['num_tasks'] = len(task_names)

    ### build model
    compound_encoder = GeoGNNModel(compound_encoder_config)
    model = DownstreamModel(model_config, compound_encoder)
    criterion = nn.BCELoss(reduction='none')
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
    # featurizer:
    #     Gen features according to the raw data and return the graph data.
    #     Collate features about the graph data and return the feed dictionary.
    # splitter:
    #     split type of the dataset:random,scaffold,random with scaffold. Here is randomsplit.
    #     `ScaffoldSplitter` will firstly order the compounds according to Bemis-Murcko scaffold, 
    #     then take the first `frac_train` proportion as the train set, the next `frac_valid` proportion as the valid set 
    #     and the rest as the test set. `ScaffoldSplitter` can better evaluate the generalization ability of the model on 
    #     out-of-distribution samples. Note that other splitters like `RandomSplitter`, `RandomScaffoldSplitter` 
    #     and `IndexSplitter` is also available."
    
    if args.task == 'data':
        print('Preprocessing data...')
        dataset = get_dataset(args.dataset_name, args.data_path, task_names)
        dataset.transform(DownstreamTransformFn(), num_workers=args.num_workers)
        dataset.save_data(args.cached_data_path)
        return
    else:
        if args.cached_data_path is None or args.cached_data_path == "":
            print('Processing data...')
            dataset = get_dataset(args.dataset_name, args.data_path, task_names)
            dataset.transform(DownstreamTransformFn(), num_workers=args.num_workers)
        else:
            print('Read preprocessing data...')
            dataset = InMemoryDataset(npz_data_path=args.cached_data_path)

    splitter = create_splitter(args.split_type)
    train_dataset, valid_dataset, test_dataset = splitter.split(
            dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    print("Train/Valid/Test num: %s/%s/%s" % (
            len(train_dataset), len(valid_dataset), len(test_dataset)))
    print('Train pos/neg ratio %s/%s' % get_pos_neg_ratio(train_dataset))
    print('Valid pos/neg ratio %s/%s' % get_pos_neg_ratio(valid_dataset))
    print('Test pos/neg ratio %s/%s' % get_pos_neg_ratio(test_dataset))

    ### start train
    # Load the train function and calculate the train loss in each epoch.
    # Here we set the epoch is in range of max epoch,you can change it if you want. 

    # Then we will calculate the train loss ,valid auc,test auc and print them.
    # Finally we save it to the model according to the dataset.
    list_val_auc, list_test_auc = [], []
    collate_fn = DownstreamCollateFn(
            atom_names=compound_encoder_config['atom_names'], 
            bond_names=compound_encoder_config['bond_names'],
            bond_float_names=compound_encoder_config['bond_float_names'],
            bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
            task_type='class')
    for epoch_id in range(args.max_epoch):
        train_loss = train(args, model, train_dataset, collate_fn, criterion, encoder_opt, head_opt)
        val_auc = evaluate(args, model, valid_dataset, collate_fn)
        test_auc = evaluate(args, model, test_dataset, collate_fn)

        list_val_auc.append(val_auc)
        list_test_auc.append(test_auc)
        test_auc_by_eval = list_test_auc[np.argmax(list_val_auc)]
        print("epoch:%s train/loss:%s" % (epoch_id, train_loss))
        print("epoch:%s val/auc:%s" % (epoch_id, val_auc))
        print("epoch:%s test/auc:%s" % (epoch_id, test_auc))
        print("epoch:%s test/auc_by_eval:%s" % (epoch_id, test_auc_by_eval))
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
        'exp_id': args.exp_id,
    }
    offset = 20
    best_epoch_id = np.argmax(list_val_auc[offset:]) + offset
    for metric, value in [
            ('test_auc', list_test_auc[best_epoch_id]),
            ('max_valid_auc', np.max(list_val_auc)),
            ('max_test_auc', np.max(list_test_auc))]:
        outs['metric'] = metric
        print('\t'.join(['FINAL'] + ["%s:%s" % (k, outs[k]) for k in outs] + [str(value)]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=['train', 'data'], default='train')

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--dataset_name", 
            choices=['bace', 'bbbp', 'clintox', 'hiv', 
                'muv', 'sider', 'tox21', 'toxcast'])
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
    parser.add_argument("--exp_id", type=int, help='used for identification only')
    args = parser.parse_args()
    
    main(args)
