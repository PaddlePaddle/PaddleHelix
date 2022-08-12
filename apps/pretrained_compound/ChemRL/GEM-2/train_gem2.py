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
# Unless required by applicable law or agreed to in writing, sosftware
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, eitdher express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Finetune:to do some downstream task
"""


import os
from os.path import join, exists, basename
import argparse
import numpy as np
from glob import glob
import csv
import json
from copy import deepcopy
import ml_collections
import logging
import time

from tensorboardX import SummaryWriter

import paddle
import paddle.nn as nn

import paddle.distributed as dist
from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients

from pahelix.datasets.inmemory_dataset import InMemoryDataset

from src.dataset import PCQMv2Dataset
from src.model import MolRegressionModel
from src.featurizer import OptimusTransformFn, OptimusCollateFn
from src.paddle_utils import dist_mean, dist_length
from src.utils import calc_parameter_size, tree_map, set_logging_level, write_to_csv, add_to_data_writer
from src.utils import ResultsCollect
from src.ema import ExponentialMovingAverage2
from src.config import make_updated_config, OPTIMUS_MODEL_CONFIG, MOL_REGRESSION_MODEL_CONFIG


def get_optimizer(args, train_config, model):
    milestones = [
            train_config.mid_step, 
            train_config.mid_step + 12, 
            train_config.mid_step + 18, 
            train_config.mid_step + 24, 
            train_config.mid_step + 30]
    gamma = 0.5
    second_scheduler = paddle.optimizer.lr.MultiStepDecay(
            learning_rate=train_config.lr, 
            milestones=milestones,
            gamma=gamma)
    scheduler = paddle.optimizer.lr.LinearWarmup(
            learning_rate=second_scheduler,
            warmup_steps=train_config.warmup_step,
            start_lr=train_config.lr * 0.1,
            end_lr=train_config.lr,
            verbose=True)
    optimizer = paddle.optimizer.Adam(
            scheduler, 
            epsilon=1e-06,
            parameters=model.parameters())
    return optimizer, scheduler


def get_train_steps_per_epoch(dataset_len):
    if args.DEBUG:
        return 20
    # add as argument
    min_data_len = paddle.to_tensor(dataset_len)
    from paddle.distributed import ReduceOp
    dist.all_reduce(min_data_len, ReduceOp.MIN)
    dataset_len = min_data_len.numpy()[0]
    logging.info(f'min dataset len: {dataset_len}')
    return int(dataset_len / args.batch_size) - 5


def get_featurizer(model_config, encoder_config):
    featurizer_dict = {
        'optimus': (OptimusTransformFn(model_config, encoder_config),
                OptimusCollateFn(model_config, encoder_config))
    }
    return featurizer_dict[model_config.model.encoder_type]


def create_model_config(args):
    model_config = make_updated_config(
            MOL_REGRESSION_MODEL_CONFIG, 
            json.load(open(args.model_config, 'r')))

    encoder_config_dict = {
        'optimus': OPTIMUS_MODEL_CONFIG,
        'lite_optimus': OPTIMUS_MODEL_CONFIG,
    }
    encoder_config = make_updated_config(
            encoder_config_dict[model_config.model.encoder_type], 
            json.load(open(args.encoder_config, 'r')))
    return model_config, encoder_config


def load_data(args, trainer_id, trainer_num, model_config, dataset_config, transform_fn):
    raw_dataset = PCQMv2Dataset(dataset_config)

    cache_dir = args.data_cache_dir
    done_file = join(args.data_cache_dir, 'done')
    if not exists(done_file):
        if trainer_id == 0:
            logging.info('load dataset')
            dataset_dict = raw_dataset.load_dataset_dict()
            if args.DEBUG:
                dataset_dict = tree_map(lambda x: x[:8], dataset_dict)
            for name in dataset_dict:
                logging.info(f'transform {name} set')
                dataset_dict[name].transform(transform_fn, num_workers=8 if args.DEBUG else 140)
                dataset_dict[name].save_data(f'{cache_dir}/{name}')
        if args.distributed:
            dist.barrier()
        open(done_file, 'w').write('')

    logging.info(f'load npz')
    train_npz_files = sorted(glob(f'{cache_dir}/train/*npz'))
    valid_npz_files = sorted(glob(f'{cache_dir}/valid/*npz'))
    if args.DEBUG:
        train_npz_files = train_npz_files[:16]
        valid_npz_files = valid_npz_files[:8]
    train_dataset = InMemoryDataset(npz_data_files=train_npz_files[trainer_id::trainer_num])
    valid_dataset = InMemoryDataset(npz_data_files=valid_npz_files[trainer_id::trainer_num]) 
    if model_config.data.get('post_transform', False):
        logging.info('post transform')
        train_dataset.transform(transform_fn, num_workers=args.num_workers)
        valid_dataset.transform(transform_fn, num_workers=args.num_workers)
    return raw_dataset, train_dataset, valid_dataset


def train(args, epoch_id, model, train_dataset, collate_fn, optimizer, train_steps, ema):
    """
    Define the train function 
    """
    model.train()
    data_gen = train_dataset.get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=True,
            collate_fn=collate_fn)

    step = 0
    s0 = time.time()
    res_collect = ResultsCollect()
    for batch in data_gen:
        label = batch['label']
        batch = tree_map(lambda x: paddle.to_tensor(x), batch)
        batch['epoch_id'] = epoch_id
        if len(label) < args.batch_size * 0.5:
            continue

        s1 = time.time()
        if args.distributed:
            with model.no_sync():
                results = model(batch)
                loss = results['loss']
                loss.backward()
            fused_allreduce_gradients(list(model.parameters()), None)
        else:
            results = model(batch)
            loss = results['loss']
            loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        # ema
        ema.update()
        s2 = time.time()

        # logging
        res_collect.add(batch, results)
        if args.DEBUG or step % 100 == 0:
            logging.info(f"step {step} {res_collect.get_result(distributed=False)} "
                f"t_data: {s1 - s0:.4f} t_train {s2 - s1:.4f}")
        s0 = time.time()
        step += 1
        if step > train_steps:
            break
    train_results = res_collect.get_result(distributed=args.distributed)
    return train_results


@paddle.no_grad()
def evaluate(args, epoch_id, model, test_dataset, collate_fn):
    """
    Define the evaluate function
    In the dataset, a proportion of labels are blank. So we use a `valid` tensor 
    to help eliminate these blank labels in both training and evaluation phase.
    We downscale the batch size of the valid dataset since there are larger mols in valid it.
    """
    model.eval()
    data_gen = test_dataset.get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=False,
            collate_fn=collate_fn)

    total_pred = []
    total_label = []
    for batch in data_gen:
        batch = tree_map(lambda x: paddle.to_tensor(x), batch) 
        results = model(batch)
        total_pred.append(results['pred'].numpy().flatten())
        total_label.append(batch['label'].numpy().flatten())
    total_pred = np.concatenate(total_pred, 0)
    total_label = np.concatenate(total_label, 0)
    mae = np.abs(total_pred - total_label)
    mean_mae = dist_mean(mae, distributed=args.distributed)
    num_sample = dist_length(mae, distributed=args.distributed)
    logging.info(f'test_epoch: {epoch_id} num: {num_sample} mae: {mean_mae}')
    return mean_mae


def main(args):
    """
    Call the configuration function of the model, build the model and load data, then start training.
    model_config:
        a json file  with the hyperparameters,such as dropout rate ,learning rate,num tasks and so on;
    num_tasks:
        it means the number of task that each dataset contains, it's related to the dataset;
    """
    def _read_json(path):
        return ml_collections.ConfigDict(json.load(open(path, 'r')))

    set_logging_level(args.logging_level)

    print(f'args:\n{args}')
    dataset_config = _read_json(args.dataset_config)
    print(f'>>> dataset_config:\n{dataset_config}')
    train_config = _read_json(args.train_config)
    print(f'>>> train_config:\n{train_config}')
    model_config, encoder_config = create_model_config(args)
    print(f'>>> model_config:\n{model_config}')
    print(f'>>> encoder_config:\n{encoder_config}')

    ### init dist
    trainer_id = 0
    trainer_num = 1
    if args.distributed:
        dist.init_parallel_env()
        trainer_id = dist.get_rank()
        trainer_num = dist.get_world_size()
    
    # recompute dropout config
    ### IMPORTANT: in order to correctly recover dropout in paddle.distributed.fleet.utils.recompute 
    paddle.seed(10000 + trainer_id)

    ### load data
    transform_fn, collate_fn = get_featurizer(model_config, encoder_config)
    raw_dataset, train_dataset, valid_dataset = load_data(
            args, trainer_id, trainer_num, model_config, dataset_config, transform_fn)
    logging.info(f"Train/Valid num: {len(train_dataset)}/{len(valid_dataset)}")
    label_stat = raw_dataset.get_label_stat()
    print(f'label_stat: {label_stat}')
    model_config.model.heads.property_regr.update(
            label_mean=list(label_stat['mean']), label_std=list(label_stat['std']))

    ### build model
    model = MolRegressionModel(model_config, encoder_config)
    print("parameter size:", calc_parameter_size(model.parameters()))
    if args.distributed:
        model = paddle.DataParallel(model)
    if not args.init_model is None and not args.init_model == "":
        model.set_state_dict(paddle.load(args.init_model))
        print('Load state_dict from %s' % args.init_model)

    ema = ExponentialMovingAverage2(model, decay=0.999)
    ema_start_step = 0 if args.DEBUG else 30

    optimizer, scheduler = get_optimizer(args, train_config, model)

    ### start train
    data_writer = None
    if dist.get_rank() == 0:
        try:    # permission denied error if without root
            data_writer = SummaryWriter(f'{args.log_dir}/tensorboard_log_dir', max_queue=0)
        except Exception as ex:
            print(f'Create data_writer failed: {ex}')
    train_steps = get_train_steps_per_epoch(len(train_dataset))
    print("train_steps per epoch : ", train_steps)
    mean_mae_list = []
    for _ in range(args.start_step):
        scheduler.step()
    for epoch_id in range(args.start_step, args.max_epoch):
        ## ema register
        if epoch_id >= ema_start_step and not ema.is_registered:
            ema.register()

        ## train
        s_time = time.time()
        scheduler.step()
        train_results = train(args, epoch_id, model, train_dataset, collate_fn, 
                optimizer, train_steps, ema)

        ## evaluate
        mean_mae = evaluate(args, epoch_id, model, valid_dataset, collate_fn)
        mean_mae_list.append(mean_mae)
        ema_mean_mae = 0
        if ema.is_registered:
            ema.apply()
            ema_mean_mae = evaluate(args, epoch_id, model, valid_dataset, collate_fn)
            ema.restore()
        
        ## logging        
        if trainer_id == 0:
            train_results.update({
                'total_batch_size': args.batch_size * trainer_num,
                "lr": scheduler.get_lr()
            })
            add_to_data_writer(data_writer, epoch_id, train_results, prefix='train')
            test_results = {
                'mean_mae': mean_mae,
                'best_val': min(mean_mae_list),
                'ema_mean_mae': ema_mean_mae,
            }
            add_to_data_writer(data_writer, epoch_id, test_results, prefix='test')
            csv_results = {'epoch': epoch_id}
            csv_results.update(train_results)
            csv_results.update(test_results)
            write_to_csv(f'{args.log_dir}/results.csv', csv_results)

        ## saving
        if trainer_id == 0:
            paddle.save(model.state_dict(), f'./{args.model_dir}/epoch_{epoch_id}.pdparams')
            if ema.is_registered:
                ema.apply()
                paddle.save(model.state_dict(), f'./{args.model_dir}/epoch_{epoch_id}_ema.pdparams')
                ema.restore()

        logging.info(f"using {(time.time() - s_time)/60} minute")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action='store_true', default=False)
    parser.add_argument("--DEBUG", action='store_true', default=False)
    parser.add_argument("--logging_level", type=str, default="DEBUG", 
            help="NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=100)

    parser.add_argument("--dataset_config", type=str)
    parser.add_argument("--data_cache_dir", type=str, default="./data_cache")
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--encoder_config", type=str)
    parser.add_argument("--train_config", type=str)
    parser.add_argument("--init_model", type=str)
    parser.add_argument("--start_step", type=int)
    parser.add_argument("--model_dir", type=str, default="./debug_models")
    parser.add_argument("--log_dir", type=str, default="./debug_log")
    args = parser.parse_args()
    
    main(args)