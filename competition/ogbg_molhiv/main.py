# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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

import torch
import sys
import os
import math
import argparse
import traceback
import re
import io
import json
import yaml
import time
import logging
from tqdm import tqdm
import numpy as np
from collections import namedtuple

from ogb.graphproppred import GraphPropPredDataset
import paddle.fluid as F
import paddle.fluid.layers as L
import pgl
from pgl.utils import paddle_helper
from pgl.utils.data.dataloader import Dataloader

from propeller import log
log.setLevel(logging.DEBUG)
import propeller.paddle as propeller
from propeller.paddle.data import Dataset as PDataset

from utils.config import prepare_config, make_dir
from utils.logger import prepare_logger, log_to_file
from utils.util import int82strarr
from dataset import MolDataset, MgfCollateFn
from model import MgfModel
import dataset as DS
import model as M

def multi_epoch_dataloader(loader, epochs):
    def _worker():
        for i in range(epochs):
            log.info("BEGIN: epoch %s ..." % i)
            for batch in loader():
                yield batch
            log.info("END: epoch %s ..." % i)
    return _worker

def train(args, pretrained_model_config=None):
    log.info("loading data")
    raw_dataset = GraphPropPredDataset(name=args.dataset_name)
    args.num_class = raw_dataset.num_tasks
    args.eval_metric = raw_dataset.eval_metric
    args.task_type = raw_dataset.task_type

    train_ds = MolDataset(args, raw_dataset)

    args.eval_steps = math.ceil(len(train_ds) / args.batch_size)
    log.info("Total %s steps (eval_steps) every epoch." % (args.eval_steps))

    fn = MgfCollateFn(args)

    train_loader = Dataloader(train_ds,
                         batch_size=args.batch_size,
                         num_workers=args.num_workers,
                         shuffle=args.shuffle,
                         stream_shuffle_size=args.shuffle_size,
                         collate_fn=fn)

    # for evaluating
    eval_train_loader = train_loader
    eval_train_loader = PDataset.from_generator_func(eval_train_loader)

    train_loader = multi_epoch_dataloader(train_loader, args.epochs)
    train_loader = PDataset.from_generator_func(train_loader)

    if args.warm_start_from is not None:
        # warm start setting
        def _fn(v):
            if not isinstance(v, F.framework.Parameter):
                return False
            if os.path.exists(os.path.join(args.warm_start_from, v.name)):
                return True
            else:
                return False
        ws = propeller.WarmStartSetting(
                predicate_fn=_fn,
                from_dir=args.warm_start_from)
    else:
        ws = None

    def cmp_fn(old, new):
        if old['eval'][args.metrics] - new['eval'][args.metrics] > 0:
            log.info("best %s eval result: %s" % (args.metrics, new['eval']))
            return True
        else:
            return False

    if args.log_id is not None:
        save_best_model = int(args.log_id) == 5
    else:
        save_best_model = True
    best_exporter = propeller.exporter.BestResultExporter(
            args.output_dir, (cmp_fn, save_best_model))

    eval_datasets = {"eval": eval_train_loader}

    propeller.train.train_and_eval(
            model_class_or_model_fn=MgfModel,
            params=pretrained_model_config,
            run_config=args,
            train_dataset=train_loader,
            eval_dataset=eval_datasets,
            warm_start_setting=ws,
            exporters=[best_exporter],
            )

def infer(args):
    log.info("loading data")
    raw_dataset = GraphPropPredDataset(name=args.dataset_name)
    args.num_class = raw_dataset.num_tasks
    args.eval_metric = raw_dataset.eval_metric
    args.task_type = raw_dataset.task_type

    test_ds = MolDataset(args, raw_dataset, mode="test")

    fn = MgfCollateFn(args, mode="test")

    test_loader = Dataloader(test_ds,
                         batch_size=args.batch_size,
                         num_workers=1,
                         collate_fn=fn)
    test_loader = PDataset.from_generator_func(test_loader)

    est = propeller.Learner(MgfModel, args, args.model_config)

    mgf_list = []
    for soft_mgf in est.predict(test_loader,
            ckpt_path=args.model_path_for_infer, split_batch=True):
        mgf_list.append(soft_mgf)

    mgf = np.concatenate(mgf_list)
    log.info("saving features")
    np.save("dataset/%s/soft_mgf_feat.npy" % (args.dataset_name.replace("-", "_")), mgf)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='gnn')
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--task_name", type=str, default="task_name")
    parser.add_argument("--infer_model", type=str, default=None)
    parser.add_argument("--log_id", type=str, default=None)
    args = parser.parse_args()

    if args.infer_model is not None:
        config = prepare_config(args.config, isCreate=False, isSave=False)
        config.model_path_for_infer = args.infer_model
        infer(config)
    else:
        config = prepare_config(args.config, isCreate=True, isSave=True)

        log_to_file(log, config.log_dir, config.log_filename)

        if config.warm_start_from is not None:
            log.info("loading model config from %s" % config.pretrained_config_file)
            pretrained_config = prepare_config(config.pretrained_config_file)
            pretrained_model_config = pretrained_config.pretrained_model_config
        else:
            pretrained_model_config = config.model_config

        config.log_id = args.log_id
        train(config, pretrained_model_config)
