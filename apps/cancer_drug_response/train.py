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
train
"""

import os
import sys
import time
import argparse
import numpy as np
import logging
from scipy.stats import pearsonr

import paddle
import paddle.nn as nn
from paddle.optimizer import Adam

import pgl
from pgl.utils.logger import log
from pgl.utils.data import Dataloader

from pahelix.utils.data_utils import load_npz_to_data_list
from pahelix.datasets.inmemory_dataset import InMemoryDataset

from data_gen import Dataset, collate_fn
from model import CDRModel


def main(args):
    """
    Model training for one epoch and return the average loss and model evaluating to monitor pcc.
    """
    paddle.set_device('gpu:{}'.format(args.device) if args.use_cuda else 'cpu')

    logging.info('Load data ...')
    dataset = InMemoryDataset(npz_data_path=args.data_path)

    train_ds = Dataset(dataset[1])
    test_ds = Dataset(dataset[0])
    train_loader = train_ds.get_data_loader(batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = test_ds.get_data_loader(batch_size=args.batch_size, collate_fn=collate_fn)

    logging.info("Data loaded.")

    model = CDRModel(args)

    optim = Adam(learning_rate=args.lr, parameters=model.parameters())
    criterion = nn.MSELoss()

    global_step = 0
    best_pcc = 0.0
    os.makedirs(args.output_path, exist_ok=True)
    best_model = os.path.join(args.output_path, 'best_model.pdparams')

    for epoch in range(1, args.epoch_num + 1):
        model.train()
        for idx, batch_data in enumerate(train_loader):
            graphs, mut, gexpr, met, label = batch_data
            g = pgl.Graph.batch(graphs).tensor()
            mut = paddle.to_tensor(mut)
            gexpr = paddle.to_tensor(gexpr)
            met = paddle.to_tensor(met)
            label = paddle.to_tensor(label)

            pred = model([g, mut, gexpr, met])
            train_loss = paddle.pow(criterion(pred[:, 0], label)[0], 0.5)
            train_loss.backward()
            train_pcc = pearsonr(pred[:, 0].numpy(), label.numpy())[0]
            optim.step()
            optim.clear_grad()

            global_step += 1
            if global_step % 500 == 0:
                message = "train: epoch %d | step %d | " % (epoch, global_step)
                message += "loss %.6f ｜ pcc %.4f" % (train_loss, train_pcc)
                log.info(message)

        result = evaluate(model, test_loader, criterion)
        message = "eval: epoch %d | step %d " % (epoch, global_step)
        for key, value in result.items():
            message += "| %s %.6f" % (key, value)
        log.info(message)

        if best_pcc < result['pcc']:
            best_pcc = result['pcc']
            paddle.save(model.state_dict(), best_model)

    log.info("best evaluating accuracy: %.6f" % best_pcc)


def evaluate(model, loader, criterion):
    """
    Evaluate the model on the test dataset and return average loss and pcc.
    """
    model.eval()
    total_loss = []
    total_pcc = []

    for idx, batch_data in enumerate(loader):
        graphs, mut, gexpr, met, label = batch_data
        g = pgl.Graph.batch(graphs).tensor()
        mut = paddle.to_tensor(mut)
        gexpr = paddle.to_tensor(gexpr)
        met = paddle.to_tensor(met)
        label = paddle.to_tensor(label)

        pred = model([g, mut, gexpr, met])
        eval_loss = paddle.pow(criterion(pred[:, 0], label)[0], 0.5)
        eval_pcc = pearsonr(pred[:, 0].numpy(), label.numpy())[0]
        total_loss.append(eval_loss.numpy())
        total_pcc.append(eval_pcc)

    total_loss = np.mean(total_loss)
    total_pcc = np.mean(total_pcc)
    model.train()

    return {"loss": total_loss, "pcc": total_pcc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/processed/')
    parser.add_argument('--output_path', type=str, default='./best_model/')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_mut', type=bool, default=True)
    parser.add_argument('--use_gexp', type=bool, default=True)
    parser.add_argument('--use_methy', type=bool, default=True)
    parser.add_argument('--layer_num', type=int, default=4)
    parser.add_argument('--units_list', type=list, default=[256, 256, 256, 100])
    parser.add_argument(
        '--gnn_type',
        type=str,
        default="gcn",
        choices=["gcn", "gin", "graphsage"])
    parser.add_argument(
        '--pool_type',
        type=str,
        default="max",
        choices=["sum", "average", "max"])
    parser.add_argument('--act', type=str, default='relu')
    parser.add_argument('--epoch_num', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()

    log.info(args)

    main(args)
