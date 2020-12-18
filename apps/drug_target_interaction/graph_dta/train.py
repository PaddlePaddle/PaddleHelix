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
train
"""

import os
import json
import shutil
import logging
import argparse
import numpy as np

import paddle
import paddle.fluid as fluid
from pgl.utils.data.dataloader import Dataloader
from pahelix.utils.paddle_utils import load_partial_params

from data_gen import DTADataset, DTACollateFunc
from model import DTAModel
from utils import default_exe_params, setup_optimizer, concordance_index

logging.basicConfig(
        format='%(asctime)s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO)


def train(args, exe, train_program, model, train_dataset):
    """Model training for one epoch and return the average loss."""
    label_name = 'KIBA' if args.use_kiba_label else 'Log10_Kd'
    collate_fn = DTACollateFunc(
        model.compound_graph_wrapper, is_inference=False,
        label_name=label_name)
    data_loader = Dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stream_shuffle_size=1000,
        collate_fn=collate_fn)

    list_loss = []
    for feed_dict in data_loader:
        train_loss, = exe.run(
                train_program, feed=feed_dict, fetch_list=[model.loss], return_numpy=False)
        list_loss.append(np.array(train_loss).mean())
    return np.mean(list_loss)


def evaluate(args, exe, test_program, model, test_dataset, best_mse,
             val_dataset=None):
    """Evaluate the model on the test dataset and return MSE and CI."""
    if args.use_val:
        assert val_dataset is not None

    label_name = 'KIBA' if args.use_kiba_label else 'Log10_Kd'
    collate_fn = DTACollateFunc(
        model.compound_graph_wrapper, is_inference=False,
        label_name=label_name)
    data_loader = Dataloader(
        test_dataset if not args.use_val else val_dataset,
        batch_size=args.batch_size,
        num_workers=1,
        collate_fn=collate_fn)

    if args.use_val:
        test_dataloader = Dataloader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=1,
            collate_fn=collate_fn)

    total_n, processed = len(test_dataset), 0
    total_pred, total_label = [], []
    for idx, feed_dict in enumerate(data_loader):
        logging.info('Evaluated {}/{}'.format(processed, total_n))
        pred, = exe.run(
                test_program, feed=feed_dict, fetch_list=[model.pred], return_numpy=False)
        total_pred.append(np.array(pred))
        total_label.append(feed_dict['label'])
        processed += total_pred[-1].shape[0]

    logging.info('Evaluated {}/{}'.format(processed, total_n))
    total_pred = np.concatenate(total_pred, 0).flatten()
    total_label = np.concatenate(total_label, 0).flatten()
    mse = ((total_label - total_pred) ** 2).mean(axis=0)

    test_mse, test_ci, ci = None, None, None
    if mse < best_mse and not args.use_val:
        # Computing CI is time consuming
        ci = concordance_index(total_label, total_pred)
    elif mse < best_mse and args.use_val:
        total_pred, total_label = [], []
        for idx, feed_dict in enumerate(test_dataloader):
            pred, = exe.run(test_program, feed=feed_dict,
                            fetch_list=[model.pred], return_numpy=False)
            total_pred.append(np.array(pred))
            total_label.append(feed_dict['label'])

        total_pred = np.concatenate(total_pred, 0).flatten()
        total_label = np.concatenate(total_label, 0).flatten()
        test_mse = ((total_label - total_pred) ** 2).mean(axis=0)
        test_ci = concordance_index(total_label, total_pred)

    if args.use_val:
        # `mse` aka `val_mse`
        # when `val_mse` > `best_mse`, test_mse = None, test_ci = None
        return mse, test_mse, test_ci
    else:
        return mse, ci


def save_metric(model_dir, epoch_id, best_mse, best_ci):
    """Save the evaluation metric to txt file."""
    metric = 'Epoch: {}, Best MSE: {}, Best CI: {}'.format(
        epoch_id, best_mse, best_ci)
    logging.info(metric)
    with open(os.path.join(model_dir, 'eval.txt'), 'w') as f:
        f.write(metric)


def main(args):
    # Enable static graph mode.
    paddle.enable_static()

    model_config = json.load(open(args.model_config, 'r'))

    exe_params = default_exe_params(args.is_distributed, args.use_cuda, args.thread_num)
    exe = exe_params['exe']

    selector = None if not args.use_val else lambda l: l[:int(len(l)*0.8)]
    train_dataset = DTADataset(
        args.train_data, exe_params['trainer_id'],
        exe_params['trainer_num'],
        max_protein_len=model_config['protein']['max_protein_len'],
        subset_selector=selector)
    test_dataset = DTADataset(
        args.test_data, exe_params['trainer_id'],
        exe_params['trainer_num'],
        max_protein_len=model_config['protein']['max_protein_len'])

    if args.use_val:
        selector = lambda l: l[int(len(l)*0.8):]
        val_dataset = DTADataset(
            args.train_data, exe_params['trainer_id'],
            exe_params['trainer_num'],
            max_protein_len=model_config['protein']['max_protein_len'],
            subset_selector=selector)

    train_program = fluid.Program()
    train_startup = fluid.Program()
    with fluid.program_guard(train_program, train_startup):
        with fluid.unique_name.guard():
            model = DTAModel(
                model_config=model_config,
                use_pretrained_compound_gnns=args.use_pretrained_compound_gnns)
            model.train()
            test_program = train_program.clone(for_test=True)
            optimizer = fluid.optimizer.Adam(learning_rate=args.lr)
            setup_optimizer(optimizer, model, args.use_cuda, args.is_distributed)
            optimizer.minimize(model.loss)

    exe.run(train_startup)
    if args.init_model is not None and args.init_model != "":
        load_partial_params(exe, args.init_model, train_program)

    config = os.path.basename(args.model_config)
    best_mse, best_mse_, best_ci, best_ep = np.inf, np.inf, 0, 0
    best_model = os.path.join(args.model_dir, 'best_model')
    for epoch_id in range(1, args.max_epoch + 1):
        logging.info('========== Epoch {} =========='.format(epoch_id))
        train_loss = train(args, exe, train_program, model, train_dataset)
        logging.info('#{} Epoch: {}, Train loss: {}'.format(
            config, epoch_id, train_loss))
        metrics = evaluate(
            args, exe, test_program, model, test_dataset, best_mse_,
            val_dataset=None if not args.use_val else val_dataset)

        if args.use_val:
            mse, test_mse, test_ci = metrics
        else:
            mse, ci = metrics

        if mse < best_mse_:
            best_ep = epoch_id
            if os.path.exists(best_model):
                shutil.rmtree(best_model)
            fluid.io.save_params(exe, best_model, train_program)

        if not args.use_val and mse < best_mse_:
            best_mse, best_mse_, best_ci = mse, mse, ci
            save_metric(args.model_dir, epoch_id, best_mse, best_ci)
        elif args.use_val and mse < best_mse_:
            best_mse, best_mse_, best_ci = test_mse, mse, test_ci
            save_metric(args.model_dir, epoch_id, best_mse, best_ci)
        else:
            logging.info('No improvement in epoch {}'.format(epoch_id))
            metric = open(os.path.join(args.model_dir, 'eval.txt'), 'r').read()
            logging.info('===== Current best:\n{}'.format(metric))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cuda", action='store_true', default=False)
    parser.add_argument('--distributed', dest='is_distributed', action='store_true')
    parser.add_argument('--use_kiba_label', action='store_true', default=False)
    parser.add_argument('--use_val', action='store_true', default=False)
    parser.add_argument('--use_pretrained_compound_gnns', action='store_true', default=False)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument('--thread_num', type=int, default=8, help='thread for cpu')
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--test_data", type=str)

    parser.add_argument("--model_config", type=str)
    parser.add_argument("--init_model", type=str)
    parser.add_argument("--model_dir", type=str)
    args = parser.parse_args()

    main(args)
