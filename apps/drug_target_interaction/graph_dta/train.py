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

from src.data_gen import DTADataset, DTACollateFunc
from src.model import DTAModel, DTAModelCriterion
from src.utils import concordance_index

logging.basicConfig(
        format='%(asctime)s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO)


def train(args, model, criterion, optimizer, dataset, collate_fn):
    """Model training for one epoch and return the average loss."""
    data_gen = dataset.get_data_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn)
    model.train()

    list_loss = []
    for graphs, proteins_token, proteins_mask, labels in data_gen:
        graphs = graphs.tensor()
        proteins_token = paddle.to_tensor(proteins_token)
        proteins_mask = paddle.to_tensor(proteins_mask)
        labels = paddle.to_tensor(labels)

        preds = model(graphs, proteins_token, proteins_mask)
        loss = criterion(preds, labels)

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        list_loss.append(loss.numpy())

    return np.mean(list_loss)


def evaluate(args, model, collate_fn, test_dataset, best_mse,
             val_dataset=None):
    """Evaluate the model on the test dataset and return MSE and CI."""
    if args.use_val:
        assert val_dataset is not None
        dataset = val_dataset
    else:
        dataset = test_dataset

    data_gen = dataset.get_data_loader(
        batch_size=args.batch_size,
        num_workers=1,
        shuffle=False,
        collate_fn=collate_fn)
    model.eval()

    total_n, processed = len(dataset), 0
    total_pred, total_label = [], []
    for idx, (graphs, proteins_token, proteins_mask, labels) in enumerate(data_gen):
        graphs = graphs.tensor()
        proteins_token = paddle.to_tensor(proteins_token)
        proteins_mask = paddle.to_tensor(proteins_mask)
        preds = model(graphs, proteins_token, proteins_mask)
        total_pred.append(preds.numpy())
        total_label.append(labels)
        processed += total_pred[-1].shape[0]

        logging.info('Evaluated {}/{}'.format(processed, total_n))

    logging.info('Evaluated {}/{}'.format(processed, total_n))
    total_pred = np.concatenate(total_pred, 0).flatten()
    total_label = np.concatenate(total_label, 0).flatten()
    mse = ((total_label - total_pred) ** 2).mean(axis=0)

    test_mse, test_ci, ci = None, None, None
    if mse < best_mse and not args.use_val:
        # Computing CI is time consuming
        ci = concordance_index(total_label, total_pred)
    elif mse < best_mse and args.use_val:
        test_data_gen = test_dataset.get_data_loader(
            batch_size=args.batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=collate_fn)

        total_pred, total_label = [], []
        for idx, (graphs, proteins_token, proteins_mask, labels) in enumerate(test_data_gen):
            graphs = graphs.tensor()
            proteins_token = paddle.to_tensor(proteins_token)
            proteins_mask = paddle.to_tensor(proteins_mask)
            preds = model(graphs, proteins_token, proteins_mask)
            total_pred.append(preds.numpy())
            total_label.append(labels)

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
    paddle.set_device(args.device)
    model_config = json.load(open(args.model_config, 'r'))

    logging.info('Load data ...')
    selector = None if not args.use_val else lambda l: l[:int(len(l)*0.8)]
    train_dataset = DTADataset(
        args.train_data,
        max_protein_len=model_config['protein']['max_protein_len'],
        subset_selector=selector)
    test_dataset = DTADataset(
        args.test_data,
        max_protein_len=model_config['protein']['max_protein_len'])

    if args.use_val:
        selector = lambda l: l[int(len(l)*0.8):]
        val_dataset = DTADataset(
            args.train_data,
            max_protein_len=model_config['protein']['max_protein_len'],
            subset_selector=selector)

    label_name = 'KIBA' if args.use_kiba_label else 'Log10_Kd'
    collate_fn = DTACollateFunc(
        model_config['compound']['atom_names'],
        model_config['compound']['bond_names'],
        is_inference=False,
        label_name=label_name)

    logging.info("Data loaded.")

    model = DTAModel(model_config)
    criterion = DTAModelCriterion()
    optimizer = paddle.optimizer.Adam(
        learning_rate=args.lr,
        parameters=model.parameters())

    os.makedirs(args.model_dir, exist_ok=True)

    best_mse, best_mse_, best_ci, best_ep = np.inf, np.inf, 0, 0
    best_model = os.path.join(args.model_dir, 'best_model.pdparams')
    cfg_name = os.path.basename(args.model_config)

    for epoch_id in range(args.max_epoch):
        logging.info('========== Epoch {} =========='.format(epoch_id))
        train_loss = train(args, model, criterion, optimizer, train_dataset, collate_fn)
        logging.info('#{} Epoch: {}, Train loss: {}'.format(
            cfg_name, epoch_id, train_loss))

        metrics = evaluate(args, model, collate_fn, test_dataset, best_mse_,
                           val_dataset=None if not args.use_val else val_dataset)

        if args.use_val:
            mse, test_mse, test_ci = metrics
        else:
            mse, ci = metrics

        if mse < best_mse_:
            best_ep = epoch_id
            paddle.save(model.state_dict(), best_model)

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
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--use_kiba_label', action='store_true', default=False)
    parser.add_argument('--use_val', action='store_true', default=False)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument('--thread_num', type=int, default=8, help='thread for cpu')
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--test_data", type=str)

    parser.add_argument("--model_config", type=str)
    parser.add_argument("--model_dir", type=str)
    args = parser.parse_args()

    main(args)
