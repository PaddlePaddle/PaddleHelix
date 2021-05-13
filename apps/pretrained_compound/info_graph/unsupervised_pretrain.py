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
Script to train and eval representations from
unsupervised pretraining of InfoGraph model
"""

import os
import sys
import json
import glob
import pickle
import argparse
from collections import namedtuple
import numpy as np
import logging
import multiprocessing as mp

logging.basicConfig(
        format='%(asctime)s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO)

import paddle
import paddle.nn as nn
import pgl

from pahelix.datasets.inmemory_dataset import InMemoryDataset

from src.model import InfoGraph, InfoGraphCriterion
from src.featurizer import MoleculeCollateFunc
from classifier import eval_on_classifiers


def train(args, model, criterion, optimizer, dataset,
          collate_fn, epoch_id):
    """Model training for one epoch and log the average loss."""

    data_gen = dataset.get_data_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=collate_fn)
    model.train()

    list_loss = []
    total_data, trained_data, batch_id = len(dataset), 0, 0
    for graphs, pos_mask, neg_mask in data_gen:
        graphs = graphs.tensor()
        pos_mask = paddle.to_tensor(pos_mask, 'float32')
        neg_mask = paddle.to_tensor(neg_mask, 'float32')

        global_repr, encoding = model(graphs)
        if criterion.prior:
            prior = paddle.uniform(
                [args.batch_size, model.embedding_dim],
                min=0.0, max=1.0)
            loss = criterion(graphs, global_repr, encoding,
                             pos_mask, neg_mask, prior)
        else:
            loss = criterion(graphs, global_repr, encoding,
                             pos_mask, neg_mask)

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        list_loss.append(loss.numpy())

        trained_data += graphs.num_graph
        if batch_id % args.log_interval == 0:
            logging.info('Epoch %d [%d/%d] train/loss:%f' % \
                         (epoch_id, trained_data, total_data, list_loss[-1]))

        batch_id += 1

    return np.mean(list_loss)


def save_embedding(args, model, dataset, collate_fn, epoch_id):
    """Save the embeddings of all the testing data for multiple classifier evaluation."""
    data_gen = dataset.get_data_loader(
        batch_size=args.batch_size,
        num_workers=1,
        shuffle=False,
        collate_fn=collate_fn)
    model.eval()


    graph_emb_lst, graph_label_lst = [], []
    for graphs, labels, valids in data_gen:
        graphs = graphs.tensor()
        graphs_emb, _ = model(graphs)
        graphs_emb = graphs_emb.numpy()

        graph_emb_lst.extend(list(graphs_emb))
        graph_label_lst.extend(list(labels))

    n = len(dataset)
    merge_data = {
        'emb': np.array(graph_emb_lst[:n]),
        'y': np.array(graph_label_lst[:n])
    }

    pkl = os.path.join(args.emb_dir, 'epoch_%s.pkl' % epoch_id)
    with open(pkl, 'wb') as f:
        pickle.dump(merge_data, f)


def parallel_eval(data_queue, pkl_lst):
    """The target function to run a multi-classifier evaluation for given embeddings."""
    for pkl in pkl_lst:
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
        metrics = eval_on_classifiers(data['emb'], data['y'], search=True)
        logging.info('{}: logreg ({:.4f}), svc ({:.4f}), '
                     'linearsvc ({:.4f}), randomforest ({:.4f})'.format(
                         os.path.basename(pkl), metrics['logreg_acc'],
                         metrics['svc_acc'], metrics['linearsvc_acc'],
                         metrics['randomforest_acc']))
        sys.stdout.flush()

        res = {
            'pkl': os.path.basename(pkl),
            'logreg': metrics['logreg_acc'],
            'svc': metrics['svc_acc'],
            'linearsvc': metrics['linearsvc_acc'],
            'randomforest': metrics['randomforest_acc'],
        }
        data_queue.put(res)


def save_eval_metric(res_collect, emb_dir):
    """Save the evaluation metrics from parallel evaluation workers."""
    base = os.path.basename(emb_dir)
    json_file = os.path.join(os.path.dirname(emb_dir), '%s_eval.json' % base)

    keys = list(res_collect.keys())
    acc_keys = []
    for k in keys:
        if type(res_collect[k][0]) is not str:
            acc_keys.append(k)
            best_acc = max(res_collect[k])
            res_collect['best_%s' % k] = best_acc
            print('Best {} acc: {:.4f}'.format(k, round(best_acc, 4)))

    res_collect['avg_acc'] = []
    for i in range(len(res_collect[acc_keys[0]])):
        sum_acc = 0
        for k in acc_keys:
            sum_acc += res_collect[k][i]
        res_collect['avg_acc'].append(sum_acc / len(acc_keys))

    best_acc = max(res_collect['avg_acc'])
    res_collect['best_avg_acc'] = best_acc
    print('Best average acc: {:.4f}'.format(round(best_acc, 4)))

    with open(json_file, 'w') as f:
        json.dump(res_collect, f)


def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)

    logging.info('Load data ...')

    dataset = InMemoryDataset(npz_data_path=os.path.join(
        args.root, args.dataset, 'processed'))
    collate_fn = MoleculeCollateFunc(
        config['atom_names'],
        config['bond_names'],
        with_graph_label=False,
        with_pos_neg_mask=True)
    eval_collate_fn = MoleculeCollateFunc(
        config['atom_names'],
        config['bond_names'],
        with_graph_label=True,
        with_pos_neg_mask=False)

    logging.info("Data loaded.")
    logging.info("Train Examples: %s" % len(dataset))
    sys.stdout.flush()

    if args.emb_dir is not None:
        # pylint: disable=E1123
        os.makedirs(args.emb_dir, exist_ok=True)

    if args.model_dir is not None:
        # pylint: disable=E1123
        os.makedirs(args.model_dir, exist_ok=True)

    model = InfoGraph(config)
    criterion = InfoGraphCriterion(config)
    optimizer = paddle.optimizer.Adam(
        learning_rate=args.lr,
        parameters=model.parameters())

    save_embedding(args, model, dataset, eval_collate_fn, -1)
    for epoch_id in range(args.max_epoch):
        train_loss = train(args, model, criterion, optimizer,
                           dataset, collate_fn, epoch_id)
        logging.info('Epoch %d, train/loss: %f' % (epoch_id, train_loss))

        pdparams = os.path.join(args.model_dir, 'epoch_%d.pdparams' % epoch_id)
        paddle.save(model.state_dict(), pdparams)

        save_embedding(args, model, dataset, eval_collate_fn, epoch_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='InfoGraph pretrain')
    parser.add_argument("--task_name", type=str, default='train')
    parser.add_argument("--use_cuda", action='store_true', default=False)
    # parser.add_argument("--data_loader", action='store_true', default=False)
    parser.add_argument("--config", type=str, default='model_configs/unsupervised_pretrain_config.json')

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--root", type=str, default='data')
    parser.add_argument("--dataset", type=str, default='mutag')
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--emb_dir", type=str)

    parser.add_argument("--eval_workers", type=int, default=10)
    parser.add_argument("--dont_save_emb", action='store_true', default=False)
    args = parser.parse_args()

    if args.task_name == 'train':
        main(args)
    elif args.task_name == 'eval':
        pkl_lst = glob.glob(os.path.join(args.emb_dir, '*.pkl'))

        proc_manager = mp.Manager()
        data_queue = proc_manager.Queue()
        proc_lst = []
        for i in range(args.eval_workers):
            ids = [j for j in range(len(pkl_lst)) if j % args.eval_workers == i]
            sub_pkl_lst = [pkl_lst[j] for j in ids]

            p = mp.Process(target=parallel_eval, args=(data_queue, sub_pkl_lst))
            p.start()
            proc_lst.append(p)

        res_collect = dict()
        to_recv = len(pkl_lst)
        while to_recv > 0:
            res = data_queue.get()
            to_recv -= 1

            for k, v in res.items():
                if k not in res_collect:
                    res_collect[k] = [v]
                else:
                    res_collect[k].append(v)
        save_eval_metric(res_collect, args.emb_dir)

        for p in proc_lst:
            p.join()
