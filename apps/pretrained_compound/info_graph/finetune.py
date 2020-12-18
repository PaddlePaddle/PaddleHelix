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
finetune
"""

import os
import sys
import json
import glob
import pickle
import shutil
import argparse
from collections import namedtuple
import numpy as np
import logging
import multiprocessing as mp

import paddle
import paddle.fluid as F
import paddle.fluid.layers as L
import pgl
from pgl.utils import paddle_helper
from pgl.graph_wrapper import GraphWrapper
from pgl.utils.data.dataloader import Dataloader
from pahelix.utils.paddle_utils import load_partial_params

from model import GINEncoder, FF, PriorDiscriminator
from data_gen import MoleculeCollateFunc
from utils import load_data, calc_rocauc_score


def create_model(args, config, graph_label):
    """Create model for given model configuration."""
    logging.info('building model')
    graph_wrapper = GraphWrapper(
        name="graph",
        node_feat=[
            ('atom_type', [None, 1], "int64"),
            ('chirality_tag', [None, 1], "int64")],
        edge_feat=[
            ('bond_type', [None, 1], "int64"),
            ('bond_direction', [None, 1], "int64")])

    encoder = GINEncoder(config)
    global_repr, patch_summary = encoder.forward(graph_wrapper)

    hid = L.fc(global_repr, config['hidden_size'], act='relu', name='finetune_fc1')
    hid = L.fc(hid, config['hidden_size'], act='relu', name='finetune_fc2')

    logits = L.fc(global_repr, args.num_tasks, name="finetune_fc3")
    loss = L.sigmoid_cross_entropy_with_logits(x=logits, label=graph_label)
    loss = L.reduce_mean(loss)
    pred = L.sigmoid(logits)

    keys = ['loss', 'graph_wrapper', 'encoder', 'graph_emb', 'pred']
    Agent = namedtuple('Agent', keys)
    return Agent(
        loss=loss,
        graph_wrapper=graph_wrapper,
        encoder=encoder,
        graph_emb=global_repr,
        pred=pred)


def train(args, exe, train_prog, agent, train_data_list, epoch_id):
    """Model training for one epoch and log the average loss."""
    collate_fn = MoleculeCollateFunc(
        agent.graph_wrapper,
        task_type='cls',
        num_cls_tasks=args.num_tasks,
        with_graph_label=True,
        with_pos_neg_mask=False)
    data_loader = Dataloader(
        train_data_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=collate_fn)

    total_data, trained_data = len(train_data_list), 0
    list_loss = []
    for batch_id, feed_dict in enumerate(data_loader):
        train_loss = exe.run(
            train_prog, feed=feed_dict, fetch_list=[agent.loss])
        train_loss = np.array(train_loss).mean()
        list_loss.append(train_loss)
        trained_data += feed_dict['graph/num_graph'][0]

        if batch_id % args.log_interval == 0:
            logging.info(
                '%s Epoch %d [%d/%d] train/loss:%f' % \
                (args.exp, epoch_id, trained_data, total_data, train_loss))

    logging.info('%s Epoch %d train/loss:%f' % \
                 (args.exp, epoch_id, np.mean(list_loss)))
    sys.stdout.flush()
    return np.mean(list_loss)


def evaluate(args, exe, test_prog, agent, test_data_list, epoch_id):
    """Evaluate the model on test dataset."""
    collate_fn = MoleculeCollateFunc(
        agent.graph_wrapper,
        task_type='cls',
        num_cls_tasks=args.num_tasks,
        with_graph_label=True,
        with_pos_neg_mask=False)
    data_loader = Dataloader(
        test_data_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_fn)

    total_data, eval_data = len(test_data_list), 0
    total_pred, total_label, total_valid = [], [], []
    for batch_id, feed_dict in enumerate(data_loader):
        pred, = exe.run(test_prog, feed=feed_dict, fetch_list=[agent.pred],
                        return_numpy=False)
        total_pred.append(np.array(pred))
        total_label.append(feed_dict['label'])
        total_valid.append(feed_dict['valid'])

    total_pred = np.concatenate(total_pred, 0)
    total_label = np.concatenate(total_label, 0)
    total_valid = np.concatenate(total_valid, 0)
    return calc_rocauc_score(total_label, total_pred, total_valid)


def main(args):
    # Enable static graph mode.
    paddle.enable_static()

    with open(args.config, 'r') as f:
        config = json.load(f)

    logging.info('Load data ...')
    train_data_list = load_data(args.train_data)
    valid_data_list = load_data(args.valid_data)
    test_data_list = load_data(args.test_data)

    logging.info("Data loaded.")
    logging.info("Train Examples: %s" % len(train_data_list))
    logging.info("Val Examples: %s" % len(valid_data_list))
    logging.info("Test Examples: %s" % len(test_data_list))
    logging.info("Num Tasks: %s" % args.num_tasks)
    sys.stdout.flush()

    train_prog = F.Program()
    test_prog = F.Program()
    startup_prog = F.Program()
    with F.program_guard(train_prog, startup_prog):
        with F.unique_name.guard():
            graph_label = L.data(
                name="label", shape=[None, args.num_tasks], dtype="float32")
            agent = create_model(args, config, graph_label)
            test_prog = train_prog.clone(for_test=True)

            opt = F.optimizer.Adam(learning_rate=args.lr)
            opt.minimize(agent.loss)

    place = F.CUDAPlace(0) if args.use_cuda else F.CPUPlace()
    exe = F.Executor(place)
    exe.run(startup_prog)

    if not args.init_model is None and not args.init_model == "":
        load_partial_params(exe, args.init_model, train_prog)
        logging.info('Loaded %s' % args.init_model)

    list_val_auc, list_test_auc, best_val_auc = [], [], 0
    best_model = os.path.join(args.model_dir, 'best_model')
    for epoch_id in range(args.max_epoch):
        train(args, exe, train_prog, agent, train_data_list, epoch_id)
        val_auc = evaluate(args, exe, test_prog, agent, valid_data_list, epoch_id)
        test_auc = evaluate(args, exe, test_prog, agent, test_data_list, epoch_id)
        list_val_auc.append(val_auc)
        list_test_auc.append(test_auc)

        if best_val_auc < val_auc:
            if os.path.exists(best_model):
                shutil.rmtree(best_model)

            F.io.save_params(exe, best_model, train_prog)
            best_val_auc = val_auc

        test_auc_by_eval = list_test_auc[np.argmax(list_val_auc)]
        logging.info('%s Epoch %d val/auc: %f' % (args.exp, epoch_id, val_auc))
        logging.info('%s Epoch %d test/auc: %f' % (args.exp, epoch_id, test_auc))
        logging.info('%s Epoch %d test/auc_by_eval: %f' % \
                     (args.exp, epoch_id, test_auc_by_eval))

    logging.info('%s final/test/auc_by_eval: %f' % (args.exp, test_auc_by_eval))

    with open(os.path.join(args.log_dir, 'metric.json'), 'w') as f:
        best_epoch = int(np.argmax(list_val_auc))
        metric = {
            'val_auc': list_val_auc,
            'test_auc': list_test_auc,
            'best_epoch': best_epoch,
            'best_test_auc': list_test_auc[best_epoch],
            'init_model': '' if args.init_model is None else args.init_model
        }
        f.write(json.dumps(metric))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='InfoGraph finetune')
    parser.add_argument("--exp", type=str)
    parser.add_argument("--use_cuda", action='store_true', default=False)
    parser.add_argument("--config", type=str, default='demos/unsupervised_pretrain_config.json')

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--root", type=str, default='dataset')
    parser.add_argument("--dataset", type=str, default='mutag')
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--valid_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--init_model", type=str, required=True)
    args = parser.parse_args()
    print(args)

    # pylint: disable=E1123
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, 'log.txt')),
            logging.StreamHandler()
        ])

    num_tasks_dict = {
        'tox21': 12,
        'hiv': 1,
        'pcba': 128,
        'muv': 17,
        'bace': 1,
        'bbbp': 1,
        'toxcast': 617,
        'sider': 27,
        'clintox': 2
    }
    args.num_tasks = num_tasks_dict[args.dataset]

    main(args)
