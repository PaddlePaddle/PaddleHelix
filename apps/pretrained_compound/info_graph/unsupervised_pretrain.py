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
import paddle.fluid as F
import paddle.fluid.layers as L
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
from paddle.fluid.incubate.fleet.base import role_maker
import pgl
from pgl.utils import paddle_helper
from pgl.graph_wrapper import GraphWrapper
from pgl.utils.data.dataloader import Dataloader
from pahelix.datasets import load_mutag_dataset, load_ptc_mr_dataset

from model import GINEncoder, FF, PriorDiscriminator
from data_gen import MoleculeCollateFunc
from classifier import eval_on_classifiers
from utils import get_positive_expectation, get_negative_expectation, load_data


def create_model(args, config):
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

    # NOTE: [num_nodes, num_graphs], bs = num_graphs
    pos_mask = L.data(name='pos_mask', shape=[-1, args.batch_size],
                      dtype='float32')
    neg_mask = L.data(name='neg_mask', shape=[-1, args.batch_size],
                      dtype='float32')

    encoder = GINEncoder(config)
    global_repr, patch_summary = encoder.forward(graph_wrapper)

    global_D = FF(encoder.embedding_dim)
    local_D = FF(encoder.embedding_dim)
    g_enc = global_D.forward(global_repr)
    l_enc = local_D.forward(patch_summary)

    res = L.matmul(l_enc, g_enc, transpose_y=True)
    E_pos = get_positive_expectation(res * pos_mask, config['measure'], average=False)
    E_pos = L.reduce_sum(E_pos) / graph_wrapper.num_nodes
    E_neg = get_negative_expectation(res * neg_mask, config['measure'], average=False)
    E_neg = L.reduce_sum(E_neg) / (graph_wrapper.num_nodes * (graph_wrapper.num_graph - 1))
    local_global_loss = E_neg - E_pos

    if config['prior']:
        prior_D = PriorDiscriminator(encoder.embedding_dim)
        prior = L.uniform_random([args.batch_size, encoder.embedding_dim],
                                 min=0.0, max=1.0)
        term_1 = L.reduce_mean(L.log(prior_D.forward(prior)))
        term_2 = L.reduce_mean(L.log(1.0 - prior_D.forward(global_repr)))
        prior_loss = - (term_1 + term_2) * config['gamma']
    else:
        prior_loss = 0

    total_loss = local_global_loss + prior_loss

    keys = ['loss', 'graph_wrapper', 'encoder', 'graph_emb']
    Agent = namedtuple('Agent', keys)
    return Agent(
        loss=total_loss,
        graph_wrapper=graph_wrapper,
        encoder=encoder,
        graph_emb=global_repr)


def train(args, exe, train_prog, agent, train_data_list, epoch_id):
    """Model training for one epoch and log the average loss."""
    collate_fn = MoleculeCollateFunc(
        agent.graph_wrapper,
        task_type='cls',
        with_graph_label=False,  # for unsupervised learning
        with_pos_neg_mask=True)
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
            logging.info('Epoch %d [%d/%d] train/loss:%f' % \
                         (epoch_id, trained_data, total_data, train_loss))

    if not args.is_fleet or fleet.worker_index() == 0:
        logging.info('Epoch %d train/loss:%f'%(epoch_id, np.mean(list_loss)))
        sys.stdout.flush()


def save_embedding(args, exe, test_prog, agent, data_list, epoch_id):
    """Save the embeddings of all the testing data for multiple classifier evaluation."""
    collate_fn = MoleculeCollateFunc(
        agent.graph_wrapper,
        task_type='cls',
        with_graph_label=True,  # save emb & label for supervised learning
        with_pos_neg_mask=True)
    data_loader = Dataloader(
        data_list,
        batch_size=args.batch_size,
        num_workers=1,
        shuffle=False,
        collate_fn=collate_fn)

    emb, y = agent.encoder.get_embeddings(
        data_loader, exe, test_prog, agent.graph_emb)
    emb, y = emb[:len(data_list)], y[:len(data_list)]
    merge_data = {'emb': emb, 'y': y}
    with open('%s/epoch_%s.pkl' % (args.emb_dir, epoch_id), 'wb') as f:
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
    # Enable static graph mode.
    paddle.enable_static()

    with open(args.config, 'r') as f:
        config = json.load(f)

    logging.info('Load data ...')
    if len(args.dataset.split(',')) > 1:
        # for large pretraining dataset, ZINC15 and ChEMBL
        # directly load the processed npz files
        train_data_list = []
        for ds in args.dataset.split(','):
            # use processed data.npz
            train_data_list.extend(
                load_data(os.path.join(args.root, ds, 'processed')))
    else:
        if args.dataset == 'mutag':
            train_data_list, _ = load_mutag_dataset(
                os.path.join(args.root, args.dataset, 'raw'))
        elif args.dataset == 'ptc_mr':
            train_data_list, _ = load_ptc_mr_dataset(
                os.path.join(args.root, args.dataset, 'raw'))
        else:
            raise ValueError('Unsupported dataset')

    if args.is_fleet:
        train_data_list = [x for i, x in enumerate(train_data_list)
                if i % fleet.worker_num() == fleet.worker_index()]
    logging.info("Data loaded.")
    logging.info("Train Examples: %s" % len(train_data_list))
    sys.stdout.flush()

    if args.emb_dir is not None:
        # pylint: disable=E1123
        os.makedirs(args.emb_dir, exist_ok=True)

    train_prog = F.Program()
    test_prog = F.Program()
    startup_prog = F.Program()
    with F.program_guard(train_prog, startup_prog):
        with F.unique_name.guard():
            agent = create_model(args, config)
            test_prog = train_prog.clone(for_test=True)

            opt = F.optimizer.Adam(learning_rate=args.lr)
            if args.is_fleet:
                dist_strategy = DistributedStrategy()
                role = role_maker.PaddleCloudRoleMaker(is_collective=True)
                fleet.init(role)
                opt = fleet.distributed_optimizer(opt, strategy=dist_strategy)
            opt.minimize(agent.loss)

    place = F.CUDAPlace(0) if args.use_cuda else F.CPUPlace()
    exe = F.Executor(place)
    exe.run(startup_prog)

    if (not args.dont_save_emb) and \
       (not args.is_fleet or fleet.worker_index() == 0):
        save_embedding(args, exe, test_prog, agent, train_data_list, -1)

    for epoch_id in range(args.max_epoch):
        train(args, exe, train_prog, agent, train_data_list, epoch_id)
        if not args.is_fleet or fleet.worker_index() == 0:
            F.io.save_params(exe, '%s/epoch%s' % (args.model_dir, epoch_id), train_prog)
            if not args.dont_save_emb:
                save_embedding(args, exe, test_prog, agent, train_data_list, epoch_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='InfoGraph pretrain')
    parser.add_argument("--task_name", type=str, default='train')
    parser.add_argument("--is_fleet", action='store_true', default=False)
    parser.add_argument("--use_cuda", action='store_true', default=False)
    # parser.add_argument("--data_loader", action='store_true', default=False)
    parser.add_argument("--config", type=str, default='demos/unsupervised_pretrain_config.json')

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--root", type=str, default='dataset')
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
