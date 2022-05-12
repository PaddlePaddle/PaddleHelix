import os
import time
import math
import argparse
import random
import numpy as np

import paddle
import paddle.nn.functional as F
from dataloader import DualDataLoader as Dataloader
from dataset import Molecule2DView, Molecule3DView
from model import GeomGCL
from utils import split_dataset_gcl
from tqdm import tqdm

def setup_seed(seed):
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@paddle.no_grad()
def evaluate(model, loader):
    model.eval()
    sum_loss = 0
    for _ in range(loader.steps):
        graph_2d, graph_3d, _ = loader.next_batch()
        feat_2d, feat_3d = model(graph_2d, graph_3d)
        loss = model.loss(feat_2d, feat_3d)
        sum_loss += loss
    return sum_loss / len(loader)


def train(args, model, trn_loader, val_loader):
    epoch_step = len(trn_loader)
    boundaries = [i for i in range(args.dec_epoch*epoch_step, args.all_steps, args.dec_epoch*epoch_step)]
    values = [args.lr * args.lr_dec_rate ** i for i in range(0, len(boundaries) + 1)]
    scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=boundaries, values=values, verbose=False)
    optim = paddle.optimizer.Adagrad(learning_rate=scheduler, parameters=model.parameters(), weight_decay=args.weight_decay) # Adagrad, Adam

    min_loss = 1e9
    running_log, use_epoch = '', 0
    print('Start training model...')
    for epoch in range(1, int(args.all_steps/epoch_step) + 1):
        sum_loss = 0.
        model.train()
        start = time.time()
        for kk in range(trn_loader.steps):
            graph_2d, graph_3d, y = trn_loader.next_batch()
            feat_2d, feat_3d = model(graph_2d, graph_3d)
            loss = model.loss(feat_2d, feat_3d)
            loss.backward()
            optim.step()
            optim.clear_grad()
            scheduler.step()
            sum_loss += loss
        
        use_loss = sum_loss/(len(trn_loader)*args.batch_size)
        end_trn = time.time()
        log = 'Epoch: %d, loss: %.6f, time: %.2f,' % (epoch, use_loss, end_trn-start)
        if args.split_val:
            val_loss = evaluate(model, val_loader)
            end_val = time.time()
            log += 'val_loss: %.6f, val_time: %.2f.\n' % (val_loss/args.batch_size, end_val-end_trn)
            use_loss = val_loss

        if use_loss < min_loss:
            use_epoch = epoch
            min_loss = use_loss
            obj = {'model': model.state_dict(), 'epoch': epoch}

        print(log)
        running_log += log
        f = open(os.path.join(args.model_dir, 'dec_%d_%f_log.txt' % (args.dec_epoch, args.lr_dec_rate)), 'w')
        f.write(running_log)
        f.close()

        if epoch - use_epoch > args.tol_epoch:
            break

    path = os.path.join(args.model_dir, 'saved_model_%d_%3f' % (args.dec_epoch, args.lr_dec_rate))
    paddle.save(obj, path)

    f = open(os.path.join(args.model_dir, 'dec_%d_%f_log.txt' % (args.dec_epoch, args.lr_dec_rate)), 'w')
    f.write(running_log + 'Saved model at the %d epoch.\n' % (use_epoch))
    f.close()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='esol')
    parser.add_argument('--model_dir', type=str, default='./runs/esol/deafult')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--split_val", type=float, default=0.1)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_dec_rate", type=float, default=0.9)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--drop_pool", type=float, default=0.2)
    parser.add_argument("--dec_epoch", type=int, default=10)
    parser.add_argument('--tol_epoch', type=int, default=50)
    parser.add_argument('--all_steps', type=int, default=40000)

    parser.add_argument("--rbf_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_convs", type=int, default=2)
    parser.add_argument("--num_pool", type=int, default=2)
    parser.add_argument("--num_dist", type=int, default=0)
    parser.add_argument("--num_angle", type=int, default=4)
    parser.add_argument('--max_dist_2d', type=float, default=3.)
    parser.add_argument('--cut_dist', type=float, default=5.)
    parser.add_argument("--spa_w", type=float, default=0.1)
    parser.add_argument("--gcl_w", type=float, default=1)
    parser.add_argument("--tau", type=float, default=0.05)

    args = parser.parse_args()
    setup_seed(args.seed)
    if not args.num_dist:
        args.num_dist = 2 if args.cut_dist <= 4 else 4
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)

    if int(args.cuda) == -1:
        paddle.set_device('cpu')
    else:
        paddle.set_device('gpu:%s' % args.cuda)
    
    if args.dataset in ['esol', 'lipop', 'freesolv']:
        args.task = 'regression'
    elif args.dataset in ['clintox', 'sider', 'tox21', 'toxcast']:
        args.task = ' classification'
    else:
        print('The dataset %s is not included.' % args.dataset)
        exit(-1)
    
    data_2d = Molecule2DView(args.data_dir, args.dataset)
    data_3d = Molecule3DView(args.data_dir, args.dataset, args.cut_dist, args.num_angle, args.num_dist)
    assert len(data_2d) == len(data_3d)
    node_in_dim = data_2d.atom_feat_dim
    edge_in_dim = data_2d.bond_feat_dim
    atom_in_dim = data_3d.atom_feat_dim
    if args.split_val:
        trn_data_2d, val_data_2d = split_dataset_gcl(data_2d, args.split_val)
        trn_data_3d, val_data_3d = split_dataset_gcl(data_3d, args.split_val)
        val_loader = Dataloader(val_data_2d, val_data_3d, args.batch_size, False)
    else:
        trn_data_2d, val_loader_2d = data_2d, None
        trn_data_3d, val_loader_3d = data_3d, None
    trn_loader = Dataloader(trn_data_2d, trn_data_3d, args.batch_size, False)
    model = GeomGCL(node_in_dim, edge_in_dim, atom_in_dim, args.rbf_dim, args.hidden_dim, \
                    args.max_dist_2d, args.cut_dist, args.spa_w, args.gcl_w, args.tau, \
                    args.num_convs, args.num_pool, args.num_dist, args.num_angle, \
                    args.dropout, args.drop_pool, activation=F.relu)
    train(args, model, trn_loader, val_loader)