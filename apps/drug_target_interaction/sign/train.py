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
"""
Training process code for Structure-aware Interactive Graph Neural Networks (SIGN).
"""
import os
import time
import math
import argparse
import random
import numpy as np

import paddle
import paddle.nn.functional as F
from pgl.utils.data import Dataloader
from dataset import ComplexDataset, collate_fn
from model import SIGN
from utils import rmse, mae, sd, pearson
from tqdm import tqdm

paddle.seed(123)

def setup_seed(seed):
    # paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

@paddle.no_grad()
def evaluate(model, loader):
    model.eval()
    y_hat_list = []
    y_list = []
    for batch_data in loader:
        a2a_g, b2a_g, b2b_gl, feats, types, counts, y = batch_data
        _, y_hat = model(a2a_g, b2a_g, b2b_gl, types, counts)
        y_hat_list += y_hat.tolist()
        y_list += y.tolist()

    y_hat = np.array(y_hat_list).reshape(-1,)
    y = np.array(y_list).reshape(-1,)
    return rmse(y, y_hat), mae(y, y_hat), sd(y, y_hat), pearson(y, y_hat)


def train(args, model, trn_loader, tst_loader, val_loader):
    # learning rate decay and optimizer
    epoch_step = len(trn_loader)
    boundaries = [i for i in range(args.dec_step, args.epochs*epoch_step, args.dec_step)]
    values = [args.lr * args.lr_dec_rate ** i for i in range(0, len(boundaries) + 1)]
    scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=boundaries, values=values, verbose=False)
    optim = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())
    # l1_loss = paddle.nn.loss.L1Loss(reduction='sum')

    rmse_val_best, res_tst_best = 1e9, ''
    running_log = ''
    print('Start training model...')
    for epoch in range(1, args.epochs + 1):
        sum_loss, sum_loss_inter = 0, 0
        model.train()
        start = time.time()
        for batch_data in tqdm(trn_loader):
            a2a_g, b2a_g, b2b_gl, feats, types, counts, y = batch_data
            feats_hat, y_hat = model(a2a_g, b2a_g, b2b_gl, types, counts)

            # loss function
            loss = F.l1_loss(y_hat, y, reduction='sum')
            loss_inter = F.l1_loss(feats_hat, feats, reduction='sum')
            loss += args.lambda_ * loss_inter
            loss.backward()
            optim.step()
            optim.clear_grad()
            scheduler.step()
    
            sum_loss += loss
            sum_loss_inter += loss_inter

        end_trn = time.time()
        rmse_val, mae_val, sd_val, r_val = evaluate(model, val_loader)
        rmse_tst, mae_tst, sd_tst, r_tst = evaluate(model, tst_loader)
        end_val = time.time()
        log = '-----------------------------------------------------------------------\n'
        log += 'Epoch: %d, loss: %.4f, loss_b: %.4f, time: %.4f, val_time: %.4f.\n' % (
                epoch, sum_loss/(epoch_step*args.batch_size), sum_loss_inter/(epoch_step*args.batch_size), end_trn-start, end_val-end_trn)
        log += 'Val - RMSE: %.6f, MAE: %.6f, SD: %.6f, R: %.6f.\n' % (rmse_val, mae_val, sd_val, r_val)
        log += 'Test - RMSE: %.6f, MAE: %.6f, SD: %.6f, R: %.6f.\n' % (rmse_tst, mae_tst, sd_tst, r_tst)
        print(log)

        if rmse_val < rmse_val_best:
            rmse_val_best = rmse_val
            res_tst_best = 'Best - RMSE: %.6f, MAE: %.6f, SD: %.6f, R: %.6f.\n' % (rmse_tst, mae_tst, sd_tst, r_tst)
            if args.save_model:
                obj = {'model': model.state_dict(), 'epoch': epoch}
                path = os.path.join(args.model_dir, 'saved_model')
                paddle.save(obj, path)
                # model.save(os.path.join(args.model_dir, 'saved_model'))

        running_log += log
        f = open(os.path.join(args.model_dir, 'log.txt'), 'w')
        f.write(running_log)
        f.close()

    f = open(os.path.join(args.model_dir, 'log.txt'), 'w')
    f.write(running_log + res_tst_best)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='pdbbind2016')
    parser.add_argument('--model_dir', type=str, default='./output/sign')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument("--save_model", action="store_true", default=True)

    parser.add_argument("--lambda_", type=float, default=1.75)
    parser.add_argument("--feat_drop", type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--lr_dec_rate", type=float, default=0.5)
    parser.add_argument("--dec_step", type=int, default=8000)
    parser.add_argument('--stop_epoch', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=300)

    parser.add_argument("--num_convs", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--infeat_dim", type=int, default=36)
    parser.add_argument("--dense_dims", type=str, default='128*4,128*2,128')

    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--cut_dist', type=float, default=5.)
    parser.add_argument('--num_angle', type=int, default=6)
    parser.add_argument('--merge_b2b', type=str, default='cat')
    parser.add_argument('--merge_b2a', type=str, default='mean')

    args = parser.parse_args()
    args.activation = F.relu
    args.dense_dims = [eval(dim) for dim in args.dense_dims.split(',')]
    if args.seed:
        setup_seed(args.seed)

    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    
    if int(args.cuda) == -1:
        paddle.set_device('cpu')
    else:
        paddle.set_device('gpu:%s' % args.cuda)
    trn_complex = ComplexDataset(args.data_dir, "%s_train" % args.dataset, args.cut_dist, args.num_angle)
    tst_complex = ComplexDataset(args.data_dir, "%s_test" % args.dataset, args.cut_dist, args.num_angle)
    val_complex = ComplexDataset(args.data_dir, "%s_val" % args.dataset, args.cut_dist, args.num_angle)
    trn_loader = Dataloader(trn_complex, args.batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn)
    tst_loader = Dataloader(tst_complex, args.batch_size, shuffle=False, num_workers=1, collate_fn=collate_fn)
    val_loader = Dataloader(val_complex, args.batch_size, shuffle=False, num_workers=1, collate_fn=collate_fn)

    model = SIGN(args)
    train(args, model, trn_loader, tst_loader, val_loader)