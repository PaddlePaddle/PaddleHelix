# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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
This file implement the training process of S-MAN model for drug-target binding affinity prediction.
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np

import paddle.fluid as fluid
import paddle.fluid.layers as fl
import pgl
from pgl.utils.logger import log, logging

from dataset import DrugDataset, split_train_valid
from metrics import *
from dataloader import GraphDataloader
from model import SMANModel


def main(args, log_res):
    """main function"""

    test_dataset = DrugDataset(
        args.data_path,
        args.dataset,
        dist_dim=args.dist_dim,
        data_flag='test',
        self_loop=args.self_loop,
        use_identity=args.use_identity)

    if args.debug:
        train_dataset = test_dataset
        valid_dataset = test_dataset
    else:
        split_train_valid(args.data_path, args.dataset, seed=args.seed)
        train_dataset = DrugDataset(
            args.data_path,
            args.dataset,
            dist_dim=args.dist_dim,
            data_flag='train_',
            self_loop=args.self_loop,
            use_identity=args.use_identity)

        valid_dataset = DrugDataset(
            args.data_path,
            args.dataset,
            dist_dim=args.dist_dim,
            data_flag='valid',
            self_loop=args.self_loop,
            use_identity=args.use_identity)

    train_loader = GraphDataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = GraphDataloader(test_dataset, batch_size=args.batch_size, shuffle=False)
    valid_loader = GraphDataloader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    place = fluid.CUDAPlace(args.gpu) if args.gpu >= 0 else fluid.CPUPlace()
    train_program = fluid.Program()
    startup_program = fluid.Program()

    with fluid.program_guard(train_program, startup_program):
        e2n_gw = pgl.graph_wrapper.GraphWrapper(
            "e2n_gw", place=place, node_feat=train_dataset[0][0][0].node_feat_info(), edge_feat=train_dataset[0][0][0].edge_feat_info())
        e2e_gw = pgl.graph_wrapper.GraphWrapper(
            "e2e_gw", place=place, node_feat=[], edge_feat=train_dataset[0][0][1].edge_feat_info())

        model = SMANModel(args, e2n_gw, e2e_gw, n_output=1)
        model.forward()

    infer_program = train_program.clone(for_test=True)

    with fluid.program_guard(train_program, startup_program):
        if args.lr_d:
            epoch_step = int(len(train_dataset) / args.batch_size) + 1
            boundaries = [
                i
                for i in range(50 * epoch_step, args.epochs * epoch_step,
                            epoch_step * 50)
            ]
            values = [args.lr * 0.5**i for i in range(0, len(boundaries) + 1)]
            lr = fl.piecewise_decay(boundaries=boundaries, values=values)
        else:
            lr = args.lr
        train_op = fluid.optimizer.Adam(lr).minimize(model.loss)

    exe = fluid.Executor(place)
    exe.run(startup_program)

    # train and evaluate
    global_step = 0
    min_rmse = 2.0
    for epoch in range(1, args.epochs + 1):
        for idx, batch_data in enumerate(train_loader):
            e2n_g, e2e_g, edges_dist, nids, eids, nlod, elod, srcs, dsts, pk = batch_data
            feed_dict = {'srcs': srcs, 'dsts':dsts, 'edges_dist': edges_dist, 'nids': nids, 'eids': eids,
                         'node_lod': nlod, 'edge_lod': elod, 'pk': pk}
            feed_dict.update(e2n_gw.to_feed(e2n_g))
            feed_dict.update(e2e_gw.to_feed(e2e_g))
            if args.lr_d:
                ret_loss, ret_lr = exe.run(
                    train_program,
                    feed=feed_dict,
                    fetch_list=[model.loss, lr])
            else:
                ret_loss = exe.run(
                    train_program,
                    feed=feed_dict,
                    fetch_list=[model.loss])
                ret_loss = ret_loss[0]
                ret_lr = lr

            global_step += 1
            if global_step % 50 == 0:
                message = "epoch %d | step %d | " % (epoch, global_step)
                message += "lr %.6f | loss %.6f" % (ret_lr, ret_loss)
                log.info(message)

        # evaluate
        # continue
        valid_res = evaluate(exe, infer_program, model, e2n_gw, e2e_gw, valid_loader)
        test_res = evaluate(exe, infer_program, model, e2n_gw, e2e_gw, test_loader)

        message = "evaluating valid result"
        for key, value in valid_res.items():
            if key not in ['pks', 'preds']:
                message += " | %s %.6f" % (key, value)
        log.info(message)

        message = "evaluating test  result"
        for key, value in test_res.items():
            if key not in ['pks', 'preds']:
                message += " | %s %.6f" % (key, value)
        log.info(message)

        if args.debug:
            break
            
        if valid_res['rmse'] < min_rmse:
            min_rmse = valid_res['rmse']
            res_msg = "epoch: "+str(epoch)+", valid rmse: "+str(min_rmse)+", test rmse: "+str(test_res['rmse'])
            log_res.info(res_msg)

            model_name = args.dataset+'_SMAN'+args.info+'_rmse_'+str(round(test_res['rmse'],6))+'_epoch_'+str(epoch)
            param_path = os.path.join(args.save_path, model_name)
            fluid.io.save_params(executor=exe, dirname=param_path, main_program=train_program)
            with open(args.output_path+model_name+'.pickle', 'wb') as f:
                pickle.dump((test_res['pks'], test_res['preds']), f)



def evaluate(exe, prog, model, e2n_gw, e2e_gw, loader):
    """evaluate"""
    total_loss = []
    pred_list = []
    pk_list = []
    # print('evaluating')
    for idx, batch_data in enumerate(loader):
        e2n_g, e2e_g, edges_dist, nids, eids, nlod, elod, srcs, dsts, pk = batch_data
        feed_dict = {'srcs': srcs, 'dsts': dsts, 'edges_dist': edges_dist, 'nids': nids, 'eids': eids,
                        'node_lod': nlod, 'edge_lod': elod, 'pk': pk}
        feed_dict.update(e2n_gw.to_feed(e2n_g))
        feed_dict.update(e2e_gw.to_feed(e2e_g))
        ret_loss, pred = exe.run(prog,
                            feed=feed_dict,
                            fetch_list=[model.loss, model.output])
        total_loss.append(ret_loss)
        pred_list += pred.tolist()
        pk_list += pk.tolist()

    total_loss = np.mean(total_loss)
    preds = np.array(pred_list).reshape(-1,)
    pks = np.array(pk_list).reshape(-1,)

    return {'loss': total_loss, 'rmse': rmse(pks,preds), 'mse': mse(pks,preds), 'mae':mae(pks,preds), 'sd':sd(pks,preds),
            'pearson': pearson(pks,preds), 'spearman': spearman(pks,preds), 'ci': ci(pks,preds), 'preds':preds, 'pks':pks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--dataset', type=str, default='v2016_LPHIN3f5t_Sp')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_path', type=str, default='../outputs/')
    parser.add_argument('--save_path', type=str, default='../runs/')
    parser.add_argument('--info', type=str, default='')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hid_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dist_dim', type=int, default=4)
    parser.add_argument(
        '--pool_type',
        type=str,
        default="sum",
        choices=["sum", "average", "max"])
    parser.add_argument('--self_loop', action='store_true')
    parser.add_argument('--use_identity', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--lr_d', action='store_true')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--drop', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()


    formatter = logging.Formatter('%(process)d - %(asctime)s - %(message)s')
    if args.log and not args.debug:
        fh = logging.FileHandler('./log_'+args.dataset+'_SMAN'+args.info, 'a')
        fh.setFormatter(formatter)
        log.addHandler(fh)
        
    log_res = logging.getLogger()
    log_res.setLevel(logging.INFO)
    fh_res = logging.FileHandler('./res_'+args.dataset+'_SMAN'+args.info, 'a')
    fh_res.setFormatter(formatter)
    log_res.addHandler(fh_res)

    if not args.debug:
        log_res.info(args)
        log.info(args)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    main(args, log_res)

