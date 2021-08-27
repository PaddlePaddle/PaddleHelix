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
This file implement the testing process of S-MAN model for drug-target binding affinity prediction.
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

from dataset import DrugDataset
from metrics import *
from dataloader import GraphDataloader
from model import SMANModel

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
            'pearson': pearson(pks,preds), 'spearman': spearman(pks,preds), 'ci': ci(pks,preds)}



parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='../data/')
parser.add_argument('--dataset', type=str, default='v2016_LPHIN3f5t_Sp')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model_path', type=str, default='../runs/v2016_LPHIN3f5t_Sp_SMAN')
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
parser.add_argument('--lr_d', action='store_true')
parser.add_argument('--log', action='store_true')
parser.add_argument('--drop', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

if not os.path.exists(args.model_path):
    print('%s not exist.' % args.model_path)
    exit(-1)

test_dataset = DrugDataset(
    args.data_path,
    args.dataset,
    dist_dim=args.dist_dim,
    data_flag='test',
    self_loop=args.self_loop,
    use_identity=args.use_identity)
test_loader = GraphDataloader(test_dataset, batch_size=args.batch_size, shuffle=False)


place = fluid.CUDAPlace(args.gpu) if args.gpu >= 0 else fluid.CPUPlace()
exe = fluid.Executor(place)
prog = fluid.default_main_program()

with fluid.program_guard(prog):
        e2n_gw = pgl.graph_wrapper.GraphWrapper(
            "e2n_gw", place=place, node_feat=test_dataset[0][0][0].node_feat_info(), edge_feat=test_dataset[0][0][0].edge_feat_info())
        e2e_gw = pgl.graph_wrapper.GraphWrapper(
            "e2e_gw", place=place, node_feat=[], edge_feat=test_dataset[0][0][1].edge_feat_info())

        model = SMANModel(args, e2n_gw, e2e_gw, n_output=1)
        model.forward()

fluid.io.load_params(executor=exe, dirname=args.model_path,
                     main_program=prog)
infer_program = prog.clone(for_test=True)

print(evaluate(exe, infer_program, model, e2n_gw, e2e_gw, test_loader))