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
Testing code for Structure-aware Interactive Graph Neural Networks (SIGN).
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

    if int(args.cuda) == -1:
        paddle.set_device('cpu')
    else:
        paddle.set_device('gpu:%s' % args.cuda)

    tst_complex = ComplexDataset(args.data_dir, "%s_test" % args.dataset, args.cut_dist, args.num_angle)
    tst_loader = Dataloader(tst_complex, args.batch_size, shuffle=False, num_workers=1, collate_fn=collate_fn)

    model = SIGN(args)
    path = os.path.join(args.model_dir, 'saved_model')
    load_state_dict = paddle.load(path)
    model.set_state_dict(load_state_dict['model'])
    rmse_tst, mae_tst, sd_tst, r_tst = evaluate(model, tst_loader)
    print('Test - RMSE: %.6f, MAE: %.6f, SD: %.6f, R: %.6f.\n' % (rmse_tst, mae_tst, sd_tst, r_tst))