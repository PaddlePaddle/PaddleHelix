#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

"""Training scripts for DeepDTA backbone."""

import os
import os.path
from os import path
import paddle
from paddle import nn
from paddle import io
import numpy as np
import pandas as pd
import json
import random
from time import time
from argparse import ArgumentParser

from model import DeepdtaModel
from preprocess import Basic_Encoder, concordance_index, mse
from get_len import get_kiba_len

# Set seed for reproduction
paddle.seed(10)

# Set device as $export CUDA_VISIBLE_DEVICES='your device number'
use_cuda = paddle.is_compiled_with_cuda()
device = 'cuda:0' if use_cuda else 'cpu'
device = device.replace('cuda', 'gpu')
device = paddle.set_device(device)

# Set loss function
loss_func = paddle.nn.MSELoss()


def training(model, training_loader, optim):
    """Training script for DeepDTA backbone model.
    
    Args:
        model: DeepDTA backbone model.
        training_loader: Dataloader of training set.
        optim: Optimizer.

    Returns:
        res_loss: Ouput training loss.
    """
    model.train()
    for _, (d_out, t_out, label) in enumerate(training_loader):
        temp = model(d_out, t_out)
        loss = loss_func(temp.astype('float32'), label.astype('float32'))

        optim.clear_grad()
        loss.backward()
        optim.step()
        res_loss = loss.numpy()[0]
    return res_loss


def predicting(model, testing_loader):
    """Predicting script for DeepDTA backbone model.
    
    Args:
        model: DeepDTA backbone model.
        testing_loader: Dataloader of validation/testing set.

    Returns:
        res_label: Output ground truth label.
        res_pred: Output prediction.
    """
    model.eval()
    with paddle.no_grad():
        test_preds, test_trues = paddle.to_tensor([]),paddle.to_tensor([])
        for _, (d_out, t_out, test_label) in enumerate(testing_loader):
            test_pred = model(d_out, t_out)
            test_trues = paddle.concat(x = [test_trues, test_label.squeeze()], axis = 0)
            test_preds = paddle.concat(x = [test_preds, test_pred.squeeze()], axis = 0)

            res_label = test_trues.numpy()
            res_pred = test_preds.numpy()
    return res_label, res_pred


def cal_len(path):
    """Calculate length of each group."""
    lines = open(path,'r').readlines()
    li = []
    for line in lines:
        li.append(int(line.strip()))
    lens = np.sum(li)
    return li, lens


def main(args):
    """Main function."""
    # Basic setting
    best_ci = 0
    best_epoch = 0
    best_train_loss = 10000
    rounds = args.rounds

    # Modeling...
    model = DeepdtaModel()

    # Optimizer
    optim = paddle.optimizer.Adam(parameters = model.parameters(), learning_rate = args.lr) # Adam

    # Load raw data
    train_data = pd.read_csv("../../Data/KIBA/CV"+str(rounds)+"/CV"+str(rounds)+"_KIBA_unseenP_seenD_train.csv")
    val_data = pd.read_csv("../../Data/KIBA/CV"+str(rounds)+"/CV"+str(rounds)+"_KIBA_unseenP_seenD_val.csv")
    test_data = pd.read_csv("../../Data/KIBA/test_KIBA_unseenP_seenD.csv")

    train_set = Basic_Encoder(train_data.index.values, train_data)
    val_set = Basic_Encoder(val_data.index.values, val_data)
    test_set = Basic_Encoder(test_data.index.values, test_data)

    # Build dataloader
    train_loader = paddle.io.DataLoader(train_set, batch_size=args.batchsize, shuffle=True)
    val_loader = paddle.io.DataLoader(val_set, batch_size=args.batchsize, shuffle=False)
    test_loader = paddle.io.DataLoader(test_set, batch_size=args.batchsize, shuffle=False)
    
    # Training...
    for epoch in range(args.epochs):
        print("===============Go for Training===============")
        train_loss = training(model, train_loader, optim)

        # Validation...
        G, P = predicting(model, val_loader)
        val_ci = concordance_index(G,P)

        val_path = "../../Data/KIBA/CV"+str(rounds)+"/CV"+str(rounds)+"_val.txt"
        # Check if kiba len file exists
        if(path.exists(val_path) == False):
            get_kiba_len()
        
        # Calculate Weighted CI, Average CI of validation set
        li, lens = cal_len(val_path)
        s = 0
        w_ci,a_ci = [],[]
        for l in li:
            try:
                w_ci.append(l*concordance_index(G[s:s+l],P[s:s+l]))
                a_ci.append(concordance_index(G[s:s+l],P[s:s+l]))
            except:
                pass
            s += l
        weight_ci, average_ci = np.sum(w_ci)/lens, np.mean(a_ci)
        
        print("===============Go for Validation===============")
        print("Weighted CI:",weight_ci)
        print("Average CI:",average_ci)
        print("Overall CI:",val_ci)

        files = open("bestResult/MolTrans_kiba_result"+str(rounds)+".txt",'a')
        files.write("val_averageCI: "+str(average_ci)+", val_weightedCI: "+str(weight_ci)+", val_overallCI: "+str(val_ci)+", train_loss: "+str(train_loss)+'\n')
        model_name = "bestModel/MolTrans_kiba_"+str(rounds)+".model"
        
        # Save the best result
        if average_ci > best_ci:
            best_ci = average_ci
            best_epoch = epoch
            best_train_loss = train_loss
            # Save best model
            print("Saving the best model...")
            paddle.save(model.state_dict(), model_name)

    print("===============Go for Testing===============")
    # Load the model
    params_dict= paddle.load(model_name)
    model.set_dict(params_dict)

    # Testing...
    test_G, test_P = predicting(model, test_loader)
    test_CI,test_MSE = concordance_index(test_G,test_P), mse(test_G,test_P)

    test_path = "../../Data/KIBA/kiba_len.txt"
    # Check if kiba len file exists
    if(path.exists(test_path) == False):
        get_kiba_len()
    # Calculate Weighted CI, Average CI of testing set
    t_li ,t_lens = cal_len(test_path)
    s = 0
    w_ci, a_ci = [], []
    for l in t_li:
        try:
            w_ci.append(l*concordance_index(G[s:s+l],P[s:s+l]))
            a_ci.append(concordance_index(G[s:s+l],P[s:s+l]))
        except:
            pass
        s += l
    test_weight_ci, test_average_ci = np.sum(w_ci)/t_lens, np.mean(a_ci)

    # Save the testing result
    files.write("test_MSE:" + str(test_MSE) + ", test_averageCI:" + str(test_average_ci) + 
                ", test_weightedCI:" + str(test_weight_ci) + ", test_overallCI:" + str(test_CI) + "\n")
    files.write("best_epoch:" + str(best_epoch + 1) + ", best_train_loss:" + str(best_train_loss) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(description='Starting...')

    parser.add_argument('--batchsize', default=256, type=int, metavar='N', help='Batch size')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='Number of total epochs')
    parser.add_argument('--rounds', default=1, type=int, metavar='N', help='The Nth round')
    parser.add_argument('--lr', default=1e-3, type=float, metavar='LR', help='Initial learning rate', dest='lr')

    args = parser.parse_args()

    beginT = time()
    print("Starting Time: {}".format(beginT))
    main(args)
    endT = time()
    print("Ending Time: {}".format(endT))
    print("Duration is: {}".format(endT - beginT))