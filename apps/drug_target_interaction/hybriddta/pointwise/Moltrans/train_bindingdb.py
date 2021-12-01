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

"""Training scripts for MolTrans backbone."""

from helper import utils
import paddle
from paddle import nn
from paddle import io
import numpy as np
import pandas as pd
import json
import os
import random
from time import time
from argparse import ArgumentParser

from model import MolTransModel
from preprocess import BindingDB_Encoder, concordance_index, mse

# Set seed for reproduction
paddle.seed(2)
np.random.seed(3)

# Set device as $export CUDA_VISIBLE_DEVICES='your device number'
use_cuda = paddle.is_compiled_with_cuda()
device = 'cuda:0' if use_cuda else 'cpu'
device = device.replace('cuda', 'gpu')
device = paddle.set_device(device)

# Set loss function
reg_loss_fn = paddle.nn.MSELoss()


def training(model, training_loader, optim):
    """Training script for MolTrans backbone model.
    
    Args:
        model: MolTrans backbone model.
        training_loader: Dataloader of training set.
        optim: Optimizer.

    Returns:
        res_loss: Ouput training loss.
    """
    model.train()
    for _, (d_out, mask_d_out, t_out, mask_t_out, label, _) in enumerate(training_loader):
        temp = model(d_out.long().cuda(), t_out.long().cuda(), mask_d_out.long().cuda(), mask_t_out.long().cuda())
        label = paddle.cast(label, "float32")
        predicts = paddle.squeeze(temp)
        loss = reg_loss_fn(predicts, label)

        optim.clear_grad()
        loss.backward()
        optim.step()
        res_loss = loss.cpu().detach().numpy()
    return res_loss


def predicting(model, testing_loader):
    """Predicting script for MolTrans backbone model.
    
    Args:
        model: MolTrans backbone model.
        testing_loader: Dataloader of validation/testing set.

    Returns:
        res_label: Output ground truth label.
        res_pred: Output prediction.
        res_group: Output groups.
    """
    model.eval()
    with paddle.no_grad():
        test_preds, test_trues, test_groups = paddle.to_tensor([]),paddle.to_tensor([]),paddle.to_tensor([])
        for _, (d_out, mask_d_out, t_out, mask_t_out, label, test_group) in enumerate(testing_loader):
            test_pred = model(d_out.long().cuda(), t_out.long().cuda(), mask_d_out.long().cuda(), mask_t_out.long().cuda())
            test_trues = paddle.concat(x = [test_trues, label.squeeze()], axis = 0)
            test_preds = paddle.concat(x = [test_preds, test_pred.squeeze()], axis = 0)
            test_groups = paddle.concat(x = [test_groups, test_group.squeeze()], axis = 0)

            res_label = test_trues.numpy()
            res_pred = test_preds.numpy()
            res_group = test_groups.numpy()

    return res_label, res_pred, res_group


def main(args):
    """Main function."""
    # Basic setting
    best_ci = 0
    best_epoch = 0
    best_train_loss = 10000
    rounds = args.rounds

    # Load model config
    model_config = json.load(open(args.model_config, 'r'))
    model = MolTransModel(model_config)
    model = model.cuda()

    # Optimizer
    optim = utils.Adam(parameters=model.parameters(), learning_rate=args.lr) # Adam

    # Load raw data
    train_data = pd.read_csv("../../Data/BindingDB/BindingDB_values_mixed_train_ki_filter.csv")
    val_data = pd.read_csv("../../Data/BindingDB/BindingDB_values_mixed_val_ki_filter.csv")
    test_data = pd.read_csv("../../Data/BindingDB/BindingDB_values_mixed_test_ki_filter.csv")

    train_set = BindingDB_Encoder(train_data.index.values, train_data)
    val_set = BindingDB_Encoder(val_data.index.values, val_data)
    test_set = BindingDB_Encoder(test_data.index.values, test_data)

    # Build dataloader
    train_loader = paddle.io.DataLoader(train_set, batch_size=args.batchsize, shuffle=True)
    val_loader = paddle.io.DataLoader(val_set, batch_size=args.batchsize, shuffle=False)
    test_loader = paddle.io.DataLoader(test_set, batch_size=args.batchsize, shuffle=False)
    
    # Training...
    for epoch in range(args.epochs):
        print("===============Go for Training===============")
        train_loss = training(model, train_loader, optim)

        # Validation...
        G, P, group_li = predicting(model, val_loader)
        val_ci = concordance_index(G,P)

        # Get length of validation set
        result = {}
        for gl in group_li:
            if result.get(gl)==None:
                result[gl] = 1
            else:
                result[gl] += 1

        lens = []
        lens.extend(result.values())

        # Skip len=1 data
        k = 0
        new_G, new_P, new_lens = [], [], []
        for ll in lens:
            if ll == 1:
                k += 1
            else:
                new_G.extend(G[k:k+ll])
                new_P.extend(P[k:k+ll])
                new_lens.append(ll)
                k += ll
        new_G,new_P = np.array(new_G),np.array(new_P)

        # Calculate Weighted CI, Average CI of validation set
        s = 0
        w_ci,a_ci = [],[]
        for l in new_lens:
            try:
                w_ci.append(l*concordance_index(new_G[s:s+l],new_P[s:s+l]))
                a_ci.append(concordance_index(new_G[s:s+l],new_P[s:s+l]))
            except:
                pass
            s += l
        weight_ci, average_ci = np.sum(w_ci)/np.sum(new_lens), np.mean(a_ci)
        print("===============Go for Validation===============")
        print("Weighted CI:",weight_ci)
        print("Average CI:",average_ci)
        print("Overall CI:",val_ci)

        files = open("bestResult/MolTrans_BindingDB_ki_result"+str(rounds)+".txt",'a')
        files.write("val_averageCI: "+str(average_ci)+", val_weightedCI: "+str(weight_ci)+", val_overallCI: "+str(val_ci)+", train_loss: "+str(train_loss)+'\n')
        model_name = "bestModel/MolTrans_BindingDB_ki_"+str(rounds)+".model"
        
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
    test_G, test_P, test_group_li = predicting(model, test_loader)
    test_CI,test_MSE = concordance_index(test_G,test_P), mse(test_G,test_P)

    # Get length of testing set
    t_result = {}
    for t_gl in test_group_li:
        if t_result.get(t_gl)==None:
            t_result[t_gl]=1
        else:
            t_result[t_gl]+=1

    t_lens = []
    t_lens.extend(t_result.values())    
        
    # Skip len=1 data
    t_k = 0
    t_new_G,t_new_P,t_new_lens = [],[],[]
    for t_ll in t_lens:
        if t_ll == 1:
            t_k += 1
        else:
            t_new_G.extend(test_G[t_k:t_k+t_ll])
            t_new_P.extend(test_P[t_k:t_k+t_ll])
            t_new_lens.append(t_ll)
            t_k += t_ll
    t_new_G,t_new_P = np.array(t_new_G),np.array(t_new_P)

    # Calculate Weighted CI, Average CI of testing set
    t_s = 0
    t_w_ci,t_a_ci = [],[]
    for t_l in t_new_lens:
        try:
            t_w_ci.append(t_l*concordance_index(t_new_G[t_s:t_s+t_l],t_new_P[t_s:t_s+t_l]))
            t_a_ci.append(concordance_index(t_new_G[t_s:t_s+t_l],t_new_P[t_s:t_s+t_l]))
        except:
            pass
        t_s += t_l
    test_weight_ci, test_average_ci = np.sum(t_w_ci)/np.sum(t_new_lens), np.mean(t_a_ci)

    # Save the testing result
    files.write("test_MSE:" + str(test_MSE) + ", test_averageCI:" + 
                str(test_average_ci) + ", test_weightedCI:" + str(test_weight_ci) + ", test_overallCI:" + str(test_CI) + "\n")
    files.write("best_epoch:" + str(best_epoch + 1) + ", best_train_loss:" + str(best_train_loss) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(description='Starting...')

    parser.add_argument('--batchsize', default=64, type=int, metavar='N', help='Batch size')
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='Number of total epochs')
    parser.add_argument('--rounds', default=1, type=int, metavar='N', help='The Nth round')
    parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='Initial learning rate', dest='lr')
    parser.add_argument('--model_config', default='./config.json', type=str, help='Model config')

    args = parser.parse_args()

    beginT = time()
    print("Starting Time: {}".format(beginT))
    main(args)
    endT = time()
    print("Ending Time: {}".format(endT))
    print("Duration is: {}".format(endT - beginT))