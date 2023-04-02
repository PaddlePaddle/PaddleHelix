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
"""
Traning scripts for regression tasks
"""

from helper import utils
import paddle
from paddle import nn
from paddle import io
from visualdl import LogWriter
import numpy as np
import pandas as pd
import json
import os
import random
from time import time
from argparse import ArgumentParser

from double_towers import MolTransModel
from preprocess import DataEncoder, concordance_index1
from util_function import (load_DAVIS, data_process, load_KIBA, load_ChEMBL_kd, load_ChEMBL_pkd,
                           load_BindingDB_kd, load_davis_dataset, load_kiba_dataset)
# Set seed for reproduction
paddle.seed(2)
np.random.seed(3)

# Whether to use GPU
#USE_GPU = True

# Set device as $export CUDA_VISIBLE_DEVICES='your device number'
use_cuda = paddle.is_compiled_with_cuda()
device = 'cuda:0' if use_cuda else 'cpu'
device = device.replace('cuda', 'gpu')
device = paddle.set_device(device)

# Set loss function
reg_loss_fn = paddle.nn.MSELoss()

# Initialize LogWriter
log_writer = LogWriter(logdir="./log")


def get_reg_db(db_name):
    """
    Get benchmark dataset for regression
    """
    if db_name.lower() == 'benchmark_davis':
        dataset = load_davis_dataset()
    elif db_name.lower() == 'benchmark_kiba':
        dataset = load_kiba_dataset()
    else:
        raise ValueError('%s not supported' % db_name)
    return dataset


def get_raw_db(db_name):
    """
    Get raw dataset for regression
    """
    # Load Davis data
    if db_name.lower() == 'raw_davis':
        X_drugs, X_targets, y = load_DAVIS(convert_to_log = True)
    # Load Kiba data
    elif db_name.lower() == 'raw_kiba':
        X_drugs, X_targets, y = load_KIBA()
    # Load ChEMBL Kd data
    elif db_name.lower() == 'raw_chembl_kd':
        X_drugs, X_targets, y = load_ChEMBL_kd()
    # Load ChEMBL pKd data
    elif db_name.lower() == 'raw_chembl_pkd':
        X_drugs, X_targets, y = load_ChEMBL_pkd()
    # Load BindingDB Kd data
    elif db_name.lower() == 'raw_bindingdb_kd':
        X_drugs, X_targets, y = load_BindingDB_kd()
    else:
        raise ValueError('%s not supported! ' % db_name)
    return X_drugs, X_targets, y


def reg_test(data_generator, model):
    """
    Test for regression task
    """
    y_pred = []
    y_label = []

    model.eval()
    for _, data in enumerate(data_generator):
        d_out, mask_d_out, t_out, mask_t_out, label = data
        temp = model(d_out.long().cuda(), t_out.long().cuda(), mask_d_out.long().cuda(), mask_t_out.long().cuda())

        label = paddle.cast(label, "float32")
        predicts = paddle.squeeze(temp, axis=1)

        loss = reg_loss_fn(predicts, label)
        predict_id = paddle.squeeze(temp).detach().cpu().numpy()
        label_id = label.to('cpu').numpy()

        y_label = y_label + label_id.flatten().tolist()
        y_pred = y_pred + predict_id.flatten().tolist()

        total_label = np.array(y_label)
        total_pred = np.array(y_pred)

        mse = ((total_label - total_pred) ** 2).mean(axis=0)
    return (mse, concordance_index1(np.array(y_label), np.array(y_pred)), loss.item())


def main(args):
    """
    Main function
    """
    # Basic setting
    optimal_mse = 10000
    log_iter = 50
    log_step = 0
    optimal_CI = 0

    # Load model config
    model_config = json.load(open(args.model_config, 'r'))
    model = MolTransModel(model_config)
    model = model.cuda()

    # Load pretrained model
    # params_dict= paddle.load('./pretrained_model/pdb2016_single_tower_1')
    # model.set_dict(params_dict)

    # Optimizer
    # scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=args.lr, warmup_steps=50, start_lr=0,
    #                                             end_lr=args.lr, verbose=False)
    optim = utils.Adam(parameters=model.parameters(), learning_rate=args.lr) # Adam
    #optim = paddle.optimizer.AdamW(learning_rate=scheduler, parameters=model.parameters(), weight_decay=0.01) # AdamW

    # Data Preparation
    # Regression Task - Raw Dataset
    # X_drugs, X_targets, y = get_raw_db(args.dataset)
    # reg_training_set, reg_validation_set, reg_testing_set = data_process(X_drugs, X_targets, y,
    #                     frac=[0.8,0.1,0.1], drug_encoding='Transformer', target_encoding='Transformer',
    #                     split_method='random_split', random_seed=1)
    # reg_training_data = DataEncoder(reg_training_set.index.values,
    #                     reg_training_set.Label.values, reg_training_set)
    # reg_train_loader = utils.BaseDataLoader(reg_training_data, batch_size=args.batchsize,
    #                     shuffle=True, drop_last=False, num_workers=args.workers)
    # reg_validation_data = DataEncoder(reg_validation_set.index.values,
    #                     reg_validation_set.Label.values, reg_validation_set)
    # reg_validation_loader = utils.BaseDataLoader(reg_validation_data, batch_size=args.batchsize,
    #                     shuffle=False, drop_last=False, num_workers=args.workers)
    # reg_testing_data = DataEncoder(reg_testing_set.index.values,
    #                     reg_testing_set.Label.values, reg_testing_set)
    # reg_testing_loader = utils.BaseDataLoader(reg_testing_data, batch_size=args.batchsize,
    #                     shuffle=False, drop_last=False, num_workers=args.workers)

    # Regression Task - Benchmark Dataset
    trainset, testset = get_reg_db(args.dataset)
    trainset_smiles = [d['smiles'] for d in trainset]
    trainset_protein = [d['protein'] for d in trainset]
    trainset_aff = [d['aff'] for d in trainset]

    testset_smiles = [d['smiles'] for d in testset]
    testset_protein = [d['protein'] for d in testset]
    testset_aff = [d['aff'] for d in testset]

    df_data_t = pd.DataFrame(zip(trainset_smiles, trainset_protein, trainset_aff))
    df_data_t.rename(columns={0:'SMILES', 1: 'Target Sequence', 2: 'Label'}, inplace=True)
    df_data_tt = pd.DataFrame(zip(testset_smiles, testset_protein, testset_aff))
    df_data_tt.rename(columns={0:'SMILES', 1: 'Target Sequence', 2: 'Label'}, inplace=True)

    reg_training_data = DataEncoder(df_data_t.index.values, df_data_t.Label.values, df_data_t)
    reg_train_loader = utils.BaseDataLoader(reg_training_data, batch_size=args.batchsize,
                                    shuffle=True, drop_last=False, num_workers=args.workers)
    reg_validation_data = DataEncoder(df_data_tt.index.values, df_data_tt.Label.values, df_data_tt)
    reg_validation_loader = utils.BaseDataLoader(reg_validation_data, batch_size=args.batchsize,
                                    shuffle=False, drop_last=False, num_workers=args.workers)

    # Initial Testing
    print("=====Go for Initial Testing=====")
    with paddle.no_grad():
        mse, CI, reg_loss = reg_test(reg_validation_loader, model)
        print("Testing result: MSE: {}, CI: {}"
              .format(mse, CI))
    
    # Training
    for epoch in range(args.epochs):
        print("=====Go for Training=====")
        model.train()
        # Regression Task
        for batch_id, data in enumerate(reg_train_loader):
            d_out, mask_d_out, t_out, mask_t_out, label = data
            temp = model(d_out.long().cuda(), t_out.long().cuda(), mask_d_out.long().cuda(), mask_t_out.long().cuda())
            label = paddle.cast(label, "float32")
            predicts = paddle.squeeze(temp)
            loss = reg_loss_fn(predicts, label)

            optim.clear_grad()
            loss.backward()
            optim.step()
            #scheduler.step()

            if batch_id % log_iter == 0:
                print("Training at epoch: {}, step: {}, loss is: {}"
                      .format(epoch, batch_id, loss.cpu().detach().numpy()))
                log_writer.add_scalar(tag="train/loss", step=log_step, value=loss.cpu().detach().numpy())
                log_step += 1
        
        # Validation
        print("=====Go for Validation=====")
        with paddle.no_grad():
            mse, CI, reg_loss = reg_test(reg_validation_loader, model)
            print("Validation at epoch: {}, MSE: {}, CI: {}, loss is: {}"
                  .format(epoch, mse, CI, reg_loss))
            log_writer.add_scalar(tag="dev/loss", step=log_step, value=reg_loss)
            log_writer.add_scalar(tag="dev/mse", step=log_step, value=mse)
            log_writer.add_scalar(tag="dev/CI", step=log_step, value=CI)
        
            # Save best model
            if mse < optimal_mse:
                optimal_mse = mse
                print("Saving the best_model with best MSE...")
                print("Best MSE: {}".format(optimal_mse))
                paddle.save(model.state_dict(), 'DAVIS_bestMSE_model_reg1')
            if CI > optimal_CI:
                optimal_CI = CI
                print("Saving the best_model with best CI...")
                print("Best CI: {}".format(optimal_CI))
                paddle.save(model.state_dict(), 'DAVIS_bestCI_model_reg1')
                
    # Print final result
    print("Best MSE: {}".format(optimal_mse))
    print("Best CI: {}".format(optimal_CI))
    paddle.save(model.state_dict(), 'DAVIS_final_model_reg1')

    # Load the model
    #params_dict= paddle.load('DAVIS_bestCI_model_reg1')
    #model.set_dict(params_dict)

    # Testing
    #print("=====Go for Testing=====")
    # Regression Task
    # with paddle.no_grad():
    #     try:
    #         mse, CI, reg_loss = reg_test(reg_testing_loader, model)
    #         print("Testing result: MSE: {}, CI: {}".format(mse, CI))
    #     except:
    #         print("Testing failed...")


if __name__ == "__main__":
    parser = ArgumentParser(description='Start Training...')
    parser.add_argument('-b', '--batchsize', default=64, type=int, metavar='N', help='Batch size')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='Number of workers')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='Number of total epochs')
    parser.add_argument('--dataset', choices=['raw_chembl_pkd', 'raw_chembl_kd', 'raw_bindingdb_kd', 'raw_davis',
                        'raw_kiba', 'benchmark_davis', 'benchmark_kiba'], default='benchmark_davis', type=str,
                        metavar='DATASET', help='Select specific dataset for your task')
    parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='Initial learning rate', dest='lr')
    parser.add_argument('--model_config', default='./config.json', type=str)
    args = parser.parse_args()

    beginT = time()
    print("Starting Time: {}".format(beginT))
    main(args)
    endT = time()
    print("Ending Time: {}".format(endT))
    print("Duration is: {}".format(endT - beginT))