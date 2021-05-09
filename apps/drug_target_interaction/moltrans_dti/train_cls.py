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
Traning scripts for classification tasks
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
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, 
                             precision_score, recall_score, auc, mean_squared_error)
from argparse import ArgumentParser

from double_towers import MolTransModel
from preprocess import DataEncoder

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
sig = paddle.nn.Sigmoid()
loss_fn = paddle.nn.BCELoss()

# Initialize LogWriter
log_writer = LogWriter(logdir="./log")


def get_cls_db(db_name):
    """
    Get benchmark dataset for classification
    """
    if db_name.lower() == 'cls_davis':
        return './dataset/classification/DAVIS'
    elif db_name.lower() == 'cls_biosnap':
        return './dataset/classification/BIOSNAP/full_data'
    elif db_name.lower() == 'cls_bindingdb':
        return './dataset/classification/BindingDB'


def cls_test(data_generator, model):
    """
    Test for classification task
    """
    y_pred = []
    y_label = []
    loss_res = 0.0
    count = 0.0

    model.eval()    
    for _, data in enumerate(data_generator):
        d_out, mask_d_out, t_out, mask_t_out, label = data
        temp = model(d_out.long().cuda(), t_out.long().cuda(), mask_d_out.long().cuda(), mask_t_out.long().cuda())
        predicts = paddle.squeeze(sig(temp))
        label = paddle.cast(label, "float32")

        loss = loss_fn(predicts, label)
        loss_res += loss
        count += 1

        predicts = predicts.detach().cpu().numpy()
        label_id = label.to('cpu').numpy()
        y_label = y_label + label_id.flatten().tolist()
        y_pred = y_pred + predicts.flatten().tolist()
    loss = loss_res / count

    fpr, tpr, threshold = roc_curve(y_label, y_pred)
    precision = tpr / (tpr + fpr)
    f1 = 2 * precision * tpr / (tpr + precision + 1e-05)
    optimal_threshold = threshold[5:][np.argmax(f1[5:])]
    print("Optimal threshold: {}".format(optimal_threshold))

    y_pred_res = [(1 if i else 0) for i in y_pred >= optimal_threshold]
    auroc = auc(fpr, tpr)
    print("AUROC: {}".format(auroc))
    print("AUPRC: {}".format(average_precision_score(y_label, y_pred)))

    cf_mat = confusion_matrix(y_label, y_pred_res)
    print("Confusion Matrix: \n{}".format(cf_mat))
    print("Precision: {}".format(precision_score(y_label, y_pred_res)))
    print("Recall: {}".format(recall_score(y_label, y_pred_res)))

    total_res = sum(sum(cf_mat))
    accuracy = (cf_mat[0, 0] + cf_mat[1, 1]) / total_res
    print("Accuracy: {}".format(accuracy))
    sensitivity = cf_mat[0, 0] / (cf_mat[0, 0] + cf_mat[0, 1])
    print("Sensitivity: {}".format(sensitivity))
    specificity = cf_mat[1, 1] / (cf_mat[1, 0] + cf_mat[1, 1])
    print("Specificity: {}".format(specificity))
    outputs = np.asarray([(1 if i else 0) for i in np.asarray(y_pred) >= 0.5])
    return (roc_auc_score(y_label, y_pred), 
            f1_score(y_label, outputs), loss.item())


def main(args):
    """
    Main function
    """
    # Basic setting
    optimal_auc = 0
    log_iter = 50
    log_step = 0

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
    data_path = get_cls_db(args.dataset)
    training_set = pd.read_csv(data_path + '/train.csv')
    validation_set = pd.read_csv(data_path + '/val.csv')
    testing_set = pd.read_csv(data_path + '/test.csv')

    training_data = DataEncoder(training_set.index.values, training_set.Label.values, training_set)
    train_loader = utils.BaseDataLoader(training_data, batch_size=args.batchsize, shuffle=True, 
                                        drop_last=False, num_workers=args.workers)
    validation_data = DataEncoder(validation_set.index.values, validation_set.Label.values, validation_set)
    validation_loader = utils.BaseDataLoader(validation_data, batch_size=args.batchsize, shuffle=False, 
                                             drop_last=False, num_workers=args.workers)
    testing_data = DataEncoder(testing_set.index.values, testing_set.Label.values, testing_set)
    testing_loader = utils.BaseDataLoader(testing_data, batch_size=args.batchsize, shuffle=False, 
                                          drop_last=False, num_workers=args.workers)
    # Initial Testing
    print("=====Start Initial Testing=====")
    with paddle.no_grad():
        auroc, f1, loss = cls_test(testing_loader, model)
        print("Initial testing set: AUROC: {}, F1: {}, Testing loss: {}".format(auroc, f1, loss))    
    
    # Training
    for epoch in range(args.epochs):
        print("=====Start Training=====")
        model.train()
        for batch_id, data in enumerate(train_loader):
            d_out, mask_d_out, t_out, mask_t_out, label = data
            temp = model(d_out.long().cuda(), t_out.long().cuda(), mask_d_out.long().cuda(), mask_t_out.long().cuda())
            label = paddle.cast(label, "float32")
            predicts = paddle.squeeze(sig(temp))
            loss = loss_fn(predicts, label)

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
        print("=====Start Validation=====")
        with paddle.no_grad():
            auroc, f1, loss = cls_test(validation_loader, model) 
            print("Validation at epoch: {}, AUROC: {}, F1: {}, loss is: {}"
                  .format(epoch, auroc, f1, loss))
            log_writer.add_scalar(tag="dev/loss", step=log_step, value=loss)
        
            # Save best model
            if auroc > optimal_auc:
                optimal_auc = auroc
                print("Saving the best_model...")
                print("Best AUROC: {}".format(optimal_auc))
                paddle.save(model.state_dict(), 'DAVIS_bestAUC_model_cls1')             
    
    print("Final AUROC: {}".format(optimal_auc))
    paddle.save(model.state_dict(), 'DAVIS_final_model_cls1')

    # Load the trained model
    params_dict= paddle.load('DAVIS_bestAUC_model_cls1')
    model.set_dict(params_dict)

    # Testing
    print("=====Start Testing=====")
    with paddle.no_grad():
        try:
            auroc, f1, loss = cls_test(testing_loader, model)
            print("Testing result: AUROC: {}, F1: {}, Testing loss is: {}".format(auroc, f1, loss))
        except:
            print("Testing failed...")


if __name__ == "__main__":
    parser = ArgumentParser(description='Start Training...')
    parser.add_argument('-b', '--batchsize', default=64, type=int, metavar='N', help='Batch size')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='Number of workers')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='Number of total epochs')
    parser.add_argument('--dataset', choices=['cls_davis', 'cls_biosnap', 'cls_bindingdb'], default='cls_davis', 
                        type=str, metavar='DATASET', help='Select specific dataset for your task')
    parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='Initial learning rate', dest='lr')
    parser.add_argument('--model_config', default='./config.json', type=str)
    args = parser.parse_args()

    beginT = time()
    print("Starting Time: {}".format(beginT))
    main(args)
    endT = time()
    print("Ending Time: {}".format(endT))
    print("Duration is: {}".format(endT - beginT))