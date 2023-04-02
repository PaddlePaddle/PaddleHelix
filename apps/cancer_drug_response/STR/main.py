#!/usr/bin/python
# coding=utf-8
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
train
"""
import time
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import logging
from scipy.stats import pearsonr
import random
import paddle
import paddle.nn as nn
from paddle.optimizer import Adam
import csv
import pgl
from pgl.utils.logger import log
from pgl.utils.data import Dataloader
from pahelix.utils.data_utils import load_npz_to_data_list
from pahelix.datasets.inmemory_dataset import InMemoryDataset
from generate_dataset import Dataset, collate_fn, Dataloader_sampler, CrossSampler, Create_Dataset, Sampler
from model import STRModel, CDRModel
from utils import pcc_cal, drugpcc_cal, cclepcc_cal
import pdb
import copy
from loss import ranking_loss
def main(args):
    """
    Model training for one epoch and return the average loss and model evaluating to monitor pcc.
    """
    t0 = time.time()
    logging.basicConfig(level = logging.DEBUG , filename = os.path.join(args.output_path, args.task + '_log.txt'), filemode='a')
    logging.info(args)
    paddle.set_device('gpu:{}'.format(args.device) if args.use_cuda else 'cpu')
    logging.info('Load data ...')
    dataset = InMemoryDataset(npz_data_path=args.data_path)
    train_ds, test_ds, ccle_datasetmaxmin, drug_datasetmaxmin = Create_Dataset(args, dataset, logging)
    max_min_dic = {'ccle':ccle_datasetmaxmin, 'drug':drug_datasetmaxmin}
    t1 = time.time()
    if args.model == 'CDR':
        model = CDRModel(args)
        logging.info('CDR model !')
        print('CDR model !')
    elif args.model == 'STR':
        model = STRModel(args)
        logging.info('STR model !')
        print('STR model !')
    else:
        print('error unknown model !')
    args.start_epoch = 0
    if args.mode == 'train':
        train(args, model, train_ds, test_ds, logging, max_min_dic)
        test_pcc(args, model, test_ds, max_min_dic)
    elif args.mode == 'test':
        test_pcc(args, model, test_ds, max_min_dic)
    elif args.mode == 'continue':
        model.set_state_dict(paddle.load(best_model))
        print(best_model,'loaded!')
        args.start_epoch = 44
    else:
        print('wrong mode !')
    t2 = time.time()
    print('load data time:',t1-t0)
    print('inference time:',t2-t1)
    return 'finshed !'

def train(args, model, train_ds, test_ds, logging, max_min_dic):
    if args.sampler == 'sampler':
        train_sampler = CrossSampler(train_ds, batch_size=args.batch_size, num_instances=args.batch_size//2)
        test_sampler = CrossSampler(test_ds, batch_size=args.batch_size, num_instances=args.batch_size//2)
        print('Use cross sampler !')
        logging.info('Use cross sampler !')
    else:
        train_sampler = Sampler(train_ds, batch_size=args.batch_size, drop_last=False)
        test_sampler = Sampler(test_ds, batch_size=args.batch_size, drop_last=False)
        print('Do not use sampler !')
        logging.info('Do not use sampler !')

    train_loader = Dataloader_sampler(train_ds, drop_last = True, batch_size=args.batch_size, collate_fn=collate_fn, sampler=train_sampler)
    test_loader = Dataloader(test_ds, shuffle=False, batch_size=args.batch_size, collate_fn=collate_fn)
    logging.info("Data loaded.")
    optim = Adam(learning_rate=args.lr, parameters=model.parameters())
    criterion = nn.MSELoss()
    global_step = 0
    best_pcc = 0.0
    os.makedirs(args.output_path, exist_ok=True)
    best_model = os.path.join(args.output_path, args.task + '_best_model.pdparams')
    end_model = os.path.join(args.output_path, args.task + '_end_model.pdparams')
    lamda = args.lamda
    print('lamda:',lamda)

    for epoch in range(args.start_epoch, args.epoch_num + 1):
        model.train()
        for idx, batch_data in enumerate(train_loader):
            graphs, mut, gexpr, met, label, _ = batch_data
            g = pgl.Graph.batch(graphs).tensor()
            mut = paddle.to_tensor(mut)
            gexpr = paddle.to_tensor(gexpr)
            met = paddle.to_tensor(met)
            label = paddle.to_tensor(label)

            pred = model([g, mut, gexpr, met])
            if args.sampler == 'sampler':
                ccle_loss = criterion(pred[:args.batch_size//2, 0], label[:args.batch_size//2])[0]  # same ccle
                drug_loss = criterion(pred[args.batch_size//2:, 0], label[args.batch_size//2:])[0] # same drug
                train_loss = ccle_loss + lamda * drug_loss
            else:
                ccle_loss, drug_loss = 0, 0
                train_loss = paddle.pow(criterion(pred[:, 0], label)[0], 0.5)

            if args.use_rankloss:
                num_pairs = label.shape[0]//2
                rank_loss = ranking_loss(pred,label,num_pairs)
                train_loss = args.beta*train_loss + (1-args.beta)*rank_loss
            train_loss.backward()
            train_pcc = pearsonr(pred[:, 0].numpy(), label.numpy())[0]
            if args.sampler == 'sampler':
                train_ccle_pcc = pearsonr(pred[:args.batch_size//2, 0].numpy(), label[:args.batch_size//2].numpy())[0]
                train_drug_pcc = pearsonr(pred[args.batch_size//2:, 0].numpy(), label[args.batch_size//2:].numpy())[0]
            else:
                train_ccle_pcc, train_drug_pcc = 0, 0
            optim.step()
            optim.clear_grad()
            global_step += 1
            if global_step % 100 == 0:
                message = "train: epoch %d | step %d | " % (epoch, global_step)
                message += "loss %.6f | pcc %.4f" % (train_loss, train_pcc)
                if args.use_rankloss:
                    message += "| rank_loss %.6f " % (rank_loss)
                logging.info(message)
                log.info(message)

        result = evaluate(args, model, test_loader, criterion, max_min_dic, epoch)
        message = "eval: epoch %d | step %d " % (epoch, global_step)
        for key, value in result.items():
            message += "| %s %.6f" % (key, value)
        logging.info(message)
        log.info(message)
        if best_pcc < result['pcc']:
            best_pcc = result['pcc']
            nepoch = epoch
            paddle.save(model.state_dict(), best_model)
        epoch_model = os.path.join(args.output_path, args.task +str(epoch)+'_model.pdparams')
        paddle.save(model.state_dict(), epoch_model)
    paddle.save(model.state_dict(), end_model)
    print('best_pcc: ',best_pcc)
    print('best_pcc in the',nepoch)
    logging.info("best evaluating accuracy: %.6f" % best_pcc)

def test_pcc(args, model, test_ds, max_min_dic):
    model.eval()
    test_loader = Dataloader(test_ds, shuffle=False, batch_size=args.batch_size, collate_fn=collate_fn)
    criterion = nn.MSELoss()
    logging.info("Data loaded.")
    print('load best model params ...')
    model.eval()
    best_model = os.path.join(args.output_path, 'STR_best_model.pdparams')
    end_model = os.path.join(args.output_path, args.task + '_end_model.pdparams')
    model.set_state_dict(paddle.load(best_model))
    print(best_model,'loaded!')
    preds = []
    labels = []
    types = []
    total_loss = []
    total_pcc = []
    out = open(os.path.join(args.output_path  ,args.task + '_end_results.csv'),'a', newline='')
    csv_write = csv.writer(out,dialect='excel')
    for idx, batch_data in enumerate(test_loader):
        graphs, mut, gexpr, met, label, cancer_type_list = batch_data
        g = pgl.Graph.batch(graphs).tensor()
        mut = paddle.to_tensor(mut)
        gexpr = paddle.to_tensor(gexpr)
        met = paddle.to_tensor(met)
        label = paddle.to_tensor(label)
        pred = model([g, mut, gexpr, met])
        pred_denorm = pred.clone()
        label_denorm = label.clone()
        if args.data_norm == 'norm':
            ccle_datasetmaxmin = max_min_dic['ccle']
            for i in range(len(cancer_type_list)):
                dis=ccle_datasetmaxmin ['Max'][cancer_type_list[i][1]]-ccle_datasetmaxmin ['Min'][cancer_type_list[i][1]]
                pred[i] = (pred[i]+5)/10*dis+ccle_datasetmaxmin ['Min'][cancer_type_list[i][1]]
                label[i] = (label[i]+5)/10*dis+ccle_datasetmaxmin ['Min'][cancer_type_list[i][1]]
                file_to_write = [cancer_type_list[i][0], cancer_type_list[i][1], cancer_type_list[i][2], pred[i].item(), label[i].item()]
                csv_write.writerow(file_to_write)

        elif args.data_norm == 'norm_drug':
            drug_datasetmaxmin = max_min_dic['drug']
            for i in range(len(cancer_type_list)):
                dis = drug_datasetmaxmin['Max'][cancer_type_list[i][0]] - drug_datasetmaxmin['Min'][cancer_type_list[i][0]]
                pred_denorm[i] = (pred_denorm[i]+5)/10*dis + drug_datasetmaxmin['Min'][cancer_type_list[i][0]]
                label_denorm[i] = (label_denorm[i]+5)/10*dis + drug_datasetmaxmin['Min'][cancer_type_list[i][0]]
                file_to_write = [cancer_type_list[i][0], cancer_type_list[i][1], cancer_type_list[i][2], pred_denorm[i].item(), label_denorm[i].item()]
                csv_write.writerow(file_to_write)
        else:
            for i in range(len(cancer_type_list)):
                file_to_write = [cancer_type_list[i][0], cancer_type_list[i][1], cancer_type_list[i][2], pred[i].item(), label[i].item()]
                csv_write.writerow(file_to_write)
        eval_loss = paddle.pow(criterion(pred[:, 0], label)[0], 0.5)
        eval_pcc = pearsonr(pred[:, 0].numpy(), label.numpy())[0]
        total_loss.append(eval_loss.numpy())
        total_pcc.append(eval_pcc)
    total_loss = np.mean(total_loss)
    total_pcc = np.mean(total_pcc)
    model.train()
    log.info("best evaluating loss: %.6f accuracy: %.6f" % (total_loss, total_pcc))


def evaluate(args, model, loader, criterion, max_min_dic, epoch):
    """
    Evaluate the model on the test dataset and return average loss and pcc.
    """
    model.eval()
    total_loss = []
    total_pcc = []
    total_ccle_pcc = []
    total_drug_pcc = []
    pred_all = []
    gt_all =[]

    pred_all_denorm, gt_all_denorm = [], []
    for idx, batch_data in enumerate(loader):
        graphs, mut, gexpr, met, label, cancer_type_list = batch_data
        g = pgl.Graph.batch(graphs).tensor()
        mut = paddle.to_tensor(mut)
        gexpr = paddle.to_tensor(gexpr)
        met = paddle.to_tensor(met)
        label = paddle.to_tensor(label)

        pred = model([g, mut, gexpr, met])
        eval_loss = paddle.pow(criterion(pred[:, 0], label)[0], 0.5)
        eval_pcc = pearsonr(pred[:, 0].numpy(), label.numpy())[0]

        total_loss.append(eval_loss.numpy())
        pred_all.append(pred[:, 0].numpy())
        gt_all.append(label.numpy())
        out = open(os.path.join(args.output_path, args.task + '_' + str(epoch) + '_end_results.csv'),'a', newline='')
        csv_write = csv.writer(out,dialect='excel')
        pred_denorm = pred.clone()
        label_denorm = label.clone()
        if args.data_norm == 'norm':
            ccle_datasetmaxmin = max_min_dic['ccle']
            for i in range(len(cancer_type_list)):
                dis = ccle_datasetmaxmin['Max'][cancer_type_list[i][1]] - ccle_datasetmaxmin['Min'][cancer_type_list[i][1]]
                pred_denorm[i] = (pred_denorm[i]+5)/10*dis + ccle_datasetmaxmin['Min'][cancer_type_list[i][1]]
                label_denorm[i] = (label_denorm[i]+5)/10*dis + ccle_datasetmaxmin['Min'][cancer_type_list[i][1]]
                file_to_write = [cancer_type_list[i][0], cancer_type_list[i][1], cancer_type_list[i][2], pred_denorm[i].item(), label_denorm[i].item()]
                csv_write.writerow(file_to_write)

        elif args.data_norm == 'norm_drug':
            drug_datasetmaxmin = max_min_dic['drug']
            for i in range(len(cancer_type_list)):
                dis = drug_datasetmaxmin['Max'][cancer_type_list[i][0]] - drug_datasetmaxmin['Min'][cancer_type_list[i][0]]
                pred_denorm[i] = (pred_denorm[i]+5)/10*dis + drug_datasetmaxmin['Min'][cancer_type_list[i][0]]
                label_denorm[i] = (label_denorm[i]+5)/10*dis + drug_datasetmaxmin['Min'][cancer_type_list[i][0]]
                file_to_write = [cancer_type_list[i][0], cancer_type_list[i][1], cancer_type_list[i][2], pred_denorm[i].item(), label_denorm[i].item()]
                csv_write.writerow(file_to_write)
        else:
            for i in range(len(cancer_type_list)):
                file_to_write = [cancer_type_list[i][0], cancer_type_list[i][1], cancer_type_list[i][2], pred[i].item(), label[i].item()]
                csv_write.writerow(file_to_write)
        pred_all_denorm.append(pred_denorm[:, 0].numpy())
        gt_all_denorm.append(label_denorm.numpy())
    pred_all = np.concatenate(pred_all,axis=0)
    gt_all = np.concatenate(gt_all,axis=0)
    total_pcc = pearsonr(pred_all, gt_all)[0]
    total_loss = np.mean(total_loss)
    if args.data_norm == 'norm':
        pred_all_denorm = np.concatenate(pred_all_denorm,axis=0)
        gt_all_denorm = np.concatenate(gt_all_denorm,axis=0)
        total_pcc_denorm = pearsonr(pred_all_denorm, gt_all_denorm)[0]
    else:
        total_pcc_denorm = 0
    pcc_drug, pcc_ccle = pcc_cal(os.path.join(args.output_path, args.task + '_' + str(epoch) + '_end_results.csv'))
    model.train()
    return {"loss": total_loss, "pcc": total_pcc, "pcc_denorm": total_pcc_denorm, 'ccle_pcc': pcc_ccle, 'drug_pcc':pcc_drug }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/processed_inference/')
    parser.add_argument('--output_path', type=str, default='./output/')
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    # train
    parser.add_argument('--task', type=str, default='debug')
    parser.add_argument('--mode', type=str, default='train',choices=["train", "test", 'continue','train_ml'])
    parser.add_argument('--model', type=str, default='STR',choices=["CDR", "STR"])
    parser.add_argument('--sampler', type=str, default='sampler',choices=["sampler", 'None'])
    parser.add_argument('--use_cuda', type=bool, default=False)
    parser.add_argument('--device', type=int, default=0)
    # data
    parser.add_argument('--use_mut', type=bool, default=False)
    parser.add_argument('--use_gexp', type=bool, default=True)
    parser.add_argument('--use_methy', type=bool, default=False)
    parser.add_argument('--data_norm', type=str, default='None',choices=["norm", 'None'])
    parser.add_argument('--split_mode', type=str, default='mix',choices=["mix", "mix_rand", "drug", "ccle", 'drug_ccle','all'])
    parser.add_argument('--split_ratio', type=float, default=0.8)
    parser.add_argument('--cross_val_num', type=int, default=0, choices=[0,1,2,3,4])
    # loss
    parser.add_argument('--use_rankloss', type=bool, default=False)
    parser.add_argument('--lamda', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.4)
    # model
    parser.add_argument('--layer_num', type=int, default=4)
    parser.add_argument('--units_list', type=list, default=[256, 256, 256, 100])
    parser.add_argument('--use_bn', type=bool, default=True)
    parser.add_argument('--gnn_type', type=str, default="gcn", choices=["gcn", "gin", "graphsage"])
    parser.add_argument('--pool_type', type=str, default="max", choices=["sum", "average", "max"])
    parser.add_argument('--act', type=str, default='relu')
    parser.add_argument('--emb_dim', type=int, default=697)

    args = parser.parse_args()
    main(args)



