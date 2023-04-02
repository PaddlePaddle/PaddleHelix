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

import numpy as np
import pandas as pd
import argparse
from paddle.io import Dataset
import paddle.distributed as dist

import paddle
import paddle.nn as nn
import random
import time
import os
from lifelines.utils import concordance_index
from random import randint, sample
from model import Model
from utils import *

import pdb

paddle.seed(10)
paddle.device.set_device("gpu")

def run (args):

        print('Load data...')
        # get the data file path and name
        data_path = args.data_path + args.dataset + '/'

        # initializen distributed env
        if args.is_parallel == 1:
            dist.init_parallel_env()

        for fold in range(args.N_runs):

            print('><<><><><><><><><><><><><><><><><><><><><><><><><<><><><><><>')

            train_file = 'BindingDB_values_mixed_' +  'train_' + args.index + '_filter.csv'
            val_file = 'BindingDB_values_mixed_' +  'val_' + args.index + '_filter.csv'
            test_file = 'BindingDB_values_mixed_' +  'test_' + args.index + '_filter.csv'

            # load the data
            train_data = pd.read_csv(data_path  + train_file)
            val_data = pd.read_csv(data_path  + val_file)
            test_data = pd.read_csv(data_path  + test_file)

            LEN_train = len(train_data)

            # load the mixed data
            if args.is_mixed:
                mixed_data_file = 'BindingDB_values_mixed_' +  'train_' + args.mixed_index + '_filter.csv'
                mixed_data = pd.read_csv(data_path + mixed_data_file)
                LEN_mixed0 = len(mixed_data)

                mixed_data_file1 = 'BindingDB_values_mixed_' +  'train_' + args.mixed_index1 + '_filter.csv'
                mixed_data1 = pd.read_csv(data_path + mixed_data_file1)

            # get the group
            qid_doc_map_train = group_by(train_data,'groupID')
            query_idx_train = qid_doc_map_train.keys()
            train_keys = np.array(list(query_idx_train))

            if args.is_mixed:
                id_doc_map_mixed = group_by(mixed_data,'groupID')
                query_idx_mixed = id_doc_map_mixed.keys()
                mixed_keys = np.array(list(query_idx_mixed))

                id_doc_map_mixed1 = group_by(mixed_data1,'groupID')
                query_idx_mixed1 = id_doc_map_mixed1.keys()
                mixed_keys1 = np.array(list(query_idx_mixed1))


            qid_doc_map_val = group_by(val_data,'groupID')
            query_idx_val = qid_doc_map_val.keys()
            val_keys = np.array(list(query_idx_val))

            qid_doc_map_test = group_by(test_data,'groupID')
            query_idx_test = qid_doc_map_test.keys()
            test_keys = np.array(list(query_idx_test))

            # get the true scores of train and mixed dataset
            true_scores = [train_data.iloc[:,-1].values[qid_doc_map_train[qid]] for qid in query_idx_train]
            if args.is_mixed:
                true_scores_mixed = [mixed_data.iloc[:,-1].values[id_doc_map_mixed[qid]] for qid in query_idx_mixed]
                true_scores_mixed1 = [mixed_data1.iloc[:,-1].values[id_doc_map_mixed1[qid]] for qid in query_idx_mixed1]
                    
            ###### get val/test dataloader
            val_index = []
            for qid in val_keys:
                val_index.append(qid_doc_map_val[qid])
            val_dataset = Data_test(val_index,val_data)
            val_dataloader = paddle.io.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False)

            test_index = []
            for qid in test_keys:
                test_index.append(qid_doc_map_test[qid])
            test_dataset = Data_test(test_index,test_data)
            test_dataloader = paddle.io.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

            if args.is_mixed:
                # concatenate the data
                train_data = train_data.append(mixed_data).append(mixed_data1)
                train_data = train_data.reset_index(drop=True)

            
            # Load model
            model = Model()

            if args.is_parallel == 1:
                model_parallel = paddle.DataParallel(model)
            else:
                model_parallel = model

            # define optimizer
            clip = nn.ClipGradByValue(min=-1, max=1)
            optimizer = paddle.optimizer.Adam(learning_rate = args.learning_rate, parameters = model.parameters(),grad_clip=clip)
            
            print('start to train the model...')
            for epoch in range(args.N_epoch):
                ##################### resampling the pairs for each epoch #####################
                train_x1_index, train_x2_index, train_scores, Y_train = sample_pairs(true_scores,K=args.sampling_N_train,eps=args.filter_threshold,seed=epoch)
                if args.is_mixed:
                    mixed_x1_index, mixed_x2_index, mixed_scores, Y_mixed = sample_pairs(true_scores_mixed,K=args.sampling_N_mixed,eps=args.filter_threshold,seed=epoch)
                    mixed_x1_index1, mixed_x2_index1, mixed_scores1, Y_mixed1 = sample_pairs(true_scores_mixed1,K=args.sampling_N_mixed1,eps=args.filter_threshold,seed=epoch)
                    # mixed all pairs from train and mixed dataset
                    temp = LEN_train
                    temp1 = LEN_mixed0
                    mixed_x1_index = [i + temp for i in mixed_x1_index]
                    mixed_x2_index = [i + temp for i in mixed_x2_index]
                    mixed_x1_index1 = [i + temp + temp1 for i in mixed_x1_index1]
                    mixed_x2_index1 = [i + temp + temp1 for i in mixed_x2_index1]

                    rain_x1_index = train_x1_index + mixed_x1_index + mixed_x1_index1
                    train_x2_index = train_x2_index + mixed_x2_index + mixed_x2_index1

                    Y_train = np.concatenate((Y_train,Y_mixed,Y_mixed1))

                ##################### resampling the pairs for each epoch #####################

                # get dataloader
                train_dataset = Data_Encoder_flow(train_x1_index, train_x2_index,Y_train, train_data)
                if args.is_parallel:
                    train_batch_sampler = paddle.io.DistributedBatchSampler(train_dataset, batch_size=args.train_batch_size, shuffle=False)
                    train_dataloader = paddle.io.DataLoader(train_dataset, batch_sampler=train_batch_sampler)
                else:
                    train_dataloader = paddle.io.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,num_workers=23)
                

                LOSS = []
                model.train()
                model_parallel.train()
                start_time = time.time()
                for batch_id, data in enumerate(train_dataloader()):
                    batch_d1 = data[0]
                    batch_t1 = data[1]
                    batch_d2 = data[2]
                    batch_t2 = data[3]
                    batch_y = data[4]


                    ###### define loss and optimization function
                    loss_ = nn.BCEWithLogitsLoss()

                    optimizer.clear_grad()
                    res = model_parallel(batch_d1,batch_t1,batch_d2,batch_t2)
                    batch_y.stop_gradient = False
                    loss = loss_(res.squeeze(1),batch_y)

                                    
                    loss.backward()
                    optimizer.step()
                                    
                    if batch_id % 100 == 0:
                        print('batch {} loss {}'.format(batch_id,loss.numpy()))
                    
                    
                    LOSS.append(float(loss.cpu().detach().numpy()))

                end_time = time.time()
                print('take time {}'.format(end_time-start_time))
                print('epoch {}: loss: {} '.format(epoch,np.mean(LOSS)))

                # test
                print('val......')
                val_average_CI, val_weighted_CI = model_eval(model,val_dataloader)
                


                if epoch == 0:
                    best_average_CI = val_weighted_CI
                    test_average_CI, test_weighted_CI = model_eval(model,test_dataloader)
                    # save the best epoch
                    paddle.save(model.state_dict(), args.save_direct + 'train_model_best' + str(fold))
                    with open(args.save_direct  + "best_results" + str(fold) + ".txt", "w") as text_file:
                        text_file.write('epoch {}: loss: {} '.format(epoch,np.mean(LOSS)) + '\n')
                        text_file.write("val Average CI is {}".format(val_average_CI) + '\n')
                        text_file.write("val weighted CI is {}".format(val_weighted_CI) + '\n')
                        text_file.write("test Average CI is {}".format(test_average_CI) + '\n')
                        text_file.write("test weighted CI is {}".format(test_weighted_CI) + '\n')
                        text_file.write('##############################################' + '\n')
                        
                if  (epoch != 0) & (val_weighted_CI >= best_average_CI):
                    best_average_CI = val_weighted_CI
                    test_average_CI, test_weighted_CI = model_eval(model,test_dataloader)
                    # save the best epoch
                    paddle.save(model.state_dict(), args.save_direct + 'train_model_best' + str(fold))
                    with open(args.save_direct  + "best_results" + str(fold) + ".txt", "w") as text_file:
                        text_file.write('epoch {}: loss: {} '.format(epoch,np.mean(LOSS)) + '\n')
                        text_file.write("val Average CI is {}".format(val_average_CI) + '\n')
                        text_file.write("val weighted CI is {}".format(val_weighted_CI) + '\n')
                        text_file.write("test Average CI is {}".format(test_average_CI) + '\n')
                        text_file.write("test weighted CI is {}".format(test_weighted_CI) + '\n')
                        text_file.write('##############################################' + '\n')

            print('###############################################################')


if __name__ == '__main__':
    ##################### set parameters #####################
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_direct", default='./output/')
    parser.add_argument("--data_path", default='../../Data_for_ALL/')
    parser.add_argument("--dataset", default='BindingDB_new')
    parser.add_argument("--is_parallel", default=True)
    parser.add_argument("--is_mixed", default=False)
    parser.add_argument("--index", default='ki')
    parser.add_argument("--mixed_index", default='kd')
    parser.add_argument("--mixed_index1", default='IC50')

    parser.add_argument("--N_runs", type=int,default=5)
    parser.add_argument("--sampling_N_train", type=int,default=10)
    parser.add_argument("--sampling_N_mixed", type=int,default=5)
    parser.add_argument("--sampling_N_mixed1", type=int,default=1)
    parser.add_argument("--filter_threshold", type=int,default=0.2)

    parser.add_argument("--train_batch_size", type=int, default=512)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--N_epoch", type=int,default=200)
    args = parser.parse_args()
    ##################### set parameters #####################


    run(args)
