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
from paddle.io import Dataset
import argparse
import paddle
import paddle.nn as nn
import random
import time
import os
from lifelines.utils import concordance_index
import paddle.distributed as dist
from model import Model
from utils import *

import pdb

paddle.seed(10)

paddle.device.set_device("gpu")

class Data_Encoder_flow(Dataset):
    def __init__(self, X1_index, X2_index,Y,data):        
        super(Data_Encoder_flow, self).__init__()
        self.X1_index = X1_index
        self.X2_index = X2_index
        self.Y = Y
        self.data = data

    def __len__(self):
        return len(self.X1_index)       
    
    def __getitem__(self, idx):   

        return_x1_index = self.X1_index[idx]
        return_x2_index = self.X2_index[idx]
        return_x1 = self.data.iloc[return_x1_index]
        return_x2 = self.data.iloc[return_x2_index]

        return_d1 = return_x1['SMILES']
        return_t1 = return_x1['Target Sequence']
        return_d2 = return_x2['SMILES']
        return_t2 = return_x2['Target Sequence']

        #Encode Label
        return_d1 = encodeDrug(return_d1,drug_dic)
        return_d2 = encodeDrug(return_d2,drug_dic)
        return_t1 = encodePro(return_t1,pro_dic)
        return_t2 = encodePro(return_t2,pro_dic)

        return_y = self.Y[idx]


        return_d1 = np.asarray(return_d1)
        return_t1 = np.asarray(return_t1)
        return_d2 = np.asarray(return_d2)
        return_t2 = np.asarray(return_t2)
        return_y = np.asarray(return_y)

        return return_d1, return_t1, return_d2, return_t2, return_y

class Data_test(Dataset):
    def __init__(self, test_index, processed_data):
        super(Data_test, self).__init__()
        self.test_index = test_index
        self.processed_data = processed_data
        self.max_len = max([len(i) for i in self.test_index])

    def __len__(self):
        return len(self.test_index)       
    
    def __getitem__(self, idx):  

        return_test_index = self.test_index[idx]
        return_data = self.processed_data.iloc[return_test_index,:]
        
        return_len = len(return_test_index)
        
        # get scores
        return_y= return_data['Label'].values.astype('float32')
        return_y = paddle.to_tensor(return_y)   
        # get featueres
        return_d = return_data['SMILES'].values
        return_t = return_data['Target Sequence'].values

        #Encode Label
        return_d = [encodeDrug(data_d,drug_dic) for data_d in return_d]
        return_t = [encodePro(data_t,pro_dic) for data_t in return_t]
        return_d = paddle.to_tensor(return_d) 
        return_t = paddle.to_tensor(return_t) 

        # pad the dataset
        if self.max_len != return_data.shape[0]:
            padded_d = paddle.zeros(shape=[self.max_len-return_d.shape[0],return_d.shape[1]]).astype('int')
            padded_t = paddle.zeros(shape=[self.max_len-return_t.shape[0],return_t.shape[1]]).astype('int')
            padded_y = paddle.zeros(shape=[self.max_len-return_y.shape[0]]).astype('float32')

            return_d = paddle.concat([return_d,padded_d],0)
            return_t = paddle.concat([return_t,padded_t],0)
            return_y = paddle.concat([return_y,padded_y],0)
                
        return return_d, return_t, return_y, return_len

def run(args):

    # initializen distributed env
    if args.is_parallel == 1:
        dist.init_parallel_env()


    CVs = ['CV1','CV2','CV3','CV4','CV5']

    data_path = args.data_path + args.dataset + '/'

    for CV in CVs:
        print('><<><><><><><><><><><><><><><><><><><><><><><><><<><><><><><>')
        print('start {}'.format(CV))

        ##################### load the data ############################
        train_file = CV + '_' + args.dataset + '_' + args.split +'_' + 'train' + '.csv'
        val_file = CV + '_' + args.dataset + '_' +  args.split + '_' + 'val' + '.csv'
        test = 'test_' + args.dataset + '_' + args.split + '.csv'
        # mixed_data_file = 'DAVIS_mixed_train_unseenP_seenD.csv'

        # load the data
        train_data = pd.read_csv(data_path + CV + '/' + train_file)
        train_data = train_data.reset_index(drop = True)
        val_data = pd.read_csv(data_path + CV + '/' + val_file)
        val_data = val_data.reset_index(drop = True)
        test_data = pd.read_csv(data_path + test)
        test_data = test_data.reset_index(drop = True)


        if args.is_mixed:
            # load the mixed data
            if args.dataset == 'DAVIS':
                mixed_dataset = 'KIBA'
            if args.dataset == 'KIBA':
                mixed_dataset = 'DAVIS'

                
            mixed_data_file = mixed_dataset + '_mixed_train_unseenP_seenD.csv'


            mixed_data = pd.read_csv(data_path + mixed_data_file)
            mixed_data = mixed_data.reset_index(drop = True)
            # remove the repeated protein sequence
            train_t = train_data['Target Sequence'].unique()
            val_t = val_data['Target Sequence'].unique()
            test_t = test_data['Target Sequence'].unique()
            mixed_t = mixed_data['Target Sequence'].unique()
            filter1 = list((set(val_t).intersection(set(mixed_t))))
            mixed_data = mixed_data[~mixed_data['Target Sequence'].isin(filter1)]
            # concatenate the data
            train_data_ALL = train_data.append(mixed_data)
            train_data_ALL = train_data_ALL.reset_index(drop = True)

        # get the group
        qid_doc_map_train = group_by(train_data,'Target ID')
        query_idx_train = qid_doc_map_train.keys()
        train_keys = np.array(list(query_idx_train))
        
        qid_doc_map_val = group_by(val_data,'Target ID')
        query_idx_val = qid_doc_map_val.keys()
        val_keys = np.array(list(query_idx_val))

        qid_doc_map_mixed = group_by(mixed_data,'Target ID')
        query_idx_mixed = qid_doc_map_mixed.keys()
        mixed_keys = np.array(list(query_idx_mixed))

        qid_doc_map_test = group_by(test_data,'Target ID')
        query_idx_test = qid_doc_map_test.keys()
        test_keys = np.array(list(query_idx_test))
        ###### get the protein group and index for train/val/test

        # get the true scores of train
        true_scores = [train_data['Label'].values[qid_doc_map_train[qid]] for qid in query_idx_train]
        if args.is_mixed:
            true_scores_mixed = [mixed_data['Label'].values[qid_doc_map_mixed[qid]] for qid in query_idx_mixed]

        
        ###### get val/test dataloader
        val_index = []
        for qid in val_keys:    
            val_index.append(qid_doc_map_val[qid])      
        val_dataset = Data_test(val_index,val_data)
        val_dataloader = paddle.io.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=True)


        test_index = []
        for qid in test_keys:    
            test_index.append(qid_doc_map_test[qid])        
        test_dataset = Data_test(test_index,test_data)
        test_dataloader = paddle.io.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)


        # Load model 
        model = Model()

        if args.is_parallel == 1:
            model_parallel = paddle.DataParallel(model)
        else:
            args.model_parallel = model

        # define optimizer
        optimizer = paddle.optimizer.Adam(learning_rate = args.learning_rate, parameters = model.parameters())
    
        print('start to train the model...')
        for epoch in range(args.N_epoch):
            ##################### resampling the pairs for each epoch #####################
            train_x1_index, train_x2_index, train_scores, Y_train = sample_pairs(true_scores,K=args.sampling_N_train,eps=args.filter_threshold,seed=epoch)
            if args.is_mixed:
                mixed_x1_index, mixed_x2_index, mixed_scores, Y_mixed = sample_pairs(true_scores_mixed,K=args.sampling_N_mixed,eps=args.filter_threshold,seed=epoch)
            ##################### resampling the pairs for each epoch #####################

            # mixed all pairs from train and mixed dataset
            len_train = len(train_x1_index)
            temp = len(train_data)
            if args.is_mixed:
                mixed_x1_index = [i + temp for i in mixed_x1_index] 
                mixed_x2_index = [i + temp for i in mixed_x2_index] 

                train_x1_index = train_x1_index + mixed_x1_index
                train_x2_index = train_x2_index + mixed_x2_index

                Y_train_data = np.concatenate((Y_train,Y_mixed))

            # get dataloader
            train_dataset = Data_Encoder_flow(train_x1_index, train_x2_index,Y_train_data, train_data_ALL)
            if args.is_parallel:
                train_batch_sampler = paddle.io.DistributedBatchSampler(train_dataset, batch_size=args.train_batch_size, shuffle=True)
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
                loss = loss_(res.squeeze(1),batch_y)
                            
                loss.backward()
                optimizer.step()
                                
                if batch_id % 100 == 0:
                    print('batch {} loss {}'.format(batch_id,loss.numpy()))
                
                LOSS.append(loss.numpy())

            end_time = time.time()
            print('take time {}'.format(end_time-start_time))
            print('epoch {}: loss: {} '.format(epoch,np.mean(LOSS)))

            # validation   
            print('validation......')     
            val_average_CI, val_weighted_CI = model_eval(model,val_dataloader)
            # test
            print('test......') 
            test_average_CI, test_weighted_CI = model_eval(model,test_dataloader)


            if epoch == 0:
                best_average_CI = val_average_CI
                # save the best epoch
                paddle.save(model.state_dict(), args.save_direct + CV + '_' + 'train_model_best' )
                with open(args.save_direct + CV + '_' + "best_results.txt", "w") as text_file:

                    text_file.write('epoch {}: loss: {} '.format(epoch,np.mean(LOSS)) + '\n')
                    text_file.write("val Average CI is {}".format(val_average_CI) + '\n')
                    text_file.write("val weighted CI is {}".format(val_weighted_CI) + '\n')

                    text_file.write("test Average CI is {}".format(test_average_CI) + '\n')
                    text_file.write("test weighted CI is {}".format(test_weighted_CI) + '\n')
                    text_file.write('##############################################' + '\n')
                    
            if  (epoch != 0) & (val_average_CI >= best_average_CI):
                best_average_CI = val_average_CI
                # save the best epoch
                paddle.save(model.state_dict(), args.save_direct + CV + '_' + 'train_model_best' )
                with open(args.save_direct + CV + '_' + "best_results.txt", "w") as text_file:
                    text_file.write('epoch {}: loss: {} '.format(epoch,np.mean(LOSS)) + '\n')
                    text_file.write("val Average CI is {}".format(val_average_CI) + '\n')
                    text_file.write("val weighted CI is {}".format(val_weighted_CI) + '\n')

                    text_file.write("test Average CI is {}".format(test_average_CI) + '\n')
                    text_file.write("test weighted CI is {}".format(test_weighted_CI) + '\n')
                    text_file.write('##############################################' + '\n') 

        print('###############################################################')





#################################################################################################
#################################################################################################
#################################################################################################
if __name__ == '__main__':

    ##################### set parameters #####################
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_direct", default='./output/')
    parser.add_argument("--data_path", default='../../../Data_for_ALL/')
    parser.add_argument("--dataset", default='DAVIS',help=' DAVIS | KIBA')
    parser.add_argument("--split", default='unseenP_seenD')

    parser.add_argument("--is_parallel", default=True)
    parser.add_argument("--is_mixed", default=True)


    parser.add_argument("--sampling_N_train", type=int,default=10)
    parser.add_argument("--sampling_N_mixed", type=int,default=5)
    parser.add_argument("--filter_threshold", type=int,default=0.2)

    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--N_epoch", type=int,default=200)
    args = parser.parse_args()
    ##################### set parameters #####################

    run(args)



