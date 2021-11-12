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

from load_data_pairwise import *
from itertools import combinations
import itertools
from random import *
import paddle
from paddle.io import Dataset
import pdb
import paddle.nn.functional as F
import paddle.nn as nn
import paddle.distributed as dist
from lifelines.utils import concordance_index
import functools
import random
import time
from double_towers import MolTransModel
import numpy as np
import pandas as pd
from preprocess import drug_encoder, target_encoder
from random import randint, sample
import argparse

print = functools.partial(print, flush=True)


# from config import *

np.random.seed(1)
paddle.seed(88)




def group_by(data, qid_index):
    """
    group documents by query-id
    :param data: input_data which contains multiple query and corresponding documents
    :param qid_index: the column num where qid locates in input data
    :return: a dict group by qid
    """
    qid_doc_map = {}
    idx = 0
    #print(data)
    for record in data:
        #print(type(record[qid_index]))
        qid_doc_map.setdefault(record[qid_index], [])
        qid_doc_map[record[qid_index]].append(idx)
        idx += 1
    return qid_doc_map



def sample_index(pairs,sampling_method = None):
    '''
    pairs: the score pairs for train or test
    
    return:
    index of x1 and x2
    '''
    x1_index = []
    x2_index = []

    for i_data in pairs:
        if sampling_method == '500 times':
            sampled_data = pd.DataFrame(i_data).sample(n=500,replace=True)            
        if sampling_method == None:
            sampled_data = pd.DataFrame(i_data)
        
        x1_index.append(sampled_data.iloc[:,0].values)
        x2_index.append(sampled_data.iloc[:,1].values)
        
    return x1_index, x2_index

def get_pairs(scores,K,eps=0.2,seed=0):
    """
    compute the ordered pairs whose firth doc has a higher value than second one.
    :param scores: given score list of documents for a particular query
    :param K: times of sampling
    :return: ordered pairs.  List of tuple, like [(1,2), (2,3), (1,3)]
    """
    pairs = []  
    random.seed(seed)
    for i in range(len(scores)):
        #for j in range(len(scores)):
        # sampling K times
        if K < 1:
            K_ = 1
        else:
            K_ = K
        
        for _ in range(K_):
            idx = random.randint(0, len(scores) - 1)
            score_diff = float(scores[i]) - float(scores[idx])
            if abs(score_diff) >  eps:
                pairs.append((i, idx, score_diff, len(scores))) 

    if K < 1:
        N_pairs = len(pairs)
        pairs = sample(pairs, int(N_pairs*K))

    return pairs



def split_pairs(order_pairs, true_scores):
    """
    split the pairs into two list, named relevant_doc and irrelevant_doc.
    relevant_doc[i] is prior to irrelevant_doc[i]

    :param order_pairs: ordered pairs of all queries
    :param ture_scores: scores of docs for each query
    :return: relevant_doc and irrelevant_doc
    """
    relevant_doc = []
    irrelevant_doc = []
    score_diff = []
    N_smiles = []
    doc_idx_base = 0
    query_num = len(order_pairs)
    for i in range(query_num):
        pair_num = len(order_pairs[i])
        docs_num = len(true_scores[i])
        for j in range(pair_num):
            d1, d2, score, N = order_pairs[i][j]
            d1 += doc_idx_base
            d2 += doc_idx_base
            relevant_doc.append(d1)
            irrelevant_doc.append(d2)
            score_diff.append(score)
            N_smiles.append(N)
        doc_idx_base += docs_num
    return relevant_doc, irrelevant_doc, score_diff, N_smiles




def filter_pairs(data,order_paris,threshold):
    # filterred the pairs which have score diff less than 0.2 
    order_paris_filtered = []
    for i_pairs in order_paris:
        pairs1_score = data[pd.DataFrame(i_pairs).iloc[:,0].values][:,1].astype('float32') 
        pairs2_score = data[pd.DataFrame(i_pairs).iloc[:,1].values][:,1].astype('float32')

        # filtered |score|<threshold
        score = pairs1_score-pairs2_score
        temp_mask = abs(score) > threshold # 0.2 threshold    
        i_pairs_filtered = np.array(i_pairs)[temp_mask].tolist()
        if len(i_pairs_filtered)>0:
            order_paris_filtered.append(i_pairs_filtered)
    return order_paris_filtered

def sample_pairs(true_scores,K,eps,seed):
    # get all the pairs after filtering based on scores
    order_paris = []
    for scores in true_scores:
        order_paris.append(get_pairs(scores,K=K,eps=eps,seed=seed))
    x1_index, x2_index, train_scores, N_smiles = split_pairs(order_paris ,true_scores)
    # change labels to binary
    Y = np.array(train_scores).astype('float32')

    Y[Y<0] = 0
    Y[Y>0] = 1

    return x1_index, x2_index, train_scores, Y

  
    
class Data_Encoder(Dataset):
    def __init__(self, X1, X2, X1_mask, X2_mask,Y):
        super(Data_Encoder, self).__init__()
        self.X1 = X1
        self.X2 = X2
        self.X1_mask = X1_mask
        self.X2_mask = X2_mask
        self.Y = Y

    def __len__(self):
        return len(self.X1)       
    
    def __getitem__(self, idx):        
        return_x1 = self.X1[idx]
        return_x2 = self.X2[idx]
        return_x1_mask = self.X1_mask[idx]
        return_x2_mask = self.X2_mask[idx]
        return_y = self.Y[idx]

        return return_x1, return_x2, return_x1_mask, return_x2_mask, return_y


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
        return_x1 = self.data[return_x1_index][2:].astype(int)
        return_x2 = self.data[return_x2_index][2:].astype(int)

        return_y = self.Y[idx]
        return return_x1, return_x2, return_y

    
       
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
        return_data_x = self.processed_data[return_test_index]
        
        return_len = len(return_test_index)
        
        # get scores
        return_data_x_y = return_data_x[:,1].astype('float32')
        return_y = paddle.to_tensor(return_data_x_y)        
        # get featueres
        return_x = return_data_x[:,2:].astype('int')
        return_x = paddle.to_tensor(return_x)
        

        # pad the dataset
        if self.max_len != return_x.shape[0]:
            padded_x = paddle.zeros(shape=[self.max_len-return_x.shape[0],return_x.shape[1]]).astype('int')
            padded_y = paddle.zeros(shape=[self.max_len-return_y.shape[0]]).astype('float32')

            return_x = paddle.concat([return_x,padded_x],0)
            return_y = paddle.concat([return_y,padded_y],0)
                
        return return_x, return_y, return_len

    
    

def model_eval(model,val_dataloader,len_SMILES,len_target):
    model.eval()
    ## validation
    CI_list = []
    weighted_CI_list = []
    weights_len = []

    for batch_id, data in enumerate(val_dataloader()):
        batch_x = data[0]
        batch_y = data[1]
        batch_len = data[2]

        # split to smiles and protein
        batch_x_smiles = batch_x[:,:,0:len_SMILES].astype('int64')
        batch_x_protein = batch_x[:,:,len_SMILES:len_SMILES+len_target].astype('int64')  
        batch_x_smiles_mask = batch_x[:,:,len_SMILES+len_target:len_SMILES+len_target+len_SMILES].astype('int64')
        batch_x_protein_mask = batch_x[:,:,len_SMILES+len_target+len_SMILES:].astype('int64')


        for i_target_score in range(batch_x.shape[0]):
            
            i_target_len = int(batch_len[i_target_score].numpy()[0])
            smiles = batch_x_smiles[i_target_score][0:i_target_len]
            target = batch_x_protein[i_target_score][0:i_target_len]
            smiles_mask = batch_x_smiles_mask[i_target_score][0:i_target_len]
            target_mask = batch_x_protein_mask[i_target_score][0:i_target_len]
            y_label = batch_y[i_target_score][0:i_target_len].numpy()

            test_DS = Data_Encoder(smiles,target,smiles_mask,target_mask,y_label)
            test_loader = paddle.io.DataLoader(test_DS, batch_size=16, shuffle=False)

            i_target_pred_scores = []
            for data_test in test_loader:
                test_smiles = data_test[0]
                test_target = data_test[1]
                test_smiles_mask = data_test[2]
                test_target_mask = data_test[3]
                test_label = data_test[4]

                pred_scores = model.forward_single(test_smiles,test_target,test_smiles_mask,test_target_mask)
                pred_scores = pred_scores.squeeze(1).numpy().tolist()
                i_target_pred_scores.extend(pred_scores)


            i_target_pred_scores = np.array(i_target_pred_scores)
            i_target_y_label = y_label

            # compute CI
            try:
                CI = concordance_index(i_target_y_label,i_target_pred_scores)
                CI_list.append(CI)
                weighted_CI_list.append(i_target_len*CI)
                weights_len.append(i_target_len)
            except:
                pass

    average_CI = np.mean(CI_list)
    weighted_CI = np.sum(weighted_CI_list)/np.sum(weights_len)

    print("Average CI is {}".format(average_CI))
    print("weighted CI is {}".format(weighted_CI))
    return average_CI, weighted_CI



def run(args):
    # initializen distributed env
    if args.is_parallel == 1:
        dist.init_parallel_env()

    data_path = args.data_path + args.dataset + '/'

    print('><<><><><><><><><><><><><><><><><><><><><><><><><<><><><><><>')
    train_file = 'BindingDB_values_mixed_' +  'train_' + args.index + '_filter.csv'
    val_file = 'BindingDB_values_mixed_' +  'val_' + args.index + '_filter.csv'
    test_file = 'BindingDB_values_mixed_' +  'test_' + args.index + '_filter.csv'

    # load the data
    r_train = pd.read_csv(data_path  + train_file)
    r_train = r_train.reset_index(drop = True) 
    r_val = pd.read_csv(data_path  + val_file)
    r_val = r_val.reset_index(drop = True) 
    r_test = pd.read_csv(data_path  + test_file)
    r_test = r_test.reset_index(drop = True) 


    # load the mixed data
    if args.is_mixed:
        mixed_data_file = 'BindingDB_values_mixed_' +  'train_' + args.mixed_index + '_filter.csv'
        r_mixed = pd.read_csv(data_path + mixed_data_file)
        r_mixed = r_mixed.reset_index(drop = True)

        mixed_data_file1 = 'BindingDB_values_mixed_' +  'train_' + args.mixed_index1 + '_filter.csv'
        r_mixed1 = pd.read_csv(data_path + mixed_data_file1)
        r_mixed1 = r_mixed1.reset_index(drop = True)



    print('Load data...')

    r_train = load_customised_BindingDB(r_train,groupID='groupID')
    r_val = load_customised_BindingDB(r_val,groupID='groupID')
    # r_train = load_customised_BindingDB(r_train,groupID='targetID')
    # r_val = load_customised_BindingDB(r_val,groupID='targetID')
    r_test = load_customised_BindingDB(r_test,groupID='groupID')
    LEN_train = len(r_train)


    print('number of train samples are {}'.format(len(r_train)))  
    print('number of val samples are {}'.format(len(r_val)))
    print('number of test samples are {}'.format(len(r_test)))
    if args.is_mixed:
        r_mixed = load_customised_BindingDB(r_mixed)
        r_mixed1 = load_customised_BindingDB(r_mixed1)
        LEN_mixed0 = len(r_mixed)
        print('number of mixed samples are {}'.format(len(r_mixed)))
        print('number of mixed samples are {}'.format(len(r_mixed1)))
    print('Load done.\n')

    

    ###### get the protein group and index for train/val/test
    qid_doc_map_train = group_by(r_train, 0)
    query_idx_train = qid_doc_map_train.keys()
    train_keys = np.array(list(query_idx_train))

    if args.is_mixed:
        id_doc_map_mixed = group_by(r_mixed,0)
        query_idx_mixed = id_doc_map_mixed.keys()
        mixed_keys = np.array(list(query_idx_mixed))

        id_doc_map_mixed1 = group_by(r_mixed1,0)
        query_idx_mixed1 = id_doc_map_mixed1.keys()
        mixed_keys1 = np.array(list(query_idx_mixed1))



    qid_doc_map_val = group_by(r_val, 0)
    query_idx_val = qid_doc_map_val.keys()
    val_keys = np.array(list(query_idx_val))

    qid_doc_map_test = group_by(r_test, 0)
    query_idx_test = qid_doc_map_test.keys()
    test_keys = np.array(list(query_idx_test))
    ###### get the protein group and index for train/val/test


    # get the true scores of train
    true_scores = [r_train[qid_doc_map_train[qid], 1] for qid in query_idx_train]
    if args.is_mixed:
        true_scores_mixed = [r_mixed[id_doc_map_mixed[qid],1] for qid in query_idx_mixed]
        true_scores_mixed1 = [r_mixed1[id_doc_map_mixed1[qid],1] for qid in query_idx_mixed1]
    
    ###### get val/test dataloader
    val_index = []
    for qid in val_keys:    
        val_index.append(qid_doc_map_val[qid])        
    val_dataset = Data_test(val_index,r_val)
    val_dataloader = paddle.io.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False)

    test_index = []
    for qid in test_keys:    
        test_index.append(qid_doc_map_test[qid])        
    test_dataset = Data_test(test_index,r_test)
    test_dataloader = paddle.io.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    if args.is_mixed:
        # concatenate the data
        r_train = np.concatenate((r_train,r_mixed,r_mixed1))
        del r_mixed
        del r_mixed1


    for fold in range(args.N_runs):        
        # Load model 
        model_config = json.load(open(args.model_config_path, 'r'))
        model = MolTransModel(model_config)

        len_SMILES = model_config['drug_max_seq']
        len_target = model_config['target_max_seq']


        if args.is_parallel == 1:
            model_parallel = paddle.DataParallel(model)
        else:
            model_parallel = model

        # define the optimizer
        clip = nn.ClipGradByValue(min=-1, max=1)
        optimizer = paddle.optimizer.AdamW(parameters=model_parallel.parameters(),weight_decay=0.01,
                                    learning_rate=args.learning_rate, grad_clip=clip)


            
        print('start to train the model...')
        for epoch in range(args.N_epoch):
            ##################### resampling the pairs for each epoch #####################
            train_x1_index, train_x2_index, train_scores, Y_train = sample_pairs(true_scores,K=args.sampling_N_train,eps=args.filter_threshold,seed=epoch)
            print('Number of training dataset is {}'.format(len(train_x1_index)))

            if args.is_mixed:
                mixed_x1_index, mixed_x2_index, mixed_scores, Y_mixed = sample_pairs(true_scores_mixed,K=args.sampling_N_mixed,eps=args.filter_threshold,seed=epoch)
                mixed_x1_index1, mixed_x2_index1, mixed_scores1, Y_mixed1 = sample_pairs(true_scores_mixed1,K=args.sampling_N_mixed1,eps=args.filter_threshold,seed=epoch)

                print('Number of mixed dataset is {}'.format(len(mixed_x1_index)))
                print('Number of mixed dataset is {}'.format(len(mixed_x1_index1)))
                # mixed all pairs from train and mixed dataset
                temp = LEN_train
                temp1 = LEN_mixed0
                mixed_x1_index = [i + temp for i in mixed_x1_index] 
                mixed_x2_index = [i + temp for i in mixed_x2_index] 
                mixed_x1_index1 = [i + temp + temp1 for i in mixed_x1_index1] 
                mixed_x2_index1 = [i + temp + temp1 for i in mixed_x2_index1] 
                

                train_x1_index = train_x1_index + mixed_x1_index + mixed_x1_index1 
                train_x2_index = train_x2_index + mixed_x2_index + mixed_x2_index1 

                Y_train = np.concatenate((Y_train,Y_mixed,Y_mixed1))

            train_dataset = Data_Encoder_flow(train_x1_index, train_x2_index, Y_train, r_train)


            if args.is_parallel:
                train_batch_sampler = paddle.io.DistributedBatchSampler(train_dataset, batch_size=args.train_batch_size, shuffle=True)
                train_dataloader = paddle.io.DataLoader(train_dataset, batch_sampler=train_batch_sampler)
            else:
                train_dataloader = paddle.io.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,num_workers=23)
            ##################### resampling the pairs for each epoch #####################

            print('***************train')
            LOSS = []
            model.train()
            model_parallel.train()
            start_time = time.time()
            for batch_id, data in enumerate(train_dataloader()):
                batch_x1 = data[0]
                batch_x2 = data[1]
                batch_y = data[2]

                ###### define loss and optimization function
                loss_ = nn.BCEWithLogitsLoss()
            
                # split to smiles and protein
                batch_x1_smiles = batch_x1[:,0:len_SMILES].astype('int64') 
                batch_x1_protein = batch_x1[:,len_SMILES:len_SMILES+len_target].astype('int64')  
                batch_x1_smiles_mask = batch_x1[:,len_SMILES+len_target:len_SMILES+len_target+len_SMILES].astype('int64')
                batch_x1_protein_mask = batch_x1[:,len_SMILES+len_target+len_SMILES:].astype('int64')

                batch_x2_smiles = batch_x2[:,0:len_SMILES].astype('int64')
                batch_x2_protein = batch_x2[:,len_SMILES:len_SMILES+len_target].astype('int64')
                batch_x2_smiles_mask = batch_x2[:,len_SMILES+len_target:len_SMILES+len_target+len_SMILES].astype('int64')
                batch_x2_protein_mask = batch_x2[:,len_SMILES+len_target+len_SMILES:].astype('int64')

                optimizer.clear_grad()
                res = model_parallel(batch_x1_smiles,batch_x1_protein,batch_x2_smiles,batch_x2_protein,batch_x1_smiles_mask,batch_x1_protein_mask,batch_x2_smiles_mask,batch_x2_protein_mask)
                
                loss = loss_(res.squeeze(1),batch_y)
                                
                loss.backward()
                optimizer.step()
                # scheduler.step()
                
                if batch_id % 100 == 0:
                    print('batch {} loss {}'.format(batch_id,loss.numpy()))
                
                LOSS.append(loss.numpy())

            end_time = time.time()
            print('take time {}'.format(end_time-start_time))
            print('epoch {}: loss: {} '.format(epoch,np.mean(LOSS)))


            # test
            print('***************test')
            val_average_CI, val_weighted_CI = model_eval(model,val_dataloader,len_SMILES,len_target)
            

            if epoch == 0:
                best_average_CI = val_weighted_CI

                test_average_CI, test_weighted_CI = model_eval(model,test_dataloader,len_SMILES,len_target)
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
                test_average_CI, test_weighted_CI = model_eval(model,test_dataloader,len_SMILES,len_target)
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
    parser.add_argument("--model_config_path", default='config.json')
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

    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--N_epoch", type=int,default=200)
    args = parser.parse_args()
    ##################### set parameters #####################

    run(args)

