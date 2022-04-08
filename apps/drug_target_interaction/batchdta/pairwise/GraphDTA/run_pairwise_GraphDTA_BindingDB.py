from itertools import combinations
import itertools
import argparse
from random import *
import random
import pdb
from lifelines.utils import concordance_index
import functools
import random
import time
import pandas as pd

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *
from processing import *


print = functools.partial(print, flush=True)

def group_by(data):
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
        qid_doc_map.setdefault(record, [])
        qid_doc_map[record].append(idx)
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

class hinge_loss(nn.Module):
    def __init__(self,threshold=1,weight=None):
        super().__init__()
        self.threshold = 1
        self.weight = weight
               
    def forward(self,predicted_score,true_score,n = None):
        # score_diff = predicted_score - true_score
        score_diff = predicted_score*true_score
        loss = self.threshold -  score_diff
        
        loss = torch.clip(loss,min=0)  
        loss = torch.square(loss) 

        if not self.weight is None:
            loss = loss * self.weight

        return 0.5*loss.mean()

def sample_pairs(true_scores,K,eps,seed):
    # get all the pairs after filtering based on scores
    order_paris = []
    for scores in true_scores:
        order_paris.append(get_pairs(scores,K=K,eps=eps,seed=seed))
    x1_index, x2_index, train_scores, N_smiles = split_pairs(order_paris ,true_scores)
    print('Number of training dataset is {}'.format(len(x1_index)))
    # change labels to binary
    Y = np.array(train_scores).astype('float32')

    Y[Y<0] = 0
    Y[Y>0] = 1

    return x1_index, x2_index, train_scores, Y

def distributed_concat(tensor, num_total_examples):    
	output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]    
	torch.distributed.all_gather(output_tensors, tensor)    
	concat = torch.cat(output_tensors, dim=0)    # truncate the dummy elements added by SequentialDistributedSampler    
	return concat[:num_total_examples]



def model_eval(model,val_dataloader,device):
    model.eval()
    ## validation
    CI_list = []
    weighted_CI_list = []
    all_true_label = []
    all_predicted_label = []
    weights_len = []
   
    with torch.no_grad():
        target_pred_scores = []
        target_y_label = []
        target_groups = []
        
        for batch_id, data in enumerate(val_dataloader):
            
            i_data, groups = data
            i_data = i_data.to(device)
            # predict
            pred_scores = model.forward_single(i_data)
            # true label
            true_label = i_data.y

            target_pred_scores.extend(pred_scores.cpu().numpy().squeeze().tolist())
            target_y_label.extend(true_label.cpu().numpy().tolist())
            target_groups.extend(groups.numpy().tolist())

        target_pred_scores = np.array(target_pred_scores)
        target_y_label = np.array(target_y_label)
        target_groups = np.array(target_groups)
        
        group_names = np.unique(target_groups)
        # loop over all the groups
        for i in group_names:
            pos = np.where(target_groups == i)

            i_target_len = len(pos[0])

            i_target_pred_scores = target_pred_scores[pos]
            i_target_y_label = target_y_label[pos]

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

    return average_CI, weighted_CI


def dist_run(rank, args, world_size, data_path, model,fold):
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    print(rank)

    ##################### load the data ############################
    train_file = 'BindingDB_values_mixed_' +  'train_' + args.index + '_filter' +'.csv'
    val_file = 'BindingDB_values_mixed_' +  'val_' + args.index + '_filter' +'.csv'
    test_file = 'BindingDB_values_mixed_' +  'test_' + args.index + '_filter' + '.csv'

    # load the data
    train_set = pd.read_csv(data_path  + '/' + train_file)
    val_set = pd.read_csv(data_path  + '/' + val_file)
    test_set = pd.read_csv(data_path  + '/' + test_file)

    # load the mixed data
    mixed_data_file = 'BindingDB_values_mixed_' +  'train_' + args.mixed_index + '_filter.csv'
    mixed_set = pd.read_csv(data_path + mixed_data_file)

    mixed_data_file1 = 'BindingDB_values_mixed_' +  'train_' + args.mixed_index1 + '_filter.csv'
    mixed_set1 = pd.read_csv(data_path + mixed_data_file1)

    # pre-processing the data
    train_set = process_data_BindingDB(train_set,2)   #group name[2], target name[3]
    val_set = process_data_BindingDB(val_set,2)
    mixed_set = process_data_BindingDB(mixed_set,2)  
    # print('pre-process mixed_set1')
    mixed_set1 = process_data_BindingDB(mixed_set1,2)  
    test_set = process_data_BindingDB(test_set,2)

    # prepare the processed data
    #train
    train_t = train_set[2]
    train_d = train_set[1]
    train_groups = train_set[0]
    train_y = train_set[3]
    train_smiles_graph = train_set[4]
    #val
    val_t = val_set[2]
    val_d = val_set[1]
    val_groups = val_set[0]
    val_y = val_set[3]
    val_smiles_graph = val_set[4]
    #mixed
    mixed_t = mixed_set[2]
    mixed_d = mixed_set[1]
    mixed_groups = mixed_set[0]
    mixed_y = mixed_set[3]
    mixed_smiles_graph = mixed_set[4]
    #mixed1
    mixed_t1 = mixed_set1[2]
    mixed_d1 = mixed_set[1]
    mixed_groups1 = mixed_set1[0]
    mixed_y1 = mixed_set1[3]
    mixed_smiles_graph1 = mixed_set1[4]
    # test
    test_t = test_set[2]
    test_d = test_set[1]
    test_groups = test_set[0]
    test_y = test_set[3]
    test_smiles_graph = test_set[4]
    ##################### load the data ############################

    # concatenate the data
    train_t_data = np.concatenate((train_t,mixed_t,mixed_t1))
    train_d_data = np.concatenate((train_d,mixed_d,mixed_d1))
    train_smiles_graph_data = {**train_smiles_graph, **mixed_smiles_graph,**mixed_smiles_graph1}
 
    # get the group and keys
    qid_doc_map_train = group_by(train_groups)
    query_idx_train = qid_doc_map_train.keys()
    train_keys = np.array(list(query_idx_train))

    qid_doc_map_val = group_by(val_groups)
    query_idx_val = qid_doc_map_val.keys()
    val_keys = np.array(list(query_idx_val))

    id_doc_map_mixed = group_by(mixed_groups)
    query_idx_mixed = id_doc_map_mixed.keys()
    mixed_keys = np.array(list(query_idx_mixed))

    id_doc_map_mixed1 = group_by(mixed_groups1)
    query_idx_mixed1 = id_doc_map_mixed1.keys()
    mixed_keys1 = np.array(list(query_idx_mixed1))
    
    qid_doc_map_test = group_by(test_groups)
    query_idx_test = qid_doc_map_test.keys()
    test_keys = np.array(list(query_idx_test))
    ###### get the protein group and index for train/val/test

    # get the true scores of train
    true_scores = [train_y[qid_doc_map_train[qid]] for qid in query_idx_train]
    true_scores_mixed = [mixed_y[id_doc_map_mixed[qid]] for qid in query_idx_mixed]
    true_scores_mixed1 = [mixed_y1[id_doc_map_mixed1[qid]] for qid in query_idx_mixed1]


    # ###### get val/test dataloader
    val_index = []
    for qid in val_keys:    
        val_index.append(qid_doc_map_val[qid])        
    val_dataset = TestDataset1(groupID=val_groups,xd=val_d,xt=val_t,y=val_y,smile_graph=val_smiles_graph)
    val_dataloader = DataLoader(val_dataset, batch_size=args.test_batch_size,shuffle=False)

    test_index = []
    for qid in test_keys:    
        test_index.append(qid_doc_map_test[qid])        
    test_dataset = TestDataset1(groupID=test_groups,xd=test_d,xt=test_t,y=test_y,smile_graph=test_smiles_graph)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size,shuffle=False)

    ###### load model
    model = model.to(rank)
    model_dist = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    # define the optimizer
    optimizer = torch.optim.Adam(model_dist.parameters(), lr=args.learning_rate)
        
    print('start to train the model...')
    for epoch in range(args.N_epoch):

        ##################### resampling the pairs for each epoch #####################
        start_time = time.time()

        train_x1_index, train_x2_index, train_scores, Y_train = sample_pairs(true_scores,K=args.sampling_N_train,eps=args.filter_threshold,seed=epoch)
        mixed_x1_index, mixed_x2_index, mixed_scores, Y_mixed = sample_pairs(true_scores_mixed,K=args.sampling_N_mixed,eps=args.filter_threshold,seed=epoch)
        mixed_x1_index1, mixed_x2_index1, mixed_scores1, Y_mixed1 = sample_pairs(true_scores_mixed,K=args.sampling_N_mixed1,eps=args.filter_threshold,seed=epoch)

        # mixed all pairs from train and mixed dataset
        len_train = len(train_x1_index)
        len_mixed = len(mixed_x1_index)
        len_mixed1 = len(mixed_x1_index1)

        onehot_train = np.zeros(len_train)
        onehot_mixed = np.ones(len_mixed)
        onehot_mixed1 = np.ones(len_mixed1)
        onehot_train_mixed = np.concatenate((onehot_train,onehot_mixed,onehot_mixed1))
        # onehot_train_mixed = np.concatenate((onehot_train,onehot_mixed))

        temp = len(train_d)
        temp1 = len(mixed_d)
        mixed_x1_index = [i + temp for i in mixed_x1_index] 
        mixed_x2_index = [i + temp for i in mixed_x2_index] 
        mixed_x1_index1 = [i + temp + temp1 for i in mixed_x1_index1] 
        mixed_x2_index1 = [i + temp + temp1 for i in mixed_x2_index1] 

        train_x1_index = train_x1_index + mixed_x1_index + mixed_x1_index1
        train_x2_index = train_x2_index + mixed_x2_index + mixed_x2_index1
        # train_x1_index = train_x1_index + mixed_x1_index 
        # train_x2_index = train_x2_index + mixed_x2_index 

        Y_train_data = np.concatenate((Y_train,Y_mixed,Y_mixed1))
        # Y_train_data = np.concatenate((Y_train,Y_mixed))

        # get dataloader
        train_dataset = TrainDataset(train_x1_index=train_x1_index,train_x2_index=train_x2_index,train_d=train_d_data, train_t=train_t_data, y=Y_train_data,onehot_train_mixed=onehot_train_mixed,smile_graph=train_smiles_graph_data)
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size,sampler=train_sampler)
        
        end_time = time.time()
        print('make pairs + sampling, take time {}'.format(end_time-start_time))
        ##################### resampling the pairs for each epoch #####################

        print('***************train')
        LOSS = []
        model.train()
        start_time = time.time()
        for batch_id, data in enumerate(train_dataloader):
            data1 = data[0].to(rank)
            data2 = data[1].to(rank)

            batch_train_mixed = data1.train_mixed 

            optimizer.zero_grad()

            output = model_dist(data1,data2)

            ture_labels = data1.y.view(-1, 1).float()

            ###### define loss and optimization function
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(output, ture_labels)
            
            loss.backward()
            optimizer.step()
            
            LOSS.append(loss.cpu().detach().numpy())
        
        end_time = time.time()
        print('take time {}'.format(end_time-start_time),flush=True)
        print('epoch {}: loss: {} '.format(epoch,np.mean(LOSS)),flush=True)

        if rank == 0:
            # test
            print('***************test')
            val_average_CI, val_weighted_CI = model_eval(model,val_dataloader,device=rank)
            print("val_Average CI is {}".format(val_average_CI),flush=True)
            print("val_weighted CI is {}".format(val_weighted_CI),flush=True)

            if epoch == 0:
                best_average_CI = val_weighted_CI
                test_average_CI, test_weighted_CI = model_eval(model,test_dataloader,device=rank)
                # save the best epoch
                torch.save(model.state_dict(), args.save_direct + 'train_model_best' + str(fold))
                with open(args.save_direct  + "best_results" + str(fold) + ".txt", "w") as text_file:
                    text_file.write('epoch {}: loss: {} '.format(epoch,np.mean(LOSS)) + '\n')
                    text_file.write("val Average CI is {}".format(val_average_CI) + '\n')
                    text_file.write("val weighted CI is {}".format(val_weighted_CI) + '\n')
                    text_file.write("test Average CI is {}".format(test_average_CI) + '\n')
                    text_file.write("test weighted CI is {}".format(test_weighted_CI) + '\n')
                    text_file.write('##############################################' + '\n')
                    
            if  (epoch != 0) & (val_weighted_CI >= best_average_CI):
                best_average_CI = val_weighted_CI
                test_average_CI, test_weighted_CI = model_eval(model,test_dataloader,device=rank)
                # save the best epoch
                torch.save(model.state_dict(), args.save_direct + 'train_model_best' + str(fold))
                with open(args.save_direct  + "best_results" + str(fold) + ".txt", "w") as text_file:
                    text_file.write('epoch {}: loss: {} '.format(epoch,np.mean(LOSS)) + '\n')
                    text_file.write("val Average CI is {}".format(val_average_CI) + '\n')
                    text_file.write("val weighted CI is {}".format(val_weighted_CI) + '\n')
                    text_file.write("test Average CI is {}".format(test_average_CI) + '\n')
                    text_file.write("test weighted CI is {}".format(test_weighted_CI) + '\n')
                    text_file.write('##############################################' + '\n') 
    

if __name__ == '__main__':
    ##################### set parameters #####################
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_direct", default='./output/')
    parser.add_argument("--data_path", default='../../Data_for_ALL/')
    parser.add_argument("--dataset", default='BindingDB_new')
    parser.add_argument("--model_name", default='GAT_GCN',help='[GATNet, GAT_GCN , GCNNet, GINConvNet]')
    parser.add_argument("--local_rank", default=0)

    parser.add_argument("--index", default='ki')
    parser.add_argument("--mixed_index", default='kd')
    parser.add_argument("--mixed_index1", default='IC50')

    parser.add_argument("--N_runs", default=5)
    parser.add_argument("--sampling_N_train", type=int,default=10)
    parser.add_argument("--sampling_N_mixed", type=int,default=5)
    parser.add_argument("--sampling_N_mixed1", type=int,default=1)
    parser.add_argument("--filter_threshold", type=int,default=0.2)

    parser.add_argument("--train_batch_size", type=int, default=512)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--N_epoch", type=int,default=200)
    args = parser.parse_args()
    ##################### set parameters #####################



    data_path = args.data_path + args.dataset + '/'

    print('><<><><><><><><><><><><><><><><><><><><><><><><><<><><><><><>')


    for fold in range(args.N_runs):
        ###### load model
        model = eval(args.model_name)()

        world_size = torch.cuda.device_count()
        print('Let\'s use', world_size, 'GPUs!')
        mp.spawn(dist_run, args=(args, world_size, data_path, model,fold), nprocs=world_size, join=True)