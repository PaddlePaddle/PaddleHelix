from itertools import combinations
import itertools
from random import *
import random
import pdb
from lifelines.utils import concordance_index
from sklearn import preprocessing
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
from processing import process_data
import argparse


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
    for record in data:
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
        for _ in range(K):
            idx = random.randint(0, len(scores) - 1)
            score_diff = float(scores[i]) - float(scores[idx])
            if abs(score_diff) >  eps:
                pairs.append((i, idx, score_diff, len(scores)))

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
    weights_len = []

    with torch.no_grad():
        
        for batch_id, data in enumerate(val_dataloader):
            i_target_len = len(data)
            i_target_pred_scores = []
            i_target_y_label = []

            # loop over all the D-T pairs in one group(T group)
            for i_data in data:
                i_data = i_data.to(device)
                pred_scores = model.forward_single(i_data)
                # get the predicted labels
                i_target_pred_scores.append(float(pred_scores))
                # get the true labels
                i_target_y_label.append(float(i_data.y.cpu()))

            i_target_pred_scores = np.array(i_target_pred_scores)
            i_target_y_label = np.array(i_target_y_label)


            # compute CI
            CI = concordance_index(i_target_y_label,i_target_pred_scores)
            CI_list.append(CI)
            weighted_CI_list.append(i_target_len*CI)
            weights_len.append(i_target_len)


    average_CI = np.mean(CI_list)
    weighted_CI = np.sum(weighted_CI_list)/np.sum(weights_len)

    return average_CI, weighted_CI


def dist_run(rank, args, world_size, train_set,mixed_set,val_set,test_set,model,CV):
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    print(rank)

    # prepare the processed data
    #train
    train_t = train_set[2]
    train_d = train_set[1]
    train_groups = train_set[0]
    train_y = train_set[3]
    train_smiles_graph = train_set[4]
    if args.is_mixed:
        #mixed
        mixed_t = mixed_set[2]
        mixed_d = mixed_set[1]
        mixed_groups = mixed_set[0]
        mixed_y = mixed_set[3]
        mixed_smiles_graph = mixed_set[4]

    del train_set
    del mixed_set
    # val
    val_t = val_set[2]
    val_d = val_set[1]
    val_groups = val_set[0]
    val_y = val_set[3]
    val_smiles_graph = val_set[4]
    # test
    test_t = test_set[2]
    test_d = test_set[1]
    test_groups = test_set[0]
    test_y = test_set[3]
    test_smiles_graph = test_set[4]
    ##################### load the data ############################

    if args.is_mixed:
        # concatenate the data
        train_t_data = np.concatenate((train_t,mixed_t))
        train_d_data = np.concatenate((train_d,mixed_d))
        train_smiles_graph_data = {**train_smiles_graph, **mixed_smiles_graph}
    else:
        train_t_data = train_t
        train_d_data = train_d
        train_smiles_graph_data = train_smiles_graph

    # get the group
    qid_doc_map_train = group_by(train_groups)
    query_idx_train = qid_doc_map_train.keys()
    train_keys = np.array(list(query_idx_train))

    if args.is_mixed:
        id_doc_map_mixed = group_by(mixed_groups)
        query_idx_mixed = id_doc_map_mixed.keys()
        mixed_keys = np.array(list(query_idx_mixed))
    
    qid_doc_map_val = group_by(val_groups)
    query_idx_val = qid_doc_map_val.keys()
    val_keys = np.array(list(query_idx_val))

    qid_doc_map_test = group_by(test_groups)
    query_idx_test = qid_doc_map_test.keys()
    test_keys = np.array(list(query_idx_test))
    ###### get the protein group and index for train/val/test

    # get the true scores of train
    true_scores = [train_y[qid_doc_map_train[qid]] for qid in query_idx_train]
    if args.is_mixed:
        true_scores_mixed = [mixed_y[id_doc_map_mixed[qid]] for qid in query_idx_mixed]

    # ###### get val/test dataloader
    val_index = []
    for qid in val_keys:
        val_index.append(qid_doc_map_val[qid])
    val_dataset = TestDataset(test_index=val_index,xd=val_d,xt=val_t,y=val_y,smile_graph=val_smiles_graph)
    val_dataloader = DataLoader(val_dataset, batch_size=args.test_batch_size,shuffle=False)
    
    test_index = []
    for qid in test_keys:
        test_index.append(qid_doc_map_test[qid])
    test_dataset = TestDataset(test_index=test_index,xd=test_d,xt=test_t,y=test_y,smile_graph=test_smiles_graph)
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
        if args.is_mixed:
            mixed_x1_index, mixed_x2_index, mixed_scores, Y_mixed = sample_pairs(true_scores_mixed,K=args.sampling_N_mixed,eps=args.filter_threshold,seed=epoch)

        # mixed all pairs from train and mixed dataset
        len_train = len(train_x1_index)
        onehot_train = np.zeros(len_train)

        if args.is_mixed:
            len_mixed1 = len(mixed_x1_index)
            onehot_mixed = np.ones(len_mixed1)
            onehot_train_mixed = np.concatenate((onehot_train,onehot_mixed))
        else:
            onehot_train_mixed = onehot_train

        if args.is_mixed:
            temp = len(train_d)
            mixed_x1_index = [i + temp for i in mixed_x1_index]
            mixed_x2_index = [i + temp for i in mixed_x2_index]

            train_x1_index = train_x1_index + mixed_x1_index
            train_x2_index = train_x2_index + mixed_x2_index

            Y_train_data = np.concatenate((Y_train,Y_mixed))
        else:
            Y_train_data = Y_train

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

            output_train = output[batch_train_mixed==0]
            output_mixed = output[batch_train_mixed==1]

            ture_labels_train = ture_labels[batch_train_mixed==0]
            ture_labels_test = ture_labels[batch_train_mixed==1]

            ###### define loss and optimization function
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(output, ture_labels)
            
            loss.backward()
            optimizer.step()
            
            if batch_id % 20 == 0:
                print('batch {} loss {}'.format(batch_id,loss.item()))
            LOSS.append(loss.cpu().detach().numpy())
        
        end_time = time.time()
        print('take time {}'.format(end_time-start_time))
        print('epoch {}: loss: {} '.format(epoch,np.mean(LOSS)))

        if rank == 0:
            # validation
            print('***************validation')
            val_average_CI, val_weighted_CI = model_eval(model,val_dataloader,device='cuda:0')
            print("val_Average CI is {}".format(val_average_CI))
            print("val_weighted CI is {}".format(val_weighted_CI))

            # test
            print('***************test')
            test_average_CI, test_weighted_CI = model_eval(model,test_dataloader,device='cuda:0')
            print("test_Average CI is {}".format(test_average_CI))
            print("test_weighted CI is {}".format(test_weighted_CI))

            if epoch == 0:
                best_average_CI = val_average_CI
                # save the best epoch
                torch.save(model.state_dict(), args.save_direct + CV + '_' + 'train_model_best' )
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
                torch.save(model.state_dict(), args.save_direct + CV + '_' + 'train_model_best' )
                with open(args.save_direct + CV + '_' + "best_results.txt", "w") as text_file:
                    text_file.write('epoch {}: loss: {} '.format(epoch,np.mean(LOSS)) + '\n')
                    text_file.write("val Average CI is {}".format(val_average_CI) + '\n')
                    text_file.write("val weighted CI is {}".format(val_weighted_CI) + '\n')

                    text_file.write("test Average CI is {}".format(test_average_CI) + '\n')
                    text_file.write("test weighted CI is {}".format(test_weighted_CI) + '\n')
                    text_file.write('##############################################' + '\n')
    


def run(args):
    print('Load data...')

    ###### load model
    model = eval(args.model_name)()
 
    CVs = ['CV1','CV2','CV3','CV4','CV5']

    data_path = args.data_path + args.dataset + '/'

    for CV in CVs:
        print('><<><><><><><><><><><><><><><><><><><><><><><><><<><><><><><>')
        print('start {}'.format(CV))

        ##################### load the data ############################
        train_file = CV + '_' + args.dataset + '_' + args.split +'_' + 'train' + '.csv'
        val_file = CV + '_' + args.dataset + '_' +  args.split + '_' + 'val' + '.csv'
        test = 'test_' + args.dataset + '_' + args.split + '.csv'

        # load the data
        train_data = pd.read_csv(data_path + CV + '/' + train_file)
        val_data = pd.read_csv(data_path + CV + '/' + val_file)
        test_data = pd.read_csv(data_path + test)

        if args.is_mixed:
            # load the mixed data
            if args.dataset == 'DAVIS':
                mixed_dataset = 'KIBA'
            if args.dataset == 'KIBA':
                mixed_dataset = 'DAVIS'

            # laod the mixed data
            mixed_data_file = mixed_dataset + '_mixed_train_unseenP_seenD.csv'
            mixed_data = pd.read_csv(data_path + mixed_data_file)
            # remove the repeated protein sequence
            val_t = val_data['Target Sequence'].unique()
            mixed_t = mixed_data['Target Sequence'].unique()
            filter1 = list((set(val_t).intersection(set(mixed_t))))
            mixed_data = mixed_data[~mixed_data['Target Sequence'].isin(filter1)]
            mixed_set = process_data(mixed_data)
        else:
            mixed_set = None
             
        # pre-processing the data
        train_set = process_data(train_data)
        val_set = process_data(val_data)
        test_set = process_data(test_data)
       
        world_size = torch.cuda.device_count()
        print('Let\'s use', world_size, 'GPUs!')
        mp.spawn(dist_run, args=(args, world_size, train_set,mixed_set,val_set,test_set,model,CV), nprocs=world_size, join=True)


if __name__ == '__main__':
    ##################### set parameters #####################
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_direct", default='./output/')
    parser.add_argument("--data_path", default='../Data_for_ALL/')
    parser.add_argument("--dataset", default='DAVIS',help=' DAVIS | KIBA')
    parser.add_argument("--model_name", default='GAT_GCN',help='[GATNet, GAT_GCN , GCNNet, GINConvNet]')
    parser.add_argument("--split", default='unseenP_seenD')
    parser.add_argument("--local_rank", default=0)


    parser.add_argument("--is_mixed", default=False)


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


