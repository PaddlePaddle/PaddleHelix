"""Training scripts for GraphDTA backbone."""

import rdkit
import torch
import sklearn
import numpy as np
import pandas as pd
import sys, os
import random
from random import shuffle
from time import time
from rdkit import Chem
import torch.nn as nn
from argparse import ArgumentParser

from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils_bindingDB import *
from preprocess import process_data

# Set ranodm seed
seed = 1
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Set loss function
loss_fn = nn.MSELoss()

# Basic settings
LOG_INTERVAL = 20


# Training script
def train(model, device, train_loader, optimizer, epoch):
    """Training script for GraphDTA backbone model.
    
    Args:
        model: DeepDTA backbone model.
        device: Device.
        train_loader: Dataloader of training set.
        optimizer: Optimizer.
        epoch: Epoch.

    Returns:
        loss: Ouput training loss.
    """
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    return loss.item()


def predicting(model, device, loader):
    """Predicting script for GraphDTA backbone model.
    
    Args:
        model: GraphDTA backbone model.
        device: Device.
        loader: Dataloader of validation/testing set.

    Returns:
        res_label: Output ground truth label.
        res_pred: Output prediction.
        res_group: Output groups.
    """
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_groups = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            res_pred = torch.cat((total_preds, output.cpu()), 0)
            res_label = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
            res_group = torch.cat((total_groups, data.g.view(-1, 1).cpu()), 0)
    return res_label.numpy().flatten(), res_pred.numpy().flatten(), res_group.numpy().flatten()


def main(args):
    """Main function."""
    # Basic settings
    best_ci = 0
    best_epoch = 0
    best_train_loss = 10000
    rounds = args.rounds

    # Set CUDA device
    cuda_name = "cuda:" + str(args.cudanum)
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

    # Modeling...
    modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][args.model]
    model_st = modeling.__name__
    print(model_st)
    model = modeling().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)    # Adam

    # Load data
    train_data = pd.read_csv("../../Data/BindingDB/BindingDB_values_mixed_train_ki_filter.csv")
    val_data = pd.read_csv("../../Data/BindingDB/BindingDB_values_mixed_val_ki_filter.csv")
    test_data = pd.read_csv("../../Data/BindingDB/BindingDB_values_mixed_test_ki_filter.csv")

    train_set = process_data(train_data)
    val_set = process_data(val_data)
    test_set = process_data(test_data)

    train_generator = TestbedDataset(root = 'dataset', dataset = 'BindingDB_train', groups=train_set[0], xd = train_set[1],
                                xt = train_set[2], y = train_set[3], smile_graph = train_set[4])
    val_generator = TestbedDataset(root = 'dataset', dataset = 'BindingDB_val', groups=val_set[0], xd = val_set[1],
                                xt = val_set[2], y = val_set[3], smile_graph = val_set[4])
    test_generator = TestbedDataset(root = 'dataset', dataset = 'BindingDB_test', groups=test_set[0], xd = test_set[1],
                                xt = test_set[2], y = test_set[3], smile_graph = test_set[4])

    # Make mini-batch processing
    train_loader = DataLoader(train_generator, batch_size = args.batchsize, shuffle = True)
    val_loader = DataLoader(val_generator, batch_size = args.batchsize, shuffle = False)
    test_loader = DataLoader(test_generator, batch_size = args.batchsize, shuffle = False)

    # Training...
    print("Training.....")
    for epoch in range(args.epochs):
        print("===============Go for Training===============")
        train_loss = train(model, device, train_loader, optimizer, epoch+1)

        # Validation...
        G, P, group_li = predicting(model, device, val_loader)
        val_ci = ci(G, P)

        # Get length of validation set
        result = {}
        for gl in group_li:
            if result.get(gl) == None:
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
        new_G, new_P = np.array(new_G), np.array(new_P)

        # Calculate Weighted CI, Average CI of validation set
        s = 0
        w_ci,a_ci = [],[]
        for l in new_lens:
            try:
                w_ci.append(l*ci(new_G[s:s+l],new_P[s:s+l]))
                a_ci.append(ci(new_G[s:s+l],new_P[s:s+l]))
            except:
                pass
            s += l
        weight_ci, average_ci = np.sum(w_ci)/np.sum(new_lens), np.mean(a_ci)
        print("===============Go for Validation===============")
        print("Weighted CI:",weight_ci)
        print("Average CI:",average_ci)
        print("Overall CI:",val_ci)

        files = open("bestResult/GraphDTA_"+model_st+"_BindingDB_ki_result"+str(args.rounds)+".txt",'a')
        files.write("val_averageCI: "+str(average_ci)+", val_weightedCI: "+str(weight_ci)+", val_overallCI: "+str(val_ci)+", train_loss: "+str(train_loss)+'\n')
        model_name = "bestModel/GraphDTA_"+model_st+"_BindingDB_ki_"+str(rounds)+".model"

        # Save the best result
        if average_ci > best_ci:
            best_ci = average_ci
            best_epoch = epoch
            best_train_loss = train_loss
            # Save best model
            print("Saving the best model...")
            torch.save(model.state_dict(), model_name)

    print("===============Go for Testing===============")
    # Load the model
    model.load_state_dict(torch.load(model_name))

    # Testing...
    test_G, test_P, test_group_li = predicting(model, device, test_loader)
    test_CI, test_MSE = ci(test_G,test_P), mse(test_G,test_P)

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
    t_new_G, t_new_P = np.array(t_new_G), np.array(t_new_P)

    # Calculate Weighted CI, Average CI of testing set
    t_s = 0
    t_w_ci,t_a_ci = [],[]
    for t_l in t_new_lens:
        try:
            t_w_ci.append(t_l*ci(t_new_G[t_s:t_s+t_l],t_new_P[t_s:t_s+t_l]))
            t_a_ci.append(ci(t_new_G[t_s:t_s+t_l],t_new_P[t_s:t_s+t_l]))
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

    parser.add_argument('--batchsize', default=512, type=int, metavar='N', help='Batch size')
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='Number of total epochs')
    parser.add_argument('--rounds', default=1, type=int, metavar='N', help='The Nth round')
    parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='Initial learning rate', dest='lr')
    parser.add_argument('--cudanum', default=0, type=int, metavar='N', help='The Nth CUDA device')
    parser.add_argument('--model', default=0, type=int, metavar='N', help='Select from GINConvNet, GATNet, GAT_GCN, GCNNet')

    args = parser.parse_args()

    beginT = time()
    print("Starting Time: {}".format(beginT))
    main(args)
    endT = time()
    print("Ending Time: {}".format(endT))
    print("Duration is: {}".format(endT - beginT))