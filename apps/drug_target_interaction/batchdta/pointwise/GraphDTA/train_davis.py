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
from utils import *
from processing import process_data

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
    """
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            res_pred = torch.cat((total_preds, output.cpu()), 0)
            res_label = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return res_label.numpy().flatten(),res_pred.numpy().flatten()

 
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
    train_data = pd.read_csv("../../Data/DAVIS/CV"+str(rounds)+"/CV"+str(rounds)+"_DAVIS_unseenP_seenD_train.csv")
    val_data = pd.read_csv("../../Data/DAVIS/CV"+str(rounds)+"/CV"+str(rounds)+"_DAVIS_unseenP_seenD_val.csv")
    test_data = pd.read_csv("../../Data/DAVIS/test_DAVIS_unseenP_seenD.csv")

    train_set = process_data(train_data, 'train')
    val_set = process_data(val_data, 'val')
    test_set = process_data(test_data, 'test')

    train_generator = TestbedDataset(root = 'dataset', dataset = 'DAVIS_train' + str(rounds), xd = train_set[0],
                                xt = train_set[1], y = train_set[2], smile_graph = train_set[3])
    val_generator = TestbedDataset(root = 'dataset', dataset = 'DAVIS_val' + str(rounds), xd = val_set[0],
                                xt = val_set[1], y = val_set[2], smile_graph = val_set[3])
    test_generator = TestbedDataset(root = 'dataset', dataset = 'DAVIS_test', xd = test_set[0],
                                xt = test_set[1], y = test_set[2], smile_graph = test_set[3])

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
        G, P = predicting(model, device, val_loader)
        val_ci = ci(G, P)

        # Calculate Weighted CI, Average CI of validation set
        lens = int(len(G)/68)
        average_ci = np.mean([ci(G[x*68:(x+1)*68],P[x*68:(x+1)*68]) for x in range(0,lens)])

        print("===============Go for Validation===============")
        print("Weighted CI:",average_ci)
        print("Average CI:",average_ci)
        print("Overall CI:",val_ci)
        
        files = open("bestResult/GraphDTA_"+model_st+"_davis_result"+str(args.rounds)+".txt",'a')
        files.write("val_averageCI: "+str(average_ci)+", val_weightedCI: "+str(average_ci)+", val_overallCI: "+str(val_ci)+", train_loss: "+str(train_loss)+'\n')
        model_name = "bestModel/GraphDTA_"+model_st+"_davis_"+str(rounds)+".model"

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
    test_G, test_P = predicting(model, device, test_loader)
    test_CI, test_MSE = ci(test_G,test_P), mse(test_G,test_P)

    # Calculate Weighted CI, Average CI of testing set
    t_lens = int(len(test_G)/68)
    test_average_ci = np.mean([ci(test_G[x*68:(x+1)*68],test_P[x*68:(x+1)*68]) for x in range(0,t_lens)])

    # Save the testing result
    files.write("test_MSE:" + str(test_MSE) + ", test_averageCI:" + 
                str(test_average_ci) + ", test_weightedCI:" + str(test_average_ci) + ", test_overallCI:" + str(test_CI) + "\n")
    files.write("best_epoch:" + str(best_epoch + 1) + ", best_train_loss:" + str(best_train_loss) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(description='Starting...')

    parser.add_argument('--batchsize', default=512, type=int, metavar='N', help='Batch size')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='Number of total epochs')
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