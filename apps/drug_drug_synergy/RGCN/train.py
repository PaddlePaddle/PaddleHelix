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
drug drug synergy model training and evaluation.
"""
import sys
import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

import networkx as nx
import warnings
warnings.filterwarnings('ignore')

import argparse

from R_model import *
from graphsage_sampling import *

from pahelix.datasets import ddi_dataset
from pahelix.datasets import dti_dataset
from pahelix.datasets import ppi_dataset

from pahelix.featurizers import het_gnn_featurizer


def train(num_subgraph, graph, label_idx, epochs, sub_neighbours=[10, 10], init=True):
    """
    Model training for one epoch and return training loss and validation loss.
    """
    sub_graph_paras = subgraph_gen(graph, label_idx, sub_neighbours) 
    sg, num_nodes, sg_eids, sg_nfeat, sub_label = sub_graph_paras['sub_graph']
    model = DDs(num_nodes, sg.node_feat['features'].shape[1], sg.edge_types, 8, num_nodes)
    model.train()
    criterion = paddle.nn.loss.CrossEntropyLoss(soft_label=True)
    epochs = epochs
    optimizer = Adam(learning_rate=0.0001,
             weight_decay=0.001,
             parameters=model.parameters())
    
    for sub_g in range(num_subgraph):
        if init:
            init = False
        else:
            sub_graph_paras = subgraph_gen(graph, label_idx, sub_neighbours) 
            sg, num_nodes, sg_eids, sg_nfeat, sub_label = sub_graph_paras['sub_graph'] 

        mask = np.zeros((num_nodes, num_nodes)).astype('float32') 
        label = paddle.to_tensor(sub_label)

        for epoch in range(1, epochs + 1):
            valid_label = negative_Sampling(sub_label)
            mask[np.where(valid_label != 0)] = 1
            mask = paddle.to_tensor(mask)

            valid_label[np.where(valid_label == -1)] = 0

            pred_prob = model(sg, paddle.to_tensor(sg.node_feat['features']))
            pred_prob = paddle.multiply(pred_prob, mask)
            train_loss = criterion(pred_prob, paddle.to_tensor(valid_label))
            train_loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            if epoch % 5 == 0:
                t = label.numpy()
                p = pred_prob.numpy()
                ground_truth = t[np.where(mask.numpy() == 1)]
                pred_prob = p[np.where(mask.numpy() == 1)]
                fpr, tpr, _ = roc_curve(y_true=ground_truth, y_score=pred_prob)
                auc_v = auc(fpr, tpr)
                print("sub_graph index : {} | epoch: {} | training loss: {:.4f} | AUC: {:.3f}".format(
                sub_g, epoch, train_loss.numpy()[0], auc_v))
                
    return model


def eval(model, graph, label, sub_neighbours, criterion):
    """
    Model evaluation  and return testing loss.
    """
    sub_graph_paras = subgraph_gen(graph, label_idx, sub_neighbours) 
    sg, num_nodes, sg_eids, sg_nfeat, sub_label = sub_graph_paras['sub_graph']
    model.eval()
    pred = model(graph, paddle.to_tensor(graph.node_feat['features']))
    label = paddle.to_tensor(sub_label)
    loss = criterion(pred, label)
    return pred, loss


def train_val_plot(training_loss, val_loss, figure_name='loss_figure.pdf'):
    """
    Plot the training loss figure.
    """
    fig, axx = plt.subplots(1, 1, figsize=(10, 6))
    axx.plot(training_loss)
    axx.plot(val_loss)
    axx.legend(['training loss', 'val loss']) 
    fig.savefig(figure_name)


def main(ddi, dti, ppi, d_feat, epochs=10, num_subgraph=20, sub_neighbours=[10, 10], cuda=False):
    """
    Args:
        -ddi: drug drug synergy score file.
        -dti: drug target interactione file.
        -ppi: protein protein interaction file.
        -d_feat: a n * 2325 dimentional matrix containing node features, n is the node size.
        -epochs: training runs.
        -num_subgraph: number of subgraphs you want to sample. 
        -sub_neighbours: number of neighbours.
        -cuda: default False.
    """
    
    #place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    paddle.set_device("gpu" if args.cuda else "cpu")
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    ddi = ddi_dataset.load_ddi_dataset(ddi)
    dti = dti_dataset.load_dti_dataset(dti)
    ppi = ppi_dataset.load_ppi_dataset(ppi)

    drug_feat = het_gnn_featurizer.DDiFeaturizer()
    value = drug_feat.collate_fn(ddi, dti, ppi, d_feat)
    hg, nodes_dict, label, label_idx = value['rt'] 
    
    trained_model = train(num_subgraph, hg, label_idx, epochs, args.sub_neighbours)

    return trained_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ddi', type=str, default='./data/DDI')
    parser.add_argument('--dti', type=str, default='./data/DTI')
    parser.add_argument('--ppi', type=str, default='./data/PPI')
    parser.add_argument('--d_feat', type=str, default='./data/all_drugs_name.fet')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_graph', type=int, default=10)
    parser.add_argument('--sub_neighbours', nargs='+', type=int, default=[10, 10])
    parser.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()
    main(args.ddi,
        args.dti,
        args.ppi,
        args.d_feat,
        args.epochs,
        args.num_graph,
        args.sub_neighbours,
        args.cuda)