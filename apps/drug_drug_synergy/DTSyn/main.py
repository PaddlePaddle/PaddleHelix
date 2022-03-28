#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from pgl.utils.data import Dataset, Dataloader
import argparse
import sys
#sys.path.append('.')
from tsnet import TSNet

from utils_no_de import *
from rdkit import Chem
import pandas as pd
import numpy as np

from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, roc_curve,  
                             precision_score, recall_score, auc, cohen_kappa_score,
                             balanced_accuracy_score, precision_recall_curve, accuracy_score)
from scipy.stats import pearsonr
from sklearn.utils import shuffle

def train(model, data_loader, lincs, loss_fn, opt):
    total_pred, total_lb = [], []
    total_loss = []
    model.train()
    for g1, g2, gm1, gm2, cell, lbs in data_loader:
        g1 = g1.tensor()
        g2 = g2.tensor()
        gm1 = paddle.to_tensor(gm1, 'int64')
        gm2 = paddle.to_tensor(gm2, 'int64')
        cell = paddle.to_tensor(cell, 'float32')
        #dea = paddle.to_tensor(dea, 'float32')
        #deb = paddle.to_tensor(deb, 'float32')
        lbs = paddle.to_tensor(lbs, 'int64')
        #batch_samples = len(lbs)
        preds = model(g1, g2, gm1, gm2, cell, lincs,  len(lbs))
        loss = loss_fn(preds, lbs)
        loss.backward()
        #print(preds.gradient())
        opt.step()
        opt.clear_grad()
        total_loss.append(loss.numpy())
    
    return np.mean(total_loss)
    
def eva(model, data_loader, lincs, loss_fn):
    model.eval()
    total_pred, total_lb = [], []
    total_loss = []
    
    for g1, g2, gm1, gm2, cell, lbs in data_loader:
        g1 = g1.tensor()
        g2 = g2.tensor()
        gm1 = paddle.to_tensor(gm1, 'int64')
        gm2 = paddle.to_tensor(gm2, 'int64')
        cell = paddle.to_tensor(cell, 'float32')
        
        lbs = paddle.to_tensor(lbs, 'int64')
        #batch_samples = len(lbs)
        preds = model(g1, g2, gm1, gm2, cell, lincs, len(lbs))
        loss = loss_fn(preds, lbs)
        total_loss.append(loss.numpy())
        total_pred.append(preds.numpy())
        total_lb.append(lbs.numpy())
    total_pred = np.concatenate(total_pred, 0)
    total_lb = np.concatenate(total_lb, 0)
    
    return total_pred, total_lb, np.mean(total_loss)

def test_auc(model, data_loader, lincs, criterion):
    test_pred, test_label, test_loss = eva(model, data_loader, lincs, criterion)
    test_prob = paddle.nn.functional.softmax(paddle.to_tensor(test_pred)).numpy()[:,1]
    pred_label = [1 if x > 0.5 else 0 for x in test_prob]
    ACC = accuracy_score(test_label, pred_label)
    BACC = balanced_accuracy_score(test_label, pred_label)
    PREC = precision_score(test_label, pred_label)
    TPR = recall_score(test_label, pred_label)
    KAPPA = cohen_kappa_score(test_label, pred_label)

    precision, recall, threshold2 = precision_recall_curve(test_label, test_prob)
    return roc_auc_score(test_label, test_prob), auc(recall, precision), test_loss, ACC, BACC, PREC, TPR, KAPPA


def collate(batch):
    d1_list , d2_list = [], []
    mask1, mask2 = [], []
    #dea, deb = [], []
    cells = []
    lbs = []
    for cd1, cd2, cell, label in batch:
        sm1, sm2 = cd1, cd2 #pub_dict[cd1], pub_dict[cd2]
        dg1, n_dg1 = smile_to_graph(sm1)
        dg2, n_dg2 = smile_to_graph(sm2)
        #dg1, n_dg1 = gem_graph(sm1)
        #dg2, n_dg2 = gem_graph(sm2)
        mask1.append(n_dg1)
        mask2.append(n_dg2)
        
        d1_list.append(dg1)
        d2_list.append(dg2)
        cells.append(cell)
        lbs.append(label)
        
    join_graph1 = pgl.Graph.batch(d1_list)
    join_mask1 = np.array(mask1)
    join_mask2 = np.array(mask2)
    join_graph2 = pgl.Graph.batch(d2_list)
    
    join_cells = np.array(cells)
    labels = np.array(lbs)
        
    return join_graph1, join_graph2, join_mask1, join_mask2, join_cells, labels 

def join_cell(ddi, cell):
    cgs = []
    for c in ddi['cell']:
        cgs.append(cell.loc[c, :].values)
    
    return cgs

def join_pert(ddi, pert):
    pta, ptb = [], []
    for i in ddi.index:
        a, b = ddi.loc[i, 'drug1'], ddi.loc[i, 'drug2']
        pta.append(pert.loc[a, :].values)
        ptb.append(pert.loc[b, :].values)

    return pta, ptb

def Pred(model, lincs, data_loader):
    model.eval()
    total_pred = []
    
    for g1, g2, gm1, gm2, cell, lbs in data_loader:
        g1 = g1.tensor()
        g2 = g2.tensor()
        gm1 = paddle.to_tensor(gm1, 'int64')
        gm2 = paddle.to_tensor(gm2, 'int64')
        cell = paddle.to_tensor(cell, 'float32')
        
        #lbs = paddle.to_tensor(lbs, 'int64')
        #batch_samples = len(lbs)
        preds = model(g1, g2, gm1, gm2, cell, lincs, len(lbs))
        
        total_pred.append(preds.numpy())
        
    total_pred = np.concatenate(total_pred, 0)
    total_prob = paddle.nn.functional.softmax(paddle.to_tensor(total_pred)).numpy()[:,1]

    return total_prob

def main(args):
    """
    Args:
        -ddi: drug drug synergy file.
        -rna: cell line gene expression file.
        -lincs: gene embeddings.
        -dropout: dropout rate for transformer blocks.
        -epochs: training epochs.
        -batch_size
        -lr: learning rate.

    """
    #paddle.set_device('cpu')
    ddi = pd.read_csv(args.ddi)
    rna = pd.read_csv(args.rna, index_col=0)
    lincs = pd.read_csv(args.lincs, index_col=0, header=None).values
    lincs = paddle.to_tensor(lincs, 'float32')
    #ddi_test = pd.read_csv(args.ddi_test)
    #print(rna.index)
    #drugs_pert = pd.read_csv(args.pert, index_col=0)
    ###################test###################
    """ddit = ddi.iloc[:512, :]
    t_cell = join_cell(ddit, rna)
    t_tr = DDsData(ddit['drug1'].values, 
            ddit['drug2'].values, 
            t_cell, 
            ddit['label'].values)
    loader_t = Dataloader(t_tr, batch_size=32, num_workers=4, collate_fn=collate)
    model = TSNet(num_drug_feat=78, 
                        num_L_feat=978, 
                        num_cell_feat=954, 
                        num_drug_out=128, 
                        coarsed_heads=4, 
                        fined_heads=4,
                        coarse_hidd=64,
                        fine_hidd=64)

    lincs = paddle.to_tensor(lincs, 'float32')
    opt = paddle.optimizer.Adam(learning_rate=args.lr, parameters=model.parameters())
    loss_fn = paddle.nn.CrossEntropyLoss()
    for g1, g2, gm1, gm2, cell, lbs in loader_t:
        g1 = g1.tensor()
        g2 = g2.tensor()
        gm1 = paddle.to_tensor(gm1, 'int64')
        gm2 = paddle.to_tensor(gm2, 'int64')
        cell = paddle.to_tensor(cell, 'float32')
        lbs = paddle.to_tensor(lbs, 'int64')
        #batch_samples = len(lbs)
        preds = model(g1, g2, gm1, gm2, cell, lincs, len(lbs))
        loss = loss_fn(preds, lbs)
        loss.backward()
        opt.step()
        opt.clear_grad()
        print('loss: {}'.format(loss))
        #print(preds)"""
    ##########################################
    ##############independent validation############
    #5-fold cross validation
    """NUM_CROSS = 5
    ddi_shuffle = shuffle(ddi)
    data_size = len(ddi)
    fold_num = int(data_size / NUM_CROSS)
    for fold in range(NUM_CROSS):
        ddi_test = ddi_shuffle.iloc[fold*fold_num:fold_num * (fold + 1), :]
        ddi_train_before = ddi_shuffle.iloc[:fold*fold_num, :]
        ddi_train_after = ddi_shuffle.iloc[fold_num * (fold + 1):, :]
        ddi_train = pd.concat([ddi_train_before, ddi_train_after])"""
    
    ddi_train = ddi.copy()
    train_cell = join_cell(ddi_train, rna)
                #train_pta, train_ptb = join_pert(ddi_train, drugs_pert)
    bt_tr = DDsData(ddi_train['drug1'].values, 
                        ddi_train['drug2'].values, 
                        train_cell, 
                        ddi_train['label'].values)  

    """test_cell = join_cell(ddi_test, rna)
            #test_pta, test_ptb = join_pert(ddi_test, drugs_pert)
        bt_test = DDsData(ddi_test['drug1'].values,   
                    ddi_test['drug2'].values, 
                    test_cell, 
                    
                    ddi_test['label'].values)"""


    loader_tr = Dataloader(bt_tr, batch_size=args.batch_size, num_workers=4, collate_fn=collate)
    #loader_test = Dataloader(bt_test, batch_size=args.batch_size, num_workers=4, collate_fn=collate)
    #loader_val = Dataloader(bt_val, batch_size=args.batch_size, num_workers=1, collate_fn=collate)

    model = TSNet(num_drug_feat=78, 
                        num_L_feat=978,
                        num_cell_feat=rna.shape[1], 
                        num_drug_out=128, 
                        coarsed_heads=4, 
                        fined_heads=4,
                        coarse_hidd=64,
                        fine_hidd=64,
                        dropout=args.dropout)
    opt = paddle.optimizer.Adam(learning_rate=args.lr, parameters=model.parameters())
    loss_fn = paddle.nn.CrossEntropyLoss()

    for e in range(args.epochs):
        train_loss = train(model, loader_tr, lincs, loss_fn, opt)
        print('Epoch {}---training loss:{}'.format(e, train_loss))
        t_auc, test_prauc, test_loss, acc, bacc, prec, tpr, kappa = test_auc(model, loader_test, lincs, loss_fn)
        print('---Testing loss:{:.4f}, AUC:{:.4f}, PRAUC:{:.4f}, ACC:{:.4f}, BACC:{:.4f}, PREC:{:.4f}, TPR:{:.4f}, KAPPA:{:.4f}'
            .format(test_loss, t_auc, test_prauc, acc, bacc, prec, tpr, kappa))
        
    #paddle.save(model.state_dict(), 'Results/xx.pdparams'.format(e+1))
    #model_params = paddle.load('Results/xx.pdparams')
    #model.set_state_dict(model_params)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("--cuda", action='store_true', default=False)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lincs", type=str, default='../data/gene_vector.csv')
    parser.add_argument("--ddi", type=str, help='using SMILES represent drugs', default='../data/ddi_dupave.csv')
    parser.add_argument("--ddi_test", type=str)
    parser.add_argument("--rna", type=str, default='../rna.csv')  

    args = parser.parse_args()
    print(args)
    main(args)