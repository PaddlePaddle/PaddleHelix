#!/usr/bin/python3                                                                                                
#-*-coding:utf-8-*- 
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from args import *
from pahelix.model_zoo.seq_vae_model import VAE
from utils import *
import paddle
from paddle.io import Dataset 
import numpy as np
import paddle.fluid as fluid
import pdb
import paddle.fluid.dygraph as dg
import paddle.nn.functional as F
import paddle.nn as nn
from paddle.optimizer.lr import LRScheduler
from visualdl import LogWriter
import rdkit
import pickle
import os
import json


def train_epoch(model, data_loader, epoch, kl_weight, config, optimizer=None):
    """
    tbd
    """
    if optimizer is None:
        model.eval()
    else:
        model.train()
        
    kl_loss_values = []
    recon_loss_values = []
    loss_values = []
        
    for batch_id, data in enumerate(data_loader()):
        # read batch data
        data_batch = data

        # forward
        kl_loss, recon_loss  = model(data_batch)
        loss = kl_weight * kl_loss + recon_loss
               
        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
        
        # Log
        kl_loss_values.append(kl_loss.numpy())
        recon_loss_values.append(recon_loss.numpy())
        loss_values.append(loss.numpy())
        lr = (optimizer.get_lr()
                  if optimizer is not None
                  else 0)

        if batch_id % 200 == 0 and batch_id > 0:
            print('batch:%s, kl_loss:%f, recon_loss:%f' % \
                (batch_id, float(np.mean(kl_loss_values)), float(np.mean(recon_loss_values))), flush=True)
               
    postfix = {
            'epoch': epoch,
            'kl_weight': kl_weight,
            'lr': lr,
            'kl_loss': np.mean(kl_loss_values),
            'recon_loss': np.mean(recon_loss_values),
            'loss': np.mean(loss_values),
            'mode': 'Eval' if optimizer is None else 'Train'}
    
    return postfix  


def train_model(config, train_dataloader, model):
    """
    tbd
    """
    # train the model
    n_epoch = config.max_epoch
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
                                learning_rate=config.lr_start)

    kl_annealer = KLAnnealer(n_epoch, config)

    # start to train 
    for epoch in range(n_epoch):
        kl_weight = kl_annealer(epoch)

        print('##########################################################################################', flush=True)
        print('EPOCH:%d'%(epoch), flush=True)

        postfix = train_epoch(model, train_dataloader, epoch, kl_weight, config, optimizer=optimizer)

        # save state_dict
        paddle.save(model.state_dict(), config.model_save + 'train_model_epoch' + str(epoch))
        paddle.save(optimizer.state_dict(), config.model_save + 'train_optimizer_epoch' + str(epoch))
        
        print('epoch:%d loss:%f kl_loss:%f recon_loss:%f' % (epoch, postfix['loss'], postfix['kl_loss'], postfix['recon_loss']), flush=True)
        print('##########################################################################################', flush=True)
 
    return model

    
def main(config):
    """
    tbd
    """
    # load model config
    model_config = json.load(open(config.model_config, 'r'))
    
    # load data    
    dataset_file = config.dataset_dir
    src_data = load_zinc_dataset(dataset_file)
    src_data = src_data[0:100]
    # load vocabulary and prepare dataloader    
    vocab = OneHotVocab.from_data(src_data)
    max_length = model_config["max_length"]
    train_dataset = StringDataset(vocab, src_data, max_length)
    train_dataloader = paddle.io.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)    

    # set GPU
    paddle.set_device(config.device)
    
    # # save vocab, config
    # pickle.dump(vocab,open(config.config_save+'vocab.pkl', 'wb'))
    # pickle.dump(config, open(config.config_save+'config.pkl', 'wb'))
    
    # build the model
    model = VAE(vocab, model_config)   
    ##################################### train the model #####################################
    model = train_model(config,train_dataloader,model)


if __name__ == '__main__':
    # load args of training settings
    parser = get_parser()
    config = parser.parse_args()

    main(config)
