#!/usr/bin/python3                                                                                                
#-*-coding:utf-8-*- 
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
train zinc
"""
from __future__ import print_function
from past.builtins import range

import os
import os.path
import sys
import numpy as np
import math
import random

from paddle.io import Dataset
import paddle
import paddle.nn.functional as F
import paddle.nn as nn

sys.path.append('./mol_common')
from cmd_args import cmd_args

from pahelix.model_zoo.sd_vae_model import MolVAE


import h5py
import json


def load_zinc_SD_data():
    """
    tbd
    """
    h5f = h5py.File(cmd_args.smiles_file, 'r')
    all_true_binary = h5f['x'][:]
    all_rule_masks = h5f['masks'][:]
    h5f.close()

    return all_true_binary, all_rule_masks


class CreateDataset(Dataset):
    """
    tbd
    """
    def __init__(self, true_binary, rule_masks):

        self.data_binary = true_binary
        self.data_masks = rule_masks
        
    def __len__(self):
        """
        Computes a number of objects in the dataset
        """
        return self.data_binary.shape[0]

    def __getitem__(self, index):
        true_binary = self.data_binary[index, :, :].astype(np.float32)
        rule_masks = self.data_masks[index, :, :].astype(np.float32)
        x_inputs = np.transpose(true_binary, [1, 0])
        
        true_binary = paddle.to_tensor(true_binary)
        rule_masks = paddle.to_tensor(rule_masks)
        x_inputs = paddle.to_tensor(x_inputs)

        return x_inputs, true_binary, rule_masks


def _train_epoch(model, data_loader, epoch, kl_weight, optimizer=None):
    """
    tbd
    """
    if optimizer is None:
        model.eval()
    else:
        model.train()
        
    kl_loss_values = []
    perplexity_loss_values = []
    loss_values = []
    
    
    for batch_id, data in enumerate(data_loader()):
        # read batch data
        x_inputs_batch, true_binary_batch, rule_masks_batch = data
        # transpose the axes of data
        true_binary_batch = paddle.transpose(true_binary_batch, (1, 0, 2))
        rule_masks_batch = paddle.transpose(rule_masks_batch, (1, 0, 2))
               
        # forward
        loss_list  = model(x_inputs_batch, true_binary_batch, rule_masks_batch,)

        if len(loss_list) == 1: # only perplexity
            perplexity = loss_list[0]
            kl_loss = paddle.to_tensor(0)
        else:
            perplexity= loss_list[0]
            kl_loss = loss_list[1]

        loss = kl_weight * kl_loss + perplexity
                
        if optimizer is not None:
            # backward
            loss.backward()
            # optimize
            optimizer.step()
            # clear gradients
            optimizer.clear_grad()
        
        # Log
        kl_loss_values.append(kl_loss.numpy()[0])
        perplexity_loss_values.append(perplexity.numpy()[0])
        loss_values.append(loss.numpy()[0])
        lr = (optimizer.get_lr()
                  if optimizer is not None
                  else 0)

        if batch_id % 200 == 0 and batch_id > 0:
            print('batch:%s, kl_loss:%f, perplexity_loss:%f' % (batch_id, float(np.mean(kl_loss_values)), \
                            float(np.mean(perplexity_loss_values))), flush=True)
            
    postfix = {
            'epoch': epoch,
            'kl_weight': kl_weight,
            'lr': lr,
            'kl_loss': np.mean(kl_loss_values),
            'perplexity_loss': np.mean(perplexity_loss_values),
            'loss': np.mean(loss_values),
            'mode': 'Eval' if optimizer is None else 'Train'}
    
    return postfix      


def _train(model, train_dataloader):
    """
    tbd
    """
    # train the model
    n_epoch = cmd_args.num_epochs

    clip_grad = nn.ClipGradByNorm(clip_norm=cmd_args.clip_grad)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), 
                                learning_rate=cmd_args.learning_rate, 
                                grad_clip=clip_grad)

    # start to train 
    for epoch in range(n_epoch):
        #kl_weight = kl_annealer(epoch)
        kl_weight = cmd_args.kl_coeff

        print('##########################################################################################', flush=True)
        print('EPOCH:%d' % (epoch), flush=True)

        postfix = _train_epoch(model, train_dataloader, epoch, kl_weight, optimizer=optimizer)

        # save state_dict
        paddle.save(model.state_dict(), cmd_args.save_dir + 'train_model_epoch' + str(epoch))
        paddle.save(optimizer.state_dict(), cmd_args.save_dir + 'train_optimizer_epoch' + str(epoch))

        print('epoch:%d loss:%f kl_loss:%f perplexity_loss:%f' % \
                        (epoch, postfix['loss'], postfix['kl_loss'], postfix['perplexity_loss']), flush=True)
        print('##########################################################################################', flush=True)

        # lr_annealer.step()

    return model


def main():
    """
    tbd
    """
    # get model config
    model_config = json.load(open(cmd_args.model_config, 'r'))
    # set gpu
    paddle.set_device(cmd_args.mode)

    all_true_binary, all_rule_masks = load_zinc_SD_data()
    all_true_binary = all_true_binary
    all_rule_masks = all_rule_masks

    # load the data loader
    train_dataset = CreateDataset(all_true_binary, all_rule_masks)
    train_dataloader = paddle.io.DataLoader(train_dataset, batch_size=cmd_args.batch_size, shuffle=True)
    
    # load model
    model = MolVAE(model_config)

    # train the model
    _train(model, train_dataloader)


if __name__ == '__main__':
    main()
