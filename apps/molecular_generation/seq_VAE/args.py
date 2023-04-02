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
"""
config args
"""


import argparse

def get_parser(parser=None):
  """
  config args
  """
  if parser is None:
    parser = argparse.ArgumentParser()
        
    # setting
    setting_arg = parser.add_argument_group('Data')
    setting_arg.add_argument('--dataset_dir', type=str, default='./data/zinc_moses/train.csv')
    setting_arg.add_argument('--model_config', type=str, default='model_config.json')
    setting_arg.add_argument('--device', type=str, default="cpu", choices=['cpu', 'gpu:0', 'gpu:1'])
    setting_arg.add_argument('--model_save', type=str, default='./results/train_models/')
    setting_arg.add_argument('--config_save', type=str, default='./results/config/')

    
    # Train
    train_arg = parser.add_argument_group('Train')
    train_arg.add_argument('--max_epoch',
                           type=int, default=1000,
                           help='Batch size')
    train_arg.add_argument('--batch_size',
                           type=int, default=1000,
                           help='Batch size')
    train_arg.add_argument('--lr_start',
                           type=float, default=3 * 1e-4,
                           help='Initial lr value')
    
    # kl annealing
    train_arg.add_argument('--kl_start',
                           type=int, default=0,
                           help='Epoch to start change kl weight from')
    train_arg.add_argument('--kl_w_start',
                           type=float, default=0,
                           help='Initial kl weight value')
    train_arg.add_argument('--kl_w_end',
                           type=float, default=0.05,
                           help='Maximum kl weight value')

    return parser
