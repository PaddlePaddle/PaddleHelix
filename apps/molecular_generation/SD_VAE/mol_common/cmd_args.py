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
cmd args
"""
import argparse


# path setting
cmd_opt = argparse.ArgumentParser(description='Argparser for molecule vae')
cmd_opt.add_argument('-save_dir', default='./results/', help='result output root')
cmd_opt.add_argument('-mode', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-saved_model', default=None, help='start from existing model')
cmd_opt.add_argument('-grammar_file', default='./data/data_SD_VAE/context_free_grammars/mol_zinc.grammar', \
						help='grammar production rules')
cmd_opt.add_argument('-info_fold', default='./data/data_SD_VAE/context_free_grammars', help='the folder saves grammer info')
cmd_opt.add_argument('-smiles_file', default='./data/data_SD_VAE/zinc/250k_rndm_zinc_drugs_clean-0.h5', help='list of smiles strings')
cmd_opt.add_argument('-model_config', default='model_config.json')
cmd_opt.add_argument('-seed', type=int, default=1, help='random seed')

# data preprocessong
cmd_opt.add_argument('-data_dump', help='location of h5 file')
cmd_opt.add_argument('-skip_deter', type=int, default=0, help='skip deterministic position')
cmd_opt.add_argument('-bondcompact', type=int, default=0, help='compact ringbond representation or not')
cmd_opt.add_argument('-data_gen_threads', default=1, type=int, help='number of threads for data generation')

# training
cmd_opt.add_argument('-loss_type', default='vanilla', help='choose loss from [perplexity | binary | vanilla]')
cmd_opt.add_argument('-num_epochs', type=int, default=500, help='number of epochs')
cmd_opt.add_argument('-batch_size', type=int, default=150, help='minibatch size')
cmd_opt.add_argument('-learning_rate', type=float, default=0.0001, help='init learning_rate')
cmd_opt.add_argument('-kl_coeff', type=float, default=1, help='coefficient for kl divergence used in vae')
cmd_opt.add_argument('-clip_grad', type=float, default=50, help='Clip gradients to this value')

cmd_args, _ = cmd_opt.parse_known_args()
