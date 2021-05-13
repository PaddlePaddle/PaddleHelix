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
sample prior
"""
import sys
import traceback
import code
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from collections import Counter
import json
from pahelix.utils.metrics.molecular_generation.metrics_ import get_all_metrics

# 0. Constants
nb_latent_point = 200
chunk_size = 100
sample_times = 100


def cal_valid_prior(model, latent_dim):
    """
    tbd
    """
    from att_model_proxy import batch_decode
    from att_model_proxy import cmd_args
    whole_valid, whole_total = 0, 0
    latent_point = np.random.normal(size=(nb_latent_point, latent_dim))
    latent_point = latent_point.astype(np.float32)

    raw_logits = model.pred_raw_logits(latent_point)
    decoded_array = batch_decode(raw_logits, True, decode_times=sample_times)

    decode_list = []
    for i in range(nb_latent_point):
        c = Counter()
        for j in range(sample_times):
            c[decoded_array[i][j]] += 1
        decoded = c.most_common(1)[0][0]
        if decoded.startswith('JUNK'):
            continue
        decode_list.append(decoded)

    metrics = get_all_metrics(gen=decode_list,k=[100,1000])
    print(metrics)

    valid_prior_save_file =  cmd_args.saved_model + '-sampled_prior.txt'
    with open(valid_prior_save_file, 'w') as fout:
        for row in decode_list:
            fout.write('%s\n' % row)
    

def main():
    """
    tbd
    """
    from att_model_proxy import cmd_args
    seed = cmd_args.seed
    np.random.seed(seed)

    from att_model_proxy import AttMolProxy as ProxyModel
    from att_model_proxy import cmd_args
    model_config = json.load(open(cmd_args.model_config, 'r'))

    model = ProxyModel()

    cal_valid_prior(model, model_config['latent_dim'])


if __name__ == '__main__':
    main()