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
reconstruct
"""
import sys
import pdb, traceback, code

from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
from att_model_proxy import cmd_args
np.random.seed(1)


# 0. Constants
nb_smiles = 200
chunk_size = 100
encode_times = 10
decode_times = 5


# 1. load the test smiles
smiles_file = cmd_args.smiles_file
smiles = [line.strip() for index, line in zip(range(nb_smiles), open(smiles_file).readlines())]


def reconstruct_single(model, smiles):
    """
    tbd
    """
    print('a chunk starts...')
    decode_result = []

    chunk = smiles
    chunk_result = [[] for _ in range(len(chunk))]
    for _encode in range(encode_times):
        z1 = model.encode(chunk, use_random=True)
        this_encode = []
        encode_id, encode_total = _encode + 1, encode_times
        for _decode in tqdm(list(range(decode_times)),
                'encode %d/%d decode' % (encode_id, encode_total)
            ):
            _result = model.decode(z1, use_random=True)
            for index, s in enumerate(_result):
                chunk_result[index].append(s)

    decode_result.extend(chunk_result)
    assert len(decode_result) == len(smiles)
    return decode_result


def reconstruct(model, smiles):
    """
    tbd
    """
    chunk_result = Parallel(n_jobs=1)(
        delayed(reconstruct_single)(model, smiles[chunk_start: chunk_start + chunk_size])
        for chunk_start in range(0, len(smiles), chunk_size)
    )

    decode_result = [_1 for _0 in chunk_result for _1 in _0]
    assert len(decode_result) == len(smiles)
    return decode_result


def save_decode_result(decode_result, filename):
    """
    tbd
    """
    with open(filename, 'w') as fout:
        for s, cand in zip(smiles, decode_result):
            print(','.join([s] + cand), file=fout)


def cal_accuracy(decode_result):
    """
    tbd
    """
    accuracy = [sum([1 for c in cand if c == s]) * 1.0 / len(cand) for s, cand in zip(smiles, decode_result)]
    junk = [sum([1 for c in cand if c.startswith('JUNK')]) * 1.0 / len(cand) for s, cand in zip(smiles, decode_result)]
    return (sum(accuracy) * 1.0 / len(accuracy)), (sum(junk) * 1.0 / len(accuracy))


def main():
    """
    tbd
    """
    from att_model_proxy import AttMolProxy as ProxyModel
    from att_model_proxy import cmd_args

    decode_result_save_file =  cmd_args.saved_model + '-reconstruct_zinc_decode_result.csv'
    accuracy_save_file =  cmd_args.saved_model + '-reconstruct_zinc_accuracy.txt'

    model = ProxyModel()

    decode_result = reconstruct(model, smiles)
    accuracy, junk = cal_accuracy(decode_result)

    print('accuracy:', accuracy, 'junk:', junk)

    save_result = True
    if save_result:
        with open(accuracy_save_file, 'w') as fout:
            print('accuracy:', accuracy, 'junk:', junk, file=fout)

        save_decode_result(decode_result, decode_result_save_file)


if __name__ == '__main__':
    cmd_args.saved_model = '../model/train_model_epoch499'
    main()
    