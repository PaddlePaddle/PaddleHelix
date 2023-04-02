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
utils
"""

import sys
import os
from os.path import exists, dirname, basename, join
import numpy as np
import pickle
import pandas as pd
import json
import time
import six
import logging
import csv

from sklearn.metrics import roc_auc_score

from .paddle_utils import dist_mean, dist_sum, dist_length


def calc_rocauc_score(labels, preds, valid):
    """compute ROC-AUC and averaged across tasks"""
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
        preds = preds.reshape(-1, 1)

    rocauc_list = []
    for i in range(labels.shape[1]):
        c_valid = valid[:, i].astype("bool")
        c_label, c_pred = labels[c_valid, i], preds[c_valid, i]
        #AUC is only defined when there is at least one positive data.
        if len(np.unique(c_label)) == 2:
            rocauc_list.append(roc_auc_score(c_label, c_pred))

    print('Valid ratio: %s' % (np.mean(valid)))
    print('Task evaluated: %s/%s' % (len(rocauc_list), labels.shape[1]))
    if len(rocauc_list) == 0:
        raise RuntimeError("No positively labeled data available. Cannot compute ROC-AUC.")

    return sum(rocauc_list)/len(rocauc_list)


def calc_rmse(labels, preds):
    return np.sqrt(np.mean((preds - labels) ** 2))


def calc_mae(labels, preds):
    return np.mean(np.abs(preds - labels))


def exempt_parameters(src_list, ref_list):
    """Remove element from src_list that is in ref_list"""
    res = []
    for x in src_list:
        flag = True
        for y in ref_list:
            if x is y:
                flag = False
                break
        if flag:
            res.append(x)
    return res


def calc_parameter_size(parameter_list):
    """Calculate the total size of `parameter_list`"""
    count = 0
    for param in parameter_list:
        count += np.prod(param.shape)
    return count


def sequence_pad(seq_list, max_len=None, pad_value=0):
    if max_len is None:
        max_len = np.max([len(seq) for seq in seq_list])

    pad_seq_list = []
    for seq in seq_list:
        if len(seq) < max_len:
            pad_shape = [max_len - len(seq)] + list(seq.shape[1:])
            pad_seq = np.concatenate([seq, np.full(pad_shape, pad_value, seq.dtype)], 0)
        else:
            pad_seq = seq[:max_len]
        pad_seq_list.append(pad_seq)

    pad_seqs = np.array(pad_seq_list)
    return pad_seqs


def edge_to_pair(edge_index, edge_feat, max_len=None):
    edge_i, edge_j = edge_index[:, 0], edge_index[:, 1]
    if max_len is None:
        max_len = np.max(edge_i)
        max_len = max(np.max(edge_j), max_len) + 1
    pair_feat = np.zeros([max_len, max_len], edge_feat.dtype)
    pair_feat[edge_i, edge_j] = edge_feat
    return pair_feat


def pair_pad(pair_list, max_len=None, pad_value=0):
    """
    pair_list: [(n1, n1), (n2, n2), ...]
    return (B, N, N, *)
    """
    if max_len is None:
        max_len = np.max([len(x) for x in pair_list])

    pad_pair_list = []
    for pair in pair_list:
        raw_shape = list(pair.shape)
        if raw_shape[0] < max_len:
            max_shape = [max_len, max_len] + raw_shape[2:]
            pad_width = [(0, x - y) for x, y in zip(max_shape, raw_shape)]
            pad_pair = np.pad(pair, pad_width, 'constant', constant_values=pad_value)
        else:
            pad_pair = pair[:max_len, :max_len]
        pad_pair_list.append(pad_pair)
    return np.array(pad_pair_list)  # (B, N, N, *)


def tree_map(f, d):
    new_d = {}
    for k in d:
        if type(d[k]) is dict:
            new_d[k] = tree_map(f, d[k])
        else:
            new_d[k] = f(d[k])
    return new_d


def tree_flatten(d):
    new_d = {}
    for k in d:
        if type(d[k]) is dict:
            cur_d = tree_flatten(d[k])
            for sub_k, sub_v in cur_d.items():
                new_d[f'{k}.{sub_k}'] = sub_v
        else:
            new_d[k] = d[k]
    return new_d


def tree_filter(key_cond, value_cond, d):
    new_d = {}
    for k in d:
        if not key_cond is None and not key_cond(k):
            continue
        if not value_cond is None and not value_cond(d[k]):
            continue

        if type(d[k]) is dict:
            cur_d = tree_filter(key_cond, value_cond, d[k])
            if len(cur_d) != 0:
                new_d[k] = cur_d
        else:
            new_d[k] = d[k]
    return new_d


def add_to_data_writer(data_writer, step, results, prefix=''):
    logging.info("step:%d %s:%s" % (step, prefix, str(results)))
    if data_writer is None:
        return
    for k, v in results.items():
        data_writer.add_scalar("%s/%s" % (prefix, k), v, step)


def write_to_csv(csv_name, value_dict):
    with open(csv_name, 'a') as f:
        csv_writer = csv.DictWriter(f, fieldnames=list(value_dict.keys()))
        if value_dict['epoch'] == 0:
            csv_writer.writeheader()
        csv_writer.writerow(value_dict)


def set_logging_level(level):
    level_dict = {
        "NOTSET": logging.NOTSET,
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s',
        level=level_dict[level],
        datefmt='%Y-%m-%d %H:%M:%S')


class ResultsCollect(object):
    def __init__(self):
        self.dict_to_mean_list = []
        self.dict_to_sum_list = []

    def add(self, batch, results, dict_to_mean={}, dict_to_sum={}):
        """
        batch, results: 
        dict_to_mean: {key: float, ...}
        dict_to_sum: {key: float, ...}
        """
        loss_dict = self._extract_loss_dict(results)
        dict_to_mean.update(loss_dict)

        if len(dict_to_mean) > 0:
            self.dict_to_mean_list.append(dict_to_mean)
        if len(dict_to_sum) > 0:
            self.dict_to_sum_list.append(dict_to_sum)

    def get_mean_result(self, distributed=False):
        result = {}
        if len(self.dict_to_mean_list) == 0:
            return result
        keys = list(self.dict_to_mean_list[0].keys())
        for k in keys:
            result[k] = dist_mean(
                    [d[k] for d in self.dict_to_mean_list],
                    distributed=distributed)
        return result
    
    def get_sum_result(self, distributed=False):
        result = {}
        if len(self.dict_to_sum_list) == 0:
            return result
        keys = list(self.dict_to_sum_list[0].keys())
        for k in keys:
            result[k] = dist_sum(
                    [d[k] for d in self.dict_to_sum_list],
                    distributed=distributed)
        return result

    def get_result(self, distributed=False):
        result = {}
        result.update(self.get_mean_result(distributed))
        result.update(self.get_sum_result(distributed))
        return result

    def _extract_loss_dict(self, results):
        """extract value with 'loss' in key"""
        res = tree_flatten(results)
        res = tree_filter(lambda k: 'loss' in k, None, res)
        res = tree_map(lambda x: x.numpy().mean(), res)
        return res
