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

from __future__ import print_function
import sys
import os
from os.path import exists, dirname
import numpy as np
import pickle
import json
import time
import six
if six.PY3:
    import _thread as thread
    from queue import Queue
else:
    import thread
    from Queue import Queue
from collections import OrderedDict
from datetime import datetime
from sklearn.metrics import roc_auc_score

from paddle import fluid

from pahelix.utils.splitters import \
    RandomSplitter, IndexSplitter, ScaffoldSplitter, RandomScaffoldSplitter
from pahelix.datasets import *
from pahelix.datasets.qm9_gdb_dataset import *


def get_downstream_task_names(dataset_name, data_path):
    """
    Get task names of downstream dataset
    """
    if dataset_name == 'bace':
        task_name = get_default_bace_task_names()
    elif dataset_name == 'bbbp':
        task_name = get_default_bbbp_task_names()
    elif dataset_name == 'clintox':
        task_name = get_default_clintox_task_names() 
    elif dataset_name == 'hiv':
        task_name = get_default_hiv_task_names() 
    elif dataset_name == 'muv':
        task_name = get_default_muv_task_names() 
    elif dataset_name == 'sider':
        task_name = get_default_sider_task_names()
    elif dataset_name == 'tox21':
        task_name = get_default_tox21_task_names()
    elif dataset_name == 'toxcast':
        task_name = get_default_toxcast_task_names(data_path)
    elif dataset_name == 'esol':
        return get_default_esol_task_names()
    elif dataset_name == 'freesolv':
        return get_default_freesolv_task_names()
    elif dataset_name == 'lipophilicity':
        return get_default_lipophilicity_task_names()
    elif dataset_name == 'qm7':
        return get_default_qm7_task_names()
    elif dataset_name == 'qm8':
        return get_default_qm8_task_names()
    elif dataset_name == 'qm9':
        return get_default_qm9_task_names()  
    elif dataset_name == 'qm9_gdb':
        return get_default_qm9_gdb_task_names()  
    else:
        raise ValueError('%s not supported' % dataset_name)

    return task_name


def get_dataset(dataset_name, data_path, task_names):
    """Return dataset according to the ``dataset_name``"""
    if dataset_name == 'bace':
        dataset = load_bace_dataset(data_path, task_names)
    elif dataset_name == 'bbbp':
        dataset = load_bbbp_dataset(data_path, task_names)
    elif dataset_name == 'clintox':
        dataset = load_clintox_dataset(data_path, task_names)
    elif dataset_name == 'hiv':
        dataset = load_hiv_dataset(data_path, task_names)
    elif dataset_name == 'muv':
        dataset = load_muv_dataset(data_path, task_names)
    elif dataset_name == 'sider':
        dataset = load_sider_dataset(data_path, task_names)
    elif dataset_name == 'tox21':
        dataset = load_tox21_dataset(data_path, task_names)
    elif dataset_name == 'toxcast':
        dataset = load_toxcast_dataset(data_path, task_names)
    elif dataset_name == 'esol':
        dataset = load_esol_dataset(data_path, task_names)
    elif dataset_name == 'freesolv':
        dataset = load_freesolv_dataset(data_path, task_names)
    elif dataset_name == 'lipophilicity':
        dataset = load_lipophilicity_dataset(data_path, task_names)
    elif dataset_name == 'qm7':
        dataset = load_qm7_dataset(data_path, task_names)
    elif dataset_name == 'qm8':
        dataset = load_qm8_dataset(data_path, task_names)
    elif dataset_name == 'qm9':
        dataset = load_qm9_dataset(data_path, task_names)
    elif dataset_name == 'qm9_gdb':
        dataset = load_qm9_gdb_dataset(data_path, task_names)
    else:
        raise ValueError('%s not supported' % dataset_name)

    return dataset


def get_dataset_stat(dataset_name, data_path, task_names):
    """tbd"""
    if dataset_name == 'esol':
        return get_esol_stat(data_path, task_names)
    elif dataset_name == 'freesolv':
        return get_freesolv_stat(data_path, task_names)
    elif dataset_name == 'lipophilicity':
        return get_lipophilicity_stat(data_path, task_names)
    elif dataset_name == 'qm7':
        return get_qm7_stat(data_path, task_names)
    elif dataset_name == 'qm8':
        return get_qm8_stat(data_path, task_names)
    elif dataset_name == 'qm9':
        return get_qm9_stat(data_path, task_names)
    elif dataset_name == 'qm9_gdb':
        return get_qm9_gdb_stat(data_path, task_names)
    else:
        raise ValueError(dataset_name)


def create_splitter(split_type):
    """Return a splitter according to the ``split_type``"""
    if split_type == 'random':
        splitter = RandomSplitter()
    elif split_type == 'index':
        splitter = IndexSplitter()
    elif split_type == 'scaffold':
        splitter = ScaffoldSplitter()
    elif split_type == 'random_scaffold':
        splitter = RandomScaffoldSplitter()
    else:
        raise ValueError('%s not supported' % split_type)
    return splitter


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
    """tbd"""
    return np.sqrt(np.mean((preds - labels) ** 2))


def calc_mae(labels, preds):
    """tbd"""
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
