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


def get_downstream_task_names(dataset_name, data_path):
    """
    get task names of downstream dataset
    """
    task_name_dict = {
        'bace': get_default_bace_task_names(),
        'bbbp': get_default_bbbp_task_names(),
        'clintox': get_default_clintox_task_names(),
        'hiv': get_default_hiv_task_names(),
        'muv': get_default_muv_task_names(),
        'sider': get_default_sider_task_names(),
        'tox21': get_default_tox21_task_names(),
        'toxcast': get_default_toxcast_task_names(data_path),
    }
    if dataset_name in task_name_dict:
        return task_name_dict[dataset_name]
    else:
        raise ValueError('%s not supported' % dataset_name)


def get_dataset(dataset_name, data_path, task_names, featurizer):
    """tbd"""
    if dataset_name == 'bace':
        dataset = load_bace_dataset(data_path, task_names, featurizer=featurizer)
    elif dataset_name == 'bbbp':
        dataset = load_bbbp_dataset(data_path, task_names, featurizer=featurizer)
    elif dataset_name == 'clintox':
        dataset = load_clintox_dataset(data_path, task_names, featurizer=featurizer)
    elif dataset_name == 'hiv':
        dataset = load_hiv_dataset(data_path, task_names, featurizer=featurizer)
    elif dataset_name == 'muv':
        dataset = load_muv_dataset(data_path, task_names, featurizer=featurizer)
    elif dataset_name == 'sider':
        dataset = load_sider_dataset(data_path, task_names, featurizer=featurizer)
    elif dataset_name == 'tox21':
        dataset = load_tox21_dataset(data_path, task_names, featurizer=featurizer)
    elif dataset_name == 'toxcast':
        dataset = load_toxcast_dataset(data_path, task_names, featurizer=featurizer)
    else:
        raise ValueError('%s not supported' % dataset_name)

    return dataset


def create_splitter(split_type):
    """tbd"""
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
    """compute ROC-AUC and averaged across tasks
    """
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
