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
Helper functions
"""

import os
import glob
import logging
import numpy as np
from paddle import fluid
from sklearn.metrics import roc_auc_score

from pahelix.utils.data_utils import load_npz_to_data_list


def get_positive_expectation(p_samples, measure, average=True):
    """Get the expectation from positive samples for given measurement."""
    if measure == 'GAN':
        Ep = - fluid.layers.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = np.log(2.0) - fluid.layers.softplus(-p_samples)
    elif measure == 'X2':
        Ep = p_samples * p_samples
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = - fluid.layers.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - fluid.layers.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise ValueError

    if average:
        return fluid.layers.reduce_sum(Ep)
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    """Get the expectation from negative samples fro given measurement."""
    if measure == 'GAN':
        Eq = fluid.layers.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = fluid.layers.softplus(-q_samples) + q_samples - np.log(2.)
    elif measure == 'X2':
        tmp = fluid.layers.sqrt(q_samples * q_samples) + 1.
        Eq = -0.5 * (tmp * tmp)
    elif measure == 'KL':
        Eq = fluid.layers.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'H2':
        Eq = fluid.layers.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise ValueError

    if average:
        return fluid.layers.reduce_sum(Eq)
    else:
        return Eq


def load_data(npz_dir):
    """Load data from multiple npz files for a given folder."""
    files = glob.glob('%s/*.npz' % npz_dir)
    data_list = []
    for f in files:
        data_list += load_npz_to_data_list(f)
    return data_list


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

    logging.info('Valid ratio: %s', np.mean(valid))
    logging.info('Task evaluated: %s/%s', len(rocauc_list), labels.shape[1])
    if len(rocauc_list) == 0:
        raise RuntimeError("No positively labeled data available. Cannot compute ROC-AUC.")

    return sum(rocauc_list)/len(rocauc_list)
