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
Metrics for sequence-based models.
"""

import numpy as np
import scipy.stats
import paddle
import paddle.nn.functional as F


class PretrainMetric(object):
    """
    Pretrain Metric.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        """
        Clear the metric.
        """
        self.example_n = 0
        self.accuracy = 0.0
        self.perplexity = 0.0

    def update(self, logit, label, cross_entropy):
        """
        Update results.
        """
        logit = paddle.to_tensor(logit)
        pred = F.softmax(logit) 
        pred = pred.numpy()
        example_n = np.sum(np.where(label != -1, 1, 0))
        self.example_n += example_n
        label = label.reshape(label.size)
        pred_label = pred.argmax(axis=1)
        acc = np.where(pred_label == label, 1, 0)
        cross_entropy = cross_entropy.reshape(cross_entropy.size)
        self.accuracy += np.sum(acc)
        self.perplexity += cross_entropy * example_n

    def show(self):
        """
        Show the metric.
        """
        print('\tExample: %d' % self.example_n)
        print('\tAccuracy: %.6f' % (self.accuracy / self.example_n))
        print('\tPerplexity: %.6f' % (np.exp(self.perplexity / self.example_n)))


class ClassificationMetric(object):
    """
    Classification Metric.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        """
        Clear the metric.
        """
        self.example_n = 0
        self.accuracy = 0.0

    def update(self, pred, label, loss):
        """
        Update results.
        """
        self.example_n += np.sum(np.where(label != -1, 1, 0))
        label = label.reshape(label.size)
        pred_label = pred.argmax(axis=1)
        pred_label = pred_label.reshape(pred_label.size)
        acc = np.where(pred_label == label, 1, 0)
        self.accuracy += np.sum(acc)

    def show(self):
        """
        Show the metric.
        """
        print('\tExample: %d' % self.example_n)
        print('\tAccuracy: %.6f' % (self.accuracy / self.example_n))


class RegressionMetric(object):
    """
    Regression Metric.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        """
        Clear the metric.
        """
        self.example_n = 0
        self.square_error = 0.0
        self.preds = []
        self.labels = []

    def update(self, pred, label, loss):
        """
        Update results.
        """
        pred = pred.reshape(pred.size)
        label = label.reshape(label.size)
        self.example_n += pred.size
        self.square_error += np.sum((pred - label)**2)
        self.preds.append(pred)
        self.labels.append(label)

    def show(self):
        """
        Show the metric.
        """
        print('\tExample: %d' % self.example_n)
        print('\tMSE: %.6f' % (self.square_error / self.example_n))
        preds = np.concatenate(self.preds)
        labels = np.concatenate(self.labels)
        print('\tSpearman\'s: %.6f' %
              scipy.stats.spearmanr(labels, preds).correlation)
    

def get_metric(task):
    """
    Get metric according to the task.
    """
    if task == 'pretrain':
        metric = PretrainMetric()
    elif task in ['seq_classification', 'classification']:
        metric = ClassificationMetric()
    elif task == 'regression':
        metric = RegressionMetric()
    return metric
    