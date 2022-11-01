# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#coding=utf-8
import paddle
import paddle.nn as nn
import numpy as np

def ranking_loss(y_pred,y_true,num_pairs):
    sample_pairs = create_random_samples(y_true, y_pred, num_pairs)
    sample_pairs = paddle.to_tensor(sample_pairs,dtype='int32')
    bs = y_true.shape[0]
    labels = -1*paddle.ones((num_pairs,1))
    list_pairs_pred = []

    for i in range(sample_pairs.shape[0]):
        list_pairs_pred.append(y_pred[sample_pairs[i]].reshape((1,2)))
        x = y_true[sample_pairs[i]]
        if x[0] > x[1]:
            labels[i,0] = 1
    y_pred_sel = paddle.concat(list_pairs_pred,axis=0)
    loss = nn.functional.margin_ranking_loss(y_pred_sel[:,0], y_pred_sel[:,1], labels, margin=0.0, reduction='mean', name=None)
    return loss

def create_random_samples(y_true, y_pred, num_pairs=10):
    bs = y_true.shape[0]
    sample_pair = np.random.randint(0, y_true.shape[0], (num_pairs * 2))
    sample_pair = np.reshape(sample_pair, (num_pairs,2))
    return sample_pair