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


import os
import sys
import numpy as np

import paddle
from pgl.utils.data import Dataset as BaseDataset
from pgl.utils.data import Dataloader

import pgl
from pgl.utils.logger import log


class Dataset(BaseDataset):
    """
    Dataset for CDR(cancer drug response)
    """

    def __init__(self, processed_data):
        self.data = processed_data
        self.keys = list(processed_data.keys())
        self.num_samples = len(processed_data[self.keys[0]])

    def __getitem__(self, idx):
        return self.data[self.keys[0]][idx], self.data[self.keys[1]][idx], self.data[self.keys[2]][idx], \
               self.data[self.keys[3]][idx], self.data[self.keys[4]][idx]

    def get_data_loader(self, batch_size, num_workers=1,
                        shuffle=False, collate_fn=None):
        """Get dataloader.
        Args:
            batch_size (int): number of data items in a batch.
            num_workers (int): number of parallel workers.
            shuffle (int): whether to shuffle yield data.
            collate_fn: callable function that processes batch data to a list of paddle tensor.
        """
        return Dataloader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=collate_fn)

    def __len__(self):
        return self.num_samples


def collate_fn(batch_data):
    """
    Collation function to distribute data to samples
    :param batch_data: batch data
    """
    graphs = []
    mut, gexpr, met, Y = [], [], [], []
    for g, mu, gex, me, y in batch_data:
        graphs.append(g)
        mut.append(mu)
        gexpr.append(gex)
        met.append(me)
        Y.append(y)
    return graphs, mut, gexpr, met, Y
