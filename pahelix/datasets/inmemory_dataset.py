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
In-memory dataset.
"""

import os
from os.path import join, exists
import numpy as np

from pgl.utils.data.dataloader import Dataloader

from pahelix.utils.data_utils import save_data_list_to_npz, load_npz_to_data_list


__all__ = ['InMemoryDataset']


class InMemoryDataset(object):
    """
    Description:
        The InMemoryDataset manages ``data_list`` which is a list of `data` and 
        the `data` is a dict of numpy ndarray. And each dict has the same keys.

        It works like a list: you can call `dataset[i] to get the i-th element of 
        the ``data_list`` and call `len(dataset)` to get the length of ``data_list``.
        
        The ``data_list`` can be cached in npz files by calling `dataset.save_data(data_path)` 
        and after that, call `InMemoryDataset(data_path)` to reload.

    Attributes:
        data_list(list): a list of dict of numpy ndarray.

    Example:
        .. code-block:: python

            data_list = [{'a': np.zeros([4, 5])}, {'a': np.zeros([7, 5])}]
            dataset = InMemoryDataset(data_list=data_list)
            print(len(dataset))
            dataset.save_data('./cached_npz')   # save data_list to ./cached_npz

            dataset2 = InMemoryDataset(npz_data_path='./cached_npz')    # will load the saved `data_list`
            print(len(dataset))
    """
    def __init__(self, 
            data_list=None,
            npz_data_path=None):
        """
        Users can either directly pass the ``data_list`` or pass the `data_path` from 
        which the cached ``data_list`` will be loaded.

        Args:
            data_list(list): a list of dict of numpy ndarray.
            data_path(str): the path to the cached npz path.
        """
        super(InMemoryDataset, self).__init__()
        assert (data_list is None) ^ (npz_data_path is None), \
                "Only data_list or npz_data_path should be set."
        self.data_list = data_list
        self.npz_data_path = npz_data_path

        if not npz_data_path is None:
            self.data_list = self._load_npz_data(npz_data_path)

    def _load_npz_data(self, data_path):
        data_list = []
        files = [f for f in os.listdir(data_path) if f.endswith('.npz')]
        for f in files:
            data_list += load_npz_to_data_list(join(data_path, f))
        return data_list

    def _save_npz_data(self, data_list, data_path, max_num_per_file=10000):
        if not exists(data_path):
            os.makedirs(data_path)
        n = len(data_list)
        for i in range(int((n - 1) / max_num_per_file) + 1):
            filename = 'part-%05d.npz' % i
            sub_data_list = self.data_list[i * max_num_per_file: (i + 1) * max_num_per_file]
            save_data_list_to_npz(join(data_path, filename), sub_data_list)

    def save_data(self, data_path):
        """
        Save the ``data_list`` to the disk specified by ``data_path`` with npz format.
        After that, call `InMemoryDataset(data_path)` to reload the ``data_list``.

        Args:
            data_path(str): the path to the cached npz path.
        """
        self._save_npz_data(self.data_list, data_path)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            dataset = InMemoryDataset(
                    data_list=[self[i] for i in range(start, stop, step)])
            return dataset
        elif isinstance(key, int) or \
                isinstance(key, np.int64) or \
                isinstance(key, np.int32):
            return self.data_list[key]
        elif isinstance(key, list):
            dataset = InMemoryDataset(
                    data_list=[self[i] for i in key])
            return dataset
        else:
            raise TypeError('Invalid argument type: %s of %s' % (type(key), key))

    def __len__(self):
        return len(self.data_list)

    def iter_batch(self, batch_size, num_workers=4, shuffle=False, collate_fn=None):
        """
        It returns an batch iterator which yields a batch of data. Firstly, a sub-list of
        `data` of size ``batch_size`` will be draw from the ``data_list``, then 
        the function ``collate_fn`` will be applied to the sub-list to create a batch and 
        yield back. This process is accelerated by multiprocess.

        Args:
            batch_size(int): the batch_size of the batch data of each yield.
            num_workers(int): the number of workers used to generate batch data. Required by 
                multiprocess.
            shuffle(bool): whether to shuffle the order of the ``data_list``.
            collate_fn(function): used to convert the sub-list of ``data_list`` to the 
                aggregated batch data.

        Yields:
            the batch data processed by ``collate_fn``.
        """
        return Dataloader(self, 
                batch_size=batch_size, 
                num_workers=num_workers, 
                shuffle=shuffle,
                collate_fn=collate_fn)
        
