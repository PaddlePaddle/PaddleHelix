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
Stream dataset.
"""

import os
from os.path import join, exists
import numpy as np

from pgl.utils.data.dataloader import Dataloader
from pgl.utils.data.dataset import StreamDataset as PglStreamDataset

from pahelix.utils.data_utils import save_data_list_to_npz, load_npz_to_data_list


__all__ = ['StreamDataset']


class StreamDataset(object):
    """tbd"""
    def __init__(self, 
            data_generator=None,
            npz_data_path=None):
        super(StreamDataset, self).__init__()
        assert (data_generator is None) ^ (npz_data_path is None), \
                "Only data_generator or npz_data_path should be set."
        self.data_generator = data_generator
        self.npz_data_path = npz_data_path

        if not npz_data_path is None:
            self.data_generator = self._load_npz_data(npz_data_path)

    def _load_npz_data(self, data_path):
        files = [file for file in os.listdir(data_path) if file.endswith('.npz')]
        for file in files:
            data_list = load_npz_to_data_list(join(data_path, file))
            for data in data_list:
                yield data

    def _save_npz_data(self, data_list, data_path, max_num_per_file=10000):
        if not exists(data_path):
            os.makedirs(data_path)

        sub_data_list = []
        count = 0
        for data in self.data_generator:
            sub_data_list.append(data)
            if len(sub_data_list) == 0:
                file = 'part-%05d.npz' % count
                save_data_list_to_npz(join(data_path, file), sub_data_list)
                sub_data_list = []
                count += 1
        if len(sub_data_list) > 0:
            file = 'part-%05d.npz' % count
            save_data_list_to_npz(join(data_path, file), sub_data_list)

    def save_data(self, data_path):
        """tbd"""
        self._save_npz_data(self.data_generator, data_path)

    def iter_batch(self, batch_size, num_workers=4, shuffle_size=1000, collate_fn=None):
        """tbd"""
        class _TempDataset(PglStreamDataset):
            def __init__(self, data_generator):
                self.data_generator = data_generator
            def __iter__(self):
                for data in self.data_generator:
                    yield data

        return Dataloader(_TempDataset(self.data_generator), 
                batch_size=batch_size, 
                num_workers=num_workers, 
                stream_shuffle_size=shuffle_size,
                collate_fn=collate_fn)
        
