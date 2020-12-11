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
Tools for data.
"""

import numpy as np
import os
import random


def save_data_list_to_npz(data_list, npz_file):
    """
    Save a list of data to the npz file. Each data is a dict 
    of numpy ndarray.

    Args:   
        data_list(list): a list of data.
        npz_file(str): the npz file location.
    """
    keys = data_list[0].keys()
    merged_data = {}
    for key in keys:
        lens = np.array([len(data[key]) for data in data_list])
        values = np.concatenate([data[key] for data in data_list], 0)
        merged_data[key] = values
        merged_data[key + '.seq_len'] = lens
    np.savez_compressed(npz_file, **merged_data)


def load_npz_to_data_list(npz_file):
    """
    Reload the data list save by :function:`save_data_list_to_npz`.

    Args:
        npz_file(str): the npz file location.

    Returns:
        data_list(list): a list of data where each data is a dict of numpy 
            ndarray.
    """
    def _split_data(values, seq_lens):
        res = []
        s = 0
        for l in seq_lens:
            res.append(values[s: s + l])
            s += l
        return res

    merged_data = np.load(npz_file)
    names = [name for name in merged_data.keys() if not name.endswith('.seq_len')]
    data_dict = {}
    for name in names:
        data_dict[name] = _split_data(merged_data[name], merged_data[name + '.seq_len'])

    data_list = []
    n = len(data_dict[names[0]])
    for i in range(n):
        data = {name:data_dict[name][i] for name in names}
        data_list.append(data)
    return data_list


def get_part_files(data_path, trainer_id, trainer_num):
    """
    Split the files in data_path so that each trainer can train from different examples.
    """
    filenames = os.listdir(data_path)
    random.shuffle(filenames)
    part_filenames = []
    for (i, filename) in enumerate(filenames):
        if i % trainer_num == trainer_id:
            part_filenames.append(data_path + '/' + filename)
    return part_filenames

