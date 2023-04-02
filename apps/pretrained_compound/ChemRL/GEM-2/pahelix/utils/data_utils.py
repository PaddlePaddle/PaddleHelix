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
| Tools for data.
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
        shape_strs = []
        flat_values = []
        for data in data_list:
            v = data[key]
            if not isinstance(data[key], np.ndarray):
                v = np.array(v)
            shape_strs.append('_'.join(map(str, v.shape)))
            flat_values.append(v.flatten())
        merged_data[key] = np.concatenate(flat_values, 0)
        merged_data[key + '.shape'] = np.array(shape_strs)
    merged_data['__npz_version__'] = np.array("2.0")
    np.savez_compressed(npz_file, **merged_data)


def load_npz_to_data_list(npz_file):
    """
    Reload the data list save by ``save_data_list_to_npz``.

    Args:
        npz_file(str): the npz file location.

    Returns:
        a list of data where each data is a dict of numpy ndarray.
    """
    def _split_data(values, seq_lens):
        res = []
        s = 0
        for l in seq_lens:
            res.append(values[s: s + l])
            s += l
        return res
    def _split_data_by_shape(flat_value, shape_strs):
        res = []
        s = 0
        for shape_str in shape_strs:
            if len(shape_str) == 0:
                l = 1
                res.append(flat_value[s])
            else:
                shape = list(map(int, shape_str.split('_')))
                l = np.prod(shape)
                res.append(flat_value[s: s + l].reshape(shape))
            s += l
        return res

    def _process_v1(merged_data):
        names = [name for name in merged_data.keys() if not name.endswith('.seq_len')]
        if '__npz_version__' in names:
            names.remove('__npz_version__')
        data_dict = {}
        for name in names:
            data_dict[name] = _split_data(merged_data[name], merged_data[name + '.seq_len'])

        data_list = []
        n = len(data_dict[names[0]])
        for i in range(n):
            data = {name:data_dict[name][i] for name in names}
            data_list.append(data)
        return data_list

    def _process_v2(merged_data):
        names = [name for name in merged_data.keys() if not name.endswith('.shape')]
        if '__npz_version__' in names:
            names.remove('__npz_version__')
        data_dict = {}
        for name in names:
            data_dict[name] = _split_data_by_shape(merged_data[name], merged_data[name + '.shape'])

        data_list = []
        n = len(data_dict[names[0]])
        for i in range(n):
            data = {name:data_dict[name][i] for name in names}
            data_list.append(data)
        return data_list

    merged_data = np.load(npz_file)
    version = merged_data.get('__npz_version__', '1.0')
    if version == '1.0':
        data_list = _process_v1(merged_data)
    elif version == '2.0':
        data_list = _process_v2(merged_data)
    else:
        raise ValueError(f'version not found: {version}')
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


if __name__ == '__main__':
    data_list = [
        {
            'a': np.arange(10).reshape(2, 5),
            'b': np.arange(4).reshape(1, 4),
            'c': 5,
        },
        {
            'a': np.arange(8).reshape(4, 2),
            'b': np.arange(5).reshape(5, 1),
            'c': 10,
        },
    ]
    npz_file = 'tmp.npz'
    print(data_list)
    save_data_list_to_npz(data_list, npz_file=npz_file)
    data_list = load_npz_to_data_list(npz_file)
    print(data_list)