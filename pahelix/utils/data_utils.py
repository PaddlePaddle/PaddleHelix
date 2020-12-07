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
Tools for compound features.
"""

import numpy as np

def save_data_list_to_npz(data_list, npz_file):
    """tbd"""
    keys = data_list[0].keys()
    merged_data = {}
    for key in keys:
        lens = np.array([len(data[key]) for data in data_list])
        values = np.concatenate([data[key] for data in data_list], 0)
        merged_data[key] = values
        merged_data[key + '.seq_len'] = lens
    np.savez_compressed(npz_file, **merged_data)


def load_npz_to_data_list(npz_file):
    """tbd"""
    def split_data(values, seq_lens):
        """tbd"""
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
        data_dict[name] = split_data(merged_data[name], merged_data[name + '.seq_len'])

    data_list = []
    n = len(data_dict[names[0]])
    for i in range(n):
        data = {name:data_dict[name][i] for name in names}
        data_list.append(data)
    return data_list