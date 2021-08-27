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
| Basic utils
"""

import numpy as np
import os
import random
import json

from pgl.utils.data import Dataloader


def mp_pool_map(list_input, func, num_workers):
    """list_output = [func(input) for input in list_input]"""
    class _CollateFn(object):
        def __init__(self, func):
            self.func = func
        def __call__(self, data_list):
            new_data_list = []
            for data in data_list:
                index, input = data
                new_data_list.append((index, self.func(input)))
            return new_data_list

    # add index
    list_new_input = [(index, x) for index, x in enumerate(list_input)]
    data_gen = Dataloader(list_new_input, 
            batch_size=8, 
            num_workers=num_workers, 
            shuffle=False,
            collate_fn=_CollateFn(func))  

    list_output = []
    for sub_outputs in data_gen:
        list_output += sub_outputs
    list_output = sorted(list_output, key=lambda x: x[0])
    # remove index
    list_output = [x[1] for x in list_output]
    return list_output


def load_json_config(path):
    """tbd"""
    return json.load(open(path, 'r'))