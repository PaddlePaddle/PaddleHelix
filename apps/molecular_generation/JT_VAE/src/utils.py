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
"""general utils"""
import json


def read_file(path):
    """Return a file into list"""
    with open(path, 'r') as f:
        data = f.read().splitlines()
    return data


def load_json_config(path):
    """Reutrn a json file into dict"""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def onek_encoding_unk(x, allowable_set):
    """One-hot embedding"""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: int(x == s), allowable_set))