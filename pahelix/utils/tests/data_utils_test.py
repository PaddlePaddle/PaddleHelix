#!/usr/bin/python
#-*-coding:utf-8-*-
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
import sys
import numpy as np
import unittest

from pahelix.utils.data_utils import load_npz_to_data_list
from pahelix.utils.data_utils import save_data_list_to_npz


class DataUtilsTest(unittest.TestCase):
    def test_data_list_to_npz(self):
        data_list = [
            {"a": np.array([1,23,4])},
            {"a": np.array([2,34,5])}
        ]
        npz_file = 'tmp.npz'
        save_data_list_to_npz(data_list, npz_file)
        reload_data_list = load_npz_to_data_list(npz_file)
        self.assertEqual(len(data_list), len(reload_data_list))
        for d1, d2 in zip(data_list, reload_data_list):
            self.assertEqual(len(d1), len(d2))
            for key in d1:
                self.assertTrue((d1[key] == d2[key]).all())


if __name__ == '__main__':
    unittest.main()