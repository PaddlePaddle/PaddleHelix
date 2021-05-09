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
"""
Paddle Container
"""

import paddle
from paddle.nn import Sequential

Sequential.tmp = Sequential.__getitem__

def __getitem__(self, name):
    """
    getitem function
    """
    if isinstance(name, slice):
        return self.__class__(*(list(self._sub_layers.values())[name]))
    else:
        if name >= len(self._sub_layers):
            raise IndexError('index {} is out of range'.format(name))
        elif name < 0:
            while name < 0:
                name += len(self._sub_layers)
        return self.tmp(name)
    

Sequential.__getitem__ = __getitem__