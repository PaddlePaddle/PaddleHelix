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
Paddle Varbase
"""

import paddle
from paddle.fluid.core import VarBase

VarBase.tmp = VarBase.__getitem__
def __getitem__(self, idx):
    """
    getitem function
    """
    if isinstance(idx, paddle.Tensor) and len(idx.shape) == 1:
        out = paddle.gather(self, idx)
        return out
    elif isinstance(idx, paddle.Tensor) and str(idx.dtype) == "VarType.BOOL":
        idx = paddle.cast(idx, "int32")
        idx = paddle.nonzero(idx)
        out = paddle.gather_nd(self, idx)
        return out
    elif isinstance(idx, tuple):
        return self.tmp(idx)
        # TODO(syf): 出来为(slice(None, None, None), slice(None, None, None), 0)
    else:
        return self.tmp(idx)
VarBase.__getitem__ = __getitem__


VarBase.setitem_tmp = VarBase.__setitem__
def __setitem__(self, idx, value):
    """
    setitem function
    """
    if isinstance(idx, paddle.Tensor) and str(idx.dtype) == "VarType.BOOL":
        value_tensor = paddle.full(self.shape, value, self.dtype)
        paddle.assign(paddle.where(idx, value_tensor, self), self)
    else:
        return self.setitem_tmp(idx, value)
VarBase.__setitem__ = __setitem__