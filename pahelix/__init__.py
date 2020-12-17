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
Initialize.
"""

from warnings import warn
def get_fluid_version():
    import paddle
    paddle_version = int(paddle.__version__.replace('.', '').split('-')[0])
    return paddle_version

paddle_version = get_fluid_version()
if paddle_version <200:
    print('Warning:\n \tYou are using paddle version less than 2.0.0, which will cause errors during the execution.\n \t To avoid that, please use paddle version >= 2.0.0rc0')
