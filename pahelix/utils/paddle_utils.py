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
| Paddle utils.
"""

import os
from os.path import exists

from paddle import fluid
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
from paddle.fluid.incubate.fleet.base import role_maker


def get_distributed_optimizer(optimizer):
    """
    Get the default collective distributed optimizer under fleet.
    """
    dist_strategy = DistributedStrategy()
    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(role)
    optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)
    return optimizer


def load_partial_params(exe, init_model, main_program):
    """
    Load partial params by checking whether it's in the :attr:`init_model` folder.

    Args:
        exe: Paddle executor.
        init_model(str): the model folder to load from.
        main_program: Paddle program.
    """
    assert exists(init_model), "[%s] cann't be found." % init_model

    def _existed_params(var):
        if not isinstance(var, fluid.framework.Parameter):
            # print("%s not existed" % var.name)
            return False
        if exists(os.path.join(init_model, var.name)):
            # print("load %s successful" % var.name)
            return True
        else:
            # print("%s not existed" % var.name)
            return False

    fluid.io.load_vars(
            exe,
            init_model,
            main_program=main_program,
            predicate=_existed_params)
    print("Load parameters from {}.".format(init_model))