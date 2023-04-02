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
Paddle Optimizer
"""

from paddle.optimizer import Momentum as Base_Momentum
from paddle.optimizer import Adam as Base_Adam
from paddle.regularizer import L2Decay


def update_parameters(parameters):
    """
    Update parameters
    """
    parameters_list = list()
    if parameters is not None:
        for items in parameters:
            if isinstance(items, dict):
                params = items["params"]
                if "lr" in items:
                    for p in params:
                        p.optimize_attr["learning_rate"] = items["lr"]
                if "weight_decay" in items:
                    for p in params:
                        p.regularizer = L2Decay(items["weight_decay"])
                parameters_list.extend(params)

            else:
                parameters_list.append(items)
    return parameters_list
                    

class Momentum(Base_Momentum):
    """
    Momentum
    """
    def __init__(self,
                 learning_rate=0.001,
                 momentum=0.9,
                 parameters=None,
                 use_nesterov=False,
                 weight_decay=None,
                 grad_clip=None,
                 name=None):
        """
        Initialization
        """
        parameters_list = update_parameters(parameters)
        super().__init__(
             learning_rate=learning_rate,
             momentum=momentum,
             parameters=parameters,
             use_nesterov=use_nesterov,
             weight_decay=weight_decay,
             grad_clip=grad_clip,
             name=name)


class Adam(Base_Adam):
    """
    Adam
    """
    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-08,
                 parameters=None,
                 weight_decay=None,
                 grad_clip=None,
                 name=None,
                 lazy_mode=False):
        """
        Initialization
        """
        parameters_list = update_parameters(parameters)
        super().__init__(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            parameters=parameters_list,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name,
            lazy_mode=lazy_mode)
