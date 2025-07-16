#   Copyright (c) 2024 PaddleHelix Authors. All Rights Reserved.
#
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# International License (the "License");  you may not use this file  except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RunModel."""
import logging

import paddle
import paddle.nn as nn
from helixfold.model import modules_all_atom

logger = logging.getLogger(__name__)

class RunModel(nn.Layer):
    """RunModel"""
    
    def __init__(self, model_config):
        super(RunModel, self).__init__()

        self.model_config = model_config

        assert model_config.model.global_config.get('all_atom_mode', False), "only support HF3"
        self.helixfold = modules_all_atom.HelixFold3(
                model_config.model.channel_num, model_config.model)
       

    def forward(self, batch, compute_loss=True):
        """forward"""

        res = self.helixfold(
                batch['feat'],
                batch['label_cropped'],
                batch['label'],
                ensemble_representations=False,
                return_representations=True,
                compute_loss=compute_loss)

        return res

    def init_params(self, params_path: str):
        """init params"""
        if params_path.endswith('.pd') or params_path.endswith('.pdparams'):
            logger.info('Load as Paddle model')
            pd_params = paddle.load(params_path)

        else:
            raise ValueError('Unsupported params file type')

        self.helixfold.set_state_dict(pd_params)