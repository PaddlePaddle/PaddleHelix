#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

"""RunModel."""

import numpy as np
import paddle
import paddle.nn as nn
import logging
import io

from helixfold.model import modules
from helixfold.model import modules_all_atom
from helixfold.model import modules_multimer
from helixfold.model.model import MSA_FEAT_DIM
from helixfold.common import residue_constants
from helixfold.model import utils
logger = logging.getLogger(__name__)

class RunModel(nn.Layer):
    """
    RunModel
    """
    def __init__(self, train_config, model_config):
        super(RunModel, self).__init__()

        self.model_config = model_config

        self.loss_rescale_with_n_res = train_config.get('loss_rescale_with_n_res', False)

        channel_num = {
            'target_feat': len(residue_constants.restypes_with_x_and_gap),
            'msa_feat': MSA_FEAT_DIM,
        }

        if model_config.model.global_config.get('all_atom_mode', False):
            self.helixfold = modules_all_atom.HelixFold3(
                model_config.model.channel_num, model_config.model)
        elif model_config.model.global_config.get('multimer_mode', False):
            channel_num['target_feat'] = len(residue_constants.restypes_with_x)
            self.helixfold = modules_multimer.HelixFold(
                channel_num, model_config.model)
        else:
            self.helixfold = modules.HelixFold(channel_num, model_config.model)

    def forward(self, batch, compute_loss=True):
        """
        all_atom_mask: (b, N_res, 37)
        """
        # print('msa_mask', batch['feat']['msa_mask'][:, 0].sum(-1), batch['feat']['msa_mask'].sum(1)[:,:10])
        # print('seq_mask', batch['feat']['seq_mask'].sum(-1))

        res = self.helixfold(
                batch['feat'],
                batch['label_cropped'],
                batch['label'],
                ensemble_representations=False,
                return_representations=True,
                compute_loss=compute_loss)
        if compute_loss:
            results, loss = res
            if self.loss_rescale_with_n_res:
                N_res = paddle.sum(batch['label']['all_atom_mask'][:, :, 0], 1)
                loss = loss * paddle.sqrt(paddle.cast(N_res, 'float32'))
            return results, loss.mean()
        else:
            return res

    def init_params(self, params_path: str):
        if params_path.endswith('.npz'):
            with open(params_path, 'rb') as f:
                params = np.load(io.BytesIO(f.read()), allow_pickle=False)
                params = dict(params)

            pd_params = utils.jax_params_to_paddle(params)
            pd_params = {
                k[len('helixfold.'):]: v
                for k, v in pd_params.items()
            }

            if self.model_config.model.global_config.fuse_attention:
                # FIXME
                utils.pd_params_merge_qkvw(pd_params)

        elif params_path.endswith('.pd') or params_path.endswith('.pdparams'):
            logger.info('Load as Paddle model')
            pd_params = paddle.load(params_path)

        else:
            raise ValueError('Unsupported params file type')

        self.helixfold.set_state_dict(pd_params)