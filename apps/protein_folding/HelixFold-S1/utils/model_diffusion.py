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

"""RunDiffusionModel."""

import numpy as np
import paddle
import paddle.nn as nn
import logging
import io

from helixfold.model import modules
from helixfold.model import modules_multimer
from helixfold.model.model import MSA_FEAT_DIM
from helixfold.common import residue_constants
from helixfold.model import utils
from helixfold.model import geometry, all_atom, r3
from utils.diffusion_modules import Diffuser, DDIMSampler
from scipy.spatial.transform import Rotation as R
logger = logging.getLogger(__name__)

class RunDiffusionModel(nn.Layer):
    """
    RunModel
    """
    def __init__(self, train_config, model_config):
        super(RunDiffusionModel, self).__init__()

        self.model_config = model_config

        self.loss_rescale_with_n_res = train_config.get('loss_rescale_with_n_res', False)

        channel_num = {
            'target_feat': len(residue_constants.restypes_with_x_and_gap),
            'msa_feat': MSA_FEAT_DIM,
        }

        if model_config.model.global_config.get('multimer_mode', False):
            channel_num['target_feat'] = len(residue_constants.restypes_with_x)
            self.helixfold = modules_multimer.HelixFold(
                channel_num, model_config.model)
        else:
            self.helixfold = modules.HelixFold(channel_num, model_config.model)

        max_diffuse_step, beta_1, beta_T = self.model_config.model.get("max_diffuse_step",1000), \
                            self.model_config.model.get("beta_1", 0.0001), \
                            self.model_config.model.get("beta_T", 0.02)
        ddim_sample_step, ddim_eta = self.model_config.model.get("ddim_sample_step", 20), \
                    self.model_config.model.get("ddim_eta",0.0)
        self.diffuser = Diffuser(T = max_diffuse_step, beta_1 = beta_1, beta_T = beta_T)
        self.diffusion_sampler = DDIMSampler(T = max_diffuse_step, beta_1 = beta_1, beta_T = beta_T,
            S = ddim_sample_step, eta = ddim_eta,  # new params for DDIM
            igso3_diffuser=self.diffuser.igso3,
            var_type='fixedsmall'
        )
        self.disable_sampling_infer = self.model_config.model.get("disable_sampling_infer", False)
        self.disable_rots_diffuse = self.model_config.model.get("disable_rots_diffuse", False)
        self.disable_trans_diffuse = self.model_config.model.get("disable_trans_diffuse", False)

    def center_translation_label(self, label):
        """
        locate translation related label to center
        """
        centered_label = {}
        for key, value in label.items():
            if key in ["all_atom_positions"]:
                centered_label[key] = value - np.mean(value,axis=-2) # [B,M,3]
            else:
                centered_label[key] = value
        return centered_label

    def forward(self, batch, compute_loss=True):
        """
        all_atom_mask: (b, N_res, 37)
        """
        batch_size, repeated_size, num_res = batch["feat"]["aatype"].shape
        if self.training:
            timestep = self.diffuser.sample_timestep(batch_size) # [batch]

            # trans
            x_0_trans = batch["label_cropped"]['backbone_affine_tensor_trans'] # [batch, num_res, 3]
            x_0_trans = x_0_trans.numpy()
            x_0_trans /= self.model_config.model.heads.structure_module.position_scale
            x_t_trans = self.diffuser.diffuse_trans(x_0_trans, t=timestep)

            # rots
            x_0_rots = batch["label_cropped"]['backbone_affine_tensor_rot'] # [batch, num_res, 3, 3]
            x_0_rots = x_0_rots.numpy()
            x_t_rots = self.diffuser.diffuse_rots(x_0_rots, t=timestep)

            # convert to tensor
            x_t_trans = paddle.to_tensor(x_t_trans.astype(np.float32))
            x_t_rots = paddle.to_tensor(x_t_rots.astype(np.float32))
            timestep = paddle.to_tensor(timestep.astype(np.int32))

            batch["feat"].update({"trans_diffused": 
                    paddle.repeat_interleave(x_t_trans.unsqueeze(1), repeated_size, axis=1)
                    })
            if not self.disable_rots_diffuse: batch["feat"].update({"rots_diffused": 
                    paddle.repeat_interleave(x_t_rots.unsqueeze(1), repeated_size, axis=1)
                    })
            if not self.disable_trans_diffuse: batch["feat"].update({"diffuse_timestep": 
                    paddle.repeat_interleave(timestep.unsqueeze(1), repeated_size, axis=1)  # repeat to avoid batch ensemble error
                    })
            
            res = self.helixfold(
                    batch['feat'],
                    batch['label_cropped'],
                    ensemble_representations=True,
                    return_representations=True,
                    compute_loss=compute_loss)
        elif self.disable_sampling_infer:
            x_t_trans = np.random.standard_normal(size=(batch_size, num_res, 3)) * 20 # random noise
            x_t_rots = R.random((batch_size* num_res)).as_matrix().reshape((batch_size, num_res, 3, 3))
            timestep = self.diffuser.last_timestep(batch_size) # [batch]

            # convert to tensor
            x_t_trans = paddle.to_tensor(x_t_trans.astype(np.float32))
            x_t_rots = paddle.to_tensor(x_t_rots.astype(np.float32))
            timestep = paddle.to_tensor(timestep.astype(np.int32))

            batch["feat"].update({"trans_diffused": 
                    paddle.repeat_interleave(x_t_trans.unsqueeze(1), repeated_size, axis=1)
                    })
            if not self.disable_rots_diffuse: batch["feat"].update({"rots_diffused": 
                    paddle.repeat_interleave(x_t_rots.unsqueeze(1), repeated_size, axis=1)
                    })
            if not self.disable_trans_diffuse: batch["feat"].update({"diffuse_timestep": 
                    paddle.repeat_interleave(timestep.unsqueeze(1), repeated_size, axis=1)  # repeat to avoid batch ensemble error
                    })
            res = self.helixfold(
                    batch['feat'],
                    batch['label_cropped'],
                    ensemble_representations=True,
                    return_representations=True,
                    compute_loss=compute_loss)  
        else: # diffusion inference
            # x_t_trans = paddle.to_tensor(np.random.standard_normal(size=(batch_size, num_res, 3)), dtype='float32') # random noise
            # x_t_rots = paddle.to_tensor(R.random((batch_size* num_res)).as_matrix().reshape((batch_size, num_res, 3, 3)), dtype='float32')
            # timestep = self.diffuser.last_timestep(batch_size) # [batch]

            x_t_trans = np.random.standard_normal(size=(batch_size, num_res, 3)) * 20 # random noise
            x_t_rots = R.random((batch_size* num_res)).as_matrix().reshape((batch_size, num_res, 3, 3))
            timestep = self.diffuser.last_timestep(batch_size) # [batch]

            # convert to tensor
            x_t_trans = paddle.to_tensor(x_t_trans.astype(np.float32))
            x_t_rots = paddle.to_tensor(x_t_rots.astype(np.float32))
            timestep = paddle.to_tensor(timestep.astype(np.int32))

            res = self.diffusion_sampler.inference(
                x_T_trans=x_t_trans, 
                x_T_rots=x_t_rots, 
                batch=batch, 
                helixfold=self.helixfold, 
                repeated_size=repeated_size,
                compute_loss=compute_loss,
                position_scale=self.model_config.model.heads.structure_module.position_scale
                )

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