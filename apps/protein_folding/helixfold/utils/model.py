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

from alphafold_paddle.model import modules


class RunModel(nn.Layer):
    """
    RunModel
    """
    def __init__(self, train_config, model_config):
        super(RunModel, self).__init__()

        self.loss_rescale_with_n_res = train_config.get('loss_rescale_with_n_res', False)

        # channel_num = {k: v.shape[-1] for k, v in self.batch.items()}
        # pylint: disable=
        channel_num = {'aatype': 106, 'residue_index': 106, 'seq_length': 1, 
            'is_distillation': 1, 'seq_mask': 106, 'msa_mask': 106, 
            'msa_row_mask': 512, 'random_crop_to_size_seed': 2, 
            'atom14_atom_exists': 14, 'residx_atom14_to_atom37': 14, 
            'residx_atom37_to_atom14': 37, 'atom37_atom_exists': 37, 
            'extra_msa': 106, 'extra_msa_mask': 106, 'extra_msa_row_mask': 1024, 
            'bert_mask': 106, 'true_msa': 106, 'extra_has_deletion': 106, 
            'extra_deletion_value': 106, 'msa_feat': 49, 'target_feat': 22}
        self.alphafold = modules.AlphaFold(channel_num, model_config.model)

    def forward(self, batch, compute_loss=True):
        """
        all_atom_mask: (b, N_res, 37)
        """
        res = self.alphafold(
                batch['feat'],
                batch['label'],
                ensemble_representations=True,
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
