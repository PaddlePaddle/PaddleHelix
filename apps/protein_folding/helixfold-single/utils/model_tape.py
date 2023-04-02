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

from alphafold_paddle.model.config import NUM_MSA_SEQ, NUM_EXTRA_SEQ
from alphafold_paddle.data.data_utils import aatype_to_sequence
from utils.utils import tree_map

from tape.others.protein_sequence_model_dynamic import ProteinEncoderModel, ProteinModel
from tape.others.dataset import transform_text_to_bert_feature, collate_bert_features



class RunTapeModel(nn.Layer):
    """
    RunModel
    """
    def __init__(self, train_config, model_config, tape_model_config, af2_model_config):
        super(RunTapeModel, self).__init__()
        self.train_config = train_config
        self.model_config = model_config
        self.tape_model_config = tape_model_config
        self.af2_model_config = af2_model_config

        self.freeze_tape = self.model_config.get('freeze_tape', False)

        self._init_tape_encoder()

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
        self.alphafold = modules.AlphaFold(channel_num, af2_model_config.model)

        def _print_attribute(key, value):
            print(f'[{self.__class__.__name__}] {key}: {value}')
        _print_attribute('freeze_tape', self.freeze_tape)
    
    def _init_tape_encoder(self):
        encoder_model = ProteinEncoderModel(self.tape_model_config, name='protein')
        self.tape_model = ProteinModel(encoder_model, self.tape_model_config)
        self.tape_single_linear = nn.Linear(
                self.tape_model_config.hidden_size,
                self.af2_model_config.model.embeddings_and_evoformer.msa_channel)
        weight_out_dim = self.model_config.last_n_weight * self.tape_model_config.head_num
        self.tape_pair_linear = nn.Linear(
                weight_out_dim,
                self.af2_model_config.model.embeddings_and_evoformer.pair_channel)

    def _create_tape_input(self, batch):
        aatypes = batch['feat']['aatype'][:, 0]    # (b, 4, num_res) -> (b, num_res)
        seq_lengths = batch['feat']['seq_length'][:, 0]    # (b, 4) -> (b)
        # convert to tape features
        data_list = []
        for aatype, seq_len in zip(aatypes, seq_lengths):
            text = aatype_to_sequence(aatype[:seq_len])
            data = transform_text_to_bert_feature(text)
            data_list.append(data)
        # collate
        tape_input = collate_bert_features(data_list)
        tape_input = tree_map(lambda x: paddle.to_tensor(x), tape_input)
        return tape_input

    def _forward_tape(self, batch):
        def _insert_recycle_dim(tensor, num_recycle):
            """shape: (d0,d1,...) -> (d0,num_recycle,d1,...)"""
            expand_shape = [1] * (len(tensor.shape) + 1)
            expand_shape[1] = num_recycle
            return paddle.tile(tensor.unsqueeze(1), expand_shape)

        tape_input = self._create_tape_input(batch)
        tape_results = self.tape_model.model.encoder_model.encoder_model(
                tape_input['sequence'], tape_input['position'],
                return_representations=True, return_last_n_weight=self.model_config.last_n_weight)
        num_recycle = batch['feat']['aatype'].shape[1]
        tape_results = tree_map(lambda x: _insert_recycle_dim(x, num_recycle), tape_results)
        if self.freeze_tape and batch['cur_step'] < self.model_config.freeze_tape_step:
            tape_results = tree_map(lambda x: x.detach(), tape_results)

        output = tape_results['output'][:, :, 1:-1]     # (b, num_recycle, num_res, d1)
        attn_weight = tape_results['attn_weight'][:, :, :, 1:-1, 1:-1].transpose([0, 1, 3, 4, 2])   # (b, num_recycle, num_res, num_res, d2)
        tape_single = self.tape_single_linear(output)   # (b, num_recycle, num_res, msa_channel)
        tape_pair = self.tape_pair_linear(attn_weight)  # (b, num_recycle, num_res, num_res, pair_channel)
        batch['feat'].update(tape_single=tape_single, tape_pair=tape_pair)
        return batch
    
    def forward(self, batch, compute_loss=True):
        """
        all_atom_mask: (b, N_res, 37)
        """
        batch = self._forward_tape(batch)
        res = self.alphafold(
                batch['feat'],
                batch['label'],
                ensemble_representations=True,
                return_representations=True,
                compute_loss=compute_loss)
        if compute_loss:
            results, loss = res
            # if self.loss_rescale_with_n_res:
            #     N_res = paddle.sum(batch['label']['all_atom_mask'][:, :, 0], 1)
            #     loss = loss * paddle.sqrt(paddle.cast(N_res, 'float32'))
            return results, loss.mean()
        else:
            return res
    
    def load_tape_params(self, tape_init_model):
        """tbd"""
        if not tape_init_model is None and tape_init_model != "":
            print(f"Load pretrain tape model from {tape_init_model}")
            self.tape_model.set_state_dict(paddle.load(tape_init_model))
    
    def load_params(self, init_model):
        """tbd"""
        if not init_model is None and init_model != "":
            print(f"Load model from {init_model}")
            self.set_state_dict(paddle.load(init_model))
    
    def save_params(self, param_path):
        paddle.save(self.state_dict(), param_path)

