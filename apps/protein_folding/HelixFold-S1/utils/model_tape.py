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

from helixfold.model import modules

from helixfold.model.config import NUM_MSA_SEQ, NUM_EXTRA_SEQ
from helixfold.data.data_utils import aatype_to_sequence
from utils.utils import tree_map, sequence_pad, pair_pad

from tape.others.models import DeBERTaEncoderModel, PretrainModel
from tape.others.dataset import transform_text_to_bert_feature, collate_bert_features, replace_fake_start_end_token


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

        self.freeze_tape = self.train_config.get('freeze_tape', False)
        self.tape_mode = self.model_config.get('tape_mode', 'solo')
        self.use_masked_msa = self.model_config.get('use_masked_msa', False)
        if self.use_masked_msa:
            assert self.tape_mode == 'G_linker', 'use_masked_msa only support G_linker so far.'

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
        self.helixfold = modules.HelixFold(channel_num, af2_model_config.model)

        def _print_attribute(key, value):
            print(f'[{self.__class__.__name__}] {key}: {value}')
        _print_attribute('freeze_tape', self.freeze_tape)
        _print_attribute('tape_mode', self.tape_mode)
        _print_attribute('use_masked_msa', self.use_masked_msa)
    
    def _init_tape_encoder(self):
        self.tape_model = PretrainModel(self.tape_model_config, task_config=None)
        self.tape_single_linear = nn.Linear(
                self.tape_model_config.hidden_size, 
                self.af2_model_config.model.embeddings_and_evoformer.msa_channel)
        weight_out_dim = self.model_config.last_n_weight * self.tape_model_config.head_num
        self.tape_pair_linear = nn.Linear(
                weight_out_dim, 
                self.af2_model_config.model.embeddings_and_evoformer.pair_channel)

    def _create_tape_input(self, batch):
        aatypes = batch['feat']['chain_aatypes'][:, 0].numpy()               # (b, 4, num_res) -> (b, num_res)
        seq_lengths = batch['feat']['chain_seq_lengths'][:, 0].numpy()       # (b, 4) -> (b)

        # convert to tape features
        data_list = []
        for i, (aatype, seq_len) in enumerate(zip(aatypes, seq_lengths)):
            text = aatype_to_sequence(aatype[:seq_len])
            # text will be pad with start and end token, num_res -> num_res + 2
            data = transform_text_to_bert_feature(text)
            data_list.append(data)
        # collate
        tape_input = collate_bert_features(data_list)
        tape_input = tree_map(lambda x: paddle.to_tensor(x), tape_input)
        return tape_input

    def _forward_tape(self, batch):
        def _insert_recycle_dim(tensor, num_recycle):
            """shape: (d0,d1,...) -> (d0,num_recycle,d1,...)"""
            return paddle.concat([tensor.unsqueeze(1)] * num_recycle, 1)

        tape_input = self._create_tape_input(batch)
        tape_input['seq_id'] = tape_input['sequence']
        if self.freeze_tape:
            print('----- freeze_tape')
            with paddle.no_grad():
                tape_results = self.tape_model.encoder_model(tape_input, return_last_n_weight=self.model_config.last_n_weight)    
                tape_results = tree_map(lambda x: x.detach(), tape_results)
        else:
            tape_results = self.tape_model.encoder_model(tape_input, return_last_n_weight=self.model_config.last_n_weight)    
        batch_size, num_recycle, crop_size = batch['feat']['aatype'].shape
        tape_results = tree_map(lambda x: _insert_recycle_dim(x, num_recycle), tape_results)
        
        output = tape_results['encoder_output'][:, :, 1:-1]     # (b, num_recycle, num_res, d1)
        attn_weight = tape_results['attn_weight'][:, :, :, 1:-1, 1:-1].transpose([0, 1, 3, 4, 2])   # (b, num_recycle, num_res, num_res, d2)

        attn_hidden_size = attn_weight.shape[-1]
        output_hidden_size = output.shape[-1]
        new_attn_weights = paddle.zeros(shape=[batch_size, num_recycle, crop_size, crop_size, attn_hidden_size])
        new_outputs = paddle.zeros(shape=[batch_size, num_recycle, crop_size, output_hidden_size])
        chain_sum = 0
        for batch_idx, cur_chain_num in enumerate(batch['feat']['chain_nums'][:, 0]):
            #######
            ## new_outputs
            #######
            cur_sample = []   # merge sequence
            cur_crop_idx = batch['feat']['crop_idx'][batch_idx][0]
            # 1. get merge sequence using seq_len.  
            # |_______|_________|_____|
            for idx in range(cur_chain_num):
                cur_chain_idx = chain_sum + idx
                cur_seq_len = batch['feat']['chain_seq_lengths'][cur_chain_idx][0]
                cur_chain_output = output[cur_chain_idx, :, :cur_seq_len]
                cur_sample.append(cur_chain_output)
            cur_sample = paddle.concat(cur_sample, axis=1)
            # 2. apply crop_idx on merge sequence to get crop_result.  
            # |_______|_________|_____| + crop_idx ==> |_________________|
            cur_sample_crop = paddle.index_select(cur_sample, cur_crop_idx, axis=1)
            new_outputs[batch_idx, :, :len(cur_crop_idx)] = cur_sample_crop

            #######
            ## new_attn_weights
            #######
            crop_idx_sum = 0       # update using asym_id feat
            crop_value_sum = 0     # update using chain_seq_lengths feat
            crop_idx_nums = batch['feat']['crop_nums'][batch_idx][0]
            for idx, crop_idx_num in enumerate(crop_idx_nums[1:]):
                cur_chain_idx = chain_sum + idx
                cur_seq_len = batch['feat']['chain_seq_lengths'][cur_chain_idx][0]
                # chain may not be selected
                if crop_idx_num == 0:
                    crop_value_sum += cur_seq_len
                    continue
                crop_start, crop_end = crop_idx_sum, crop_idx_sum + crop_idx_num
                cur_crop_val = batch['feat']['crop_idx'][batch_idx][0][crop_start:crop_end]
                cur_crop_val -= crop_value_sum
                cur_attn_weight = paddle.index_select(attn_weight[cur_chain_idx], cur_crop_val, axis=1)   
                cur_attn_weight = paddle.index_select(cur_attn_weight, cur_crop_val, axis=2)
                new_attn_weights[batch_idx, :, crop_start:crop_end, crop_start:crop_end] = cur_attn_weight
                crop_idx_sum += crop_idx_num
                crop_value_sum += cur_seq_len
            chain_sum += cur_chain_num
        
        tape_single = self.tape_single_linear(new_outputs)   # (b, num_recycle, num_res, msa_channel)
        tape_pair = self.tape_pair_linear(new_attn_weights)  # (b, num_recycle, num_res, num_res, pair_channel)
        batch['feat'].update(tape_single=tape_single, tape_pair=tape_pair)

        return batch

    def _create_tapeG_input(self, batch):
        def _slice_1st_recycle(x):
            return x[:, 0].numpy()
        G_len = 30
        aatypes = _slice_1st_recycle(batch['feat']['chain_aatypes'])         # (b_n, num_res)
        seq_lengths = _slice_1st_recycle(batch['feat']['chain_seq_lengths']) # (b_n)
        chain_nums = _slice_1st_recycle(batch['feat']['chain_nums'])    # (b)

        # convert to tape features
        data_list = []
        unG_index_list = []
        offset = 0
        for batch_i, chain_num in enumerate(chain_nums):
            concat_text = ''
            G_unmask = []
            for subbatch_i in range(offset, offset + chain_num):
                text = aatype_to_sequence(aatypes[subbatch_i][:seq_lengths[subbatch_i]])
                concat_text += text
                G_unmask.append(np.ones(len(text)))
                if subbatch_i < offset + chain_num - 1:
                    concat_text += 'G' * G_len
                    G_unmask.append(np.zeros(G_len))
            G_unmask = np.concatenate(G_unmask, 0)
            unG_index = np.where(G_unmask == 1)[0]

            # text will be pad with start and end token, num_res -> num_res + 2
            data = transform_text_to_bert_feature(concat_text)
            data_list.append(data)
            unG_index_list.append(unG_index)

            offset += chain_num
        # collate
        tape_input = collate_bert_features(data_list)
        tape_input = tree_map(lambda x: paddle.to_tensor(x), tape_input)
        return tape_input, unG_index_list

    def _forward_tapeG(self, batch):
        def _insert_recycle_dim(tensor, num_recycle):
            """shape: (d0,d1,...) -> (d0,num_recycle,d1,...)"""
            return paddle.concat([tensor.unsqueeze(1)] * num_recycle, 1)

        tape_input, unG_index_list = self._create_tapeG_input(batch)
        if self.training and self.use_masked_msa:
            tape_input['seq_id'] = tape_input['masked_sequence']
        else:
            tape_input['seq_id'] = tape_input['sequence']
        if self.freeze_tape:
            print('----- freeze_tape')
            with paddle.no_grad():
                tape_results = self.tape_model.encoder_model(tape_input, return_last_n_weight=self.model_config.last_n_weight)    
                tape_results = tree_map(lambda x: x.detach(), tape_results)
        else:
            tape_results = self.tape_model.encoder_model(tape_input, return_last_n_weight=self.model_config.last_n_weight)    
        batch_size, num_recycle, crop_size = batch['feat']['aatype'].shape

        tape_output = tape_results['encoder_output']     # (b, num_res+n_G+2, d1)
        tape_weight = tape_results['attn_weight'].transpose([0, 2, 3, 1])     # (b, num_head, num_res+n_G+2, num_res+n_G+2) -> (b, num_res+n_G+2, num_res+n_G+2, num_head)
        new_single_list = []
        new_pair_list = []
        if self.use_masked_msa:
            new_label_list = []
        for batch_i in range(batch_size):
            unG_index = paddle.to_tensor(unG_index_list[batch_i], 'int64')
            cur_crop_idx = batch['feat']['crop_idx'][batch_i][0]

            new_single = tape_output[batch_i, 1: -1][unG_index][cur_crop_idx]     # (num_res, d1)
            new_pair = tape_weight[batch_i, 1: -1, 1: -1]           # (num_res+n_G, num_res+n_G, num_head)
            new_pair = paddle.index_select(paddle.index_select(
                    new_pair[unG_index][cur_crop_idx], unG_index, 1), cur_crop_idx, 1)   # (num_res, num_res, num_head)
            new_single_list.append(sequence_pad(new_single, crop_size))
            new_pair_list.append(pair_pad(new_pair, crop_size))
            if self.use_masked_msa:
                new_label = tape_input['label'][batch_i, 1: -1][unG_index][cur_crop_idx]     # (num_res,)
                new_label_list.append(sequence_pad(new_label, crop_size, pad_value=-1))

        new_single_list = _insert_recycle_dim(paddle.stack(new_single_list), num_recycle)     # (b, num_recycle, num_res, d1)
        new_pair_list = _insert_recycle_dim(paddle.stack(new_pair_list), num_recycle)     # (b, num_recycle, num_res, num_res, num_head)
        if self.use_masked_msa:
            new_label_list = _insert_recycle_dim(paddle.stack(new_label_list), num_recycle)     # (b, num_recycle, num_res, num_res, num_head)
            ## replace the true_msa
            true_msa = new_label_list.unsqueeze([2])
            batch['feat']['true_msa'] = paddle.cast(paddle.clip(true_msa, min=0), batch['feat']['true_msa'].dtype)
            batch['feat']['bert_mask'] = paddle.cast(true_msa != -1, batch['feat']['bert_mask'].dtype)
        tape_single = self.tape_single_linear(new_single_list)   # (b, num_recycle, num_res, msa_channel)
        tape_pair = self.tape_pair_linear(new_pair_list)  # (b, num_recycle, num_res, num_res, pair_channel)
        batch['feat'].update(tape_single=tape_single, tape_pair=tape_pair)
        return batch
    
    def forward(self, batch, compute_loss=True):
        """
        all_atom_mask: (b, N_res, 37)
        """
        if self.tape_mode == 'solo':
            batch = self._forward_tape(batch)
        elif self.tape_mode == 'G_linker':
            batch = self._forward_tapeG(batch)
        else:
            raise ValueError(self.tape_mode)
        res = self.helixfold(
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

