#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""Updated modules for helixfold-3 all-atom model"""

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import gc
from copy import deepcopy

FLUID_DEPRECATED = not hasattr(paddle, 'fluid')
if FLUID_DEPRECATED:
    from paddle.base.framework import _dygraph_tracer
else:
    from paddle.fluid.framework import _dygraph_tracer
import logging
from helixfold.model.modules import (
    TriangleMultiplication,
    TriangleAttention,
    OuterProductMean,
    DistogramHead,
    Dropout,
    recompute_wrapper,
    softmax_cross_entropy,
)
from helixfold.model import diffusion
from helixfold.model.diffusion import (
    AtomAttentionEncoder,
    AtomAttentionDecoder,
    AttentionPairBias,
    RelativePositionEncoding,
)

from helixfold.model.utils import subbatch, tree_map
from helixfold.model.utils import get_all_atom_confidence_metrics, get_all_atom_confidence_metrics_pd
from helixfold.model import chain_align_np_aa, lddt, modules
from helixfold.common import residue_constants
from utils.utils import get_structure_module_bf16_op_list, random_choice_top_k_index_with_prob, \
                        find_top_k_upper_triangle_indices
from utils.interface_utils import InterfaceInfo
import time

class HelixFold3(nn.Layer):
    """HelixFold-3 all-atom model
    """
    def __init__(self, channel_num, config):
        super(HelixFold3, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = config.global_config

        self.input_embedder = InputEmbedder(
            self.channel_num, self.config.input_embedder, self.global_config)

        if self.global_config.get('dist_model', False):
            from helixfold.model.dist.modules_all_atom_dap import DistEmbeddingsAndPairformer 
            self.embeddings_and_pairformer = DistEmbeddingsAndPairformer(
                self.channel_num,
                self.config.embeddings_and_pairformer,
                self.global_config)
        else:
            self.embeddings_and_pairformer = EmbeddingsAndPairformer(
                self.channel_num,
                self.config.embeddings_and_pairformer,
                self.global_config)

        # For the reason of batch_dim tile, we put relative_positional_encoding
        # before diffusion module
        self.diff_rel_pos_encoding = RelativePositionEncoding(
            self.channel_num,
            self.config.heads.diffusion_module.diffusion_conditioning.relative_position_encoding,
            self.global_config)
        if self.global_config.get('dist_model', False):   
            from helixfold.model.dist.diffusion_dap import DistDiffusionModule 
            self.diffusion_module = DistDiffusionModule(
                self.channel_num,
                self.config.heads.diffusion_module,
                self.global_config)
        else:
            self.diffusion_module = diffusion.DiffusionModule(
                self.channel_num,
                self.config.heads.diffusion_module,
                self.global_config)
            # self.diffusion_module = paddle.incubate.jit.inference(self.diffusion_module)
        self.distogram = DistogramHead(
            self.channel_num,
            self.config.heads.distogram,
            self.global_config)

        if self.config.heads.confidence_head.weight > 0:
            if self.global_config.get('dist_model', False):
                from helixfold.model.dist.modules_all_atom_dap import DistConfidenceHead
                self.confidence_head = DistConfidenceHead(
                    self.channel_num,
                    self.config.heads.confidence_head,
                    self.global_config)
            else:
                self.confidence_head = ConfidenceHead(
                    self.channel_num,
                    self.config.heads.confidence_head,
                    self.global_config)

        self.gen_interface = self.config.get('gen_interface', False) # whether generate interface
        self.add_interface = self.config.get('add_interface', False) # whether add interface info to the pair representation

        if self.gen_interface:
            self.interface_head = InterfaceHead(self.channel_num, self.config.heads.interface_head, 
                                self.global_config)

        if self.add_interface: 
            self.interface_info_project = nn.Linear(
                    1, self.channel_num['token_pair_channel'],
                    bias_attr=False)


    def forward(self,
                batch,
                label_cropped,
                label,
                return_representations=False,
                ensemble_representations=True,
                compute_loss=True):
        t0 = time.time()
        single_inputs_act, single_init_act, pair_init_act = \
            self.input_embedder(batch)

        single_act = paddle.zeros_like(single_init_act)
        pair_act = paddle.zeros_like(pair_init_act)

        if 'num_iter_recycling' in batch:
            # Training trick: dynamic recycling number
            num_iter = batch['num_iter_recycling'].numpy()[0, 0]
            num_iter = min(int(num_iter), self.config.num_recycle)
        else:
            num_iter = self.config.num_recycle

        seq_mask = batch['seq_mask']
        masks = {
            'msa': batch['msa_mask'],
            'pair': seq_mask.unsqueeze(axis=1) * seq_mask.unsqueeze(axis=2)
        }

        for recycle_idx in range(1 + num_iter):
            single_act, pair_act = single_act.detach(), pair_act.detach()
            single_act, pair_act = self.embeddings_and_pairformer(
                batch, pair_init_act, pair_act,
                single_inputs_act, single_init_act, single_act,
                masks)

        representations = {
            'single_inputs': single_inputs_act,
            'single': single_act,
            'pair': pair_act
        }
        t1 = time.time()
        if not self.training:
            del pair_init_act, single_init_act, masks, seq_mask
            gc.collect()

        if self.training and _dygraph_tracer()._amp_dtype == "bfloat16":
            with paddle.amp.auto_cast(enable=False):
                bf16 = paddle.base.core.VarDesc.VarType.BF16 if FLUID_DEPRECATED else paddle.fluid.core.VarDesc.VarType.BF16
                for key, value in representations.items():
                    if isinstance(value, paddle.Tensor) and value.dtype in [bf16]:
                        temp_value = value.cast('float32')
                        temp_value.stop_gradient = value.stop_gradient
                        representations[key] = temp_value
                for key, value in batch.items():
                    if isinstance(value, paddle.Tensor) and value.dtype in [bf16]:
                        temp_value = value.cast('float32')
                        temp_value.stop_gradient = value.stop_gradient
                        batch[key] = temp_value
                ret, total_loss = self._forward_heads(representations, batch, label_cropped, label, compute_loss)
        else:
            ret, total_loss = self._forward_heads(representations, batch, label_cropped, label, compute_loss)

        t2 = time.time()
        ret["duration"] = {
            "representations": t1 - t0,
            "forward_heads": t2 - t1,
            "total": t2 - t0}
        if compute_loss:
            return ret, total_loss
        else:
            return ret
    
    def _forward_heads(self, representations, batch, label_cropped, label, compute_loss):
        total_loss = 0.
        ret = {}

        # TODO: add is_dna_aa, is_rna_aa and is_ligand_aa in data_loader
        if not 'is_dna_aa' in batch:
            for name in ['is_protein', 'is_dna', 'is_rna', 'is_ligand']:
                batch[f'{name}_aa'] = paddle.stack([x[index] for x, index 
                        in zip(batch[name], batch['ref_token2atom_idx'])])

        ## Generate interface
        if self.gen_interface:
            interface_ret = self.interface_head(representations, batch, label_cropped, label, compute_loss)
            # gen_interface_info = interface_ret['gen_interface_info']
            # gen_interface_indices = interface_ret['gen_interface_indices']
            gen_interfaces = interface_ret['gen_interfaces']
            if compute_loss and 'loss' in interface_ret:
                total_loss += interface_ret['loss'] * self.config.heads.interface_head.weight
            ret['interface_head'] = interface_ret
  
        ## Merge interface_info
        if self.add_interface:
            # gather interface info for each case in the batch
            interface_info_batch = []
            if 'interface_source' in batch: 
                for batch_i in range(len(batch['interface_source'])):
                    interface_source = batch['interface_source'][batch_i]
                    if interface_source == 'gen' and self.gen_interface and \
                        len(gen_interfaces) > batch_i and len(gen_interfaces[batch_i]) > 0:
                        gen_interfaces_repeat0 = gen_interfaces[batch_i][0]
                        gen_interfaces_repeat0_map = gen_interfaces_repeat0.interface_map().astype('float32') # (n_token, n_token)
                        gen_interfaces_repeat0_map_tensor = paddle.to_tensor(gen_interfaces_repeat0_map, 
                                                                        dtype=representations['pair'].dtype)
                        interface_info_batch.append(gen_interfaces_repeat0_map_tensor)
                    elif interface_source == 'gt':
                        interface_info_batch.append(batch['interface_info_sample'][batch_i].cast(representations['pair'].dtype))
                    else:
                        interface_info_batch.append(paddle.zeros(
                                            representations['pair'].shape[1:3],
                                            dtype=representations['pair'].dtype))
            else:
                for _ in range(len(batch['asym_id'])):
                    interface_info_batch.append(paddle.zeros(
                                            representations['pair'].shape[1:3],
                                            dtype=representations['pair'].dtype))
            interface_info = paddle.stack(interface_info_batch) # (batch_size, n_token, n_token)

            # merge the interface_info and the pair representations
            representations['interface_info_encoding'] = self.interface_info_project(
                    interface_info.unsqueeze(axis=-1))

        ## diffusion_module head
        # prepare needed keys
        if self.training:
            diff_batch_size = self.config.heads.diffusion_module.diff_batch_size
            # on which noise will be added
            names_needed = ['all_atom_pos', 'all_atom_pos_mask', 
                        'frame_mask', 'frame_ai_indice', 'frame_bi_indice', 'frame_ci_indice']
            batch.update({k: label_cropped[k] for k in names_needed})
            batch['all_atom_pos_mask'] = batch['all_atom_pos_mask'].cast(batch['all_atom_pos'].dtype)
        else:
            diff_batch_size = self.config.heads.diffusion_module.test_diff_batch_size
            batch['all_atom_pos_mask'] = paddle.ones_like(label['all_atom_pos_mask']).cast(label['all_atom_pos'].dtype)

        # use for-loop instead of inserting diff_batch_size after batch dimention
        max_len_diff_batch = 3_500 # FIXME(zhukunrui): replace with dynamic size accordingly
        diff_batch_size_loop = 1 if batch['asym_id'].shape[1] < max_len_diff_batch else diff_batch_size
        diff_batch_size = diff_batch_size // diff_batch_size_loop

        # Insert diffusion dim: (B, *) -> (B, diff_batch, *)
        #   Don't tile "pair_act" and "rel_pos_encoding" until it's really needed.
        diff_repr = {
            'rel_pos_encoding': self.diff_rel_pos_encoding(batch),
            **representations}
        diff_repr = diffusion.insert_diff_batch_dim(diff_repr, diff_batch_size, 
                special_keys=['pair', 'rel_pos_encoding', 'interface_info_encoding'])
        diff_batch = diffusion.insert_diff_batch_dim(batch, diff_batch_size,
                special_keys=['interface_info', 'pocket_info'])
        diff_label = diffusion.insert_diff_batch_dim(label, diff_batch_size)

        # iterate over batch dim
        sample_ret_list = []
        for batch_i in range(len(batch['asym_id'])):
          sample_ret_to_cat = []
          for _ in range(diff_batch_size_loop):  # swithc between diff_batch and for_loop according to n_tok size
            sample_batch = tree_map(lambda x: x[batch_i], diff_batch)   # dict of (diff_batch, *)
            sample_repr = tree_map(lambda x: x[batch_i], diff_repr)
            sample_ret = self.diffusion_module(sample_repr, sample_batch)   # dict of (diff_batch, *)
            if compute_loss:
                ### do multi chain permutation align to generate new label
                sample_label = tree_map(lambda x: x[batch_i], diff_label)
                sample_aligned_label = chain_align_np_aa.multi_chain_permutation_align(
                        sample_batch, sample_label, sample_ret) 
                loss_output = self.diffusion_module.loss(sample_ret, sample_batch, sample_aligned_label)
                sample_ret.update(loss_output)

            sample_ret_to_cat.append(sample_ret)  # list of dict (diff_batch, *) or (1, *)
          # concat sample_ret along dim of diff_batch
          # unify (diff_batch_size_loop, 1, * ) and (1, diff_batch, *) into (diff_batch, *)
          with paddle.amp.auto_cast(enable=False):
            sample_ret_keys = sample_ret_to_cat[0].keys()
            sample_ret_concated = {}
            for key in sample_ret_keys:
                sample_ret_concated[key] = paddle.concat( # (diff_batch, *)
                    [sample_ret_item[key] for sample_ret_item in sample_ret_to_cat], 
                    axis=0)
            sample_ret_list.append(sample_ret_concated) # list of dict with shape (diff_batch, *)

        with paddle.amp.auto_cast(enable=False):
            diff_ret = merge_list_to_dict(sample_ret_list)    # dict of (B, diff_batch, *)
        if compute_loss:
            cur_loss = diff_ret['loss'].mean([1]) # (B, diff_batch) -> (B,)
            total_loss += cur_loss * self.config.heads.diffusion_module.weight
        ret['diffusion_module'] = diff_ret
        
        ## distogram
        distogram_preds = self.distogram(representations, batch)
        if compute_loss:
            loss_output = self.distogram.loss(
                distogram_preds, label_cropped)
            ret['distogram'] = loss_output
            total_loss += loss_output['loss'] * self.config.heads.distogram.weight

        ## confidence model
        call_conf_head = False
        if self.config.heads.confidence_head.weight > 0:
            call_conf_head = True
            if self.training and paddle.all(label_cropped['resolution'] == 0):
                    call_conf_head = False
        if call_conf_head:
            # prepare needed keys and rollout
            representations['rel_pos_encoding'] = self.diff_rel_pos_encoding(batch)
            representations = tree_map(lambda x: x.detach(), representations)
            if self.training:
                names_needed = ['all_atom_pos', 'all_atom_pos_mask', 
                        'all_centra_token_indice', 'all_centra_token_indice_mask',
                        'frame_mask', 'frame_ai_indice', 'frame_bi_indice', 'frame_ci_indice']
                batch.update({k: label_cropped[k] for k in names_needed})
                batch['all_atom_pos_mask'] = batch['all_atom_pos_mask'].cast(
                        batch['all_atom_pos'].dtype)
                with paddle.no_grad():
                    rollout_ret = self.diffusion_module.sample_diffusion(
                            representations, batch, step_num=10)    # dict of (B, *)
                diff_rollout_ret = diffusion.insert_diff_batch_dim(rollout_ret, 1)    # dict of (B, 1, *)
            else:
                names_needed = ['frame_mask', 'all_centra_token_indice',
                        'all_centra_token_indice_mask']
                batch.update({k: label[k] for k in names_needed})
                diff_rollout_ret = diff_ret      # dict of (B, diff_batch, *)
            # Insert diffusion dim: (B, *) -> (B, diff_batch, *)
            diff_batch_size = diff_rollout_ret['final_atom_positions'].shape[1]
            ''' disable diff_batch_dim insert
            diff_repr = diffusion.insert_diff_batch_dim(representations, diff_batch_size, 
                    special_keys=['pair', 'rel_pos_encoding'])
            diff_batch = diffusion.insert_diff_batch_dim(batch, diff_batch_size)
            diff_label = diffusion.insert_diff_batch_dim(label, diff_batch_size)
            '''                        
            diff_repr = representations
            diff_batch = batch
            diff_label = label
            # iterate over batch dim and diffusion dim
            sample_conf_ret_loflist = []
            for batch_i in range(len(batch['asym_id'])):
                sample_conf_ret_loflist.append([])
                for _ in range(diff_batch_size):
                    diff_i = 0 # repeatly run on the same slide
                    _slice_sample = lambda x: x[diff_i: diff_i + 1]
                    _slice_sample_w_dim = lambda x: x[batch_i][diff_i: diff_i + 1]
                    sample_batch = tree_map(_slice_sample, diff_batch)  # dict of (1, *)
                    sample_repr = tree_map(_slice_sample, diff_repr)
                    sample_rollout_ret = tree_map(_slice_sample_w_dim, diff_rollout_ret)
                    
                    sample_conf_ret = self.confidence_head(sample_repr, sample_batch, sample_rollout_ret, distogram_preds)
                    if compute_loss:
                        sample_label = tree_map(_slice_sample, diff_label)
                        sample_aligned_label = chain_align_np_aa.multi_chain_permutation_align(
                                sample_batch, sample_label, sample_rollout_ret)
                        sample_aligned_label['resolution'] = label['resolution']
                        loss_output = self.confidence_head.loss(
                                sample_conf_ret, sample_rollout_ret, sample_batch, sample_aligned_label)
                        sample_conf_ret.update(loss_output)
                    # drop keys to reduce memory
                    for k in ['logits_pae', 'logits_pde', 'logits_plddt', 'logits_resolved']: # , 'pae']:
                        if k in sample_conf_ret: del sample_conf_ret[k]
                    # remove batch_dim, which =1
                    sample_conf_ret = tree_map(lambda x: x.squeeze(0), sample_conf_ret)
                    sample_conf_ret_loflist[batch_i].append(sample_conf_ret)
            
            with paddle.amp.auto_cast(enable=False):
                ## NOTE: Becareful, this operation will prevent auto cast to bf16, which will cause the precision loss in ptm, iptm and etc.
                conf_ret = merge_loflist_to_dict(sample_conf_ret_loflist)    # dict of (B, diff_batch, *)
            
            if compute_loss:
                cur_loss = conf_ret['loss'].mean([1]) # (B, diff_batch) -> (B,)
                total_loss += cur_loss * self.config.heads.confidence_head.weight
            ret['confidence_head'] = conf_ret
        
        return ret, total_loss


class InputEmbedder(nn.Layer):
    """InputEmbedder

    Algorithm 2: Construct an initial 1D embedding
        +
    Algorithm 1, line 1-5

    """
    def __init__(self, channel_num, config, global_config):
        super(InputEmbedder, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        self.atom_attention_encoder = AtomAttentionEncoder(
            self.channel_num,
            self.config.atom_encoder,
            self.global_config)

        self.single_project = nn.Linear(
            self.channel_num['token_channel'] + 32 + 32 + 1,
            self.channel_num['token_channel'],
            bias_attr=False)

        self.single_to_pair_project = nn.Linear(
            self.channel_num['token_channel'] + 32 + 32 + 1,
            self.channel_num['token_pair_channel'],
            bias_attr=False)

        self.relative_position_encoding = RelativePositionEncoding(
            self.channel_num,
            self.config.relative_position_encoding,
            self.global_config)

        self.token_bond_project = nn.Linear(
            1, self.channel_num['token_pair_channel'],
            bias_attr=False)
        self.add_pocket = self.config.get('add_pocket', False)
        self.add_constrains = self.config.get('add_constrains', False)
        if self.add_pocket:
            self.pocket_info_project = nn.Linear(
                1, self.channel_num['token_pair_channel'],
                bias_attr=False)
        if self.add_constrains:
            self.constrains_token_info_project = nn.Linear(
                20, self.channel_num['token_pair_channel'],
                bias_attr=False)
            self.constrains_chain_info_project = nn.Linear(
                20, self.channel_num['token_pair_channel'],
                bias_attr=False)


        self.add_interface = self.config.get('add_interface', False)
        if self.add_interface:
            self.interface_info_project = nn.Linear(
                1, self.channel_num['token_pair_channel'],
                bias_attr=False)

        

    def forward(self, batch):
        ai, _, _, _ = self.atom_attention_encoder(batch, None, None, None)

        restype = nn.functional.one_hot(batch['restype'], 32)
        single_inputs_act = paddle.concat(
            [ai, restype, batch['profile'],
             batch['deletion_mean'].unsqueeze(axis=-1)], axis=-1)

        single_init_act = self.single_project(single_inputs_act)

        single_to_pair = self.single_to_pair_project(single_inputs_act)
        pair_init_act = single_to_pair.unsqueeze(axis=-3) + \
            single_to_pair.unsqueeze(axis=-2)
        pair_init_act += self.relative_position_encoding(batch)
        pair_init_act += self.token_bond_project(
            batch['token_bonds'].unsqueeze(axis=-1)
            )

        if self.add_pocket:
            if 'pocket_info' in batch:
                pocket_info = batch['pocket_info']
            else:
                # NOTE: The way empty pocket_info is made
                # need to align with how train data is made
                pocket_info = paddle.zeros(
                        pair_init_act.shape[:3],
                        dtype=pair_init_act.dtype)
            pair_init_act += self.pocket_info_project(
                    pocket_info.unsqueeze(axis=-1))

        if self.add_constrains:
            logging_marker = "[constrains_info]"
            if 'constrains_token_info' in batch:
                constrains_token_info = batch['constrains_token_info']
                constrains_token_pairs = batch['constrains_token_pairs']
                constrains_chain_info = batch['constrains_chain_info']
                constrains_chain_pairs = batch['constrains_chain_pairs']
                logging.info(f"{logging_marker} using constrains_token_info: " \
                             + f"{constrains_token_info.shape} {constrains_token_info.mean()}")
                logging.info(f"{logging_marker} using constrains_token_pairs: {constrains_token_pairs}")
                logging.info(f"{logging_marker} using constrains_chain_info: " \
                             + f"{constrains_chain_info.shape} {constrains_chain_info.mean()}")
                logging.info(f"{logging_marker} using constrains_chain_pairs: {constrains_chain_pairs}")
            else:
                # NOTE: The way empty pocket_info is made
                # need to align with how train data is made
                constrains_token_info = paddle.zeros(
                        pair_init_act.shape[:3] + [20],  # [batch, n_tok, n_tok, 20]
                        dtype=pair_init_act.dtype)
                constrains_token_pairs = -1 * paddle.ones((restype.shape[0], 5, 2))
                constrains_chain_info = paddle.zeros(
                        pair_init_act.shape[:3] + [20],  # [batch, n_tok, n_tok, 20]
                        dtype=pair_init_act.dtype)
                constrains_chain_pairs = -1 * paddle.ones((restype.shape[0], 5, 2))
                batch['constrains_token_info'], batch['constrains_token_pairs'] = \
                    constrains_token_info, constrains_token_pairs
                batch['constrains_chain_info'], batch['constrains_chain_pairs'] = \
                    constrains_chain_info, constrains_chain_pairs
                logging.info(f"{logging_marker} empty constrains_info: " \
                             + f"{batch['constrains_token_info'].mean()} {batch['constrains_chain_info'].mean()}")
                logging.info(f"{logging_marker} empty constrains_pairs: " \
                             + f"{batch['constrains_token_pairs']}  {batch['constrains_chain_pairs']}")
            pair_init_act += self.constrains_token_info_project(
                    constrains_token_info)
            pair_init_act += self.constrains_chain_info_project(
                    constrains_chain_info)
        if self.add_interface:
            if 'interface_info' in batch:
                interface_info_batch = batch['interface_info']
                if len(interface_info_batch.shape) > 3:
                    # If there are multiple interface_info, use the first one
                    interface_info_batch = interface_info_batch[:,:,:,0]
            else:
                # NOTE: The way empty interface_info is made
                # need to align with how train data is made
                interface_info_batch = paddle.zeros(
                        pair_init_act.shape[:3],
                        dtype=pair_init_act.dtype)
            pair_init_act += self.interface_info_project(
                    interface_info_batch.unsqueeze(axis=-1))

        return single_inputs_act, single_init_act, pair_init_act


class EmbeddingsAndPairformer(nn.Layer):
    """Template module, MSA module, Pairformer

    Algorithm 1, line 8-13

    """

    def __init__(self, channel_num, config, global_config):
        super(EmbeddingsAndPairformer, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        pair_channel = channel_num['token_pair_channel']
        self.pair_norm = nn.LayerNorm(pair_channel)
        self.pair_project = nn.Linear(
            pair_channel, pair_channel, bias_attr=False)

        single_channel = channel_num['token_channel']
        self.single_norm = nn.LayerNorm(single_channel)
        self.single_project = nn.Linear(
            single_channel, single_channel, bias_attr=False)

        self.template_embedder = TemplateEmbedder(
            channel_num,
            config.template_module,
            global_config)
        self.msa_module = MsaModule(
            channel_num,
            config.msa_module,
            global_config)

        self.pairformer_stack = nn.LayerList()
        for _ in range(self.config.pairformer.num_block):
            self.pairformer_stack.append(Pairformer(
                self.channel_num, self.config.pairformer,
                self.global_config))

    def forward(self, batch, pair_init_act, pair_act,
                single_inputs_act, single_init_act, single_act,
                masks):
        pair_act = pair_init_act + self.pair_project(self.pair_norm(pair_act))
        pair_act += self.template_embedder(batch, pair_act)
        pair_act += self.msa_module(batch, pair_act, single_inputs_act, masks)

        single_act = single_init_act + self.single_project(
            self.single_norm(single_act))

        single_act, pair_act = stack_forward_with_recompute(
            self.pairformer_stack,
            (single_act, pair_act),
            (masks,),
            is_recompute=self.training)

        return single_act, pair_act


class TemplateEmbedder(nn.Layer):
    """Template module

    Algorithm 16

    """
    def __init__(self, channel_num, config, global_config):
        super(TemplateEmbedder, self).__init__()

        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        self.pair_norm = nn.LayerNorm(self.channel_num['token_pair_channel'])
        self.pair_project = nn.Linear(
            self.channel_num['token_pair_channel'],
            self.config.num_channel,
            bias_attr=False)

        self.template_project = nn.Linear(
            # template_distogram (32), backbone_frame_mask (1)
            # template_unit_vector (3), pseudo_beta_mask (1)
            # template_restype_i & j (32 * 2)
            39 + 1 + 3 + 1 + 32 * 2,
            self.config.num_channel,
            bias_attr=False)

        self.pairformer_stack = nn.LayerList()
        for _ in range(self.config.pairformer_stack.num_block):
            self.pairformer_stack.append(Pairformer(
                {
                    # NOTE: `modules.*` use `pair_channel`,
                    # while `modules_all_atom.*` use `token_pair_channel`
                    'pair_channel': self.config.num_channel,
                    'token_pair_channel': self.config.num_channel,

                    # NOTE: placeholder, as pairformer in template embedder
                    # only use pair_act
                    'token_channel': self.channel_num['token_channel'],
                },
                self.config.pairformer_stack,
                self.global_config,
                pair_only=True))

        self.out_norm = nn.LayerNorm(self.config.num_channel)
        self.out_projection = nn.Linear(
            self.config.num_channel,
            self.channel_num['token_pair_channel'],
            bias_attr=False)

    def forward(self, batch, pair_act):
        backbone_frame_mask = batch['template_backbone_frame_mask']
        out_act = 0.
        total_num_templates = backbone_frame_mask.shape[1]
        if self.training:
            p0 = 1.0 / (total_num_templates + 1) #TODO: move default value to config
            if hasattr(self.config, 'zero_prob'):
                p0 = self.config.zero_prob
            assert 0 < p0 < 1.0
            p_sample = [p0] + [(1.0 - p0) / total_num_templates] * total_num_templates
            rand_num_temp = np.random.choice(total_num_templates + 1, 1, p=p_sample)
            num_templates = min(rand_num_temp, self.config.max_templates)
            indices = np.random.choice(total_num_templates, 
                    num_templates, replace=False)
        else:
            num_templates = min(total_num_templates, self.config.max_templates)
            indices = list(range(num_templates))

        asym_id = batch['asym_id']
        multichain_mask = asym_id.unsqueeze(axis=-1) == \
            asym_id.unsqueeze(axis=-2)
        
        if num_templates == 0:
            # no template feature, return 0 tensor
            out_act = paddle.zeros(multichain_mask.shape + [self.config.num_channel])
            out_pair_act = self.out_projection(nn.functional.relu(out_act))
            return out_pair_act
        for t in indices:
            temp_distogram = modules.dgram_from_positions(
                batch['template_pseudo_beta'][:, t],
                num_bins=39,
                min_bin=3.25,
                max_bin=50.75
            )

            backbone_frame_mask_2d = backbone_frame_mask[:, t].unsqueeze(axis=-1) * \
                backbone_frame_mask[:, t].unsqueeze(axis=-2)

            pseudo_beta_mask = batch['template_pseudo_beta_mask'][:, t]
            pseudo_beta_mask_2d = pseudo_beta_mask.unsqueeze(axis=-1) * \
                pseudo_beta_mask.unsqueeze(axis=-2)

            act = paddle.concat([
                temp_distogram,
                backbone_frame_mask_2d.unsqueeze(axis=-1),
                batch['template_unit_vector'][:, t],
                pseudo_beta_mask_2d.unsqueeze(axis=-1)], axis=-1)

            multichain_mask = paddle.cast(multichain_mask, act.dtype)
            # 使用 template_inter_mask 替换原 multichain_mask， 并作用于pairformer_stack网络
            multichain_mask = batch['template_inter_mask'][:,t]
            act *= multichain_mask.unsqueeze(-1)

            if 'template_restype' in batch:
                restype = batch['template_restype'][:, t]
            else:
                restype = batch['template_aatype'][:, t]

            restype_i = paddle.expand_as(
                restype.unsqueeze(axis=-2), backbone_frame_mask_2d)
            restype_j = paddle.expand_as(
                restype.unsqueeze(axis=-1), backbone_frame_mask_2d)

            restype_i = nn.functional.one_hot(restype_i, 32)
            restype_j = nn.functional.one_hot(restype_j, 32)
            act_t = paddle.concat([act, restype_i, restype_j], axis=-1)

            pair_act_t = self.pair_project(self.pair_norm(pair_act))
            v_t = pair_act_t + self.template_project(act_t)

            # multichain_mask 接入 template_inter_mask
            residual = stack_forward_with_recompute(
                self.pairformer_stack,
                v_t,
                ({'pair': multichain_mask},),
                is_recompute=self.training)
            v_t += residual
            out_act += self.out_norm(v_t)

        out_act /= num_templates
        out_pair_act = self.out_projection(nn.functional.relu(out_act))
        return out_pair_act


class MsaModule(nn.Layer):
    """MSA module

    Algorithm 8

    """
    def __init__(self, channel_num, config, global_config):
        super(MsaModule, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        self.msa_project = nn.Linear(
            32 + 1 + 1, self.config.msa_channel, bias_attr=False)

        # 449 = 32 (restype) + 32 (profile) + 1 (deletion_mean) + 384 (token_channel)
        self.single_project = nn.Linear(
            449, self.config.msa_channel, bias_attr=False)

        self.evoformer_stack = nn.LayerList()
        for _ in range(self.config.num_block):
            self.evoformer_stack.append(EvoformerV3(
                self.channel_num,
                self.config,
                self.global_config))

    def forward(self, batch, pair_act, single_inputs_act, masks):
        indices = paddle.randperm(batch['msa'].shape[1])
        indices = indices[:self.config.msa_depth]

        msa_mask = paddle.index_select(masks['msa'], indices, axis=1)
        msa_feat = self._create_msa_feature(batch, indices)
        msa_act = self.msa_project(msa_feat)
        msa_act += paddle.unsqueeze(
            self.single_project(single_inputs_act), axis=1)

        msa_act, pair_act = stack_forward_with_recompute(
            self.evoformer_stack,
            (msa_act, pair_act),
            ({
                'msa': msa_mask,
                'pair': masks['pair'],
            },),
            is_recompute=self.training)
        return pair_act

    def _create_msa_feature(self, batch, indices):
        msa = paddle.index_select(batch['msa'], indices, axis=1)
        has_deletion = paddle.index_select(
            batch['has_deletion'], indices, axis=1)
        deletion_value = paddle.index_select(
            batch['deletion_value'], indices, axis=1)

        msa_1hot = nn.functional.one_hot(msa, 32)
        msa_feat = [
            msa_1hot,
            has_deletion.unsqueeze(axis=-1),
            deletion_value.unsqueeze(axis=-1)
        ]
        return paddle.concat(msa_feat, axis=-1)


class Pairformer(nn.Layer):
    """Pairformer

    Algorithm 17, line 2-8

    """

    def __init__(self, channel_num, config, global_config,
                 pair_only=False):
        super(Pairformer, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config
        self.pair_only = pair_only

        use_dropout_nd = self.global_config.get('use_dropout_nd', False)

        self.triangle_multiplication_outgoing = TriangleMultiplication(
            self.channel_num,
            self.config.triangle_multiplication_outgoing,
            self.global_config,
            name='triangle_multiplication_outgoing')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_multiplication_outgoing)
        self.triangle_outgoing_dropout = nn.Dropout(
            dropout_rate, axis=dropout_axis) \
            if not use_dropout_nd else Dropout(
                    dropout_rate, axis=dropout_axis)

        self.triangle_multiplication_incoming = TriangleMultiplication(
            self.channel_num,
            self.config.triangle_multiplication_incoming,
            self.global_config,
            name='triangle_multiplication_incoming')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_multiplication_incoming)
        self.triangle_incoming_dropout = nn.Dropout(
            dropout_rate, axis=dropout_axis) \
            if not use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.triangle_attention_starting_node = TriangleAttention(
            self.channel_num,
            self.config.triangle_attention_starting_node,
            self.global_config,
            name='triangle_attention_starting_node')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_attention_starting_node)
        self.triangle_starting_dropout = nn.Dropout(
            dropout_rate, axis=dropout_axis) \
            if not use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.triangle_attention_ending_node = TriangleAttention(
            self.channel_num,
            self.config.triangle_attention_ending_node,
            self.global_config,
            name='triangle_attention_ending_node')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_attention_ending_node)
        self.triangle_ending_dropout = nn.Dropout(
            dropout_rate, axis=dropout_axis) \
            if not use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.pair_transition = TransitionV3(
            self.channel_num,
            self.config.pair_transition,
            self.global_config,
            'pair_transition')

        if not self.pair_only:
            self.single_attention_with_pair_bias = AttentionPairBias(
                self.channel_num['token_channel'],
                self.channel_num['token_channel'],
                self.channel_num['token_pair_channel'],
                self.config.single_attention_with_pair_bias.num_head,
                False)

            self.single_transition = TransitionV3(
                self.channel_num,
                self.config.single_transition,
                self.global_config,
                'single_transition')

        if self.pair_only:
            setattr(self, 'forward', self._forward_pair)
        else:
            setattr(self, 'forward', self._forward_all)

    def _forward_pair(self, pair_act, masks):
        pair_mask = masks['pair']
        residual = self.triangle_multiplication_outgoing(pair_act, pair_mask)
        pair_act += self.triangle_outgoing_dropout(residual)
        residual = self.triangle_multiplication_incoming(pair_act, pair_mask)
        pair_act += self.triangle_incoming_dropout(residual)

        residual = self.triangle_attention_starting_node(pair_act, pair_mask)
        pair_act += self.triangle_starting_dropout(residual)

        residual = self.triangle_attention_ending_node(pair_act, pair_mask)
        pair_act += self.triangle_ending_dropout(residual)

        pair_act += self.pair_transition(pair_act)

        return pair_act

    def _forward_all(self, single_act, pair_act, masks):
        pair_act = self._forward_pair(pair_act, masks)

        pair_mask = masks['pair']
        beta = paddle.zeros_like(pair_mask)
        single_act += self.single_attention_with_pair_bias(
            single_act, None, pair_act, beta)
        single_act += self.single_transition(single_act)

        return single_act, pair_act

    def _parse_dropout_params(self, module):
        dropout_rate = 0.0 if self.global_config.deterministic else \
            module.config.dropout_rate
        dropout_axis = None
        if module.config.shared_dropout:
            dropout_axis = {
                'per_row': [0, 2, 3],
                'per_column': [0, 1, 3],
            }[module.config.orientation]

        return dropout_rate, dropout_axis


class TransitionV3(nn.Layer):
    """Transition layer v3 in HelixFold3

    Algorithm 11
    """

    def __init__(self, channel_num, config, global_config,
                 transition_type):
        super(TransitionV3, self).__init__()
        assert transition_type in ['pair_transition', 'single_transition', 'msa_transition']
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config
        self.transition_type = transition_type

        # FIXME: @xueyang02, may have different channel size
        # as af3 use pairformer at different modules
        if transition_type == 'pair_transition':
            in_dim = channel_num['token_pair_channel']
        elif transition_type == 'single_transition':
            in_dim = channel_num['token_channel']
        elif transition_type == 'msa_transition':
            in_dim = channel_num['msa_channel']

        self.input_layer_norm = nn.LayerNorm(in_dim)

        nc = int(in_dim * self.config.num_intermediate_factor)
        self.proj_a = nn.Linear(
            in_dim, nc,
            bias_attr=False,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.KaimingNormal()))
        self.proj_b = nn.Linear(
            in_dim, nc,
            bias_attr=False,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.KaimingNormal()))

        if self.global_config.zero_init:
            last_init = nn.initializer.Constant(0.0)
        else:
            last_init = nn.initializer.TruncatedNormal()

        self.proj_out = nn.Linear(
            nc, in_dim,
            bias_attr=False,
            weight_attr=paddle.ParamAttr(initializer=last_init))

        self.swish = nn.Swish()

    def forward(self, x):
        x = self.input_layer_norm(x)

        def _transition_fn(x):
            a = self.proj_a(x)
            b = self.proj_b(x)
            return self.proj_out(self.swish(a) * b)

        if not self.training:
            sb_transition = subbatch(
                _transition_fn, [0], [1],
                self.global_config.subbatch_size, 1)
            x = sb_transition(x)

        else:
            x = _transition_fn(x)

        return x


class EvoformerV3(nn.Layer):
    """Modified evoformer for HelixFold-3 in MSA module

    Algorithm 8, line 6-13
    """
    def __init__(self, channel_num, config, global_config):
        super(EvoformerV3, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        use_dropout_nd = self.global_config.get('use_dropout_nd', False)

        self.msa_subbatch = self.config.get('msa_subbatch', 0)
        self.use_msa_dynamic_subbatch = self.config.get('use_msa_dynamic_subbatch', False)

        self.outer_product_mean = OuterProductMean(
            self.channel_num,
            self.config.outer_product_mean,
            self.global_config,
            False,
            name='outer_product_mean')

        self.msa_pair_weighted_averaging = MSAPairWeightedAveraging(
            self.channel_num,
            self.config.msa_pair_weighted_averaging,
            self.global_config)

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.msa_pair_weighted_averaging)
        self.msa_averaging_dropout = nn.Dropout(
            dropout_rate, axis=dropout_axis) \
            if not use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.msa_transition = TransitionV3(
            self.channel_num,
            self.config.msa_transition,
            self.global_config,
            'msa_transition')

        self.triangle_multiplication_outgoing = TriangleMultiplication(
            self.channel_num,
            self.config.triangle_multiplication_outgoing,
            self.global_config,
            name='triangle_multiplication_outgoing')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_multiplication_outgoing)
        self.triangle_outgoing_dropout = nn.Dropout(
            dropout_rate, axis=dropout_axis) \
            if not use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.triangle_multiplication_incoming = TriangleMultiplication(
            self.channel_num,
            self.config.triangle_multiplication_incoming,
            self.global_config,
            name='triangle_multiplication_incoming')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_multiplication_incoming)
        self.triangle_incoming_dropout = nn.Dropout(
            dropout_rate, axis=dropout_axis) \
            if not use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.triangle_attention_starting_node = TriangleAttention(
            self.channel_num,
            self.config.triangle_attention_starting_node,
            self.global_config,
            name='triangle_attention_starting_node')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_attention_starting_node)
        self.triangle_starting_dropout = nn.Dropout(
            dropout_rate, axis=dropout_axis) \
            if not use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.triangle_attention_ending_node = TriangleAttention(
            self.channel_num,
            self.config.triangle_attention_ending_node,
            self.global_config,
            name='triangle_attention_ending_node')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_attention_ending_node)
        self.triangle_ending_dropout = nn.Dropout(
            dropout_rate, axis=dropout_axis) \
            if not use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.pair_transition = TransitionV3(
            self.channel_num,
            self.config.pair_transition,
            self.global_config,
            'pair_transition')

    def forward(self, msa_act, pair_act, masks):
        msa_mask, pair_mask = masks['msa'], masks['pair']

        pair_act += self.outer_product_mean(msa_act, msa_mask)

        msa_depth, token_len = msa_act.shape[1:3]
        msa_subbatch_size = self._get_msa_subbatch_size(msa_depth, token_len)

        if msa_subbatch_size > 0:
            logging.info(f"using msa_subbatch with size {msa_subbatch_size}")
            # subbatch enable
            # TODO: Temporarily work around enisum bugs for large dimension tensors
            # TODO: Relationship bw len(token) and subbatch size should be experimentally determined.
            all_residual_batch = []
            for start in range(0, msa_depth, msa_subbatch_size):
                end = min(start + msa_subbatch_size, msa_depth)
                msa_act_batch = msa_act[:, start:end]
                residual_batch = self.msa_pair_weighted_averaging(
                                    msa_act_batch, pair_act, pair_mask)
                all_residual_batch.append(residual_batch)
            residual = paddle.concat(all_residual_batch, axis=1)
        else:  
            residual = self.msa_pair_weighted_averaging(
                msa_act, pair_act, pair_mask)
        msa_act += self.msa_averaging_dropout(residual)
        msa_act += self.msa_transition(msa_act)

        residual = self.triangle_multiplication_outgoing(pair_act, pair_mask)
        pair_act += self.triangle_outgoing_dropout(residual)

        residual = self.triangle_multiplication_incoming(pair_act, pair_mask)
        pair_act += pair_act + self.triangle_incoming_dropout(residual)

        residual = self.triangle_attention_starting_node(pair_act, pair_mask)
        pair_act += self.triangle_starting_dropout(residual)

        residual = self.triangle_attention_ending_node(pair_act, pair_mask)
        pair_act += self.triangle_ending_dropout(residual)

        pair_act += self.pair_transition(pair_act)

        return msa_act, pair_act

    def _get_msa_subbatch_size(self, msa_depth, token_len):
        if self.msa_subbatch != 0:
            # forcedly determined by configuration
            return self.msa_subbatch
        if not self.use_msa_dynamic_subbatch:
            # use full batch size
            return 0
        # dynamically determined by token length
        # TODO: Relationship bw len(token) and subbatch size should be experimentally determined.
        if token_len >= 5000:
            return 1024
        elif token_len >= 2000:
            return 2048
        else:
            return msa_depth

    def _parse_dropout_params(self, module):
        dropout_rate = 0.0 if self.global_config.deterministic else \
            module.config.dropout_rate
        dropout_axis = None
        if module.config.shared_dropout:
            dropout_axis = {
                'per_row': [0, 2, 3],
                'per_column': [0, 1, 3],
            }[module.config.orientation]

        return dropout_rate, dropout_axis


class MSAPairWeightedAveraging(nn.Layer):
    """MSA per-row attention biased by the pair representation.

    A modified `modules.MSARowAttentionWithPairBias` for HelixFold3

    Algorithm 10
    """
    def __init__(self, channel_num, config, global_config):
        super(MSAPairWeightedAveraging, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        msa_channel = channel_num['msa_channel']
        pair_channel = channel_num['pair_channel']

        self.query_norm = nn.LayerNorm(msa_channel)
        self.feat_2d_norm = nn.LayerNorm(pair_channel)

        self.v_proj_w = paddle.create_parameter(
            [msa_channel, config.num_head, config.num_channel], 'float32',
            default_initializer=nn.initializer.XavierUniform())
        self.bias_proj_w = paddle.create_parameter(
            [pair_channel, config.num_head], 'float32',
            default_initializer=nn.initializer.Normal(
                std=1. / np.sqrt(pair_channel)))
        self.gating_proj_w = paddle.create_parameter(
            [msa_channel, config.num_head, config.num_channel], 'float32',
            default_initializer=nn.initializer.Constant(0.0))
        self.out_proj_w = paddle.create_parameter(
            [config.num_head, config.num_channel, msa_channel], 'float32',
            default_initializer=nn.initializer.XavierUniform())

    def forward(self, msa_act, pair_act, pair_mask):
        mask_bias = 1e9 * (pair_mask - 1.)
        mask_bias = paddle.unsqueeze(mask_bias, axis=[1])

        pair_act = self.feat_2d_norm(pair_act)
        msa_act = self.query_norm(msa_act)

        v = paddle.einsum('nbqa,ahc->nbqhc', msa_act, self.v_proj_w)
        bias = paddle.einsum('nqkc,ch->nhqk', pair_act, self.bias_proj_w)
        gating = paddle.einsum('nbqa,ahc->nbqhc', msa_act, self.gating_proj_w)
        gating = nn.functional.sigmoid(gating)

        weights = nn.functional.softmax(bias + mask_bias)
        weighted_avg = paddle.einsum('nhqk,nbkhc->nbqhc', weights, v)
        out_act = gating * weighted_avg
        out_act = paddle.einsum('nbqhc,hco->nbqo', out_act, self.out_proj_w)
        return out_act


class ConfidenceHead(nn.Layer):
    """Confidence head 
    Algorithm 31
    """
    def __init__(self, channel_num, config, global_config):
        super(ConfidenceHead, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        # TODO: v_bins different from the paper
        self.v_bins = paddle.arange(0, 22, 0.5)
        token_channel = channel_num['token_channel']
        token_pair_channel = channel_num['token_pair_channel']

        self.atom_encoder = AtomAttentionEncoder(
                channel_num, self.config.atom_encoder, self.global_config)
        self.ln_s = nn.LayerNorm(token_channel * 2 + 32 + 32 + 1)
        self.lin_s_left = nn.Linear(
                token_channel * 2 + 32 + 32 + 1, token_pair_channel)
        self.lin_s_right = nn.Linear(
                token_channel * 2 + 32 + 32 + 1, token_pair_channel)
        self.lin_dij = nn.Linear(len(self.v_bins) + 1, token_pair_channel)

        self.pairformer_stack = nn.LayerList()
        for _ in range(self.config.pairformer.num_block):
            self.pairformer_stack.append(Pairformer(
                self.channel_num, self.config.pairformer,
                self.global_config))
        
        self.atom_decoder = AtomAttentionDecoder(
                channel_num, self.config.atom_decoder, self.global_config)
        self.ln_pae = nn.LayerNorm(token_pair_channel)
        self.lin_pae = nn.Linear(token_pair_channel, self.config.b_pae)
        self.ln_pde = nn.LayerNorm(token_pair_channel)
        self.lin_pde = nn.Linear(token_pair_channel, self.config.b_pde)
        self.ln_si = nn.LayerNorm(token_channel)
        self.lin_plddt = nn.Linear(token_pair_channel, self.config.b_plddt)
        self.lin_resolved = nn.Linear(token_pair_channel, 2)

    def _atom_value_to_token(self, atom_value, token_indice):
        token_value = paddle.stack([v[index] for v, index 
                in zip(atom_value, token_indice)])     # (B, N_token, 3)
        return token_value

    def _loss(self, error, stride, num_bins, logits, mask):
        error_index = paddle.clip(paddle.floor(error / stride), 
                0, num_bins - 1).cast('int64')
        loss = F.cross_entropy(
                logits, error_index, reduction='none')
        reduce_axis = list(range(1, len(mask.shape)))
        loss = (loss * mask).sum(reduce_axis) / (mask.sum(reduce_axis) + 1e-8)
        return loss

    def forward(self, representations, batch, rollout_value, distogram_res):
        """forward"""
        s_inputs = representations['single_inputs']  # (B, N_token, d1)
        si = representations['single'] # (B, N_token, d1)
        zij = representations['pair']   # (B, N_token, N_token, d2)
        xl_pred = rollout_value['final_atom_positions'] # (B, N_atom, 3)
        xl_mask = rollout_value['final_atom_mask']
        xl_pred = diffusion.CentreRandomAugmentation(xl_pred, xl_mask)
        
        ## encode
        ai, ql_skip, cl_skip, p_lm_skip = self.atom_encoder(
                feature=batch, rl=xl_pred / self.config.sigma_data, s_trunk=si, zij=zij) 
        ai = self.ln_s(paddle.concat([ai, s_inputs], -1))
        zij += self.lin_s_left(ai)[:, :, None] + self.lin_s_right(ai)[:, None]
        rep_pos = self._atom_value_to_token(xl_pred, batch['all_centra_token_indice'])
        dij = points_self_dist(rep_pos)
        zij += self.lin_dij(one_hot(dij, self.v_bins))

        ## pairformer
        masks = {
            'msa': batch['msa_mask'],
            'pair': batch['seq_mask'].unsqueeze(axis=1) * batch['seq_mask'].unsqueeze(axis=2),
        }
        si, zij = stack_forward_with_recompute(
            self.pairformer_stack,
            (si, zij),
            (masks,),
            is_recompute=self.training)

        ## decode
        logits_pae = self.lin_pae(self.ln_pae(zij))
        logits_pde = self.lin_pde(self.ln_pde(zij + zij.transpose([0, 2, 1, 3])))
        atom_token_uid = batch['ref_token2atom_idx']
        atom_mask = paddle.ones_like(atom_token_uid)
        si = self.ln_si(si)
        al = self.atom_decoder(si, ql_skip, cl_skip, p_lm_skip,
                                     atom_token_uid, atom_mask)
        logits_plddt = self.lin_plddt(al)
        logits_resolved = self.lin_resolved(al)
        ret = {
            'logits_pae': logits_pae,   # (B, N_token, N_token, b_pae)
            'logits_pde': logits_pde,   # (B, N_token, N_token, b_pde)
            'logits_plddt': logits_plddt,   # (B, N_atom, b_plddt)
            'logits_resolved': logits_resolved, # (B, N_atom, 2)
            'distogram': distogram_res,
        }
        if not self.training:
            metrics = self.get_metrics(ret, rollout_value, batch)
            ret.update(metrics)
            ret.pop('distogram') ## delete distogram for saving memory
        return ret
    
    def loss(self, value, rollout_value, batch, label):
        """
        gt_atom_pos: label['all_atom_pos'] is from permutation alignment,
            while batch['all_atom_pos'] is from label_cropped.
        """
        pred_atom_pos = rollout_value['final_atom_positions']
        atom_mask = label['all_atom_pos_mask']
        gt_atom_pos = diffusion.get_x_gt_aligned(
                label['all_atom_pos'], 
                batch['all_atom_pos'], 
                pred_atom_pos, 
                w=paddle.ones_like(atom_mask), 
                mask=atom_mask)
        ## permutation align within each ligand
        gt_atom_pos = chain_align_np_aa.ligand_permutation_align(
                batch, pred_atom_pos, gt_atom_pos, atom_mask)

        ## filter by resolution
        resolution_mask = 1.0
        if self.config.filter_by_resolution:
            # NMR & distillation have resolution = 0
            resolution = label['resolution'].squeeze(-1)
            resolution_mask = paddle.cast((resolution >= self.config.min_resolution)
                    & (resolution <= self.config.max_resolution), 'float32')    # (B,)

        ret = {}
        total_loss = 0.0

        ## atom-level plddt
        loss_plddt = self._loss_plddt(batch, value, 
                pred_atom_pos, gt_atom_pos, atom_mask)
        loss_plddt *= resolution_mask
        ret['loss_plddt'] = loss_plddt
        total_loss += loss_plddt

        ## atom-level resolved
        loss_resolved = self._loss_resolved(batch, value, atom_mask)
        loss_resolved *= resolution_mask
        ret['loss_resolved'] = loss_resolved
        total_loss += loss_resolved

        ## token-token pae
        loss_pae = self._loss_pae(batch, value, pred_atom_pos, gt_atom_pos)
        loss_pae *= resolution_mask
        ret['loss_pae'] = loss_pae
        total_loss += loss_pae

        ## token-token pde
        loss_pde = self._loss_pde(batch, value, pred_atom_pos, gt_atom_pos)
        loss_pde *= resolution_mask
        ret['loss_pde'] = loss_pde
        total_loss += loss_pde

        ret['loss'] = total_loss
        return ret
    
    def _loss_plddt(self, batch, value, pred_atom_pos, gt_atom_pos, atom_mask):
        cutoff = (batch['is_protein_aa'] * 15.0 + batch['is_dna_aa'] * 30.0 
                + batch['is_rna_aa'] * 30.0)
        lddt_score = lddt.lddt(
                predicted_points=pred_atom_pos,
                true_points=gt_atom_pos,
                true_points_mask=atom_mask[..., None],
                cutoff=cutoff,
                per_residue=True).detach()  # (B, N_atom)
        lddt_score.stop_gradient = True
        loss_plddt = self._loss(
                lddt_score, 
                1.0 / self.config.b_plddt, 
                self.config.b_plddt, 
                value['logits_plddt'], 
                atom_mask)
        return loss_plddt
    
    def _loss_resolved(self, batch, value, atom_mask):
        # TODO: atom-level feature are not padded for batch=1
        #  otherwise, please provide key `atom_exist`
        assert atom_mask.shape[0] == 1
        atom_exist = paddle.ones_like(atom_mask)
        loss_resolved = F.cross_entropy(
                value['logits_resolved'], 
                atom_mask.cast('int64'), 
                reduction='none')   # (B, N_atom)
        loss_resolved = (loss_resolved * atom_exist).sum([1]) / (
                atom_exist.sum([1]) + 1e-8)
        return loss_resolved

    def _loss_pae(self, batch, value, pred_atom_pos, gt_atom_pos):
        frame_mask = batch['frame_mask']    # (B, N_token)
        pair_mask = frame_mask[:, :, None] * frame_mask[:, None]
        ae = lddt.alignment_error(
                pred_atom_pos, 
                gt_atom_pos,
                frame_ai_indice=batch['frame_ai_indice'],
                frame_bi_indice=batch['frame_bi_indice'],
                frame_ci_indice=batch['frame_ci_indice'],
                frame_mask=frame_mask)
        ae.stop_gradient = True
        loss_pae = self._loss(
                ae, 
                self.config.stride_pae, 
                self.config.b_pae, 
                value['logits_pae'], 
                pair_mask)
        return loss_pae

    def _loss_pde(self, batch, value, pred_atom_pos, gt_atom_pos):
        pair_mask = (batch['all_centra_token_indice_mask'][:, :, None]
                * batch['all_centra_token_indice_mask'][:, None])    # (B, N_token, N_token)
        rep_pos = self._atom_value_to_token(
                pred_atom_pos, batch['all_centra_token_indice'])    # (B, N_token, 3)
        gt_rep_pos = self._atom_value_to_token(
                gt_atom_pos, batch['all_centra_token_indice'])
        dist_error = paddle.abs(points_self_dist(rep_pos) - 
                points_self_dist(gt_rep_pos))   # (B, N_token, N_token)
        dist_error.stop_gradient = True
        loss_pde = self._loss(
                dist_error, 
                self.config.stride_pde, 
                self.config.b_pde, 
                value['logits_pde'], 
                pair_mask)  # (B, N_token, N_token)
        return loss_pde

    def get_metrics(self, logit_value, structure_value, batch):
        """
        Args:
            logits_plddt: (B, N_atom, b_plddt)
            logits_pae: (B, N_token, N_token, b_pae)
        
        Returns:
            atom_plddts: (B, N_atom)
            mean_plddt: (B,)
            chain_plddt_asym_id: (B, n_chain)
            chain_plddt: (B, n_chain)

            chain_inter_pde_asym_id: (B, n_chain)
            chain_inter_pde: (B, n_chain)

            pae: (B, N_token, N_token)
            ptm: (B,)
            iptm: (B,)
            has_clash: (B,)
            ranking_confidence: (B,)
            chain_pair_asym_ids: (B, n_chain)
            chain_pair_iptm: (B, n_chain, n_chain)
            chain_pair_mask: (B, n_chain, n_chain)
        """
        B = logit_value['logits_pae'].shape[0]
        breaks_pae = paddle.linspace(0., 
                self.config.stride_pae * self.config.b_pae,
                self.config.b_pae - 1)
        breaks_pde = paddle.linspace(0., 
            self.config.stride_pde * self.config.b_pde,
            self.config.b_pde - 1)
        inputs = {
            'token2atom_idx': batch['ref_token2atom_idx'],
            'frame_mask': batch['frame_mask'],
            'asym_id': batch['asym_id'],

            'breaks_pae': paddle.tile(breaks_pae, [B, 1]),
            'breaks_pde': paddle.tile(breaks_pde, [B, 1]),
            'perm_asym_id': batch['perm_asym_id'],
            'is_polymer_chain': ((batch['is_protein_aa'] + 
                    batch['is_dna_aa'] + batch['is_rna_aa']) > 0),
            'ref_token2atom_idx': batch['ref_token2atom_idx'],
            'is_ligand': batch['is_ligand'],
            'is_ligand_aa': batch['is_ligand_aa'],
            **logit_value,
            **structure_value,
        }

        ret_list = []
        for i in range(B):
            # numpy version
            # cur_input = tree_map(lambda x: x[i].astype('float32').numpy(), inputs)
            # ret = get_all_atom_confidence_metrics(cur_input)

            # paddle version
            with paddle.amp.auto_cast(enable=False):
                ## NOTE: Becareful, this operation will prevent auto cast to bf16, which will cause the precision loss in ptm, iptm and etc.
                cur_input_pd = tree_map(lambda x: x[i].astype('float32'), inputs)
                ret = get_all_atom_confidence_metrics_pd(cur_input_pd)
            ret_list.append(ret)

        metrics = {}
        for k, v in ret_list[0].items():
            metrics[k] = paddle.to_tensor(np.stack([r[k] for r in ret_list]))
        return metrics


class InterfaceHead(nn.Layer):
    """Head to predict the probability distribution of token pairs on interface."""
    def __init__(self, channel_num, config, global_config):
        super(InterfaceHead, self).__init__()
        self.config = config
        self.channel_num = channel_num
        self.global_config = global_config

        self.fusion_layer = nn.Linear(1, self.channel_num['pair_channel'], bias_attr=False)

        self.pairformer_stack = nn.LayerList()
        for _ in range(self.config.pairformer.num_block):
            self.pairformer_stack.append(Pairformer(
                self.channel_num, self.config.pairformer,
                self.global_config))

        self.half_logits = nn.Linear(
            self.channel_num['pair_channel'], 2, name='half_logits')

        self.train_with_mix_interface = self.config.get('train_with_mix_interface', False) # whether add interface info to the pair representation
        # whether update the representation during interface generation. If true, the representation 
        # used in step i+1 is from the output of step i. Otherwise, all steps use the same representation.
        self.update_repr_during_interface_gen = self.config.get('update_repr_during_interface_gen', True)
        self.save_intermediate_interface_infos = self.config.get('save_intermediate_interface_infos', False)
        self.save_interface_info_maps = self.config.get('save_interface_info_maps', False)

        print(f'[DEBUG] InterfaceHead, save_intermediate_interface_infos: {self.save_intermediate_interface_infos}')
        print(f'[DEBUG] InterfaceHead, save_interface_info_maps: {self.save_interface_info_maps}')

    
    def _gen_interface_batch_i_in_repeats(self, representations_batch_i, interface_mask_batch_i, masks, 
                                interface_gen_repeats, interface_size, interface_seed, interface_temperature_batch_i,
                                mixed_interface_batch_i, gt_interface_batch_i, valid_len, compute_loss):

        token_num = representations_batch_i['pair'].shape[1]
        
        loss_batch_i = 0.

        gen_interface_list = []
        # gen_interface_indices = paddle.ones((interface_gen_repeats, interface_size, 2),
        #                                             dtype='int64') * (-1) # -1 means invalid index

        # store the generated interface map
        gen_interface_info_batch_i = [] 
        # store intermediate probability and interface sample history
        pred_interface_prob_in_steps_batch_i = []
        interface_info_in_steps_batch_i = []
        # store token-pairs that have been sampled in previous repeats
        sampled_interface_across_repeats = np.zeros((1, token_num, token_num), dtype='float32')

        for repeat_j in range(interface_gen_repeats):
            if compute_loss:
                remaining_gt_interface_batch_i = gt_interface_batch_i.clone()

            pred_interface_prob_in_steps_batch_i_repeat_j = []
            interface_info_in_steps_batch_i_repeat_j = [] 

            gen_interface_info_batch_i_repeat_j = paddle.zeros(
                                                    (1, token_num, token_num),
                                                    dtype=representations_batch_i['pair'].dtype)

            interface_info_obj = InterfaceInfo(token_num, probability=1.0)

            for interface_i in range(interface_size):
                # predict probability of next token
                representations_batch_i_, pred_interface_logits_batch_i = self._pred_interface_prob(
                                            representations_batch_i, gen_interface_info_batch_i_repeat_j, masks)
                if self.update_repr_during_interface_gen:
                    representations_batch_i = representations_batch_i_
                    # stop gradient backprop at representation tensors
                    for key, value in representations_batch_i.items():
                        if isinstance(value, paddle.Tensor):
                            value.stop_gradient = True

                pred_interface_prob_batch_i = nn.functional.softmax(pred_interface_logits_batch_i / interface_temperature_batch_i)[:,:,:,1]
                pred_interface_prob_batch_i = pred_interface_prob_batch_i.astype('float32').numpy()

                # mask out invalid token pairs
                pred_interface_prob_batch_i *= interface_mask_batch_i.astype('float32').numpy()
                pred_interface_prob_in_steps_batch_i_repeat_j.append(pred_interface_prob_batch_i[0])

                # mask out sampled history
                pred_interface_prob_batch_i *= (1 - sampled_interface_across_repeats)

                # NOTE: compute_loss should not be true in interferencing senarios, 
                # because when compute_loss is true, the generated interface is 
                # actually sampled from GT interface.
                if compute_loss:    
                    # compute the interface_prob prediction loss
                    if interface_i == 0 and self.train_with_mix_interface:
                        # when no sampled token pair as condition is given, use the mixed interface as label
                        interface_loss = self.loss(
                                pred_interface_logits_batch_i, mixed_interface_batch_i, interface_mask_batch_i)      
                    else:
                        interface_loss = self.loss(
                                pred_interface_logits_batch_i, remaining_gt_interface_batch_i, interface_mask_batch_i)       
                    loss_batch_i += interface_loss

                    has_nonzero = paddle.any(remaining_gt_interface_batch_i != 0)
                    if has_nonzero:
                        # Find the coordinates of positive items  
                        coords = list(np.where(remaining_gt_interface_batch_i[0].numpy()))
                        # print(f'[DEBUG]: coords: {coords}')

                        # Randomly sample one of the positive items
                        seed = int(interface_seed + valid_len + interface_i) if interface_seed is not None else None
                        idx = np.random.default_rng(seed).choice(
                            range(len(coords[0])), 1, replace=False)
                        token_id_1, token_id_2 = coords[0][idx], coords[1][idx]
                        
                        # Add the sampled item to gen_interface_info
                        gen_interface_info_batch_i_repeat_j = gen_interface_info_batch_i_repeat_j.clone()
                        gen_interface_info_batch_i_repeat_j[0, token_id_1, token_id_2] = 1.
                        gen_interface_info_batch_i_repeat_j[0, token_id_2, token_id_1] = 1.
                        gen_interface_info_batch_i_repeat_j.stop_gradient = True

                        # Add the sampled indices to the interface indices tensor
                        interface_info_obj.add_node([token_id_1, token_id_2], match_annotation=1)
                        # update interface probability
                        token_pair_prob = pred_interface_prob_batch_i[0, token_id_1, token_id_2]
                        interface_info_obj.set_probability(interface_info_obj.probability * token_pair_prob)
                        # gen_interface_indices[repeat_j, interface_i, 0] = token_id_1
                        # gen_interface_indices[repeat_j, interface_i, 1] = token_id_2
                        
                        # Remove the sampled item from remaining_gt_interface
                        remaining_gt_interface_batch_i = remaining_gt_interface_batch_i.clone()
                        remaining_gt_interface_batch_i[0, token_id_1, token_id_2] = 0.
                        remaining_gt_interface_batch_i[0, token_id_2, token_id_1] = 0.
                        remaining_gt_interface_batch_i.stop_gradient = True

                    else:
                        print("[DEBUG] Interface sampling: No token pairs left to sample.")

                else:       
                    pred_interface_prob_batch_i_flat = pred_interface_prob_batch_i.flatten()
                    if pred_interface_prob_batch_i_flat.sum() <= 0:
                        # no probibal new token pair in the interface
                        continue

                    # sample a new token pair from predicted probability distribution
                    seed = int(interface_seed + valid_len + interface_i) if interface_seed is not None else None
                    idx, prob = random_choice_top_k_index_with_prob(pred_interface_prob_batch_i_flat, k=valid_len, seed=seed)

                    # pred_interface_prob_batch_i_flat /= pred_interface_prob_batch_i_flat.sum()
                    # idx = np.random.default_rng(seed).choice(
                    #             range(len(pred_interface_prob_batch_i_flat)), p=pred_interface_prob_batch_i_flat)

                    token_num = pred_interface_prob_batch_i.shape[1]
                    token_id_1 = int(idx // token_num)
                    token_id_2 = int(idx % token_num)

                    # add the sampled token-pair to gen_interface_info
                    gen_interface_info_batch_i_repeat_j[0, token_id_1, token_id_2] = 1.
                    gen_interface_info_batch_i_repeat_j[0, token_id_2, token_id_1] = 1.

                    # Add the sampled indices to the interface indices tensor
                    if gt_interface_batch_i is not None:
                        match_annotation = gt_interface_batch_i[0, token_id_1, token_id_2].item()
                    else:
                        match_annotation = -1
                    interface_info_obj.add_node([token_id_1, token_id_2], match_annotation=match_annotation)
                    # update interface probability
                    token_pair_prob = pred_interface_prob_batch_i[0, token_id_1, token_id_2]
                    interface_info_obj.set_probability(interface_info_obj.probability * token_pair_prob) 
                    # gen_interface_indices[repeat_j, interface_i, 0] = token_id_1
                    # gen_interface_indices[repeat_j, interface_i, 1] = token_id_2

                    # add to the sampled token pairs accross all repeats
                    masking_neighborhood_width = 1  # masking token pairs in neighborhood
                    sampled_interface_across_repeats[0, 
                                max(token_id_1-masking_neighborhood_width,0): token_id_1+masking_neighborhood_width+1, 
                                max(token_id_2-masking_neighborhood_width,0): token_id_2+masking_neighborhood_width+1] = 1.
                    sampled_interface_across_repeats[0, 
                                max(token_id_2-masking_neighborhood_width,0): token_id_2+masking_neighborhood_width+1, 
                                max(token_id_1-masking_neighborhood_width,0): token_id_1+masking_neighborhood_width+1] = 1.

                interface_info_in_steps_batch_i_repeat_j.append(gen_interface_info_batch_i_repeat_j[0].astype('float32').numpy())

            gen_interface_list.append(interface_info_obj)
            
            if self.save_interface_info_maps:
                gen_interface_info_batch_i.append(gen_interface_info_batch_i_repeat_j)
            if self.save_intermediate_interface_infos:
                pred_interface_prob_in_steps_batch_i.append(pred_interface_prob_in_steps_batch_i_repeat_j)
                interface_info_in_steps_batch_i.append(interface_info_in_steps_batch_i_repeat_j)

        return gen_interface_list, gen_interface_info_batch_i, \
                pred_interface_prob_in_steps_batch_i, interface_info_in_steps_batch_i, \
                loss_batch_i


    def _gen_interface_batch_i_in_beam(self, representations_batch_i, interface_mask_batch_i, masks, 
                                interface_size, beam_size,
                                mixed_interface_batch_i, gt_interface_batch_i, valid_len):

        token_num = representations_batch_i['pair'].shape[1]
        
        loss_batch_i = 0.

        root_interface = InterfaceInfo(n_token=token_num, probability=1.0)

        parents_dict = {root_interface.get_interface_key(): root_interface, } 
        peers_dict = dict()

        pred_interface_prob_in_step0 = []

        for interface_i in range(interface_size):
            for parent_key, parent in parents_dict.items():
                
                # get the parent interface map
                parent_interface_info_array = parent.interface_map()[None].astype(np.float32)
                parent_interface_info = paddle.to_tensor(parent_interface_info_array, 
                                                          dtype=representations_batch_i['pair'].dtype)
                
                # predict probability of next token
                representations_batch_i_, pred_interface_logits_batch_i = self._pred_interface_prob(
                                            representations_batch_i, parent_interface_info, masks)

                pred_interface_prob_batch_i = nn.functional.softmax(pred_interface_logits_batch_i)[:,:,:,1]
                pred_interface_prob_batch_i = pred_interface_prob_batch_i.astype('float32').numpy()

                # mask out invalid token pairs
                pred_interface_prob_batch_i *= interface_mask_batch_i.astype('float32').numpy()

                if interface_i == 0:
                    # save the predicted interface probability in step 0
                    pred_interface_prob_in_step0.append(pred_interface_prob_batch_i)

                # mask out parent node indices
                pred_interface_prob_batch_i *= (1.0 - parent_interface_info_array)
                    
                pred_interface_prob_batch_i_flat = pred_interface_prob_batch_i.flatten()
                if pred_interface_prob_batch_i_flat.sum() <= 0:
                    # no probibal new token pair in the interface
                    continue

                # find token pairs with highest probabilities
                top_indices, top_scores = find_top_k_upper_triangle_indices(pred_interface_prob_batch_i[0], beam_size)
                for index_, score_ in zip(top_indices, top_scores):
                    child = deepcopy(parent)
                    if gt_interface_batch_i is not None:
                        match_annotation = gt_interface_batch_i[0, index_[0], index_[1]].item()
                    else:
                        match_annotation = -1

                    child.add_node([index_[0], index_[1]], match_annotation)
                    child.set_probability(parent.probability * score_)
                    child_key = child.get_interface_key()
                    if child_key in peers_dict:
                        peers_dict[child_key].set_probability(peers_dict[child_key].probability + child.probability)
                    else:
                        peers_dict[child_key] = child

            # assign peers to parents
            parents_dict = peers_dict
            peers_dict = dict()

        # sort the interfaces by scores
        interface_items = sorted(parents_dict.items(), key=lambda item: item[1].probability)[::-1]
        gen_interfaces = [item[1] for item in interface_items]

        return gen_interfaces, pred_interface_prob_in_step0

    def forward(self, representations, batch, label_cropped, label, 
                        compute_loss):
        
        batch_size = representations['pair'].shape[0]
        token_num = representations['pair'].shape[1]

        intermediate_interface_infos = {'pred_interface_prob_in_steps': [], 'interface_info_in_steps': []}
        loss_batch = [] 

        interface_size_batch = batch.get('interface_gen_size', [0] * batch_size)
        interface_seed_batch = batch.get('interface_gen_seed', [None] * batch_size)
        interface_temperature_batch = batch.get('interface_gen_temperature', [1.0] * batch_size)
        interface_gen_repeats_batch = batch.get('interface_gen_repeats', [1] * batch_size)
        interface_sampling_method_batch = batch.get('interface_sampling_method', ['in_repeats'] * batch_size)
        interface_sampling_beam_batch = batch.get('interface_sampling_beam', [10] * batch_size)

        if isinstance(interface_size_batch, paddle.Tensor):
            interface_size_batch = interface_size_batch.numpy()

        if isinstance(interface_seed_batch, paddle.Tensor):
            interface_seed_batch = interface_seed_batch.numpy()

        if isinstance(interface_gen_repeats_batch, paddle.Tensor):
            interface_gen_repeats_batch = interface_gen_repeats_batch.numpy()

        if isinstance(interface_sampling_beam_batch, paddle.Tensor):
            interface_sampling_beam_batch = interface_sampling_beam_batch.numpy()

        gen_interfaces_batch_list = []

        for batch_i in range(batch_size):

            uniq_asym_ids = np.unique(batch['asym_id'][batch_i].numpy())
            if len(uniq_asym_ids) < 2:
                # not a multimer
                gen_interfaces_batch_list.append([])
                continue

            if isinstance(interface_size_batch, paddle.Tensor):
                interface_size = interface_size_batch[batch_i].item()
            else:
                interface_size = interface_size_batch[batch_i]
            
            if isinstance(interface_seed_batch, paddle.Tensor):
                interface_seed = interface_seed_batch[batch_i].item()
            else:
                interface_seed = interface_seed_batch[batch_i]

            interface_gen_repeats = interface_gen_repeats_batch[batch_i]
            interface_sampling_method = interface_sampling_method_batch[batch_i]
            interface_sampling_beam =interface_sampling_beam_batch[batch_i]

            representations_batch_i = {k: v[batch_i:batch_i+1] for k, v in representations.items()}
    
            if 'interface_info_full' in batch:
                # get ground-truth interface
                gt_interface_batch_i = batch['interface_info_full'][batch_i:batch_i+1]
                gt_interface_batch_i.stop_gradient = True

            else:
                gt_interface_batch_i = None

            if 'interface_info_mix' in batch:
                # get mixed interface from homo chains
                mixed_interface_batch_i = batch['interface_info_mix'][batch_i:batch_i+1]
                mixed_interface_batch_i.stop_gradient = True
            else:
                mixed_interface_batch_i = None

            masks = {
                'msa': batch['msa_mask'][batch_i:batch_i+1],
                'pair': batch['seq_mask'][batch_i:batch_i+1].unsqueeze(axis=1) * \
                        batch['seq_mask'][batch_i:batch_i+1].unsqueeze(axis=2),
            }

            interface_mask_batch_i = batch['interface_mask'][batch_i:batch_i+1]
            interface_mask_batch_i = interface_mask_batch_i * masks['pair']

            interface_temperature_batch_i = interface_temperature_batch[batch_i]

            valid_len = label['all_centra_token_indice_mask'][batch_i].numpy().sum()

            if interface_sampling_method == 'in_repeats' or compute_loss:

                gen_interfaces, gen_interface_info_batch_i_list, \
                pred_interface_prob_in_steps_batch_i, interface_info_in_steps_batch_i, \
                loss_batch_i = \
                    self._gen_interface_batch_i_in_repeats(representations_batch_i, interface_mask_batch_i, masks, 
                                    interface_gen_repeats, interface_size, interface_seed, interface_temperature_batch_i,
                                    mixed_interface_batch_i, gt_interface_batch_i, valid_len, compute_loss)

                gen_interfaces_batch_list.append(gen_interfaces)

                if self.save_interface_info_maps:
                    ## TODO:  @liuyang, need to check this code?? 
                    gen_interface_info_batch_list.append(paddle.concat(gen_interface_info_batch_i_list, axis=0))
                if self.save_intermediate_interface_infos:
                    intermediate_interface_infos['pred_interface_prob_in_steps'].append(pred_interface_prob_in_steps_batch_i)
                    intermediate_interface_infos['interface_info_in_steps'].append(interface_info_in_steps_batch_i)

                if isinstance(loss_batch_i, paddle.Tensor):
                    loss_batch.append(loss_batch_i)

            elif interface_sampling_method == 'in_beam':
                gen_interfaces, pred_interface_prob_in_step0 = self._gen_interface_batch_i_in_beam(
                                representations_batch_i, interface_mask_batch_i, masks, 
                                interface_size, interface_sampling_beam,
                                mixed_interface_batch_i, gt_interface_batch_i, valid_len)
                
                gen_interfaces_batch_list.append(gen_interfaces)

                if self.save_intermediate_interface_infos:
                    intermediate_interface_infos['pred_interface_prob_in_steps'].append(pred_interface_prob_in_step0)

            else:
                raise ValueError(f'Unsupported interface sampling_method: {interface_sampling_method}')

        ret = {
            'gen_interfaces': gen_interfaces_batch_list,
            # 'gen_interface_indices': gen_interface_indices_batch_list
        }

        if self.save_intermediate_interface_infos:
            ret['intermediate_interface_infos'] = intermediate_interface_infos

        if compute_loss and len(loss_batch) > 0:
            loss = paddle.concat(loss_batch, axis=0)
            ret['loss'] = loss

        return ret


    def _pred_interface_prob(self, representations, sampled_interface_info, masks):

        single_act = representations['single']
        pair_act = representations['pair']

        # fusion sampled interface info into pair representation
        pair_act = pair_act + self.fusion_layer(sampled_interface_info.unsqueeze(axis=-1))

        single_act, pair_act = stack_forward_with_recompute(
            self.pairformer_stack,
            (single_act, pair_act),
            (masks,),
            is_recompute=self.training)

        half_logits = self.half_logits(pair_act)
        logits = half_logits + paddle.transpose(half_logits, perm=[0, 2, 1, 3])

        representations['single'] = single_act
        representations['pair'] = pair_act

        return representations, logits

    def build_interface_info_from_index(self, n_token, interface_info_indices):

        k = interface_info_indices.shape[0] # interface_size(k), indices(2)
        interface_info = paddle.zeros((n_token, n_token), dtype='float32')
        for i in range(k):
            row, col = interface_info_indices[i]
            if row >= 0 and col >= 0:
                interface_info[row, col] = 1.0
                interface_info[col, row] = 1.0

        return interface_info
    
    def loss(self, logits, label, mask=None):

        if mask is None:
            mask = paddle.ones_like(label)
        
        reduce_axis = list(range(1, len(label.shape)))
        loss = F.cross_entropy(
                logits, label.cast('int64'), reduction='none')
        return (loss * mask).sum(reduce_axis) / (mask.sum(reduce_axis) + 1e-8)
    

def one_hot(value, v_bins):
    """
    Args:
        value: (*)
        v_bins: (M)
    Returns:
        (*, M)
    """
    num_bins = v_bins.shape[0] + 1
    value = value[..., None]
    counts = paddle.sum(value > v_bins, -1)
    emb = F.one_hot(counts, num_classes=num_bins)
    return emb


def points_self_dist(x):
    """x: (..., n, 3)"""
    dist2 = paddle.sum((x.unsqueeze(-2) - x.unsqueeze(-3)) ** 2, -1)
    return paddle.sqrt(dist2 + 1e-8) # (..., n, n)


def stack_forward_with_recompute(stack, iter_args, common_args, is_recompute=True):
    ret = iter_args
    for iteration in stack:
        if isinstance(ret, paddle.Tensor):
            ret = (ret,)
        ret = recompute_wrapper(
            iteration,
            *ret,
            *common_args,
            is_recompute=is_recompute)

    return ret


def merge_list_to_dict(sample_list):
    """
    merge sample list into a batch dict
    """
    batch = {}
    for k in sample_list[0]:
        if isinstance(sample_list[0][k], paddle.Tensor):
            batch[k] = paddle.stack([sr[k] for sr in sample_list])
        else:
            batch[k] = [sr[k] for sr in sample_list]
    return batch


def merge_loflist_to_dict(sample_loflist):
    """
    merge sample list of list into a batch dict
    """
    batch = {}
    for k in sample_loflist[0][0]:
        if isinstance(sample_loflist[0][0][k], paddle.Tensor):
            batch[k] = paddle.stack([paddle.stack([s[k] for s in slist])
                    for slist in sample_loflist])
        else:
            batch[k] = [[s[k] for s in slist] 
                    for slist in sample_loflist]
    return batch
