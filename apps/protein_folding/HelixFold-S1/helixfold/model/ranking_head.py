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

import paddle
import paddle.nn as nn
import numpy as np

"""Ranking Head implemented for ranking model"""
class RankingHead(nn.Layer):
    def __init__(self, channel_num, config, global_config, name='ranking_head'):
        super(RankingHead, self).__init__()
        self.config = config
        self.channel_num = channel_num
        self.global_config = global_config
        self.pae_value_length = 1
        self.logits = nn.Linear(self.pae_value_length, self.config.num_bins, name='logits')

    def _softmax_cross_entropy(self, logits, labels):
        """Computes softmax cross entropy given logits and one-hot class labels."""
        loss = -paddle.sum(labels * paddle.nn.functional.log_softmax(logits), axis=-1)
        return loss

    def _margin_ranking_loss(self, output1, output2, label, margin=0.00001):
        return paddle.max(paddle.Tensor([margin]), -label * (output1 - output2))

    def _sigmoid_cross_entropy(self, logits, labels):
        """Computes sigmoid cross entropy given logits and multiple class labels."""
        log_p = paddle.nn.functional.log_sigmoid(logits)
        # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter is more numerically stable
        log_not_p = paddle.nn.functional.log_sigmoid(-logits)
        loss = -labels * log_p - (1. - labels) * log_not_p
        return loss

    def forward(self, representation, batch):
        protein_chain_asym_id = 1
        pos_peptide_chain_asym_id = 2
        neg_peptide_chain_asym_id = 3

        protein_mask = paddle.where(batch['asym_id'][0]==protein_chain_asym_id, 1, 0)
        pos_peptide_mask = paddle.where(batch['asym_id'][0]==pos_peptide_chain_asym_id, 1, 0)
        neg_peptide_mask = paddle.where(batch['asym_id'][0]==neg_peptide_chain_asym_id, 1, 0)

        logits = representation['logits'][0]
        breaks = representation['breaks']
        aligned_confidence_probs = nn.Softmax(axis=-1)(logits)
        dist = (breaks[1]-breaks[0])
        bin_centers = breaks + dist / 2
        bin_centers = paddle.concat([bin_centers, bin_centers[-1]+dist], axis=0)
        paes = paddle.sum(aligned_confidence_probs * bin_centers, axis=-1)

        pae1, pae2 = None, None
        # TODO
        if paddle.any(pos_peptide_mask == 1):
            paes_pos_prot = paddle.sum((paes * pos_peptide_mask[..., None]) * protein_mask[..., None, :], keepdim=True)
            paes_prot_pos = paddle.sum((paes * protein_mask[..., None]) * pos_peptide_mask[..., None, :], keepdim=True)
            pae1 = 0.1 * (paes_pos_prot + paes_prot_pos)[0] / \
                (2 * paddle.sum(pos_peptide_mask) * paddle.sum(protein_mask))

        if paddle.any(neg_peptide_mask == 1):
            paes_neg_prot = paddle.sum((paes * neg_peptide_mask[..., None]) * protein_mask[..., None, :], keepdim=True)
            paes_prot_neg = paddle.sum((paes * protein_mask[..., None]) * neg_peptide_mask[..., None, :], keepdim=True)
            pae2 = 0.1 * (paes_neg_prot + paes_prot_neg)[0] / \
                 (2 * paddle.sum(neg_peptide_mask) * paddle.sum(protein_mask))

        binder_logits1 = self.logits(pae1) if pae1 is not None else None
        binder_logits2 = self.logits(pae2) if pae2 is not None else None

        # label=1 indicates the first peptide is a positive sample whereas the second one is a negative
        return {'binder_logits1': binder_logits1, 'binder_logits2': binder_logits2}

    def loss(self, value) -> dict:
        binder_logits1 = value['binder_logits1'] if 'binder_logits1' in value else None
        binder_logits2 = value['binder_logits2'] if 'binder_logits2' in value else None
        
        import ipdb; ipdb.set_trace()
        if binder_logits1 is None:
            label = paddle.Tensor([0])
            loss = self._sigmoid_cross_entropy(binder_logits2, label)
        elif binder_logits2 is None:
            label = paddle.Tensor([1])
            loss = self._sigmoid_cross_entropy(binder_logits1, label)
        else:
            label = paddle.Tensor([1])
            loss = self._margin_ranking_loss(binder_logits1, binder_logits2, label)
        return {'loss': loss}
