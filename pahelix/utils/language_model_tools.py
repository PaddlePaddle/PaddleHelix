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
Tools for language models.
"""

from copy import copy
import numpy as np
import random

def apply_bert_mask(token_ids, tokenizer):
    """
    Apply BERT mask to the token_ids.

    Args:
        token_ids: The list of token ids.

    Returns:
        masked_token_ids: The list of masked token ids.
        labels: The labels for traininig BERT.
    """
    vocab_size = len(tokenizer.vocab)
    masked_token_ids = copy(token_ids)
    special_token_id_n = len(tokenizer.special_token_ids)

    # 80% chance to change to mask token,
    # 10% chance to change to random token,
    # 10% chance to keep current token
    prob = np.random.random(token_ids.size)
    mask_token_ids = np.ones(token_ids.size) * tokenizer.mask_token_id
    random_token_ids = np.random.randint(
            low=special_token_id_n, high=vocab_size, size=token_ids.size)
    replace_token_ids1 = np.where(prob < 0.8, mask_token_ids, random_token_ids)
    replace_token_ids2 = np.where(prob < 0.9, replace_token_ids1, token_ids)

    # 15% change to replace the tokens
    prob = np.random.random(token_ids.size)
    masked_token_ids = np.where(
            np.all([token_ids >= special_token_id_n, prob < 0.15], axis=0),
            replace_token_ids2, token_ids)
    labels = np.where(
            np.all([token_ids >= special_token_id_n, prob < 0.15], axis=0),
            token_ids, -1)

    return masked_token_ids, labels
