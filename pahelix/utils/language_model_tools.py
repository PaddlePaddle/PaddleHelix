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
| Tools for language models.
"""

from copy import copy
import numpy as np
import random

def apply_bert_mask(inputs, pad_mask, tokenizer):
    """
    Apply BERT mask to the token_ids.

    Args:
        token_ids: The list of token ids.

    Returns:
        masked_token_ids: The list of masked token ids.
        labels: The labels for traininig BERT.
    """
    vocab_size = len(tokenizer.vocab)
    bert_mask = np.random.uniform(size=inputs.shape) < 0.15
    bert_mask &= pad_mask

    masked_inputs = inputs * ~bert_mask
    random_uniform = np.random.uniform(size=inputs.shape)
    token_bert_mask = random_uniform < 0.8
    random_bert_mask = random_uniform > 0.9
    true_bert_mask = ~token_bert_mask & ~random_bert_mask

    token_bert_mask = token_bert_mask & bert_mask
    random_bert_mask = random_bert_mask & bert_mask
    true_bert_mask = true_bert_mask & bert_mask

    masked_inputs += tokenizer.mask_token_id * token_bert_mask

    masked_inputs += np.random.randint(0, vocab_size, size=(inputs.shape)) * random_bert_mask
    masked_inputs += inputs * true_bert_mask

    labels = np.where(bert_mask, inputs, -1)

    return masked_inputs, labels