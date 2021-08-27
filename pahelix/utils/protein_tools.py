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
Tools for protein features.
"""

from collections import OrderedDict
from enum import Enum

class ProteinTokenizer(object):
    """
    Protein Tokenizer.
    """
    padding_token = '<pad>'
    mask_token = '<mask>'
    start_token = class_token = '<cls>'
    end_token = seperate_token = '<sep>'
    unknown_token = '<unk>'

    padding_token_id = 0
    mask_token_id = 1
    start_token_id = class_token_id = 2
    end_token_id = seperate_token_id = 3
    unknown_token_id = 4

    special_token_ids = [padding_token_id, mask_token_id, start_token_id, end_token_id, unknown_token_id]

    vocab = OrderedDict([
        (padding_token, 0),
        (mask_token, 1),
        (class_token, 2),
        (seperate_token, 3),
        (unknown_token, 4),
        ('A', 5),
        ('B', 6),
        ('C', 7),
        ('D', 8),
        ('E', 9),
        ('F', 10),
        ('G', 11),
        ('H', 12),
        ('I', 13),
        ('K', 14),
        ('L', 15),
        ('M', 16),
        ('N', 17),
        ('O', 18),
        ('P', 19),
        ('Q', 20),
        ('R', 21),
        ('S', 22),
        ('T', 23),
        ('U', 24),
        ('V', 25),
        ('W', 26),
        ('X', 27),
        ('Y', 28),
        ('Z', 29)])

    def tokenize(self, sequence):
        """
        Split the sequence into token list.

        Args:
            sequence: The sequence to be tokenized.

        Returns:
            tokens: The token lists.
        """
        return [x for x in sequence]

    def convert_token_to_id(self, token):
        """ 
        Converts a token to an id.

        Args:
            token: Token.

        Returns:
            id: The id of the input token.
        """
        if token not in self.vocab:
            return ProteinTokenizer.unknown_token_id
        else:
            return ProteinTokenizer.vocab[token]

    def convert_tokens_to_ids(self, tokens):
        """
        Convert multiple tokens to ids.
        
        Args:
            tokens: The list of tokens.

        Returns:
            ids: The id list of the input tokens.
        """
        return [self.convert_token_to_id(token) for token in tokens]

    def gen_token_ids(self, sequence):
        """
        Generate the list of token ids according the input sequence.

        Args:
            sequence: Sequence to be tokenized.

        Returns:
            token_ids: The list of token ids.
        """
        tokens = []
        tokens.append(ProteinTokenizer.start_token)
        tokens.extend(self.tokenize(sequence))
        tokens.append(ProteinTokenizer.end_token)
        token_ids = self.convert_tokens_to_ids(tokens)
        return token_ids


