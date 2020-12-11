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
Multiple protein datasets.
"""

import numpy as np
from pahelix.utils.tokenizers import Tokenizer, VocabType

class Pfam(object):
    """
    Class for pfam dataset.
    """
    def __init__(self, vocab_type=VocabType.PROTEIN):
        self.tokenizer = Tokenizer(vocab_type=vocab_type)
        self.clear()

    def gen_sequence_data(self, data):
        """
        Genearte sequence data.
        """
        amino_acids = data['amino_acids']
        token_ids = self.tokenizer.gen_token_ids(amino_acids)
        return token_ids

    def append(self, data):
        """
        Append data.
        """
        token_ids = self.gen_sequence_data(data)
        self.token_ids.extend(token_ids)
        self.lengths.append(len(token_ids))

    def clear(self):
        """
        Clear data.
        """
        self.token_ids = []
        self.lengths = []

    def save_npz(self, filename):
        """
        Save data to npz format file.
        """
        np.savez('%s' % filename, 
                token_ids=np.array(self.token_ids, dtype='int8'),
                lengths=np.array(self.lengths, dtype='int64'))


class SecondStructure(object):
    """
    Class for second structure dataset.
    """
    def __init__(self, vocab_type=VocabType.PROTEIN):
        self.tokenizer = Tokenizer(vocab_type=vocab_type)
        self.clear()

    def gen_sequence_data(self, data):
        """
        Genearte sequence data.
        """
        amino_acids = data['amino_acids']
        token_ids = self.tokenizer.gen_token_ids(amino_acids)
        labels3 = [0] + data['ss3'] + [0]
        labels8 = [0] + data['ss8'] + [0]
        return token_ids, labels3, labels8

    def append(self, data):
        """
        Append data.
        """
        token_ids, labels3, labels8 = self.gen_sequence_data(data)
        self.token_ids.extend(token_ids)
        self.labels3.extend(labels3)
        self.labels8.extend(labels8)
        self.lengths.append(len(token_ids))

    def clear(self):
        """
        Clear data.
        """
        self.token_ids = []
        self.labels3 = []
        self.labels8 = []
        self.lengths = []

    def save_npz(self, filename):
        """
        Save data to npz format file.
        """
        np.savez('%s' % filename, 
                token_ids=np.array(self.token_ids, dtype='int8'),
                labels3=np.array(self.labels3, dtype='int8'),
                labels8=np.array(self.labels8, dtype='int8'),
                lengths=np.array(self.lengths, dtype='int64'))
        

class RemoteHomology(object):
    """
    Class for remote homology dataset.
    """
    def __init__(self, vocab_type=VocabType.PROTEIN):
        self.tokenizer = Tokenizer(vocab_type=vocab_type)
        self.clear()

    def gen_sequence_data(self, data):
        """
        Genearte sequence data.
        """
        amino_acids = data['amino_acids']
        token_ids = self.tokenizer.gen_token_ids(amino_acids)
        label = data['fold_label']
        return token_ids, label

    def append(self, data):
        """
        Append data.
        """
        token_ids, labels = self.gen_sequence_data(data)
        self.token_ids.extend(token_ids)
        self.labels.extend(labels)
        self.lengths.append(len(token_ids))

    def clear(self):
        """
        Clear data.
        """
        self.token_ids = []
        self.labels = []
        self.lengths = []

    def save_npz(self, filename):
        """
        Save data to npz format file.
        """
        np.savez('%s' % filename, 
                token_ids=np.array(self.token_ids, dtype='int8'),
                labels=np.array(self.labels, dtype='int8'),
                lengths=np.array(self.lengths, dtype='int64'))


class Fluorescence(object):
    """
    Class for fluorescene dataset.
    """
    def __init__(self, vocab_type=VocabType.PROTEIN):
        self.tokenizer = Tokenizer(vocab_type=vocab_type)
        self.clear()

    def gen_sequence_data(self, data):
        """
        Genearte sequence data.
        """
        amino_acids = data['amino_acids']
        label = data['log_fluorescence']
        token_ids = self.tokenizer.gen_token_ids(amino_acids)
        return token_ids, label

    def append(self, data):
        """
        Append data.
        """
        token_ids, labels = self.gen_sequence_data(data)
        self.token_ids.extend(token_ids)
        self.labels.extend(labels)
        self.lengths.append(len(token_ids))

    def clear(self):
        """
        Clear data.
        """
        self.token_ids = []
        self.labels = []
        self.lengths = []

    def save_npz(self, filename):
        """
        Save data to npz format file.
        """
        np.savez('%s' % filename, 
                token_ids=np.array(self.token_ids, dtype='int8'),
                labels=np.array(self.labels, dtype='int8'),
                lengths=np.array(self.lengths, dtype='int64'))


class Stability(object):
    """
    Class for stability dataset.
    """
    def __init__(self, vocab_type=VocabType.PROTEIN):
        self.tokenizer = Tokenizer(vocab_type=vocab_type)
        self.clear()

    def gen_sequence_data(self, data):
        """
        Genearte sequence data.
        """
        amino_acids = data['amino_acids']
        label = data['stability_score']
        token_ids = self.tokenizer.gen_token_ids(amino_acids)
        return token_ids, label

    def append(self, data):
        """
        Append data.
        """
        token_ids, labels = self.gen_sequence_data(data)
        self.token_ids.extend(token_ids)
        self.labels.extend(labels)
        self.lengths.append(len(token_ids))

    def clear(self):
        """
        Clear data.
        """
        self.token_ids = []
        self.labels = []
        self.lengths = []

    def save_npz(self, filename):
        """
        Save data to npz format file.
        """
        np.savez('%s' % filename, 
                token_ids=np.array(self.token_ids, dtype='int8'),
                labels=np.array(self.labels, dtype='int8'),
                lengths=np.array(self.lengths, dtype='int64'))

