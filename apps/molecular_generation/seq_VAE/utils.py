#!/usr/bin/python3                                                                                                
#-*-coding:utf-8-*- 
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
utils
"""

from paddle.io import Dataset
import paddle
from paddle.optimizer.lr import LRScheduler
import math
import pdb
import numpy as np

def load_zinc_dataset(filename):
    """
    To load the data files.
    """
    with open(filename, "r") as f:
        data = f.read().splitlines()
    return data


class SpecialTokens(object):
    """
    define the begain, end, pad, unknown character.
    """
    bos = '<bos>'
    eos = '<eos>'
    pad = '<pad>'
    unk = '<unk>'

    
class CharVocab(object):
    """
    The vocabulary.
    """
    @classmethod
    def from_data(cls, data, *args, **kwargs):
        """
        get the data
        """
        chars = set()
        for string in data:
            chars.update(string)

        return cls(chars, *args, **kwargs)

    def __init__(self, chars, ss=None):
        if ss is None:
            ss=SpecialTokens

        if (ss.bos in chars) or (ss.eos in chars) or \
                (ss.pad in chars) or (ss.unk in chars):
            raise ValueError('SpecialTokens in chars')

        all_syms = sorted(list(chars)) + [ss.bos, ss.eos, ss.pad, ss.unk]

        self.ss = ss
        self.c2i = {c: i for i, c in enumerate(all_syms)}
        self.i2c = {i: c for i, c in enumerate(all_syms)}

    def __len__(self):
        # the length of string dictionary
        return len(self.c2i)

    @property
    def bos(self):
        """
        begain of sentence
        """
        return self.c2i[self.ss.bos]

    @property
    def eos(self):
        """
        end of sentence
        """
        return self.c2i[self.ss.eos]

    @property
    def pad(self):
        """
        padding
        """
        return self.c2i[self.ss.pad]

    @property
    def unk(self):
        """
        unknown
        """
        return self.c2i[self.ss.unk]

    def char2id(self, char):
        """
        convert character to id
        """
        if char not in self.c2i:
            return self.unk

        return self.c2i[char]

    def id2char(self, id):
        """
        convert id to character
        """
        if id not in self.i2c:
            return self.ss.unk

        return self.i2c[id]

    def string2ids(self, string, add_bos=False, add_eos=False):
        """
        convert string to id
        """

        ids = [self.char2id(c) for c in string]

        if add_bos:
            ids = [self.bos] + ids
        if add_eos:
            ids = ids + [self.eos]

        return ids

    def ids2string(self, ids, rem_bos=True, rem_eos=True):
        """
        convert id to string
        """
        if len(ids) == 0:
            return ''
        if rem_bos and ids[0] == self.bos:
            ids = ids[1:]
        if rem_eos and ids[-1] == self.eos:
            ids = ids[:-1]

        string = ''.join([self.id2char(id) for id in ids])

        return string
    

class OneHotVocab(CharVocab):
    """
    The one hot vocabulary.
    """
    def __init__(self, *args, **kwargs):
        super(OneHotVocab, self).__init__(*args, **kwargs)
        #self.vectors = paddle.eye(len(self.c2i))
        self.vectors = np.eye(len(self.c2i), dtype='float32')

            
class StringDataset(Dataset):
    """
    create paddle dataset.
    """
    def __init__(self, vocab, data, max_length):
        """
        Creates a convenient Dataset with SMILES tokinization

        Arguments:
            vocab: CharVocab instance for tokenization
            data (list): SMILES strings for the dataset
        """
        self.vocab = vocab
        self.tokens = [vocab.string2ids(s) for s in data]
        self.data = data
        self.bos = vocab.bos
        self.eos = vocab.eos
        
        self.lengths = [len(s) + 2 for s in self.tokens]
        self.max_sequence = max_length

    def __len__(self):
        """
        Computes a number of objects in the dataset
        """
        return len(self.tokens)

    def __getitem__(self, index):
        """
        Prepares paddle tensors with a given SMILES.

        Arguments:
            index (int): index of SMILES in the original dataset

        Returns:
            A tuple (with_bos, with_eos, smiles), where
            * with_bos is a paddle.int64 tensor of SMILES tokens with
                BOS (beginning of a sentence) token
            * with_eos is a paddle.int64long tensor of SMILES tokens with
                EOS (end of a sentence) token
            * smiles is an original SMILES from the dataset
        """
        tokens = self.tokens[index]        
        return_data = paddle.to_tensor([self.bos] + tokens + [self.eos], dtype='int64')

        padded_return_data = paddle.concat([return_data, \
            paddle.to_tensor([self.vocab.pad] * (self.max_sequence - len(return_data)))])
                

        return padded_return_data, self.lengths[index]
    
    
class KLAnnealer(object):
    """
    KL annealing.
    """
    def __init__(self, n_epoch, config):
        self.i_start = config.kl_start
        self.w_start = config.kl_w_start
        self.w_max = config.kl_w_end
        self.n_epoch = n_epoch

        self.inc = (self.w_max - self.w_start) / (self.n_epoch - self.i_start)

    def __call__(self, i):
        k = (i - self.i_start) if i >= self.i_start else 0
        return self.w_start + k * self.inc

        