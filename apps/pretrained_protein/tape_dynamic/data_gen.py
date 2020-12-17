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
DataLoader generator for the sequence-based pretrain models for protein.
"""

import os
import math
import random

import numpy as np
import paddle


class SecondaryStructureDataset(paddle.io.Dataset):
    """SecondaryStructure Dataset
    """
    def __init__(self, base_path, mode='train', num_classes=3):
        """Init function of dataset

        Args:
            base_path (str): the data path.
            mode (str, optional): 'train', 'test' or 'dev'. Defaults to 'train'.
            num_classes (int, optional): label class. Defaults to 3.

        Raises:
            RuntimeError: [description]
        """
        self.mode = mode
        assert num_classes in [
            3, 8
        ], "Number of class should be 3 or 8, but got %d" % num_classes
        self.num_classes = num_classes

        if self.mode == 'train':
            data_dir = os.path.join(base_path, 'train')
        elif self.mode in ['test', 'dev']:
            data_dir = os.path.join(base_path, 'valid')
        else:
            raise RuntimeError(
                "Unknown mode %s, it should be train, dev or test." % mode)

        self.examples = self._read_file(data_dir)

    def _read_file(self, data_dir):
        """Read file from disk

        Args:
            data_dir (str): data dir.

        Returns:
            list: list of data
        """
        filenames = os.listdir(data_dir)
        examples = []
        for filename in filenames:
            file_path = os.path.join(data_dir, filename)
            data = np.load(file_path)
            token_ids = data['token_ids']
            token_ids = token_ids.tolist()
            if self.num_classes == 3:
                labels = data['labels3']
            else:
                labels = data['labels8']
            labels = labels.tolist()
            lengths = data['lengths']
            lengths = lengths.tolist()

            offset = 0
            size = len(lengths)
            for i in range(size):
                # one sample whose length, tokens and label
                length = lengths[i]
                tokens = token_ids[offset:offset +
                                   length]  # .reshape(length, 1)
                label = labels[offset:offset + length]  # .reshape(length, 1)
                examples.append((tokens, length, label))

                offset += length

        if self.mode == 'train':
            random.shuffle(examples)
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def pad_to_max_seq_len(texts, max_seq_len, pad_token_id=0):
    """
    Padded the texts to the max sequence length if the length of text is lower than it.
    Unless it truncates the text.

    Args:
        texts(obj:`list`): Texts which contrains a sequence of word ids.
        max_seq_len(obj:`int`): Max sequence length.
        pad_token_id(obj:`int`, optinal, defaults to 0) : The pad token index.
    """
    for index, text in enumerate(texts):
        seq_len = len(text)
        if seq_len < max_seq_len:
            padded_tokens = [
                pad_token_id for _ in range(max_seq_len - seq_len)
            ]
            new_text = text + padded_tokens
            texts[index] = new_text
        elif seq_len > max_seq_len:
            new_text = text[:max_seq_len]
            texts[index] = new_text


def generate_batch(batch, pad_token_id=0, return_label=True):
    """
    Generates a batch whose text will be padded to the max sequence length in the batch.

    Args:
        batch(obj:`List[Example]`) : One batch, which contains texts, labels and the true sequence lengths.
        pad_token_id(obj:`int`, optinal, defaults to 0) : The pad token index.

    Returns:
        batch(:obj:`Tuple[list]`): The batch data which contains texts, seq_lens and labels.
    """
    seq_lens = [entry[1] for entry in batch]

    batch_max_seq_len = max(seq_lens)
    texts = [entry[0] for entry in batch]
    labels = [entry[2] for entry in batch]
    pad_to_max_seq_len(texts, batch_max_seq_len, pad_token_id)
    # -1 is the pad label token id
    pad_to_max_seq_len(labels, batch_max_seq_len, pad_token_id=-1)
    texts = np.array(texts)
    seq_lens = np.array(seq_lens)
    labels = np.array(labels)  # .reshape(-1, batch_max_seq_len, 1)
    return texts, seq_lens, labels


def create_dataloader(dataset, mode='train', batch_size=1, pad_token_id=0):
    """Create dataloader

    Args:
        dataset (obj:`paddle.io.dataset`): an instance of  `paddle.io.dataset`.
        mode (str, optional): 'train', 'test' or 'dev'. Defaults to 'train'.
        batch_size (int, optional): batch size. Defaults to 1.
        pad_token_id (int, optional): the id of PAD token. Defaults to 0.

    Returns:
        Dataloader: an instance of `paddle.io.Dataloader`.
    """
    shuffle = True if mode == 'train' else False
    if mode == 'train':
        sampler = paddle.io.DistributedBatchSampler(dataset,
                                                    batch_size=batch_size,
                                                    shuffle=shuffle)
    else:
        sampler = paddle.io.BatchSampler(dataset,
                                         shuffle=shuffle,
                                         batch_size=batch_size)
    dataloader = paddle.io.DataLoader(dataset,
                                      batch_sampler=sampler,
                                      return_list=True,
                                      collate_fn=lambda batch: generate_batch(
                                          batch, pad_token_id=pad_token_id))
    return dataloader
