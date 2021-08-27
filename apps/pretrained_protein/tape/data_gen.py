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

import math
import numpy as np
import paddle
from pahelix.utils.language_model_tools import apply_bert_mask
from pahelix.utils.protein_tools import ProteinTokenizer
from paddle.io import IterableDataset, DataLoader, get_worker_info

boundaries = np.array([100, 200, 300, 400, 600, 900, 1000, 1200, 1300, 2000, 3000])
batch_sizes = np.array([160, 160, 128, 96, 64, 32, 24, 16, 16, 8, 1, 1])

class PfamDataset(IterableDataset):
    """
    Pfam Dataset
    """
    def __init__(self, data_dir, pad_token_id):
        self.data_dir = data_dir
        self.pad_token_id = pad_token_id

        # set self.start & self.end if DataLoader.num_workers != 0
        # self.start = 0
        # self.end = 100000

    def __iter__(self):
        buckets = [[] for i in range(len(batch_sizes))]

        filenames = os.listdir(self.data_dir)
        for filename in filenames:
            file_path = os.path.join(self.data_dir, filename)
            data = np.load(file_path)
            token_ids = data["token_ids"].tolist()
            lengths = data['lengths']
            offsets = np.zeros(lengths.shape[0], dtype=int)
            for i in range(1, len(lengths)):
                offsets[i] = offsets[i - 1] + lengths[i - 1]

            for i in range(lengths.size):
                # bucket mechanism
                cur_len = lengths[i]
                offset = offsets[i]
                text = token_ids[offset:offset + cur_len]
                pos = np.arange(1, lengths[i]+1).tolist()
                cur_bucket = np.digitize(cur_len, boundaries)
                buckets[cur_bucket].append((text, cur_len, pos))
                if len(buckets[cur_bucket]) == batch_sizes[cur_bucket]:
                    bucket = self._do_pad_mask(buckets[cur_bucket])
                    yield bucket
                    buckets[cur_bucket] = []

        for bucket in buckets:
            if bucket:
                bucket = self._do_pad_mask(bucket)
                yield bucket

    def _do_pad_mask(self, bucket):
        """
        do padding & bert masking for pfam data
        """
        seq_lens = [entry[1] for entry in bucket]
        batch_max_seq_len = max(seq_lens)
        texts = [entry[0] for entry in bucket]
        pad_to_max_seq_len(texts, batch_max_seq_len, self.pad_token_id)

        batch_size = len(texts)
        texts = np.array(texts)
        seq_lens = np.array(seq_lens)
        indices = np.tile(np.arange(batch_max_seq_len)[None, :], (batch_size, 1))
        pad_mask = indices < seq_lens[:, None]

        masked_token_ids, labels = apply_bert_mask(texts, pad_mask, ProteinTokenizer)
        masked_token_ids = masked_token_ids.tolist()
        labels = labels.tolist()

        mask_bucket = []
        for i in range(batch_size):
            mask_bucket.append((masked_token_ids[i], bucket[i][1], bucket[i][2], labels[i]))

        return mask_bucket


class SequenceDataset(IterableDataset):
    """
    Dataset for sequence classification/regression tasks.
    """
    def __init__(self, data_dir, label_name):
        self.data_dir = data_dir
        self.label_name = label_name

    def __iter__(self):
        buckets = [[] for i in range(len(batch_sizes))]
        filenames = os.listdir(self.data_dir)
        for filename in filenames:
            file_path = os.path.join(self.data_dir, filename)
            data = np.load(file_path)
            token_ids = data['token_ids'].tolist()
            labels = data[self.label_name].tolist()
            lengths = data['lengths']

            offsets = np.zeros(lengths.shape[0], dtype=int)
            for i in range(1, len(lengths)):
                offsets[i] = offsets[i - 1] + lengths[i - 1]

            for i in range(lengths.size):
                # bucket mechanism
                cur_len = lengths[i]
                offset = offsets[i]
                text = token_ids[offset:offset + cur_len]
                label = labels[offset:offset + cur_len]
                pos = np.arange(1, lengths[i] + 1).tolist()
                cur_bucket = np.digitize(cur_len, boundaries)
                buckets[cur_bucket].append((text, cur_len, pos, label))
                if len(buckets[cur_bucket]) == batch_sizes[cur_bucket]:
                    yield buckets[cur_bucket]
                    buckets[cur_bucket] = []

        for bucket in buckets:
            if bucket:
                yield bucket


class NormalDataset(IterableDataset):
    """
    Dataset for classification/regression tasks.
    """
    def __init__(self, data_dir, label_name):
        self.data_dir = data_dir
        self.label_name = label_name

    def __iter__(self):
        buckets = [[] for i in range(len(batch_sizes))]
        filenames = os.listdir(self.data_dir)
        for filename in filenames:
            file_path = os.path.join(self.data_dir, filename)
            data = np.load(file_path)
            token_ids = data['token_ids'].tolist()
            labels = data[self.label_name].tolist()
            lengths = data['lengths']

            offset = 0
            for i in range(lengths.size):
                # bucket mechanism
                text = token_ids[offset:offset + lengths[i]]
                pos = np.arange(1, lengths[i]+1).tolist()
                label = labels[i:i + 1][0]
                cur_bucket = np.digitize(lengths[i], boundaries)
                buckets[cur_bucket].append((text, lengths[i], pos, label))
                if len(buckets[cur_bucket]) == batch_sizes[cur_bucket]:
                    yield buckets[cur_bucket]
                    buckets[cur_bucket] = []
                offset += lengths[i]

        for bucket in buckets:
            if bucket:
                yield bucket


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


def generate_batch(batch, model_config, pad_token_id=0, pad_label=True):
    """
    Generates a batch whose text will be padded to the max sequence length in the batch.

    Args:
        batch(obj:`List[Example]`) : One batch, which contains texts, labels and the true sequence lengths.
        pad_token_id(obj:`int`, optinal, defaults to 0) : The pad token index.

    Returns:
        batch(:obj:`Tuple[list]`): The batch data which contains texts, seq_lens and labels.
    """
    batch = batch[0]
    seq_lens = [entry[1] for entry in batch]
    batch_max_seq_len = max(seq_lens)
    texts = [entry[0] for entry in batch]
    pos = [entry[2] for entry in batch]
    labels = [entry[3] for entry in batch]
    pad_to_max_seq_len(texts, batch_max_seq_len, pad_token_id)
    pad_to_max_seq_len(pos, batch_max_seq_len, pad_token_id)
    if pad_label:
        pad_to_max_seq_len(labels, batch_max_seq_len, pad_token_id=-1)

    texts = np.array(texts, dtype="int")
    pos = np.array(pos)
    labels = np.array(labels)

    task = model_config['task']
    if task in ['pretrain', 'seq_classification', 'classification']:
        labels = labels.astype(np.int64)
    else:
        labels = labels.astype(np.float32)

    return texts, pos, labels


def worker_init_fn(worker_id):
    """
    split dataset for workers in DataLoader
    """
    worker_info = get_worker_info()

    dataset = worker_info.dataset
    start = dataset.start
    end = dataset.end
    num_per_worker = int(
        math.ceil((end - start) / float(worker_info.num_workers)))

    worker_id = worker_info.id
    dataset.start = start + worker_id * num_per_worker
    dataset.end = min(dataset.start + num_per_worker, end)


def create_dataloader(data_dir, model_config):
    """
    DataLoader for dataset
    """
    task = model_config['task']
    pad_token_id = ProteinTokenizer.padding_token_id
    pad_label = True
    if task == 'pretrain':
        dataset = PfamDataset(data_dir, pad_token_id=pad_token_id)
    elif task == 'seq_classification':
        label_name = model_config.get('label_name', 'labels')
        dataset = SequenceDataset(data_dir, label_name)
    elif task in ['regression', 'classification']:
        label_name = model_config.get('label_name', 'labels')
        dataset = NormalDataset(data_dir, label_name)
        pad_label=False
    else:
        raise NameError('Task %s is unsupport.' % task)

    dataloader = paddle.io.DataLoader(dataset,
                                      num_workers=0,
                                      return_list=True,
                                      drop_last=True,
                                      collate_fn=lambda batch: generate_batch(
                                          batch, model_config, pad_token_id=pad_token_id, pad_label=pad_label),
                                      worker_init_fn=worker_init_fn)
    return dataloader
