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
Train sequence-based models for protein.
"""
import os
import time

import argparse
import json
import numpy as np
import paddle
import paddle.nn.functional as F
from protein_sequence_model_dynamic import LstmSeqClassificationModel,\
    TransformerSeqClassificationModel
from pahelix.utils.protein_tools import ProteinTokenizer

from data_gen import create_dataloader, SecondaryStructureDataset
from metrics import ClassificationMetric


@paddle.no_grad()
def eval(model, valid_dataloader, criterion, metric):
    """
    Given a dataset, it evals model and computes the metric accuarcy.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        valid_dataloader(obj:`paddle.io.DataLoader`): The dev dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.

    """
    model.eval()
    metric.clear()

    loss_all = np.array([], dtype=np.float32)
    for texts, seq_lens, labels in valid_dataloader:
        labels = labels.reshape([-1, 1])
        mask = labels != -1
        logits = model(texts, seq_lens)
        logits = logits.reshape([-1, logits.shape[-1]])
        # mask, remove the effect of 'PAD'
        mask = paddle.cast(mask, dtype='float32')
        inf_tensor = paddle.full(shape=mask.shape,
                                 dtype='float32',
                                 fill_value=-1. * 1e12)
        logits = paddle.multiply(logits, mask) + paddle.multiply(
            inf_tensor, (1 - mask))
        probs = F.softmax(logits, axis=1)
        loss = criterion(probs, labels)
        loss_all = np.append(loss_all, loss.numpy())

        probs = probs.numpy()
        labels = labels.numpy()
        metric.update(probs, labels)
    metric.show()
    metric.clear()
    loss_avg = np.mean(loss_all)
    return loss_avg


def main(args):
    """The main function 

    Args:
        args (args): configs

    Raises:
        ValueError: if the args is invalid.
    """
    model_config = json.load(open(args.model_config, 'r'))
    paddle.set_device("gpu" if args.use_cuda else "cpu")
    rank = 0
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
        # the current process id
        rank = paddle.distributed.get_rank()

    train_dataset = SecondaryStructureDataset(base_path=args.train_data,
                                              mode='train',
                                              num_classes=3)
    train_loader = create_dataloader(
        train_dataset,
        mode='train',
        batch_size=args.batch_size,
        pad_token_id=ProteinTokenizer.padding_token_id)
    test_dataset = SecondaryStructureDataset(base_path=args.train_data,
                                             mode='test',
                                             num_classes=3)
    test_loader = create_dataloader(
        test_dataset,
        mode='test',
        batch_size=args.batch_size,
        pad_token_id=ProteinTokenizer.padding_token_id)

    if model_config["model_type"] == "transformer":
        model = TransformerSeqClassificationModel(
            vocab_size=len(ProteinTokenizer.vocab),
            num_class=model_config['class_num'],
            emb_dim=model_config['hidden_size'])
    elif model_config["model_type"] == "lstm":
        model = LstmSeqClassificationModel(vocab_size=len(
            ProteinTokenizer.vocab),
            num_class=model_config['class_num'],
            emb_dim=model_config['hidden_size'])
    else:
        raise ValueError("Not avaliable {}".format(model_config["model_type"]))

    if os.path.exists(args.init_model):
        param_state_dict = paddle.load(args.init_model)
        model.set_dict(param_state_dict)
        print("Loaded model parameters from %s" % args.init_model)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    if args.warmup_steps > 0:
        lr_scheduler = paddle.optimizer.lr.NoamDecay(
            1 / (args.warmup_steps * (args.lr ** 2)), args.warmup_steps)
    else:
        lr_scheduler = args.lr

    max_grad_norm = 0.1
    grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=max_grad_norm)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=model_config['weight_decay'],
        grad_clip=grad_clip)

    criterion = paddle.nn.CrossEntropyLoss(ignore_index=-1)
    metric = ClassificationMetric()

    for epoch in range(args.epoch):
        loss_sum = 0
        model.train()
        start_time = time.time()
        for i, (texts, seq_lens, labels) in enumerate(train_loader, start=1):
            labels = labels.reshape([-1, 1])
            mask = labels != -1
            logits = model(texts, seq_lens)
            logits = logits.reshape([-1, logits.shape[-1]])
            # mask, remove the effect of 'PAD'
            mask = paddle.cast(mask, dtype='float32')
            inf_tensor = paddle.full(shape=mask.shape,
                                     dtype='float32',
                                     fill_value=-1. * 1e12)
            logits = paddle.multiply(logits, mask) + paddle.multiply(
                inf_tensor, (1 - mask))
            probs = F.softmax(logits, axis=1)
            loss = criterion(probs, labels)
            loss_sum += loss.numpy()
            loss.backward()
            optimizer.step()
            optimizer.clear_gradients()
            probs = probs.numpy()
            labels = labels.numpy()
            metric.update(probs, labels)
            if i % 10 == 0:
                print('epoch %d, step %d, avg loss %.5f' %
                      (epoch, i, loss_sum / 10))
                metric.show()
                loss_sum = 0
                metric.clear()

        if rank == 0:
            print('Test:')
            avg_loss = eval(model, test_loader, criterion, metric)
            print("Average loss: %.5f" % avg_loss)

            print("Save model epoch%d." % epoch)
            param_path = os.path.join(args.model_dir, 'epoch%d' % epoch,
                                      'saved_params.pdparams')
            opt_path = os.path.join(args.model_dir, 'epoch%d' % epoch,
                                    'saved_opt.pdopt')
            paddle.save(model.state_dict(), param_path)
            paddle.save(optimizer.state_dict(), opt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cuda", action='store_true', default=True)
    parser.add_argument('--model_dir', default='./models')
    parser.add_argument('--init_model', default='')
    parser.add_argument('--train_data', default='./data')
    parser.add_argument('--test_data', default='./test_data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--reg', type=float, default=1e-4)
    parser.add_argument('--thread_num', type=int,
                        default=8, help='thread for cpu')
    parser.add_argument('--warmup_steps', type=int, default=-1)
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument(
        '--distributed', dest='is_distributed', action='store_true')

    args = parser.parse_args()
    main(args)
