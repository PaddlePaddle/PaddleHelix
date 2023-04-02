"""
paddle_predict
"""

import os
import time
import sys

import argparse
import json
import codecs
import numpy as np
import random
import paddle
import paddle.nn.functional as F
from pahelix.model_zoo.protein_sequence_model import ProteinEncoderModel, ProteinModel, ProteinCriterion
from pahelix.utils.protein_tools import ProteinTokenizer

from data_gen import create_dataloader, pad_to_max_seq_len
from metrics import get_metric
from paddle.distributed import fleet

def show_results(examples, pred, task):
    """
    Show the results.
    """
    if task == 'classification':
        pred_label = pred.argmax(axis=-1)
        for (i, example) in enumerate(examples):
            print('%s: %.3f' % (example, pred_label[i]))
    elif task == 'seq_classification':
        pred_label = pred.argmax(axis=-1)
        for i, example in enumerate(examples):
            print('example: %s ' % example)
            cur_pred_label = pred_label[i][1:len(example) + 1]
            print('pred: %s' % ' '.join([str(val) for val in cur_pred_label]))
    elif task == 'regression':
        for (i, example) in enumerate(examples):
            print('%s: %.3f' % (example, pred[i]))

def main(args):
    """
    main function
    """

    model_config = json.load(open(args.model_config, 'r'))
    if args.use_cuda:
        paddle.set_device("gpu")
    else:
        paddle.set_device("cpu")

    encoder_model = ProteinEncoderModel(model_config, name='protein')
    model = ProteinModel(encoder_model, model_config)
    model.load_dict(paddle.load(args.predict_model))

    tokenizer = ProteinTokenizer()
    examples = []
    with codecs.open(args.predict_data) as f_read:
        for line in f_read:
            if len(line.strip()) == 0:
                continue
            examples.append(line.strip())

    example_ids = [tokenizer.gen_token_ids(example) for example in examples]
    max_seq_len = max([len(example_id) for example_id in example_ids])
    pos = [list(range(1, len(example_id) + 1)) for example_id in example_ids]
    pad_to_max_seq_len(example_ids, max_seq_len)
    pad_to_max_seq_len(pos, max_seq_len)

    texts = paddle.to_tensor(example_ids)
    pos = paddle.to_tensor(pos)
    pred = model(texts, pos)
    pred = pred.numpy()

    show_results(examples, pred, model_config['task'])
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_data", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--predict_model", type=str, default='')
    parser.add_argument("--use_cuda", action='store_true', default=False)
    args = parser.parse_args()

    main(args)
