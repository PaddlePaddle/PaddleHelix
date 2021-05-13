"""
paddle_eval
"""

import os
import time
import sys

import argparse
import json
import codecs
import numpy as np
import paddle
import paddle.nn.functional as F
from others.protein_sequence_model_dynamic import ProteinEncoderModel, ProteinModel, ProteinCriterion
from others.protein_tools import ProteinTokenizer

from others.data_gen import create_dataloader
from others.metrics import get_metric
from paddle.distributed import fleet

@paddle.no_grad()
def eval(model, eval_loader, criterion, metric):
    """
    eval function
    """
    model.eval()
    metric.clear()

    loss_all = np.array([], dtype=np.float32)
    for i, (text, pos, label) in enumerate(eval_loader, start=1):
        pred = model(text, pos)
        label = label.reshape([-1, 1])
        pred = pred.reshape([-1, pred.shape[-1]])

        loss = criterion.cal_loss(pred, label)
        loss_all = np.append(loss_all, loss.numpy())

        pred = pred.numpy()
        label = label.numpy()
        loss = loss.numpy()
        metric.update(pred, label, loss)

    print("eval_metric: ")
    metric.show()
    print("eval_metric finished!")
    metric.clear()
    loss_avg = np.mean(loss_all)
    return loss_avg


def main(args):
    """
    main function
    """
    model_config = json.load(open(args.model_config, 'r'))
    paddle.set_device("gpu")
    strategy = fleet.DistributedStrategy()
    fleet.init(is_collective=True, strategy=strategy)

    eval_loader = create_dataloader(
        data_dir=args.eval_data,
        model_config=model_config)

    encoder_model = ProteinEncoderModel(model_config, name='protein')
    model = ProteinModel(encoder_model, model_config)
    model = fleet.distributed_model(model)
    model.load_dict(paddle.load(args.eval_model))

    criterion = ProteinCriterion(model_config)
    metric = get_metric(model_config['task'])
    eval_cur_loss = eval(model, eval_loader, criterion, metric)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_data", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--eval_model", type=str, default='')
    args = parser.parse_args()

    main(args)
