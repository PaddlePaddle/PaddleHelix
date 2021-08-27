"""
paddle_train
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

from data_gen import create_dataloader
from metrics import get_metric
from paddle.distributed import fleet

@paddle.no_grad()
def eval(model, valid_loader, criterion, metric):
    """
    eval function
    """
    model.eval()
    metric.clear()

    loss_all = np.array([], dtype=np.float32)
    for i, (text, pos, label) in enumerate(valid_loader, start=1):
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
    if args.use_cuda:
        paddle.set_device("gpu")
    else:
        paddle.set_device("cpu")

    if args.is_distributed:
        strategy = fleet.DistributedStrategy()
        fleet.init(is_collective=args.use_cuda, strategy=strategy)

    train_loader = create_dataloader(
        data_dir=args.train_data,
        model_config=model_config)

    valid_loader = create_dataloader(
        data_dir=args.valid_data,
        model_config=model_config)

    encoder_model = ProteinEncoderModel(model_config, name='protein')
    model = ProteinModel(encoder_model, model_config)
    if args.is_distributed:
        model = fleet.distributed_model(model)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]

    grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=1e-4,
        epsilon=1e-06,
        weight_decay=0.01,
        parameters=model.parameters(),
        apply_decay_param_fun=lambda x: x in decay_params)

    if args.is_distributed:
        optimizer = fleet.distributed_optimizer(optimizer)
    criterion = ProteinCriterion(model_config)
    metric = get_metric(model_config['task'])

    if args.init_model:
        print("load init_model")
        # for hot_start
        if args.hot_start == 'hot_start':
            model.load_dict(paddle.load(args.init_model))
        # for pre_train
        else:
            encoder_model.load_dict(paddle.load(args.init_model))

    train_sum_loss = 0
    valid_min_loss = 10000
    steps_per_epoch = 20
    cur_step = 0
    while True:
        model.train()
        for (text, pos, label) in train_loader:
            # print("text: ", text)
            cur_step += 1
            pred = model(text, pos)
            label = label.reshape([-1, 1])
            pred = pred.reshape([-1, pred.shape[-1]])
            loss = criterion.cal_loss(pred, label)

            print("loss: ", loss)
            train_sum_loss += loss.numpy()
            loss.backward()
            optimizer.minimize(loss)
            model.clear_gradients()

            pred = pred.numpy()
            label = label.numpy()
            loss = loss.numpy()
            metric.update(pred, label, loss)
            if cur_step % 10 == 0:
                print('step %d, avg loss %.5f' % (cur_step, train_sum_loss / 10))
                metric.show()
                train_sum_loss = 0
                metric.clear()

            # save best_model
            if cur_step % steps_per_epoch == 0:
                print("eval begin_time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                valid_cur_loss = eval(model, valid_loader, criterion, metric)
                print("valid_cur_loss: ", valid_cur_loss)
                print("eval end_time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                if valid_cur_loss < valid_min_loss:
                    print("%s Save best model step_%d." % \
                            (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), cur_step))
                    paddle.save(encoder_model.state_dict(), 'models/epoch_best_encoder.pdparams')
                    paddle.save(model.state_dict(), 'models/epoch_best.pdparams')
                    valid_min_loss = valid_cur_loss

                    os.system("cp -rf models/epoch_best.pdparams models/step_%d.pdparams" % (cur_step))
                    os.system("cp -rf models/epoch_best_encoder.pdparams models/step_%d_encoder.pdparams" % (cur_step))
                model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--valid_data", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--init_model", type=str, default='')
    parser.add_argument("--hot_start", type=str, default='hot_start')
    parser.add_argument("--use_cuda", action='store_true', default=False)
    parser.add_argument("--is_distributed", action='store_true', default=False)
    args = parser.parse_args()

    main(args)
