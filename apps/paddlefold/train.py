#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

"""Training scripts."""

import os
from os.path import exists, join, dirname
import time
import sys
import argparse
import numpy as np
import random
import json
import ml_collections
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

import paddle
from paddle import distributed as dist
from tensorboardX import SummaryWriter

from utils.utils import get_model_parameter_size, add_to_data_writer, upload_to_hadoop, csv_print
from utils.metric import ResultsCollect
from utils.model import RunModel
from utils.dataset import LoopedBatchSampler, AF2Dataset, AF2TestDataset, AF2DistillDataset
from utils.param_fuse import get_fused_params
from utils.distributed import grad_sync, param_sync
from utils.misc import TrainLogger
from alphafold_paddle.model import config


MAX_EVAL_SIZE = int(os.environ.get('MAX_EVAL_SIZE', 1400))
print(f'[ENV] MAX_EVAL_SIZE:{MAX_EVAL_SIZE}')


def time_me():
    paddle.device.cuda.synchronize()
    return time.time()

def worker_init_fn(worker_id):
    """ set seed in subproces for dataloader when num_worker > 0"""
    np.random.seed(args.seed + worker_id)
    random.seed(args.seed + worker_id)

def init_seed(seed):
    """ set seed for reproduct results"""
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_optimizer(opt_config, model):
    if opt_config.grad_clip == 0:
        grad_clip = None
    else:
        grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=float(opt_config.grad_clip))
    if 'decay' in opt_config:
        second_scheduler = paddle.optimizer.lr.StepDecay(
                learning_rate=opt_config.lr, 
                step_size=opt_config.decay.step_size,
                gamma=opt_config.decay.gamma)
    else:
        second_scheduler = opt_config.lr
    lr_scheduler = paddle.optimizer.lr.LinearWarmup(
            learning_rate=second_scheduler, 
            warmup_steps=opt_config.warmup_steps, 
            start_lr=opt_config.lr * 0.01, 
            end_lr=opt_config.lr, 
            verbose=False)

    fused_parameters = get_fused_params(model.parameters())
    optimizer = paddle.optimizer.Adam(
            learning_rate=lr_scheduler, 
            epsilon=1e-06,
            grad_clip=grad_clip,
            parameters=[{
                'params': fused_parameters,
            }])
    return optimizer, lr_scheduler


def get_bf16_op_list():
    """tbd."""

    black_list = {"softmax", "reduce_sum"}
    white_list = {"concat", "elementwise_add", "elementwise_div", "elementwise_mul", "elementwise_sub", "fill_any_like", "fill_constant", "gather", "gaussian_random",
        "layer_norm", "log_softmax", "matmul_v2", "p_norm", "py_layer", "relu", "scale", "sigmoid", "slice", "softplus", "split", "sqrt", "square", "stack",
        "sum", "transpose2"}
    return black_list, white_list


def add_dyna_features(train_config, model_config, batch, step):
    """add `num_iter_recycling` and `use_clamped_fape`"""
    random_key = 32
    shape = batch['feat']['aatype'].shape[:2]

    num_iter_recycling = np.random.default_rng(random_key + step).integers(
            model_config.model.num_recycle + 1)
    batch['feat']['num_iter_recycling'] = paddle.full(shape, num_iter_recycling)
    print(f'\tAdd dyna feature num_iter_recycling: {num_iter_recycling}')

    if train_config.unclamped_fape:
        if np.random.default_rng(random_key + step).uniform() < 0.1:
            batch['label']['use_clamped_fape'] = paddle.full(shape, 0.0)
            print(f'\tAdd dyna label use_clamped_fape: 0.0')


def check_batch(batch, max_length=None):
    """print data shapes and check max_length"""
    def _print(k, d):
        if k in d:
            print(f'\t{k}: {d[k].shape}')

    logging.info(f'Get protein_name: {batch["name"]}')    
    for k in ['aatype', 'msa_feat', 'extra_msa', 'masked_msa_only']:
        _print(k, batch["feat"])
    for k in ['all_atom_positions']:
        _print(k, batch["label"])

    L = batch["feat"]['aatype'].shape[2]
    if not max_length is None and L > max_length:
        print(f'\tskip {batch["name"]} due to two long length')
        return False
    return True


@paddle.no_grad()
def eval(args, model, eval_dataset, compute_loss, cache_dir=None):
    """evaluate a given dataset"""
    model.eval()
    data_loader = paddle.io.DataLoader(
            dataset=eval_dataset,
            batch_size=1,
            drop_last=False,
            num_workers=0)
    res_collect = ResultsCollect(
            eval_tm_score=True, 
            tm_score_bin=args.tm_score_bin,
            lddt_score_bin=args.lddt_score_bin,
            cache_dir=cache_dir, distributed=args.distributed)
    s0 = time_me()
    for i, batch in enumerate(data_loader):
        if not check_batch(batch, max_length=MAX_EVAL_SIZE):
            continue
        s1 = time_me()
        res = model(batch, compute_loss=compute_loss)
        if compute_loss:
            results, loss = res
        else:
            results, loss = res, np.zeros([1])
        s2 = time_me()

        extra_dict = {'loss': np.array(loss)[0], 'data_time': s1 - s0, 'train_time': s2 - s1}
        res_collect.add(batch, results, extra_dict)
        print(f'Test_step: {i} loss: {extra_dict}')
        s0 = time_me()

    res = res_collect.get_result()
    return res


def full_eval(args, cur_step, model, valid_dataset, test_dataset_dict, data_writer):
    # eval valid set
    if not valid_dataset is None:
        logging.info(f'[Main] Train_step: {cur_step} evaluate valid set ==========')
        valid_results = eval(
                args,
                model, 
                valid_dataset, 
                compute_loss=False,
                cache_dir=f'output/valid_pdbs/{cur_step}')
        add_to_data_writer(data_writer, cur_step, valid_results, prefix='valid')
        csv_print({**valid_results, '0-VALID': '0-VALID'})
        logging.info(f'[Main] Train_step: {cur_step} evaluate valid finish ==========')

    # eval test set
    logging.info(f'[Main] Train_step: {cur_step} evaluate test set ==========')
    for name, test_dataset in test_dataset_dict.items():
        test_result = eval(
                args,
                model, 
                test_dataset, 
                compute_loss=False,
                cache_dir=f'output/test_pdbs-{name}/{cur_step}')
        add_to_data_writer(data_writer, cur_step, test_result, prefix='test-' + name)
        csv_print({**test_result, f'0-TEST-{name}': f'0-TEST-{name}'})
    logging.info(f'[Main] Train_step: {cur_step} evaluate test finish ==========')


def train(args, cur_step, model, train_data_gen, distill_data_gen, train_config, model_config, lr_scheduler, optimizer, res_collect, train_logger):
    model.train()
    # fetch data
    logging.info(f'[Main] Train_step: {cur_step} fetch_data')
    s0 = time_me()
    batch = None
    if distill_data_gen:
        rand_distill = np.random.random()
        batch = next(distill_data_gen) if rand_distill > 0.25 else next(train_data_gen) 
    else:
        batch = next(train_data_gen)
    if not check_batch(batch):
        return
    add_dyna_features(train_config, model_config, batch, cur_step)

    # train
    def _forward_with_precision(batch):
        if args.precision == "bf16":
            black_list, white_list = get_bf16_op_list()
            with paddle.amp.auto_cast(level='O1', custom_white_list=white_list, custom_black_list=black_list, dtype='bfloat16'):
                return model(batch)
        elif args.precision == "fp32":
            return model(batch)
        else:
            raise ValueError("Please choose precision from bf16 and fp32! ")

    s1 = time_me()
    logging.info(f'[Main] Train_step: {cur_step} train')

    results, loss = _forward_with_precision(batch)

    s2 = time_me()
    loss.backward()

    s3 = time_me()
    if args.distributed and cur_step % args.gradient_merge_k_steps == 0:
        grad_sync(optimizer._param_groups, None)

    s4 = time_me()
    if cur_step % args.gradient_merge_k_steps == 0:
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()

    if args.precision == "bf16":
        loss = loss.cast("float32")
        
    s5 = time_me()

    train_logger.update("loss", loss.item())
    train_logger.update("reader_cost", s1 - s0)
    train_logger.update("forward_cost", s2 - s1)
    train_logger.update("backward_cost", s3 - s2)
    train_logger.update("gradsync_cost", s4 - s3)
    train_logger.update("update_cost", s5 - s4)
    train_logger.update("batch_cost", s5 - s0)
    train_logger.update("protein", args.batch_size * dist.get_world_size())

    if cur_step % args.gradient_merge_k_steps == 0:

        train_logger.update("avg_loss", train_logger.mean("loss"))

        log_msg = f"[Main] Train_step: {cur_step}, " + train_logger.msg()
        extra_dict = train_logger.state_dict()

        train_logger.reset("loss")

        res_collect.add(batch, results, extra_dict)
        logging.info(log_msg)


def main(args):
    """main function"""
    print(f'>>> args:\n{args}')
    data_config = ml_collections.ConfigDict(json.load(open(args.data_config, 'r')))
    print(f'>>> data_config:\n{data_config}')
    train_config = ml_collections.ConfigDict(json.load(open(args.train_config, 'r')))
    print(f'>>> train_config:\n{train_config}')

    ### check paddle version
    if args.distributed:
        assert paddle.fluid.core.is_compiled_with_dist(), "Please using the paddle version compiled with distribute."
    args.distributed = args.distributed and dist.get_world_size() > 1

    rank = 0
    world_size = 1
    if args.distributed:
        dist.init_parallel_env()
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    ### set seed for reproduce experiment results
    if args.seed is not None:
        args.seed += rank
        init_seed(args.seed)
    
    ### create model
    model_config = config.model_config(args.model_name)
    model = RunModel(train_config, model_config)
    single_model = model
    if args.distributed:
        # broadcast param to other ranks when using distributed data parallel
        param_sync(model, src_rank=0)

    if rank == 0:
        # print("model:", model)
        print("model size:", get_model_parameter_size(model))

    if (not args.init_model is None) and (not args.init_model == ""):
        print(f"Load pretrain model from {args.init_model}")
        single_model.alphafold.set_state_dict(paddle.load(args.init_model))
    
    optimizer, lr_scheduler = get_optimizer(train_config.optimizer, model)
    
    ### load dataset
    if not args.only_test:
        train_dataset = AF2Dataset(
                model_config=model_config,
                data_config=data_config.train,
                trainer_id=rank, 
                trainer_num=world_size,
                crop_size=train_config.crop_size,
                is_pad_if_crop=True,
                delete_msa_block=True,
                is_shuffle=True)
    if 'valid' in data_config:
        valid_dataset = AF2Dataset(
                model_config=model_config,
                data_config=data_config.valid,
                trainer_id=rank, 
                trainer_num=world_size,
                is_shuffle=False)
    else:
        valid_dataset = None
    test_dataset_dict = {}
    if 'test' in data_config:
        for test_name in data_config.test:
            test_dataset_dict[test_name] = AF2TestDataset(
                    model_config=model_config,
                    data_config=data_config.test[test_name],
                    trainer_id=rank, 
                    trainer_num=world_size)
    distill_dataset = None
    if 'distill' in data_config:
        distill_dataset = AF2DistillDataset(
                model_config=model_config,
                data_config=data_config.distill,
                trainer_id=rank,
                trainer_num=world_size,
                crop_size=train_config.crop_size,
                is_pad_if_crop=True,
                delete_msa_block=True,
                is_shuffle=True)


    ### if only_test
    if args.only_test:
        full_eval(args, args.start_step, model, valid_dataset, test_dataset_dict, None)
        exit(0)

    ### create data loader
    train_loader = paddle.io.DataLoader(
            dataset=train_dataset,
            batch_sampler=LoopedBatchSampler(
                dataset=train_dataset, 
                shuffle=True, 
                batch_size=args.batch_size, 
                drop_last=False),
            num_workers=args.num_workers,
            worker_init_fn=worker_init_fn if args.seed is not None else None)
    train_data_gen = iter(train_loader)

    distill_data_gen = None
    if distill_dataset:
        distill_loader = paddle.io.DataLoader(
                dataset=distill_dataset,
                batch_sampler=LoopedBatchSampler(
                    dataset=distill_dataset,
                    shuffle=True,
                    batch_size=args.batch_size,
                    drop_last=False),
                num_workers=args.num_workers,
                worker_init_fn=worker_init_fn if args.seed is not None else None)
        distill_data_gen = iter(distill_loader)

    ### start training
    if rank == 0:
        try:    # permission denied error if without root
            data_writer = SummaryWriter(f'{args.log_dir}/tensorboard_log_dir', max_queue=0)
        except Exception as ex:
            print(f'Create data_writer failed: {ex}')
            data_writer = None
    else:
        data_writer = None

    train_logger = TrainLogger()

    res_collect = ResultsCollect()
    cur_step = args.start_step
    for _ in range(cur_step):
        lr_scheduler.step()
    logging.info('[Main] Start training.')
    while True:
        # reset train log info
        if cur_step == 5:
            train_logger.reset()

        if cur_step > args.train_step:
            break

        # train
        train(args, cur_step, model, train_data_gen, distill_data_gen, train_config, model_config, \
                lr_scheduler, optimizer, res_collect, train_logger)
        if cur_step % args.log_step == 0:
            train_results = res_collect.get_result()
            train_results['lr'] = lr_scheduler.get_lr()
            train_results['batch_size'] = args.batch_size * world_size
            add_to_data_writer(data_writer, cur_step, train_results, prefix='train')
            res_collect = ResultsCollect()

        # evaluate
        if cur_step % args.eval_step == 0:
            full_eval(args, cur_step, model, valid_dataset, test_dataset_dict, data_writer)

        # save params
        if cur_step % args.save_step == 0 and rank == 0:
            paddle.save(single_model.alphafold.state_dict(), f'{args.model_dir}/step_{cur_step}.pdparams')
            if args.paddlecloud:
                upload_to_hadoop(args, cur_step)
        
        cur_step += 1
        sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action='store_true', default=False)
    parser.add_argument("--paddlecloud", action='store_true', default=False)
    parser.add_argument("--only_test", action='store_true', default=False)
    parser.add_argument("--seed", type=int, default=None, help="set seed for reproduce experiment results, None is do not set seed")

    parser.add_argument("--tm_score_bin", type=str, help="path to tm_score bin")
    parser.add_argument("--lddt_score_bin", type=str, help="path to lddt bin")

    parser.add_argument("--data_config", type=str, help="path to data config")
    parser.add_argument("--train_config", type=str, help='path to train config')
    parser.add_argument("--model_name", type=str, help='used to choose model config')
    parser.add_argument("--init_model", type=str, default='')
    parser.add_argument("--precision", type=str, choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--train_step", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gradient_merge_k_steps", type=int, default=1)

    parser.add_argument("--model_dir", type=str, default='./models')
    parser.add_argument("--log_dir", type=str, default='./log')
    parser.add_argument("--log_step", type=int, default=20)
    parser.add_argument("--eval_step", type=int, default=200)
    parser.add_argument("--save_step", type=int, default=200)
    args = parser.parse_args()

    main(args)
