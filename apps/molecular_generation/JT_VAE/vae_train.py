#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""JTVAE training process"""
import paddle
import numpy as np
import argparse
import rdkit
import sys
from pgl.utils.data.dataloader import Dataloader
from src.datautils import JtnnDataSet, JtnnCollateFn
from src.jtnn_vae import JTNNVAE
from src.vocab import Vocab, get_vocab
from src.utils import load_json_config

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--config', required=True)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--load_epoch', type=int, default=0)
parser.add_argument('--use_gpu', type=eval, default=True)

args = parser.parse_args()

paddle.set_device('gpu') if args.use_gpu else paddle.set_device('cpu')
config = load_json_config(args.config)
vocab = get_vocab(args.vocab)
vocab = Vocab(vocab)
model = JTNNVAE(vocab, config['hidden_size'],
                config['latent_size'], config['depthT'], config['depthG'])
model.train()
if args.load_epoch > 0:
    paddle.load(args.save_dir + "/model.iter-" + str(args.load_epoch))


scheduler = paddle.optimizer.lr.ExponentialDecay(
    learning_rate=config['lr'], 
    gamma=config['anneal_rate'], 
    verbose=True)
clip = paddle.nn.ClipGradByNorm(clip_norm=config['clip_norm'])
optimizer = paddle.optimizer.Adam(
    parameters=model.parameters(),
    learning_rate=scheduler,
    grad_clip=clip)

train_dataset = JtnnDataSet(args.train)
collate_fn = JtnnCollateFn(vocab, True)
data_loader = Dataloader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    stream_shuffle_size=100,
    collate_fn=collate_fn)


total_step = args.load_epoch
beta = config['beta']
meters = np.zeros(4)
for epoch in range(args.epoch):
    for batch in data_loader:
        total_step += 1
        res = model(batch, beta)
        loss = res['loss']
        kl_div = res['kl_div']
        wacc = res['word_acc'] 
        tacc = res['topo_acc'] 
        sacc = res['assm_acc']
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])

        if total_step % config['print_iter'] == 0:
            meters /= config['print_iter']
            print("[%d] Beta: %.3f, KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f" % (
            total_step, beta, meters[0], meters[1], meters[2], meters[3]))
            sys.stdout.flush()
            meters *= 0

        if total_step % config['save_iter'] == 0:
            paddle.save(model.state_dict(), args.save_dir +
                        "/model.iter-" + str(total_step))
            paddle.save(optimizer.state_dict(), args.save_dir + "/train_optimizer.iter-" + str(total_step))

        if total_step % config['anneal_iter'] == 0:
            scheduler.step()

        if total_step % config['kl_anneal_iter'] == 0 and total_step >= config['warmup']:
            beta = min(config['max_beta'], beta + config['step_beta'])





