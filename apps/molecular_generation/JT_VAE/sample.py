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
"""sample smiles from latent space"""
import rdkit
import argparse
import paddle
from src.vocab import Vocab, get_vocab
from src.utils import load_json_config
from src.jtnn_vae import JTNNVAE
from src.vocab import Vocab
lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--nsample', type=int, required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--config', required=True)

args = parser.parse_args()
config = load_json_config(args.config)
vocab = get_vocab(args.vocab)
vocab = Vocab(vocab)
model = JTNNVAE(vocab, config['hidden_size'], config['latent_size'],
                config['depthT'], config['depthG'])
train_model_params = paddle.load(args.model)
model.set_state_dict(train_model_params)
model.eval()

res = []
for i in range(args.nsample):
    smi = model.sample_prior()
    print(i, smi)
    res.append(smi)
with open(args.output, 'w')as f:
    for smi in res:
        f.write(smi + '\n')
