import pdb
from layers.static_backbones import StaticEvoformer
from config import model_config
import paddle as pd
from paddle.jit import to_static, save
from paddle.static import InputSpec
from paddle import inference as pdinfer
import time
import os
import numpy as np
from tqdm import tqdm
import pickle
from argparse import ArgumentParser as Parser
import warnings

model_prefix = 'evoformer'
parser = Parser('[pd.infer] UT of static helixfold.{}'.format(model_prefix))
parser.add_argument('--n_cpus', type=int, default=64, help='physical cores used during pd.infer')
args = parser.parse_args()
n_cpus = args.n_cpus
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

cfg = model_config('model_1')
c = cfg['model']['embeddings_and_evoformer']
gc = cfg['model']['global_config']

n_warm = 3
n_iter = 13
channel_num = {
            'target_feat': 22,
            'msa_feat': 49,
            'extra_msa_channel': c.extra_msa_channel,
            'msa_channel': c.msa_channel,
            'pair_channel': c.pair_channel,
            'seq_channel': c.seq_channel
        }
### create sample input
len_dim = 1024 # 1024=350GB
feed_dict = {
  'msa_activations': np.ones([1, 508, len_dim, 256], dtype='float32'),
  'extra_pair_act': np.ones([1, len_dim, len_dim, 128], dtype='float32'),
  'msa_mask': np.ones([1, 508, len_dim], dtype='float32'),
  'mask_2d': np.ones([1, len_dim, len_dim], dtype='float32')
}

print('# [INFO] build and save static graph of {}'.format(model_prefix))
model = StaticEvoformer(
  config=c, 
  global_config=gc,
  feed_dict=feed_dict,
  channel_num=channel_num,
  n_cpus=n_cpus,
  module_prefix='evoformer',
  root_weights='static_modules',
  is_pdinfer_init=False)

print('# [INFO] inference on static graph')
# run
dts = 0.
for i in tqdm(range(n_iter)):
  t0 = time.time()
  outputs = model(feed_dict)
  print('# [INFO] output of {}:'.format(model_prefix))
  for k, v in outputs.items():
    print('{} -> {}'.format(k, v.shape))
  dt = time.time() - t0
  if i >= n_warm:
    dts += dt
print('# [INFO] avg inference time = {}'.format(dts/(n_iter-n_warm)))