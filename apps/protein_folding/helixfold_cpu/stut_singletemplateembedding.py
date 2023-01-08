from layers.static_backbones import StaticSingleTemplateEmbedding
from config import model_config
import paddle as pd
import numpy as np
from tqdm import tqdm
import time
import os
from argparse import ArgumentParser as Parser


parser = Parser('# [INFO] UT of static helixfold.singletemplateembedding')
parser.add_argument('--n_cpus', type=int, default=64, help='physical cores used during pd.infer')
args = parser.parse_args()
n_cpus = args.n_cpus

n_warm = 3
n_iter = 13
cfg = model_config('model_1')
c = cfg['model']['embeddings_and_evoformer']
gc = cfg['model']['global_config']
len_dim = 206
feed_dict = {
  'msa_mask': np.ones([1, 508, len_dim], dtype='float32'),
  'torsion_angles_mask': np.ones([1, 4, len_dim, 7], dtype='float32'),
  'msa_activations_raw': np.ones([1, 508, len_dim, 256], dtype='float32'),
  'template_features': np.ones([1, 4, len_dim, 57], dtype='float32')
}
channel_num = {
  'target_feat': 22,
  'msa_feat': 49,
  'extra_msa_channel': c.extra_msa_channel,
  'msa_channel': c.msa_channel,
  'pair_channel': c.pair_channel,
  'seq_channel': c.seq_channel
}
model = StaticSingleTemplateEmbedding(
  config=c,
  global_config=gc,
  feed_dict=feed_dict,
  channel_num=channel_num,
  n_cpus=n_cpus,
  module_prefix='singletemplateembedding',
  root_weights='static_modules',
  is_pdinfer_init=False
)

# run
dts = 0.
for i in tqdm(range(n_iter)):
  t0 = time.time()
  outputs = model(feed_dict)
  dt = time.time() - t0
  if i >= n_warm:
    dts += dt
print('# [INFO] avg inference time = {}'.format(dts/(n_iter-n_warm)))

for name, output in outputs.items():
  print('# [INFO] {} -> {}'.format(
    name, output.shape
  ))
