from layers.static_backbones import StaticEvoformerIteration
from config import model_config
import numpy as np
from tqdm import tqdm
import time
from argparse import ArgumentParser as Parser


parser = Parser('# [INFO] UT of static helixfold.evoformeriteration')
parser.add_argument('--n_cpus', type=int, default=64, help='physical cores used during pd.infer')
args = parser.parse_args()
n_cpus = args.n_cpus

n_warm = 3
n_iter = 13
cfg = model_config('model_1')
c = cfg['model']['embeddings_and_evoformer']['evoformer']
gc = cfg['model']['global_config']
len_dim = 1024
feed_dict = {
  'msa_act':np.ones([1, 512, len_dim, 256], dtype='float32'),
  'pair_act':np.ones([1, len_dim, len_dim, 128], dtype='float32'),
  'msa_mask':np.ones([1, 512, len_dim], dtype='float32'),
  'pair_mask':np.ones([1, len_dim, len_dim], dtype='float32')
}
channel_num={'msa_channel':256, 'pair_channel':128}
is_extra_msa = False
model = StaticEvoformerIteration(
  config=c,
  global_config=gc,
  feed_dict=feed_dict,
  channel_num=channel_num,
  n_cpus=n_cpus,
  is_extra_msa=is_extra_msa,
  module_prefix='evoformeriteration',
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
for k, v in outputs.items():
  print('{} -> {}'.format(k, v.shape))
