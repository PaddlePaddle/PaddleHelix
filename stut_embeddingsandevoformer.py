from layers.static_subnets import StaticEmbeddingsAndEvoformer
from config import model_config
import time
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser as Parser

parser = Parser('[pd.infer] static UT of pdinfer.StaticEmbeddingsAndEvoformer')
parser.add_argument('--n_cpus', type=int, default=64, help='physical cores used during pd.infer')
args = parser.parse_args()
n_cpus = args.n_cpus

cfg = model_config('model_1')
c = cfg['model']['embeddings_and_evoformer']
gc = cfg['model']['global_config']

n_warm = 1
n_iter = 2
ignore_eval = False
root_weights = 'static_modules'
module_prefix = 'embeddingsandevoformer'
channel_num = {
            'target_feat': 22,
            'msa_feat': 49,
            'extra_msa_channel': c.extra_msa_channel,
            'msa_channel': c.msa_channel,
            'pair_channel': c.pair_channel,
            'seq_channel': c.seq_channel
        }
### create sample input
len_dim = 512
feed_dict = {
  'target_feat': np.ones([1, len_dim, 22], dtype='float32'),
  'msa_feat': np.ones([1, 508, len_dim, 49], dtype='float32'),
  'seq_mask': np.ones([1, len_dim], dtype='float32'),
  'aatype': np.ones([1, len_dim], dtype='float32'),
  'residue_index': np.ones([1, len_dim], dtype='float32'),
  'template_mask': np.ones([1, 4], dtype='float32'),
  'template_aatype': np.ones([1, 4, len_dim], dtype="int32"), # define
  'template_pseudo_beta_mask': np.ones([1, 4, len_dim], dtype='float32'),
  'template_pseudo_beta': np.ones([1, 4, len_dim, 3], dtype='float32'),
  'template_all_atom_positions': np.ones([1, 4, len_dim, 37, 3], dtype='float32'),
  'template_all_atom_masks': np.ones([1, 4, len_dim, 37], dtype='float32'),
  'extra_msa': np.ones([1, 5120, len_dim], dtype='float32'),
  'extra_has_deletion': np.ones([1, 5120, len_dim], dtype='float32'),
  'extra_deletion_value': np.ones([1, 5120, len_dim], dtype='float32'),
  'extra_msa_mask': np.ones([1, 5120, len_dim], dtype='float32'),
  'msa_mask': np.ones([1, 508, len_dim], dtype='float32'),
  'prev_pos': np.ones([1, len_dim, 37, 3], dtype='float32'),
  'prev_msa_first_row': np.ones([1, len_dim, 256], dtype='float32'),
  'prev_pair': np.ones([1, len_dim, len_dim, 128], dtype='float32'),
  # 'torsion_angles_sin_cos': pd.ones([1, 4, len_dim, 7, 2]),
  # 'alt_torsion_angles_sin_cos': pd.ones([1, 4, len_dim, 7, 2]),
  # 'torsion_angles_mask': pd.ones([1, 4, len_dim, 7])
}

print('# [INFO] build and save static graph of Attention')
model = StaticEmbeddingsAndEvoformer(
  config=c, 
  global_config=gc,
  seq_len=len_dim,
  channel_num=channel_num,
  n_cpus=n_cpus,
  module_prefix=module_prefix,
  root_weights=root_weights,
  is_pdinfer_init=False)

print('# [INFO] inference on static graph')
# run
dts = 0.
for i in tqdm(range(n_iter)):
  t0 = time.time()
  outputs = model(feed_dict)
  print('# [INFO] output of {}:'.format(module_prefix))
  for k, v in outputs.items():
    print('{} -> {}'.format(k, v.shape))
  dt = time.time() - t0
  if i >= n_warm:
    dts += dt
print('# [INFO] avg inference time = {}'.format(dts/(n_iter-n_warm)))
