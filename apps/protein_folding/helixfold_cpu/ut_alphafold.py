from layers.net import AlphaFold
from config import model_config
import time
from tqdm import tqdm
import paddle
from argparse import ArgumentParser as Parser

parser = Parser('[pd.infer] dynamic UT of pdinfer.HelixFold')

cfg = model_config('model_1')
c = cfg['model']

n_warm = 0
n_iter = 1
ignore_eval = False

### create sample input
len_dim = 40 
feed_dict = {
  'target_feat': paddle.ones([1, 4, len_dim, 22], dtype='float32'),
  'msa_feat': paddle.ones([1, 4, 508, len_dim, 49], dtype='float32'),
  'seq_mask': paddle.ones([1, 4, len_dim], dtype='float32'),
  'seq_length': paddle.ones([1, 4, len_dim], dtype='int32'),
  'aatype': paddle.ones([1, 4, len_dim], dtype='float32'),
  'residue_index': paddle.ones([1, 4, len_dim], dtype='float32'),
  'template_mask': paddle.ones([1, 4, 4], dtype='float32'),
  'template_aatype': paddle.ones([1, 4, 4, len_dim], dtype="int32"), # define
  'template_pseudo_beta_mask': paddle.ones([1, 4, 4, len_dim], dtype='float32'),
  'template_pseudo_beta': paddle.ones([1, 4, 4, len_dim, 3], dtype='float32'),
  'template_all_atom_positions': paddle.ones([1, 4, 4, len_dim, 37, 3], dtype='float32'),
  'template_all_atom_masks': paddle.ones([1, 4, 4, len_dim, 37], dtype='float32'),
  'extra_msa': paddle.ones([1, 4, 5120, len_dim], dtype='float32'),
  'extra_has_deletion': paddle.ones([1, 4, 5120, len_dim], dtype='float32'),
  'extra_deletion_value': paddle.ones([1, 4, 5120, len_dim], dtype='float32'),
  'extra_msa_mask': paddle.ones([1, 4, 5120, len_dim], dtype='float32'),
  'msa_mask': paddle.ones([1, 4, 508, len_dim], dtype='float32'),
  'prev_pos': paddle.ones([1, 4, len_dim, 37, 3], dtype='float32'),
  'prev_msa_first_row': paddle.ones([1, 4, len_dim, 256], dtype='float32'),
  'prev_pair': paddle.ones([1, 4, len_dim, len_dim, 128], dtype='float32'),
  'atom14_atom_exists': paddle.ones([1, 4, len_dim, 14], dtype='float32'),
  'atom37_atom_exists': paddle.ones([1, 4, len_dim, 37], dtype='float32'),
  'residx_atom37_to_atom14': paddle.ones([1, 4, len_dim, 37], dtype='float32')
}

print('# [INFO] build dynamic graph of HelixFold')
model = AlphaFold(config=c)
model.eval()

print('# [INFO] inference on dynamic graph')
# run
dts = 0.
for i in tqdm(range(n_iter)):
  t0 = time.time()
  with paddle.no_grad():
    outputs = model(feed_dict, False)
  dt = time.time() - t0
  if i >= n_warm:
    dts += dt
print('# [INFO] avg inference time = {}'.format(dts/(n_iter-n_warm)))
