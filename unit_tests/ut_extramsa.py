import pdb
from layers.backbones import ExtraMsa
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

model_prefix = 'extramsa'
parser = Parser('[pd.infer] UT of pdpd.{}'.format(model_prefix))
parser.add_argument('--n_cpus', type=int, default=64, help='physical cores used during pd.infer')
args = parser.parse_args()
n_cpus = args.n_cpus
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

cfg = model_config('model_1')
c = cfg['model']['embeddings_and_evoformer']
gc = cfg['model']['global_config']

n_warm = 3
n_iter = 13
ignore_eval = True
is_dynamic_input = False
prefix_weights = 'dynamic_params/{}'.format(model_prefix) if is_dynamic_input else 'static_params/{}'.format(model_prefix)
f_topo = prefix_weights + '.pdmodel'
f_params = prefix_weights + '.pdiparams'

TARGET_FEAT_DIM = 22
MSA_FEAT_DIM = 49
channel_num = {
            'target_feat': TARGET_FEAT_DIM,
            'msa_feat': MSA_FEAT_DIM,
            'extra_msa_channel': c.extra_msa_channel,
            'msa_channel': c.msa_channel,
            'pair_channel': c.pair_channel,
            'seq_channel': c.seq_channel
        }
### create sample input
len_dim = 10
batch = {
  'extra_msa': pd.ones([1, 5120, len_dim]),
  'extra_has_deletion': pd.ones([1, 5120, len_dim]),
  'extra_deletion_value': pd.ones([1, 5120, len_dim]),
  'extra_msa_mask': pd.ones([1, 5120, len_dim]),
  'pair_activations': pd.ones([1, len_dim, len_dim, 128]),
  'mask_2d': pd.ones([1, len_dim, len_dim])
}


print('# [INFO] build and save static graph of {}'.format(model_prefix))
model = ExtraMsa(channel_num, c, gc)
model.eval()
if not os.path.isfile(f_topo):
  if is_dynamic_input:
    len_dim = None
  net = to_static(model, input_spec=[
    InputSpec(shape=[1, 5120, len_dim],name='extra_msa'),
    InputSpec(shape=[1, 5120, len_dim],name='extra_has_deletion'),
    InputSpec(shape=[1, 5120, len_dim],name='extra_deletion_value'),
    InputSpec(shape=[1, 5120, len_dim],name='extra_msa_mask'),
    InputSpec(shape=[1, len_dim, len_dim, 128],name='pair_activations'),
    InputSpec(shape=[1, len_dim, len_dim],name='mask_2d')
  ])
  save(net, prefix_weights)


print('# [INFO] inference on dynamic graph')
warnings.filterwarnings('ignore', 'DAP communication')
if not ignore_eval:
  dy_dts = 0.
  with pd.no_grad():
    for i in tqdm(range(n_iter)):
      t0 = time.time()
      _ = model(**batch)
      t1 = time.time()
      dt = t1 - t0
      if i >= n_warm:
        dy_dts += dt
warnings.resetwarnings()

dts = 0.
pd_cfg = pdinfer.Config(f_topo, f_params)

print('# [INFO] inference on static graph')
### optimization based on intel architecture
pd_cfg.set_cpu_math_library_num_threads(n_cpus)
pd_cfg.enable_mkldnn()
#pd_cfg.enable_memory_optim() # no change in perf. or memory
if is_dynamic_input:
  pd_cfg.set_mkldnn_cache_capacity(1)
predictor = pdinfer.create_predictor(pd_cfg)

# 创建输入样例
extra_msa_1 = np.ones([1, 5120, len_dim], dtype='float32')
extra_has_deletion_1 = np.ones([1, 5120, len_dim], dtype='float32')
extra_deletion_value_1 = np.ones([1, 5120, len_dim], dtype='float32')
extra_msa_mask_1 = np.ones([1, 5120, len_dim], dtype='float32')
pair_activations_1 = np.ones([1, len_dim, len_dim, 128], dtype='float32')
mask_2d_1 = np.ones([1, len_dim, len_dim], dtype='float32')

# 获取输入轴
input_names = predictor.get_input_names()
inputl_exms = predictor.get_input_handle('extra_msa')
inputl_exha = predictor.get_input_handle('extra_has_deletion')
inputl_exde = predictor.get_input_handle('extra_deletion_value')
inputl_exmm = predictor.get_input_handle('extra_msa_mask')
inputl_pact = predictor.get_input_handle('pair_activations')
inputl_ms2d  = predictor.get_input_handle('mask_2d')

# # 变形输入轴
# if is_dynamic_input:
#   print('# [INFO] re-organize dynamic axes')
#   inputl_q.reshape(q_mat1.shape)
#   inputl_m.reshape(m_mat1.shape)
#   inputl_b.reshape(bias1.shape)

# 获取输出轴
output_names = predictor.get_output_names()
print(output_names)
outputls = {k:predictor.get_output_handle(k) for k in output_names}

# run
dts = 0.
for i in tqdm(range(n_iter)):
  t0 = time.time()
  inputl_exms.copy_from_cpu(extra_msa_1)
  inputl_exha.copy_from_cpu(extra_has_deletion_1)
  inputl_exde.copy_from_cpu(extra_deletion_value_1)
  inputl_exmm.copy_from_cpu(extra_msa_mask_1)
  inputl_pact.copy_from_cpu(pair_activations_1)
  inputl_ms2d.copy_from_cpu(mask_2d_1)

  predictor.run()
  output_shapes = {k:outputls[k].copy_to_cpu().shape for k in output_names}
  print(output_shapes)
  t1 = time.time()
  dt = t1 - t0
  if i >= n_warm:
    dts += dt

if not ignore_eval:
  print('# [dynamic-graph] avg inference time = {}'.format(dy_dts/(n_iter-n_warm)))
print('# [static-graph] avg inference time = {}'.format(dts/(n_iter-n_warm)))

# print(output.shape)