import pdb
from layers.head import MaskedMsaHead
from config import model_config
import paddle as pd
from paddle.jit import to_static, save
from paddle.static import InputSpec
from paddle import inference as pdinfer
import time
import os
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser as Parser

parser = Parser('[pd.infer] UT of pdpd.maskedmsahead')
parser.add_argument('--n_cpus', type=int, default=64, help='physical cores used during pd.infer')
args = parser.parse_args()
n_cpus = args.n_cpus

cfg = model_config('model_1')
c = cfg['model']['heads']['masked_msa']
gc = cfg['model']['global_config']
### create sample input
# masked_msa
len_dim = 206
msa_representation = pd.ones([1, 508, len_dim, 256])

n_warm = 3
n_iter = 13
ignore_eval = True
is_dynamic_input = False
prefix_weights = 'dynamic_params/maskedmashead' if is_dynamic_input else 'static_params/maskedmashead'
f_topo = prefix_weights + '.pdmodel'
f_params = prefix_weights + '.pdiparams'

TARGET_FEAT_DIM = 22
MSA_FEAT_DIM = 49
channel_num = {
            'target_feat': TARGET_FEAT_DIM,
            'msa_feat': MSA_FEAT_DIM,
            'msa_channel': 256,
        }

print('# [INFO] build and save static graph of Attention')
model = MaskedMsaHead(channel_num, c, gc)
model.eval()
if not os.path.isfile(f_topo):
  if is_dynamic_input:
    len_dim = None
  net = to_static(model, input_spec=[
    InputSpec(shape=[1, 508, len_dim, 256]),
  ])
  save(net, prefix_weights)


print('# [INFO] inference on dynamic graph')
if not ignore_eval:
  dy_dts = 0.
  with pd.no_grad():
    for i in tqdm(range(n_iter)):
      t0 = time.time()
      _ = model(msa_representation)
      t1 = time.time()
      dt = t1 - t0
      if i >= n_warm:
        dy_dts += dt

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
msa_representation1 = np.ones([1, 508, len_dim, 256], dtype='float32')

# 获取输入轴
input_names = predictor.get_input_names() # ['q_data', 'm_data', 'bias']
inputl_msa = predictor.get_input_handle('msa_representation')

# 变形输入轴
if is_dynamic_input:
  print('# [INFO] re-organize dynamic axes')
  inputl_q.reshape(q_mat1.shape)
  inputl_m.reshape(m_mat1.shape)
  inputl_b.reshape(bias1.shape)

# 获取输出轴
output_names = predictor.get_output_names() # ['tmp_2']
outputl = predictor.get_output_handle('tmp_2')

# run
dts = 0.
for i in range(n_iter):
  t0 = time.time()
  inputl_q.copy_from_cpu(q_mat1)
  inputl_m.copy_from_cpu(m_mat1)
  inputl_b.copy_from_cpu(bias1)
  predictor.run()
  output = outputl.copy_to_cpu()
  t1 = time.time()
  dt = t1 - t0
  if i >= n_warm:
    dts += dt

if not ignore_eval:
  print('# [dynamic-graph] avg inference time = {}'.format(dy_dts/(n_iter-n_warm)))
print('# [static-graph] avg inference time = {}'.format(dts/(n_iter-n_warm)))

print(output.shape)