import pdb
from layers.basics import GlobalAttention
from config import model_config
import paddle as pd
from paddle.jit import to_static, save
from paddle.static import InputSpec
from paddle import inference as pdinfer
import time
import os
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser as Parser

parser = Parser('[pd.infer] UT of pdpd.GolbalAttention')
parser.add_argument('--n_cpus', type=int, required=True, help='physical cores used during pd.infer')
args = parser.parse_args()
n_cpus = args.n_cpus

cfg = model_config('model_1')
c = cfg['model']['embeddings_and_evoformer']['evoformer']['msa_column_attention']
gc = cfg['model']['global_config']

n_warm = 3
n_iter = 13
ignore_eval = False
is_dynamic_input = False
prefix_weights = 'dynamic_params/globalattention' if is_dynamic_input else 'static_params/globalattention'
f_topo = prefix_weights + '.pdmodel'
f_params = prefix_weights + '.pdiparams'
### setup model
model = GlobalAttention(c,gc,64,64,64)
model.eval()
if not os.path.isfile(f_topo):
  if is_dynamic_input:
    len_dim = None
  else:
    len_dim = 764 # None
  net = to_static(model, input_spec=[
    InputSpec(shape=[1, len_dim, 5120, 64]),
    InputSpec(shape=[1, len_dim, 5120, 64]),
    InputSpec(shape=[1, len_dim, 5120, 1])
  ])
  save(net, prefix_weights)
else:
  len_dim = 764
### create sample input
msa_act = pd.ones([1, 764, 5120, 64])
msa_mask = pd.ones([1, 764, 5120, 1])
# bias = pd.ones([1, 764, 1, 1, 5120])  # not used

print('# [INFO] inference on dynamic graph')
if not ignore_eval:
  dy_dts = 0.
  with pd.no_grad():
    for i in tqdm(range(n_iter)):
      t0 = time.time()
      _ = model(msa_act, msa_act, msa_mask)
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
msa_act1 = np.ones([1, len_dim, 5120, 64], dtype='float32')
msa_mask1 = np.ones([1, len_dim, 5120, 1], dtype='float32')
# bias1 = np.ones([1, 1, 1, 1, 4], dtype='float32')

# 获取输入轴
input_names = predictor.get_input_names() # ['q_data', 'm_data', 'q_mask']
inputl_q = predictor.get_input_handle('q_data')
inputl_m = predictor.get_input_handle('m_data')
inputl_b = predictor.get_input_handle('q_mask')

# 变形输入轴
if is_dynamic_input:
  print('# [INFO] re-organize dynamic axes')
  inputl_q.reshape(msa_act1.shape)
  inputl_m.reshape(msa_act1.shape)
  inputl_b.reshape(msa_mask1.shape)

# 获取输出轴
output_names = predictor.get_output_names() # ['tmp_10']
outputl = predictor.get_output_handle('tmp_10')

# run
dts = 0.
for i in tqdm(range(n_iter)):
  t0 = time.time()
  inputl_q.copy_from_cpu(msa_act1)
  inputl_m.copy_from_cpu(msa_act1)
  inputl_b.copy_from_cpu(msa_mask1)
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