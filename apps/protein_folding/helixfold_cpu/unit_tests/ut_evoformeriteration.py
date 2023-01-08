import pdb
from layers.backbones import EvoformerIteration
from config import model_config
import paddle as pd
from paddle.jit import to_static, save
from paddle.static import InputSpec
from paddle import inference as pdinfer
import time
import os
import numpy as np
from argparse import ArgumentParser as Parser
from tqdm import tqdm


parser = Parser('[pd.infer] UT of helixfold.evoformeriteration')
parser.add_argument('--n_cpus', type=int, default=64, help='physical cores used during pd.infer')
args = parser.parse_args()
n_cpus = args.n_cpus

cfg = model_config('model_1')
c = cfg['model']['embeddings_and_evoformer']['evoformer']
gc = cfg['model']['global_config']
len_dim = 1024
msa_act = pd.ones([1, 512, len_dim, 256])
msa_mask = pd.ones([1, 512, len_dim])
pair_act = pd.ones([1, len_dim, len_dim, 128])
pair_mask = pd.ones([1, len_dim, len_dim])
n_warm = 3
n_iter = 13
ignore_eval = True
is_dynamic_input = False
module_prefix = 'evoformeriteration'
prefix_weights = 'dynamic_params/evoformeriteration' if is_dynamic_input else 'static_params/evoformeriteration'
f_topo = prefix_weights + '.pdmodel'
f_params = prefix_weights + '.pdiparams'
channel_num = {'msa_channel':256, 'pair_channel':128}

print('# [INFO] build and save static graph of Evoformeriteration')
model = EvoformerIteration(channel_num, c, gc, is_extra_msa=False)
model.eval()
if not os.path.isfile(f_topo):
  if is_dynamic_input:
    len_dim = None
  net = to_static(model, input_spec=[
    InputSpec(shape=[1, 512, len_dim, 256]),
    InputSpec(shape=[1, len_dim, len_dim, 128]),
    InputSpec(shape=[1, 512, len_dim]),
    InputSpec(shape=[1, len_dim, len_dim])
  ])
  save(net, prefix_weights)


print('# [INFO] inference on dynamic graph')
if not ignore_eval:
  dy_dts = 0.
  with pd.no_grad():
    for i in tqdm(range(n_iter)):
      t0 = time.time()
      msa_act, pair_act = model(msa_act, pair_act, msa_mask, pair_mask)
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
msa_act1 = np.ones([1, 512, len_dim, 256], dtype='float32')
msa_mask1 = np.ones([1, 512, len_dim], dtype='float32')
pair_act1 = np.ones([1, len_dim, len_dim, 128], dtype='float32')
pair_mask1 = np.ones([1, len_dim, len_dim], dtype='float32')

# 获取输入轴
input_names = predictor.get_input_names()
inputl_sa = predictor.get_input_handle('msa_act')
inputl_sm = predictor.get_input_handle('msa_mask')
inputl_pa = predictor.get_input_handle('pair_act')
inputl_pm = predictor.get_input_handle('pair_mask')

# 变形输入轴
if is_dynamic_input:
  print('# [INFO] re-organize dynamic axes')
  inputl_sa.reshape(msa_act1.shape)
  inputl_sm.reshape(msa_mask1.shape)
  inputl_pa.reshape(pair_act1.shape)
  inputl_pm.reshape(pair_mask1.shape)

# 获取输出轴
output_names = predictor.get_output_names() # ['tmp_2']
outputl = predictor.get_output_handle('tmp_2')

# run
dts = 0.
for i in tqdm(range(n_iter)):
  t0 = time.time()
  inputl_sa.copy_from_cpu(msa_act1)
  inputl_sm.copy_from_cpu(msa_mask1)
  inputl_pa.copy_from_cpu(pair_act1)
  inputl_pm.copy_from_cpu(pair_mask1)
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