from layers.embeddings import TemplatePair
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

parser = Parser('[pd.infer] UT of pdpd.Attention')
parser.add_argument('--n_cpus', type=int, default=64, help='physical cores used during pd.infer')
args = parser.parse_args()
n_cpus = args.n_cpus

channel_num = {
            'target_feat': 22,
            'msa_feat': 49,
            'extra_msa_channel': 64,
            'msa_channel': 256,
            'pair_channel': 128,
            'seq_channel': 384,
            'template_pair': 88, 
        }
cfg = model_config('model_1')
c = cfg['model']['embeddings_and_evoformer']['template']['template_pair_stack']
gc = cfg['model']['global_config']

n_warm = 3
n_iter = 10 + n_warm
ignore_eval = False
is_dynamic_input = False
prefix_weights = 'dynamic_params/templatepair' if is_dynamic_input else 'static_params/templatepair'
f_topo = prefix_weights + '.pdmodel'
f_params = prefix_weights + '.pdiparams'

### create sample input
len_dim = 206
pair_act  = pd.ones([1, len_dim, len_dim, 64])
pair_mask = pd.ones([1, len_dim, len_dim])

print('# [INFO] build and save static graph of Attention')
model = TemplatePair(channel_num, c, gc)
model.eval()
if not os.path.isfile(f_topo):
  if is_dynamic_input:
    len_dim = None
  net = to_static(model, input_spec=[
    InputSpec(shape=[1, len_dim, len_dim, 64]),
    InputSpec(shape=[1, len_dim, len_dim])
  ])
  save(net, prefix_weights)

print('# [INFO] inference on dynamic graph')
if not ignore_eval:
  dy_dts = 0.
  with pd.no_grad():
    for i in tqdm(range(n_iter)):
      t0 = time.time()
      _ = model(pair_act, pair_mask)
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
pair_act1 = np.ones([1, len_dim, len_dim, 64], dtype='float32')
pair_mask1 = np.ones([1, len_dim, len_dim], dtype='float32')

# 获取输入轴
input_names = predictor.get_input_names() # ['pair_act1', 'pair_mask1']
inputl_a = predictor.get_input_handle('pair_act')
inputl_b = predictor.get_input_handle('pair_mask')

# 变形输入轴
if is_dynamic_input:
  print('# [INFO] re-organize dynamic axes')
  inputl_a.reshape(pair_act1.shape)
  inputl_b.reshape(pair_mask1.shape)

# 获取输出轴
output_names = predictor.get_output_names() # ['tmp_0']
outputl = predictor.get_output_handle(output_names[0])

# run
dts = 0.
for i in tqdm(range(n_iter)):
  t0 = time.time()
  inputl_a.copy_from_cpu(pair_act1)
  inputl_b.copy_from_cpu(pair_mask1)

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