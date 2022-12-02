from layers.head import PredictedLDDTHead
from config import model_config
import paddle as pd
from paddle.jit import to_static, save
from paddle.static import InputSpec
from paddle import inference as pdinfer
import time
from tqdm import tqdm
import os
import numpy as np
from argparse import ArgumentParser as Parser

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
            'template_pair': 85, 
        }
cfg = model_config('model_1')
c = cfg['model']['heads']['predicted_lddt']
gc = cfg['model']['global_config']

n_warm = 3
n_iter = 1000 + n_warm
force_static_cvt = True
ignore_eval = False
is_dynamic_input = False
model_name = 'plddt' # 'plddt'
prefix_weights = 'dynamic_params/' + model_name if is_dynamic_input else 'static_params/' + model_name
f_topo = prefix_weights + '.pdmodel'
f_params = prefix_weights + '.pdiparams'

### create sample input
len_dim = 206
structure_module = pd.ones([1, 384])

print('# [INFO] build and save static graph of Attention')
model = PredictedLDDTHead(channel_num, c, gc)
model.eval()
if not os.path.isfile(f_topo) or force_static_cvt:
  if is_dynamic_input:
    len_dim = None
  net = to_static(model, input_spec=[InputSpec(shape=[1, 384])])
  save(net, prefix_weights)

print('# [INFO] inference on dynamic graph')
if not ignore_eval:
  dy_dts = 0.
  with pd.no_grad():
    for i in tqdm(range(n_iter)):
      t0 = time.time()
      _ = model(structure_module)
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
structure_module1 = np.ones([1, 384], dtype="float32")

# 获取输入轴
input_names = predictor.get_input_names()
inputl_structure = predictor.get_input_handle('structure_module')


# 变形输入轴
if is_dynamic_input:
  print('# [INFO] re-organize dynamic axes')
  inputl_structure.reshape(structure_module1.shape)

# 获取输出轴
output_names = predictor.get_output_names() # ['tmp_0']
outputl = predictor.get_output_handle(output_names[0])

# run
dts = 0.
for i in tqdm(range(n_iter)):
  t0 = time.time()
  inputl_structure.copy_from_cpu(structure_module1)
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