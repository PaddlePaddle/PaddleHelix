from layers.embeddings import TemplateEmbedding
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
c = cfg['model']['embeddings_and_evoformer']['template']
gc = cfg['model']['global_config']

n_warm = 3
n_iter = 10 + n_warm
force_static_cvt = True
ignore_eval = False
is_dynamic_input = False
model_name = 'templateembedding'
prefix_weights = 'static_params/' + model_name if is_dynamic_input else 'static_params/' + model_name
f_topo = prefix_weights + '.pdmodel'
f_params = prefix_weights + '.pdiparams'

### create sample input
len_dim = 206
query_embedding = pd.ones([1, len_dim, len_dim, 128])
template_mask = pd.ones([1, 4])
template_aatype = pd.ones([1, len_dim], dtype='int32')
template_pseudo_beta_mask = pd.ones([1, len_dim])
template_pseudo_beta = pd.ones([1, len_dim, 3])
template_all_atom_positions = pd.ones([1, len_dim, 37, 3])
template_all_atom_masks = pd.ones([1, len_dim, 37])
mask_2d = pd.ones([1, len_dim, len_dim])

print('# [INFO] build and save static graph of Attention')
model = TemplateEmbedding(channel_num, c, gc)
model.eval()

print('# [INFO] inference on dynamic graph')
if not ignore_eval:
  dy_dts = 0.
  with pd.no_grad():
    for i in tqdm(range(n_iter)):
      t0 = time.time()
      _ = model(query_embedding,
                template_mask,
                template_aatype, 
                template_pseudo_beta_mask, 
                template_pseudo_beta, 
                template_all_atom_positions, 
                template_all_atom_masks, 
                mask_2d)
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
query_embedding1 = np.ones([len_dim, len_dim, 128], dtype="float32")
template_mask1 = np.ones([4])
template_aatype1 = np.ones([4, len_dim], dtype="int32")
template_pseudo_beta_mask1 = np.ones([4, len_dim], dtype="float32")
template_pseudo_beta1 = np.ones([4, len_dim, 3], dtype="float32")
template_all_atom_positions1 = np.ones([4, len_dim, 37, 3], dtype="float32")
template_all_atom_masks1 = np.ones([4, len_dim, 37], dtype="float32")
mask_2d1 = np.ones([len_dim, len_dim], dtype="float32")

# 获取输入轴
input_names = predictor.get_input_names()
inputl_qe = predictor.get_input_handle('query_embedding')
inputl_tm = predictor.get_input_handle('template_mask')
inputl_ta = predictor.get_input_handle('template_aatype')
inputl_tpbm = predictor.get_input_handle('template_pseudo_beta_mask')
inputl_tpb = predictor.get_input_handle('template_pseudo_beta')
inputl_taap = predictor.get_input_handle('template_all_atom_positions')
inputl_taam = predictor.get_input_handle('template_all_atom_masks')
inputl_m2d = predictor.get_input_handle('mask_2d')

# 变形输入轴
if is_dynamic_input:
  print('# [INFO] re-organize dynamic axes')
  inputl_qe.reshape(query_embedding1.shape)
  inputl_tm.reshape(template_mask1.shape)
  inputl_ta.reshape(template_aatype1.shape)
  inputl_tpbm.reshape(template_pseudo_beta_mask1.shape)
  inputl_tpb.reshape(template_pseudo_beta1.shape)
  inputl_taap.reshape(template_all_atom_positions1.shape)
  inputl_taam.reshape(template_all_atom_masks1.shape)
  inputl_m2d.reshape(mask_2d1.shape)

# 获取输出轴
output_names = predictor.get_output_names() # ['tmp_0']
outputl = predictor.get_output_handle(output_names[0])

# run
dts = 0.
for i in tqdm(range(n_iter)):
  t0 = time.time()
  inputl_qe.copy_from_cpu(query_embedding1)
  inputl_tm.copy_from_cpu(template_mask1)
  inputl_ta.copy_from_cpu(template_aatype1)
  inputl_tpbm.copy_from_cpu(template_pseudo_beta_mask1)
  inputl_tpb.copy_from_cpu(template_pseudo_beta1)
  inputl_taap.copy_from_cpu(template_all_atom_positions1)
  inputl_taam.copy_from_cpu(template_all_atom_masks1)
  inputl_m2d.copy_from_cpu(mask_2d1)
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