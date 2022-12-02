import pdb
from layers.subnets import EmbeddingsAndEvoformer
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

parser = Parser('[pd.infer] UT of pdpd.Attention')
parser.add_argument('--n_cpus', type=int, default=64, help='physical cores used during pd.infer')
args = parser.parse_args()
n_cpus = args.n_cpus

cfg = model_config('model_1')
c = cfg['model']['embeddings_and_evoformer']
gc = cfg['model']['global_config']

n_warm = 3
n_iter = 13
ignore_eval = False
is_dynamic_input = False
prefix_weights = 'dynamic_params/embeddingsandevoformer' if is_dynamic_input else 'static_params/embeddingsandevoformer'
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
with open('/home/yangw/experiments/helix_fold/T1042/model_1_input.pkl', 'rb') as h:
  sample = pickle.load(h)
len_dim = 40
batch = {
  'target_feat': pd.ones([1, len_dim, 22]),
  'msa_feat': pd.ones([1, 508, len_dim, 49]), # 508 -> 512
  'seq_mask': pd.ones([1, len_dim]),
  'aatype': pd.ones([1, len_dim]),
  'prev_pos': pd.ones([1, len_dim, 37, 3]),
  'prev_msa_first_row': pd.ones([1, len_dim, 256]),
  'prev_pair': pd.ones([1, len_dim, len_dim, 128]),
  'residue_index': pd.ones([1, len_dim]),
  'template_mask': pd.ones([1, 4]),
  'template_aatype': pd.ones([1, 4, len_dim], dtype="int32"), # define
  'template_pseudo_beta_mask': pd.ones([1, 4, len_dim]),
  'template_pseudo_beta': pd.ones([1, 4, len_dim, 3]),
  'template_all_atom_positions': pd.ones([1, 4, len_dim, 37, 3]),
  'template_all_atom_masks': pd.ones([1, 4, len_dim, 37]),
  'extra_msa': pd.ones([1, 5120, len_dim]),
  'extra_has_deletion': pd.ones([1, 5120, len_dim]),
  'extra_deletion_value': pd.ones([1, 5120, len_dim]),
  'extra_msa_mask': pd.ones([1, 5120, len_dim]),
  'msa_mask': pd.ones([1, 508, len_dim]),
  # 'torsion_angles_sin_cos': pd.ones([1, 4, len_dim, 7, 2]),
  # 'alt_torsion_angles_sin_cos': pd.ones([1, 4, len_dim, 7, 2]),
  # 'torsion_angles_mask': pd.ones([1, 4, len_dim, 7])
}


print('# [INFO] build and save static graph of Attention')
model = EmbeddingsAndEvoformer(channel_num, c, gc)
model.eval()
with pd.no_grad():
  _ = model(**batch)
if not os.path.isfile(f_topo):
  if is_dynamic_input:
    len_dim = None
  net = to_static(model, input_spec=[
    InputSpec(shape=[1, len_dim, 22],name='target_feat'),
    InputSpec(shape=[1, 508, len_dim, 49],name='msa_feat'),
    InputSpec(shape=[1, len_dim],name='seq_mask'),
    InputSpec(shape=[1, len_dim],name='aatype'),
    InputSpec(shape=[1, len_dim],name='residue_index'),
    InputSpec(shape=[1, 4],name='template_mask'),
    InputSpec(shape=[1, 4, len_dim],name='template_aatype', dtype="int32"),
    InputSpec(shape=[1, 4, len_dim],name='template_pseudo_beta_mask'),
    InputSpec(shape=[1, 4, len_dim, 3],name='template_pseudo_beta'),
    InputSpec(shape=[1, 4, len_dim, 37, 3],name='template_all_atom_positions'),
    InputSpec(shape=[1, 4, len_dim, 37],name='template_all_atom_masks'),
    InputSpec(shape=[1, 5120, len_dim],name='extra_msa'),
    InputSpec(shape=[1, 5120, len_dim],name='extra_has_deletion'),
    InputSpec(shape=[1, 5120, len_dim],name='extra_deletion_value'),
    InputSpec(shape=[1, 5120, len_dim],name='extra_msa_mask'),
    InputSpec(shape=[1, 508, len_dim],name='msa_mask'),
    InputSpec(shape=[1, len_dim, 37, 3],name='prev_pos'),
    InputSpec(shape=[1, len_dim, 256],name='prev_msa_first_row'),
    InputSpec(shape=[1, len_dim, len_dim, 128],name='prev_pair'),
    # InputSpec(shape=[1, 4, len_dim, 7, 2]),
    # InputSpec(shape=[1, 4, len_dim, 7, 2]),
    # InputSpec(shape=[1, 4, len_dim, 7])
  ])
  save(net, prefix_weights)


print('# [INFO] inference on dynamic graph')
if not ignore_eval:
  dy_dts = 0.
  with pd.no_grad():
    for i in range(n_iter):
      t0 = time.time()
      _ = model(**batch)
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
target_feat_1 = np.ones([1, len_dim, 22], dtype='float32')
msa_feat_1 = np.ones([1, 508, len_dim, 49], dtype='float32')
seq_mask_1 = np.ones([1, len_dim], dtype='float32')
aatype_1 = np.ones([1, len_dim], dtype='int32')
prev_pos_1 = np.ones([1, len_dim, 37, 3], dtype='float32')
prev_msa_first_row_1 = np.ones([1, len_dim, 256], dtype='float32')
prev_pair_1 = np.ones([1, len_dim, len_dim, 128], dtype='float32')
residue_index_1 = np.ones([1, len_dim], dtype='float32')
template_mask_1 = np.ones([1, 4], dtype='float32')
template_aatype_1 = np.ones([1, 4, len_dim], dtype="int32")
template_pseudo_beta_mask_1 = np.ones([1, 4, len_dim], dtype='float32')
template_pseudo_beta_1 = np.ones([1, 4, len_dim, 3], dtype='float32')
template_all_atom_positions_1 = np.ones([1, 4, len_dim, 37, 3], dtype='float32')
template_all_atom_masks_1 = np.ones([1, 4, len_dim, 37], dtype='float32')
extra_msa_1 = np.ones([1, 5120, len_dim], dtype='float32')
extra_has_deletion_1 = np.ones([1, 5120, len_dim], dtype='float32')
extra_deletion_value_1 = np.ones([1, 5120, len_dim], dtype='float32')
extra_msa_mask_1 = np.ones([1, 5120, len_dim], dtype='float32')
msa_mask_1 = np.ones([1, 508, len_dim], dtype='float32')

# 获取输入轴
input_names = predictor.get_input_names()
inputl_ta   = predictor.get_input_handle('target_feat')
inputl_msf  = predictor.get_input_handle('msa_feat')
inputl_se   = predictor.get_input_handle('seq_mask')
inputl_aa   = predictor.get_input_handle('aatype')
inputl_prpo = predictor.get_input_handle('prev_pos')
inputl_prms = predictor.get_input_handle('prev_msa_first_row')
inputl_prpa = predictor.get_input_handle('prev_pair')
inputl_re   = predictor.get_input_handle('residue_index')
inputl_tema = predictor.get_input_handle('template_mask')
inputl_teaa = predictor.get_input_handle('template_aatype')
inputl_tepm = predictor.get_input_handle('template_pseudo_beta_mask')
inputl_tepb = predictor.get_input_handle('template_pseudo_beta')
inputl_teap = predictor.get_input_handle('template_all_atom_positions')
inputl_team = predictor.get_input_handle('template_all_atom_masks')
inputl_exms = predictor.get_input_handle('extra_msa')
inputl_exha = predictor.get_input_handle('extra_has_deletion')
inputl_exde = predictor.get_input_handle('extra_deletion_value')
inputl_exmm = predictor.get_input_handle('extra_msa_mask')
inputl_msm  = predictor.get_input_handle('msa_mask')

# # 变形输入轴
# if is_dynamic_input:
#   print('# [INFO] re-organize dynamic axes')
#   inputl_q.reshape(q_mat1.shape)
#   inputl_m.reshape(m_mat1.shape)
#   inputl_b.reshape(bias1.shape)

# 获取输出轴
output_names = predictor.get_output_names()
outputl = predictor.get_output_handle('tmp_2')

# run
dts = 0.
for i in tqdm(range(n_iter)):
  t0 = time.time()
  inputl_ta.copy_from_cpu(target_feat_1)
  inputl_msf.copy_from_cpu(msa_feat_1)
  inputl_se.copy_from_cpu(seq_mask_1)
  inputl_aa.copy_from_cpu(aatype_1)
  inputl_prpo.copy_from_cpu(prev_pos_1)
  inputl_prms.copy_from_cpu(prev_msa_first_row_1)
  inputl_prpa.copy_from_cpu(prev_pair_1)
  inputl_re.copy_from_cpu(residue_index_1)
  inputl_tema.copy_from_cpu(template_mask_1)
  inputl_teaa.copy_from_cpu(template_aatype_1)
  inputl_tepm.copy_from_cpu(template_pseudo_beta_mask_1)
  inputl_tepb.copy_from_cpu(template_pseudo_beta_1)
  inputl_teap.copy_from_cpu(template_all_atom_positions_1)
  inputl_team.copy_from_cpu(template_all_atom_masks_1)
  inputl_exms.copy_from_cpu(extra_msa_1)
  inputl_exha.copy_from_cpu(extra_has_deletion_1)
  inputl_exde.copy_from_cpu(extra_deletion_value_1)
  inputl_exmm.copy_from_cpu(extra_msa_mask_1)
  inputl_msm.copy_from_cpu(msa_mask_1)

  predictor.run()
  output = outputl.copy_to_cpu()
  t1 = time.time()
  dt = t1 - t0
  if i >= n_warm:
    dts += dt

# if not ignore_eval:
#   print('# [dynamic-graph] avg inference time = {}'.format(dy_dts/(n_iter-n_warm)))
# print('# [static-graph] avg inference time = {}'.format(dts/(n_iter-n_warm)))

# print(output.shape)