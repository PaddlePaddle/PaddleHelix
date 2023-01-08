import paddle as pd
from paddle import nn
from paddle.jit import to_static, save
from paddle.static import InputSpec
from paddle import inference as pdinfer
import os
import pdb
import warnings


class StaticModule(object):
  def __init__(self, 
    config:dict, 
    global_config:dict,
    pdmodule:nn.Layer,
    feed_dict:dict, 
    module_prefix:str,
    n_cpus:int,
    channel_num:dict, 
    root_weights:str = 'static_params',
    is_pdinfer_init:bool = True
  ) -> None:
    # basic hyper-params
    self.c = config
    self.gc = global_config
    self.module_prefix = module_prefix
    self.channel_num = channel_num

    f_topo = os.path.join(root_weights, '{}.pdmodel'.format(self.module_prefix))
    f_bin  = os.path.join(root_weights, '{}.pdiparams'.format(self.module_prefix))
    if not (os.path.isfile(f_topo) and os.path.isfile(f_bin)):
      print('# [statc_basics] build static model')
      pdmodule.eval()
      warnings.filterwarnings('ignore', 'DAP comm')
      specs = [InputSpec(shape=list(v.shape), dtype=v.dtype, name=k) for k, v in feed_dict.items()]
      net = to_static(pdmodule, input_spec=specs)
      save(net, os.path.join(root_weights,self.module_prefix))
      warnings.resetwarnings()
    
    print('# [statc_basics] build executor of static model')
    pd_cfg = pdinfer.Config(f_topo, f_bin)
    if not is_pdinfer_init:
      pd_cfg.set_cpu_math_library_num_threads(n_cpus)
      pd_cfg.enable_mkldnn()
    self.predictor = pdinfer.create_predictor(pd_cfg)

    print('# [statc_basics] build input ports')
    self.input_names = self.predictor.get_input_names()
    self.input_ports = {}
    for k in self.input_names:
      assert k in feed_dict.keys()
      self.input_ports[k] = self.predictor.get_input_handle(k)
    
    print('# [statc_basics] build output ports')
    self.output_names = self.predictor.get_output_names()
    self.output_ports = {}
    for k in self.output_names:
      self.output_ports[k] = self.predictor.get_output_handle(k)

  def __call__(self, feed_dict:dict) -> dict:
    for k, input_port in self.input_ports.items():
      input_port.copy_from_cpu(feed_dict[k])
    self.predictor.run()
    return {k:output_port.copy_to_cpu() for k, output_port in self.output_ports.items()}


class JitModule(object):
  def __init__(self, 
    config:dict, 
    global_config:dict,
    pdmodule:nn.Layer,
    feed_dict:dict, 
    module_prefix:str,
    n_cpus:int,
    channel_num:dict, 
    root_weights:str = 'static_params',
    is_pdinfer_init:bool = True
  ) -> None:
    # basic hyper-params
    self.c = config
    self.gc = global_config
    self.module_prefix = module_prefix
    self.channel_num = channel_num
    self.n_cpus = n_cpus
    self.is_pdinfer_init = is_pdinfer_init

    self.f_topo = os.path.join(root_weights, '{}.pdmodel'.format(self.module_prefix))
    self.f_bin  = os.path.join(root_weights, '{}.pdiparams'.format(self.module_prefix))
    if not (os.path.isfile(self.f_topo) and os.path.isfile(self.f_bin)):
      print('# [statc_basics] build static model')
      pdmodule.eval()
      warnings.filterwarnings('ignore', 'DAP comm')
      specs = [InputSpec(shape=list(v.shape), dtype=v.dtype, name=k) for k, v in feed_dict.items()]
      net = to_static(pdmodule, input_spec=specs)
      save(net, os.path.join(root_weights,self.module_prefix))
      warnings.resetwarnings()
  
  
  def __call__(self, feed_dict:dict) -> dict:
    # print('# [{}.basics] build JIT graph'.format(self.module_prefix))
    pd_cfg = pdinfer.Config(self.f_topo, self.f_bin)
    if not self.is_pdinfer_init:
      pd_cfg.set_cpu_math_library_num_threads(self.n_cpus)
      pd_cfg.enable_mkldnn()
    predictor = pdinfer.create_predictor(pd_cfg)

    # print('# [{}.basics] build input ports'.format(self.module_prefix))
    input_names = predictor.get_input_names()
    input_ports = {}
    for k in input_names:
      assert k in feed_dict.keys()
      input_ports[k] = predictor.get_input_handle(k)
    
    # print('# [{}.basics] build output ports'.format(self.module_prefix))
    output_names = predictor.get_output_names()
    output_ports = {}
    for k in output_names:
      output_ports[k] = predictor.get_output_handle(k)
    for k, input_port in input_ports.items():
      input_port.copy_from_cpu(feed_dict[k])
    predictor.run()
    return {k:pd.Tensor(output_port.copy_to_cpu()) for k, output_port in output_ports.items()}
