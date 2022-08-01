import os
from os.path import basename
import numpy as np
import logging

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
###### sharding dependencies
from paddle.distributed.sharding import group_sharded_parallel, save_group_sharded_model
from paddle.fluid.dygraph.parallel import sync_params_buffers
from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients
######

def get_model_parameter_size(model):
    """tbd"""
    size = 0
    for param in model.parameters():
        size += np.product(param.shape)
    return size


def static_params_to_dygraph(model, static_tensor_dict):
    """Simple tool for convert static paramters to dygraph paramters dict.
    **NOTE** The model must both support static graph and dygraph mode.
    Args:
        model (nn.Layer): the model of a neural network.
        static_tensor_dict (string): path of which locate the saved paramters in static mode.
            Usualy load by `paddle.static.load_program_state`.
    Returns:
        [tensor dict]: a state dict the same as the dygraph mode.
    """
    state_dict = model.state_dict()
    # static_tensor_dict = paddle.static.load_program_state(static_params_path)

    ret_dict = dict()
    for n, p in state_dict.items():
        if p.name not in static_tensor_dict:
            logging.info("%s paramter is missing from you state dict." % n)
            continue
        ret_dict[n] = static_tensor_dict[p.name]

    return ret_dict


def dist_all_reduce(x, return_num=False, distributed=False):
    """tbd"""
    n = len(x)
    x_sum = 0 if n == 0 else np.sum(x)
    if distributed:
        n = dist.all_reduce(paddle.to_tensor(n, dtype='int64')).numpy()[0]
        x_sum = dist.all_reduce(paddle.to_tensor(x_sum, dtype='float32')).numpy()[0]
    x_mean = 0 if n == 0 else x_sum / n
    if return_num:
        return x_mean, n
    else:
        return x_mean


def dist_mean(x, distributed=False):
    """tbd"""
    n = len(x)
    x_sum = 0 if n == 0 else np.sum(x)
    if distributed:
        n = dist.all_reduce(paddle.to_tensor(n, dtype='int64')).numpy()[0]
        x_sum = dist.all_reduce(paddle.to_tensor(x_sum, dtype='float32')).numpy()[0]
    x_mean = 0 if n == 0 else x_sum / n
    return x_mean


def dist_sum(x, distributed=False):
    """tbd"""
    n = len(x)
    x_sum = 0 if n == 0 else np.sum(x)
    if distributed:
        x_sum = dist.all_reduce(paddle.to_tensor(x_sum, dtype='float32')).numpy()[0]
    return x_sum


def dist_length(x, distributed=False):
    """tbd"""
    n = len(x)
    if distributed:
        n = dist.all_reduce(paddle.to_tensor(n, dtype='int64')).numpy()[0]
    return n


def set_logging_level(level):
    level_dict = {
        "NOTSET": logging.NOTSET,
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s',
        level=level_dict[level],
        datefmt='%Y-%m-%d %H:%M:%S')


def add_to_data_writer(data_writer, step, results, prefix=''):
    """tbd"""
    logging.info("step:%d %s:%s" % (step, prefix, str(results)))
    if data_writer is None:
        return
    for k, v in results.items():
        data_writer.add_scalar("%s/%s" % (prefix, k), v, step)


class DistWrapper(object):
    def __init__(self, dp_degree, sharding_degree):
        self.dp_degree = dp_degree
        self.sharding_degree = sharding_degree

    def get_strategy(self):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": self.dp_degree,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": self.sharding_degree
        }
        return strategy

    def get_hcg(self):
        return fleet.get_hybrid_communicate_group()

    def get_dp_group(self):
        return self.get_hcg().get_data_parallel_group()
    
    def get_sharding_group(self):
        return self.get_hcg().get_sharding_parallel_group()

    def get_rank_and_world_size(self):
        hcg = self.get_hcg()
        global_rank = hcg.get_global_rank()
        mp_rank = hcg.get_model_parallel_rank()
        pp_rank = hcg.get_stage_id()
        dp_rank = hcg.get_data_parallel_rank()
        sharding_rank = hcg.get_sharding_parallel_rank()

        sharding_size = hcg.get_sharding_parallel_world_size()
        data_world_rank = dp_rank * sharding_size + sharding_rank
        data_world_size = self.dp_degree * self.sharding_degree
        return data_world_rank, data_world_size
    
    def wrap_model(self, model, optimizer, scaler=None):
        dp_group = self.get_dp_group()
        sharding_group = self.get_sharding_group()

        if self.dp_degree > 1:
            # sync_params_buffers(model, comm_group=dp_group, src_rank=dp_group.ranks[0])
            for param in model.parameters():
                paddle.distributed.broadcast(param,
                    src=dp_group.ranks[0],
                    group=dp_group,
                    use_calc_stream=True)

        if self.sharding_degree > 1:
            model, optimizer, scaler = group_sharded_parallel(model, optimizer, level="p_g_os", scaler=None, group=sharding_group)
        return model, optimizer

    def sync_grad(self, model):
        hcg = self.get_hcg()
        dp_group = self.get_dp_group()

        if self.dp_degree > 1:
            fused_allreduce_gradients(model.parameters(), hcg)
            for p in model.parameters():
                if hasattr(p, "bw_storage"):
                    assert p.grad is None, "This case shouldn't happen."
                    p.bw_storage.scale_(1.0 / dp_group.nranks)
                    paddle.distributed.all_reduce(p.bw_storage, group=dp_group)
