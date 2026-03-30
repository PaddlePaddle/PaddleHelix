#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils."""

import os
import re
from os.path import basename, exists
import subprocess
import shutil
import numbers
import tarfile
from collections.abc import Mapping, Sequence
import numpy as np
import paddle
from paddle.framework import core
import pickle


import shutil
import tarfile
from pathlib import Path

import paddle.distributed as dist
from utils.interface_utils import InterfaceInfo
from datetime import datetime

def get_custom_amp_list():
    """tbd."""

    black_list = {"reduce_sum"}
    white_list = {
        "one_hot", 
        "concat",
        "dropout_nd",
        "einsum",
        "elementwise_add",
        "elementwise_div",
        "elementwise_mul",
        "elementwise_sub",
        "fill_any_like",
        "fill_constant",
        "fused_gate_attention",
        "fused_gemm_epilogue",
        "gather",
        "gaussian_random",
        "layer_norm",
        "log_softmax",
        "matmul_v2",
        "p_norm",
        "py_layer",
        "relu",
        "scale",
        "sigmoid",
        "slice",
        "softmax",
        "softplus",
        "split",
        "split_with_num",
        "sqrt",
        "square",
        "stack",
        "sum",
        "transpose2",
        "unsqueeze2",
        "unstack",
        "where"
    }
    return black_list, white_list


def get_structure_module_bf16_op_list():
    black_list = {
        "clip",
        "dropout_nd",
        "elementwise_add",
        "elementwise_div",
        "elementwise_mul",
        "elementwise_sub",
        "fill_any_like",
        "fill_constant",
        "fused_gate_attention",
        "fused_gemm_epilogue",
        "gaussian_random",
        "linspace",
        "log_softmax",
        "p_norm",
        "py_layer",
        "reduce_mean",
        "reduce_min",
        "reduce_prod",
        "reduce_sum",
        "scale",
        "sigmoid",
        "softmax",
        "softplus",
        "sqrt",
        "square",
        "squared_l2_norm",
        "sum",
        "uniform_random",
    }
    white_list = {
        "abs",
        "bitwise_or",
        "concat",
        "elementwise_max",
        "elementwise_min",
        "equal", 
        "eye",
        "gather",
        "greater_than",
        "layer_norm",
        "less_than",
        "matmul_v2",
        "one_hot_v2",
        "reduce_max",
        "relu",
        "reshape2",
        "slice", 
        "split",
        "squeeze2",
        "stack",
        "transpose2",
        "unsqueeze2",
        "tile",
    }
    return black_list, white_list 


def get_model_parameter_size(model):
    """tbd"""
    size = 0
    for param in model.parameters():
        size += np.product(param.shape)
    return size


def tree_map(f, d):
    new_d = {}
    for k in d:
        if type(d[k]) is dict:
            new_d[k] = tree_map(f, d[k])
        else:
            new_d[k] = f(d[k])
    return new_d


def tree_flatten(d):
    new_d = {}
    for k in d:
        if type(d[k]) is dict:
            cur_d = tree_flatten(d[k])
            for sub_k, sub_v in cur_d.items():
                new_d[f'{k}.{sub_k}'] = sub_v
        else:
            new_d[k] = d[k]
    return new_d


def tree_filter(key_cond, value_cond, d):
    new_d = {}
    for k in d:
        if not key_cond is None and not key_cond(k):
            continue
        if not value_cond is None and not value_cond(d[k]):
            continue

        if type(d[k]) is dict:
            cur_d = tree_filter(key_cond, value_cond, d[k])
            if len(cur_d) != 0:
                new_d[k] = cur_d
        else:
            new_d[k] = d[k]
    return new_d


def add_to_data_writer(data_writer, step, results, prefix=''):
    """tbd"""
    print("step:%d %s:%s" % (step, prefix, str(results)))
    if data_writer is None:
        return
    for k, v in results.items():
        data_writer.add_scalar("%s/%s" % (prefix, k), v, step)


def upload_to_hadoop(args, cur_step):
    def _upload_file(local_file, hadoop_dir):
        assert len(hadoop_dir) > 10, \
            f"hadoop_dir ({hadoop_dir}) is too short"
        if not exists(local_file):
            return
        file_name = basename(local_file)
        os.system(f"{hadoop_fs} -mkdir {hadoop_dir}")
        os.system(f"{hadoop_fs} -rmr {hadoop_dir}/{file_name}")
        # upload file in background
        os.system(f"{hadoop_fs} -put {local_file} {hadoop_dir}/{file_name} &")    

    hadoop_fs = os.environ["HADOOP_FS"]
    output_path = os.environ["OUTPUT_PATH"]

    # upload models
    _upload_file(
            f'{args.model_dir}/step_{cur_step}.pdparams', 
            f'{output_path}/models')
    _upload_file(
            f'{args.model_dir}/step_{cur_step}_ema.pdparams', 
            f'{output_path}/models')

    # upload tensorboard log
    files = os.listdir(f'{args.log_dir}/tensorboard_log_dir')
    for file in files:
        _upload_file(
                f'{args.log_dir}/tensorboard_log_dir/{file}', 
                f'{output_path}/log/tensorboard_log_dir')


def upload_interface_probs_to_hadoop(log_dir, cur_step):
    def _upload_file(local_file, hadoop_dir):
        assert len(hadoop_dir) > 10, \
            f"hadoop_dir ({hadoop_dir}) is too short"
        if not exists(local_file):
            return
        file_name = basename(local_file)
        os.system(f"{hadoop_fs} -mkdir -p {hadoop_dir}")
        os.system(f"{hadoop_fs} -rmr {hadoop_dir}/{file_name}")
        # upload file in background
        os.system(f"{hadoop_fs} -put {local_file} {hadoop_dir}/{file_name} &")    

    hadoop_fs = os.environ["HADOOP_FS"]
    output_path = os.environ["OUTPUT_PATH"]

    # Find and upload interface_probs tar files
    interface_probs_files = find_interface_probs_files(log_dir, cur_step)
    for subdir, tar_file_path in interface_probs_files.items():
        _upload_file(
                tar_file_path,
                f'{output_path}/interface_probs/{subdir}/{cur_step}')


def find_interface_probs_files(log_dir, cur_step):
    result = {}

    # 遍历log_dir目录下的所有子目录
    for subdir in os.listdir(log_dir):
        subdir_path = os.path.join(log_dir, subdir)
        step_dir_path = os.path.join(subdir_path, str(cur_step))

        # 检查是否符合{log_dir}/XXX/{cur_step}的模式
        if os.path.isdir(step_dir_path):
            repeat_dirs = [d for d in os.listdir(step_dir_path) if d.startswith('repeat_')]

            # 如果找到repeat_子目录
            if repeat_dirs:
                interface_prob_avg_dir = os.path.join(step_dir_path, 'interface_prob_avg')
                os.makedirs(interface_prob_avg_dir, exist_ok=True)

                file_dict = {}

                for repeat_dir in repeat_dirs:
                    repeat_dir_path = os.path.join(step_dir_path, repeat_dir)

                    # 寻找以-interface-prob-step0.pkl为后缀的文件
                    for file in os.listdir(repeat_dir_path):
                        if file.endswith('-interface-prob-step0.pkl'):
                            file_path = os.path.join(repeat_dir_path, file)
                            relative_file_path = os.path.join('interface_prob_avg', file)

                            # Load the numpy array from the .pkl file
                            with open(file_path, 'rb') as f:
                                arr = pickle.load(f)

                            # Add or average the numpy array
                            if file in file_dict:
                                file_dict[file].append(arr)
                            else:
                                file_dict[file] = [arr]

                # Compute the average and save the files
                for file, arrays in file_dict.items():
                    avg_array = np.mean(arrays, axis=0)
                    avg_file_path = os.path.join(interface_prob_avg_dir, file)
                    with open(avg_file_path, 'wb') as f:
                        pickle.dump(avg_array, f)

                # 归档为tar文件
                tar_file_path = os.path.join(step_dir_path, f"interface_prob_avg.tar")
                with tarfile.open(tar_file_path, "w") as tar:
                    tar.add(interface_prob_avg_dir, arcname=os.path.basename(interface_prob_avg_dir))

                # 添加到结果字典
                result[subdir] = tar_file_path

    return result


def download_interface_prob_files(hdfs_dir, local_dir, step_num=None):
    hadoop_fs = os.environ["HADOOP_FS"]

    if step_num is not None:
        # Use the provided step_num as the directory to download
        max_dir_name = str(step_num)
        max_dir_path = f"{hdfs_dir}/{max_dir_name}"
    else:
        # List all subdirectories in the HDFS directory
        list_command = f"{hadoop_fs} -ls {hdfs_dir}"
        result = subprocess.run(list_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print(f"Error listing directory: {result.stderr}")
            return
        
        numeric_dirs = {}
        for line in result.stdout.splitlines():
            print(line)
            parts = line.split()
            if len(parts) > 0 and os.path.basename(parts[-1]).isdigit():
                base_name_number = int(os.path.basename(parts[-1]))
                numeric_dirs[base_name_number] = parts[-1]
        
        if not numeric_dirs:
            print("No numeric directories found.")
            return
        
        # Find the directory name with the largest numerical value
        max_dir_name = str(max(numeric_dirs.keys()))
        max_dir_path = f"{hdfs_dir}/{max_dir_name}"

    # Check if interface_prob_avg.tar exists in max_dir_path
    check_command = f"{hadoop_fs} -ls {max_dir_path}"
    check_result = subprocess.run(check_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check_result.returncode != 0:
        print(f"Error checking directory: {check_result.stderr}")
        return
    contains_target_file = any(line.endswith('interface_prob_avg.tar') for line in check_result.stdout.splitlines())
    if not contains_target_file:
        print(f"Directory {max_dir_name} does not contain 'interface_prob_avg.tar'. Skipping download.")
        return
    
    # Ensure the local target directory exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    local_max_dir_path = os.path.join(local_dir, max_dir_name)
    if not os.path.exists(local_max_dir_path):
        rank_id = dist.get_rank()
        if rank_id == 0:
            # Download the directory to the local target directory
            print(f"Downloading directory {max_dir_path} to {local_max_dir_path}...")
            get_command = f"{hadoop_fs} -get {max_dir_path} {local_max_dir_path}"
            get_result = subprocess.run(get_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if get_result.returncode != 0:
                print(f"Error downloading directory: {get_result.stderr}")
                return

            # Find and extract tar files
            for root, _, files in os.walk(local_max_dir_path):
                for file in files:
                    if file.endswith('.tar'):
                        tar_file_path = os.path.join(root, file)
                        print(f"Extracting {tar_file_path}...")
                        with tarfile.open(tar_file_path, 'r') as tar:
                            tar.extractall(root)

        # wait the main process finishing the downloading
        dist.barrier()
        print(f"[{get_timestamp()}] The interface prob files have been downloaded and extracted.")
    else:
        print(f"Directory {max_dir_name} already exists in {local_dir}.")
    
    return

def csv_print(d):
    keys = sorted(list(d.keys()))
    values = [str(d[k]) for k in keys]
    print(' '.join([str(x) for x in keys]))
    print(' '.join([str(x) for x in values]))


def multimer_collate_fn(data_list):
    chain_features = ['chain_aatypes', 'chain_seq_lengths']
    not_cat_features = ['crop_idx', 'crop_nums', 'chain_seq_lengths', 'msa', 'deletion_matrix', 'cluster_bias_mask']
    batch = {}
    for k1 in data_list[0].keys():
        if k1 == "label":
            batch[k1] = {}
            for k2 in data_list[0][k1].keys():
                batch[k1][k2] = [data[k1][k2] for data in data_list]
        elif isinstance(data_list[0][k1], dict):
            batch[k1] = {}
            for k2 in data_list[0][k1].keys():
                if k2 in chain_features:
                    continue
                if k2 in not_cat_features:
                    batch[k1][k2] = [data[k1][k2] for data in data_list]
                else:
                    batch[k1][k2] = np.concatenate([np.expand_dims(data[k1][k2], axis=0) for data in data_list], axis=0)
        elif isinstance(data_list[0][k1], str):
            batch[k1] = [data[k1] for data in data_list]

    for k in chain_features:
        if k not in batch['feat']:
            continue
        batch['feat'][k] = np.concatenate([data['feat'][k] for data in data_list], axis=1)
    
    if 'chain_aatypes' in batch['feat']:
        batch['feat']['chain_aatypes'] = \
            batch['feat']['chain_aatypes'].transpose([1, 0, 2])

    if 'chain_seq_lengths' in batch['feat']:
        batch['feat']['chain_seq_lengths'] = \
            batch['feat']['chain_seq_lengths'].transpose([1, 0])

    return batch


def all_atom_collate_fn(batch, name=None):
    """
    token-level features are of the same length,
    but atom-level features are of different length.

    Args:
        batch(list of sample data): batch should be a list of sample data.

    Returns:
        Batched data: batched each number, numpy array and paddle.Tensor
                      in input data.
    """
    def _pad_to_same_0_dim(array_list, pad_type='const', value=0):
        if array_list[0].ndim == 0:
            return array_list
        max_0dim = np.max([len(i) for i in array_list])
        res = []
        for i, array in enumerate(array_list):
            pad_shape = [max_0dim - len(array)] + list(array.shape[1:])
            if pad_type == 'const': 
                pad_tensor = np.full(pad_shape, value, dtype=array.dtype)
            elif pad_type == 'last':
                pad_tensor = np.full(pad_shape, array[-1], dtype=array.dtype)
            else:
                raise ValueError(pad_type)
            res.append(np.concatenate([array, pad_tensor], 0))
        return res

    sample = batch[0]
    if isinstance(sample, np.ndarray):
        batch = _pad_to_same_0_dim(batch)
        batch = np.stack(batch, axis=0)
        return batch
    elif isinstance(sample, (paddle.Tensor, core.eager.Tensor)):
        return paddle.stack(batch, axis=0)
    elif isinstance(sample, numbers.Number):
        batch = np.array(batch)
        return batch
    elif isinstance(sample, (str, bytes, InterfaceInfo)):
        return batch
    elif isinstance(sample, Mapping):
        new_sample = {}
        for key in sample:
            if key == 'ref_token2atom_idx':
                v = np.stack(_pad_to_same_0_dim(
                        [d[key] for d in batch], pad_type='last'), axis=0)
            else:
                v = all_atom_collate_fn([d[key] for d in batch], key)
            new_sample[key] = v
        return new_sample
    elif isinstance(sample, Sequence):
        sample_fields_num = len(sample)
        if not all(len(sample) == sample_fields_num for sample in iter(batch)):
            raise RuntimeError(
                "fileds number not same among samples in a batch"
            )
        return [all_atom_collate_fn(fields) for fields in zip(*batch)]

    print(f'[DEBUG] Sample with error type: {sample}, name={name} type= {type(sample)}')

    raise TypeError(
        "batch data con only contains: tensor, numpy.ndarray, "
        "dict, list, number, but got {}".format(type(sample))
    )


def sequence_pad(seq_array, max_size, pad_value=0):
    """
    seq_array: (L, *)
    return:
        (max_size, *)
    """
    L = list(seq_array.shape)[0]
    extra_dim = list(seq_array.shape)[1:]
    if max_size > L:
        pad_array = paddle.full([max_size - L] + extra_dim, pad_value, seq_array.dtype)
        return paddle.concat([seq_array, pad_array], 0)
    return seq_array


def pair_pad(pair_array, max_size, pad_value=0):
    """
    pair_array: (L, L, *)
    return:
        (max_size, max_size, *)
    """
    L = list(pair_array.shape)[0]
    extra_dim = list(pair_array.shape)[2:]
    assert L == pair_array.shape[1], pair_array.shape
    if max_size > L:
        pad_array = paddle.full([max_size - L, L] + extra_dim, pad_value, pair_array.dtype)
        new_array = paddle.concat([pair_array, pad_array], 0)  # (max_size, L, *)
        pad_array = paddle.full([max_size, max_size - L] + extra_dim, pad_value, pair_array.dtype)
        new_array = paddle.concat([new_array, pad_array], 1)  # (max_size, max_size, *)
        return new_array
    return pair_array


def get_grad_norm(parameters, norm_type=2):
    """get_grad_norm"""
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]

    support_norm_type = [float("inf"), 0, 1, 2]
    if norm_type not in support_norm_type:
        raise ValueError(f'norm_type only support {support_norm_type}')

    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return paddle.to_tensor(0.0)
    if norm_type == float("inf"):
        norms = [g.detach().abs().max() for g in grads]
        total_norm = (
            norms[0] if len(norms) == 1 else paddle.max(paddle.stack(norms))
        )
    else:
        total_norm = paddle.linalg.norm(
            paddle.stack(
                [paddle.linalg.norm(g.detach(), norm_type) for g in grads]
            ),
            norm_type,
        )
    return total_norm


def print_grad_statistics(named_parameters, prefix_list):
    """print_grad_statistics"""
    prefix_param_dict = {k: [] for k in prefix_list}
    for name, p in named_parameters:
        if p.grad is None:
            continue
        for prefix in prefix_list:
            if name.startswith(prefix):
                prefix_param_dict[prefix].append(p.grad.detach().std())
    
    print(f'Gradient statistics:')
    for prefix, values in prefix_param_dict.items():
        if len(prefix_param_dict[prefix]) > 0:
            print(f"{prefix}: {float(paddle.stack(values).mean()):.6f}")

def find_top_k_upper_triangle_indices(arr, K):
    # Get the size of the array
    N = arr.shape[0]
    
    # Get the indices of the upper triangle, excluding the diagonal
    upper_indices = np.triu_indices(N, k=1)
    
    # Extract the values from the upper triangle
    upper_values = arr[upper_indices]
    
    # Find the indices of the top K largest values
    top_k_indices = np.argsort(upper_values)[-K:][::-1]

    top_k_values = upper_values[top_k_indices]
    
    # Convert these indices back to the original array indices
    result_indices = np.vstack((upper_indices[0][top_k_indices], upper_indices[1][top_k_indices])).T
    
    return result_indices, top_k_values


def find_max_numeric_directory(base_dir):
    # Regular expression to match directories named with only digits
    numeric_dir_pattern = re.compile(r'^\d+$')
    
    max_dir_name = None
    max_dir_value = -1
    
    # Traverse the directory
    for entry in os.scandir(base_dir):
        if entry.is_dir():
            dir_name = entry.name
            if numeric_dir_pattern.match(dir_name):
                dir_value = int(dir_name)
                if dir_value > max_dir_value:
                    max_dir_value = dir_value
                    max_dir_name = os.path.join(base_dir, dir_name)
    
    return max_dir_name

def get_timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            

def random_choice_top_k_index_with_prob(probabilities, k=None, seed=None):
    """
    Randomly choose an index from the top K indices with the highest probabilities.
    
    Parameters:
    probabilities (list or numpy array): A list or array of probabilities.
    k (int): The number of top indices to consider.
    
    Returns:
    tuple: A tuple containing the chosen index and its corresponding probability.
    """
    
    # Sort indices based on probabilities in descending order
    sorted_indices = np.argsort(probabilities)[::-1]
    
    # Extract top K indices and their corresponding probabilities
    if k is None:
        k = len(probabilities)
    top_k_indices = sorted_indices[:k]
    top_k_probabilities = probabilities[top_k_indices]
    
    # Normalize the top K probabilities so they sum to 1
    top_k_probabilities /= top_k_probabilities.sum()
    
    # Randomly choose an index from the top K indices based on their normalized probabilities
    chosen_index_in_top_k = np.random.default_rng(seed).choice(len(top_k_indices), p=top_k_probabilities)
    
    # Get the original index and its corresponding probability from the input array
    chosen_original_index = top_k_indices[chosen_index_in_top_k]
    chosen_probability = probabilities[chosen_original_index]
    
    # Return the chosen index and its corresponding probability
    return chosen_original_index, chosen_probability

def find_top_k_upper_triangle_indices(arr, K):
    # Get the size of the array
    N = arr.shape[0]
    
    # Get the indices of the upper triangle, excluding the diagonal
    upper_indices = np.triu_indices(N, k=1)
    
    # Extract the values from the upper triangle
    upper_values = arr[upper_indices]
    
    # Find the indices of the top K largest values
    top_k_indices = np.argsort(upper_values)[-K:][::-1]

    top_k_values = upper_values[top_k_indices]
    
    # Convert these indices back to the original array indices
    result_indices = np.vstack((upper_indices[0][top_k_indices], upper_indices[1][top_k_indices])).T
    
    return result_indices, top_k_values
