#!/bin/bash

source ~/.bashrc

batch_size="16"
lr="0.001"
regularization="0"
thread_num="8" # thread_num is for cpu, please set CUDA_VISIBLE_DEVICES for gpu
warmup_steps="0"
model_type="lstm" # candidate model_types: transformer, lstm, resnet
task="secondary_structure" # candidate tasks: pfam, secondary_structure, remote_homology, fluorescence, stability
model_config="./${model_type}_${task}_config.json"
model_dir="./models"
use_cuda="true" # candidates: true/false
distributed="false" # candidates: true/false
train_data="./toy_data/${task}/npz"
test_data="./toy_data/${task}/npz/valid"

# export PYTHONPATH="../../../../"

if [ "${distributed}" == "true" ]; then
    if [ "${use_cuda}" == "true" ]; then
        export FLAGS_sync_nccl_allreduce=1
        export FLAGS_fuse_parameter_memory_size=64
        export CUDA_VISIBLE_DEVICES="0,1"

        python -m paddle.distributed.launch \
            --log_dir log_dirs \
            ../train.py \
                --train_data ${train_data} \
                --test_data ${test_data} \
                --lr ${lr} \
                --thread_num ${thread_num} \
                --warmup_steps ${warmup_steps} \
                --batch_size ${batch_size} \
                --model_config ${model_config} \
                --model_dir ${model_dir} \
                --use_cuda \
                --distributed
    else
        echo "Only gpu is supported for distributed mode at present."
    fi
else
    if [ "${use_cuda}" == "true" ]; then
        export CUDA_VISIBLE_DEVICES="2"
        python ../train.py \
                --train_data ${train_data} \
                --test_data ${test_data} \
                --lr ${lr} \
                --thread_num ${thread_num} \
                --warmup_steps ${warmup_steps} \
                --batch_size ${batch_size} \
                --model_config ${model_config} \
                --model_dir ${model_dir} \
                --use_cuda
    else
        python ../train.py \
                --train_data ${train_data} \
                --test_data ${test_data} \
                --lr ${lr} \
                --thread_num ${thread_num} \
                --warmup_steps ${warmup_steps} \
                --batch_size ${batch_size} \
                --model_config ${model_config} \
                --model_dir ${model_dir}
    fi
fi