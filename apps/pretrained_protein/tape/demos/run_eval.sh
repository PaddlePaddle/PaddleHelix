#!/bin/bash

source ~/.bashrc

batch_size="256"
thread_num="8" # thread_num is for cpu, please set CUDA_VISIBLE_DEVICES for gpu
model_type="transformer" # candidate model_types: transformer, lstm, resnet
task="fluorescence" # candidate tasks: pfam, secondary_structure, remote_homology, fluorescence, stability
model_config="./${model_type}_${task}_config.json"
init_model="./models/epoch0"
use_cuda="true" # candidates: true/false
test_data="./toy_data/${task}/npz/valid"

export PYTHONPATH="../../../../"

if [ "${use_cuda}" == "true" ]; then
    export CUDA_VISIBLE_DEVICES="4"
    python ../eval.py \
            --test_data ${test_data} \
            --thread_num ${thread_num} \
            --batch_size ${batch_size} \
            --model_config ${model_config} \
            --init_model ${init_model} \
            --use_cuda
else
    python ../eval.py \
            --test_data ${test_data} \
            --thread_num ${thread_num} \
            --batch_size ${batch_size} \
            --model_config ${model_config} \
            --init_model ${init_model}
fi

