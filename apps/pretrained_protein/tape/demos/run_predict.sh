#!/bin/bash

source ~/.bashrc

batch_size="2"
thread_num="8" # thread_num is for cpu, please set CUDA_VISIBLE_DEVICES for gpu
model_type="transformer" # candidate model_types: transformer, lstm, resnet
task="fluorescence" # candidate tasks: secondary_structure, remote_homology, fluorescence, stability
model_config="./${model_type}_${task}_config.json"
init_model="./models/epoch5"
use_cuda="true" # candidates: true/false

export PYTHONPATH="../../../../"

if [ "${use_cuda}" == "true" ]; then
    export CUDA_VISIBLE_DEVICES="1"
    cat demo_animo_acid_sequences | \
    python ../predict.py \
            --thread_num ${thread_num} \
            --batch_size ${batch_size} \
            --model_config ${model_config} \
            --init_model ${init_model} \
            --use_cuda
else
    cat demo_animo_acid_sequences | \
    python ../predict.py \
            --thread_num ${thread_num} \
            --batch_size ${batch_size} \
            --model_config ${model_config} \
            --init_model ${init_model}
fi

