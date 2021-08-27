#!/bin/bash

model_type="resnet" # candidate model_types: transformer, lstm, resnet
task="secondary_structure" # candidate tasks: pfam, secondary_structure, remote_homology, fluorescence, stability
model_config="./configs/${model_type}_${task}_config.json"

train_data="./secondary_structure_toy_data/"
valid_data="./secondary_structure_toy_data/"

export PYTHONPATH="../../../"

python train.py \
        --train_data ${train_data} \
        --valid_data ${valid_data} \
        --model_config ${model_config} \
        --use_cuda
