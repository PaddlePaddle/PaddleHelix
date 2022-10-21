#!/bin/bash
cd $(dirname $0)
cd ..

train_pcqm(){
    mkdir -p log/$exp_name model/$exp_name
    python3 train_gem2.py  \
        --inference \
        --batch_size=$batch_size \
        --num_workers=10 \
        --max_epoch=150 \
        --dataset_config=$dataset_config \
        --data_cache_dir=$data_cache_dir \
        --model_config=$model_config \
        --encoder_config=$encoder_config \
        --train_config=$train_config \
        --init_model=$init_model \
        --start_step=$start_step \
        --model_dir=./model/$exp_name \
        --log_dir=./log/$exp_name
}


exp_name=""
batch_size=32
dataset_config="configs/dataset_configs/pcqmv2.json"
data_cache_dir="../data/pcqm4m-v2-rdkit3d"
model_config="configs/model_configs/mol_regr-optimus-mae.json"
encoder_config="configs/model_configs/opt3d_l12_c256.json"
init_model="./model/gem2_l12_c256.pdparams"
train_config="configs/train_configs/lr4e-4-mid40.json"
start_step=0
train_pcqm