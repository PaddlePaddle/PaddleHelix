#!/bin/bash
cd $(dirname $0)
cd ..

############
# config
############
root="data"

train() {
    local dataset=$1
    local model_dir=$2
    local extra_args=${@:3}

    python train.py --device gpu \
           --train_data "$root/$dataset/processed/train/" \
           --test_data "$root/$dataset/processed/test/" \
           --model_config $config \
           --model_dir $model_dir \
           $extra_args
}

dataset=$1
config=$2

if [[ ! -e $config ]]; then
    echo "Cannot find "$config
    exit 1
fi

config_filename=$(basename "$config")
config_name="${config_filename%.*}"
model_dir="model_dir/"$dataset"_"$config_name
train $dataset $model_dir ${@:3}

# for dataset in "davis"
# do
#     config_filename=$(basename "$config")
#     config_name="${config_filename%.*}"
#     model_dir="model_dir/"$dataset"_"$config_name
#     train $dataset $model_dir
# done
