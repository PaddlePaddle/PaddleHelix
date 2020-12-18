#!/bin/bash
cd $(dirname $0)

############
# config
############
runs=3
root="data"
config="unsupervised_pretrain_config.json"

############
# use ctrl+c to kill parallel exec
############
trap killgroup SIGINT
killgroup(){
  echo killing...
  kill -9 0
}


train() {
    local dataset=$1
    local model_dir=$2
    local emb_dir=$3

    python ../unsupervised_pretrain.py \
           --task_name train \
           --use_cuda \
           --root $root \
           --dataset $dataset \
           --model_dir $model_dir \
           --emb_dir $emb_dir \
           --config $config
}

eval_emb() {
    local model_dir=$1
    local emb_dir=$2

    python ../unsupervised_pretrain.py \
           --task_name eval \
           --model_dir $model_dir \
           --emb_dir $emb_dir
}

for dataset in "mutag" "ptc_mr"
do
    echo "=================================================="
    echo "Train and eval on "$dataset

    for run_id in $(seq 1 $runs)
    do
        echo "=================================================="
        echo "Run-"$run_id
        model_dir="model_dir/"$dataset"/run_"$run_id
        emb_dir="emb_dir/"$dataset"/run_"$run_id
        train $dataset $model_dir $emb_dir
        eval_emb $model_dir $emb_dir
    done
done
