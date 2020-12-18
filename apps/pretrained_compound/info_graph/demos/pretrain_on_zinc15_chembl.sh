#!/bin/bash
cd $(dirname $0)

############
# config
############
root="data"
dataset="chembl_filtered,zinc_standard_agent"
config="config_on_zinc15_chembl.json"

############
# use ctrl+c to kill parallel exec
############
trap killgroup SIGINT
killgroup(){
  echo killing...
  kill -9 0
}

python ../unsupervised_pretrain.py \
       --task_name train \
       --use_cuda \
       --root $root \
       --dataset $dataset \
       --batch_size 1024 \
       --num_workers 16 \
       --model_dir "model_dir/zinc15_chembl_pretrain" \
       --config $config \
       --dont_save_emb
