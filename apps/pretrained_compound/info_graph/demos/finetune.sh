#!/bin/bash
cd $(dirname $0)

############
# config
############
root="/mnt/xueyang/Datasets/chem_dataset"
config="config_on_zinc15_chembl.json"

############
# use ctrl+c to kill parallel exec
############
trap killgroup SIGINT
killgroup(){
  echo killing...
  kill -9 0
}


exp=$1
init_model=$2
# dataset=$3

# python ../finetune.py \
#        --exp $exp \
#        --use_cuda \
#        --config $config \
#        --root $root \
#        --dataset $dataset \
#        --train_data "$root/$dataset/scaffold_npz/train" \
#        --valid_data "$root/$dataset/scaffold_npz/valid" \
#        --test_data "$root/$dataset/scaffold_npz/test" \
#        --model_dir "model_dir/${dataset}_finetune_"$exp \
#        --log_dir "log_dir/${dataset}_finetune_"$exp \
#        --init_model $init_model


for dataset in "bbbp" "tox21" "toxcast" "sider" "clintox" "muv" "hiv" "bace"
do
    echo "==================== $dataset ===================="
    python ../finetune.py \
           --exp $exp \
           --use_cuda \
           --config $config \
           --root $root \
           --dataset $dataset \
           --train_data "$root/$dataset/scaffold_npz/train" \
           --valid_data "$root/$dataset/scaffold_npz/valid" \
           --test_data "$root/$dataset/scaffold_npz/test" \
           --model_dir "model_dir/${dataset}_finetune_"$exp \
           --log_dir "log_dir/${dataset}_finetune_"$exp \
           --init_model $init_model
done
