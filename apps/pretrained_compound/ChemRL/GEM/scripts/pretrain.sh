#!/bin/bash
cd $(dirname $0)
cd ..

source ~/.bashrc
source ./scripts/utils.sh

root_path="../../../.."
export PYTHONPATH="$(pwd)/$root_path/":$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

### download demo data
wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/compound_datasets/demo_zinc_smiles.tgz
tar xzf demo_zinc_smiles.tgz

### start pretrain
compound_encoder_config="model_configs/geognn_l8.json"
model_config="model_configs/pretrain_gem.json"
dataset="zinc"
data_path="./demo_zinc_smiles"
python pretrain.py \
		--batch_size=256 \
		--num_workers=4 \
		--max_epoch=50 \
		--lr=1e-3 \
		--dropout_rate=0.2 \
		--dataset=$dataset \
		--data_path=$data_path \
		--compound_encoder_config=$compound_encoder_config \
		--model_config=$model_config \
		--model_dir=$root_path/output/chemrl_gem/pretrain_models/$dataset
