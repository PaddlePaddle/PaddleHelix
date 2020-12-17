#!/bin/bash
cd $(dirname $0)
cd ..

source ~/.bashrc

root_path="../../.."
export FLAGS_fraction_of_gpu_memory_to_use=0.6
export PYTHONPATH="$(pwd)/$root_path/":$PYTHONPATH

dataset=chembl_filtered
data_path=$root_path/data/chem_dataset/$dataset/raw

CUDA_VISIBLE_DEVICES=0 python pretrain_supervised.py \
		--use_cuda \
		--max_epoch=100 \
		--data_path=$data_path \
		--model_config=gnn_model.json \
		--model_dir=$root_path/output/pretrain_gnns/pretrain_supervised/$dataset
