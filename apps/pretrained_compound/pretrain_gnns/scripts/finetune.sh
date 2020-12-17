#!/bin/bash
cd $(dirname $0)
cd ..

source ~/.bashrc

root_path="../../.."
export FLAGS_fraction_of_gpu_memory_to_use=0.6
export PYTHONPATH="$(pwd)/$root_path/":$PYTHONPATH

datasets="bace bbbp clintox hiv muv sider tox21 toxcast"
for dataset in $datasets; do
	data_path=$root_path/data/chem_dataset/$dataset/raw

	CUDA_VISIBLE_DEVICES=0 python finetune.py \
			--use_cuda \
			--batch_size=128 \
			--max_epoch=100 \
			--dataset_name=$dataset \
			--data_path=$data_path \
			--split_type=scaffold \
			--model_config=gnn_model.json \
			--model_dir=$root_path/output/pretrain_gnns/finetune/$dataset
	if [ $? -ne 0 ]; then
		echo "[FAILED] finetune $dataset"
		exit 1
	fi
done

