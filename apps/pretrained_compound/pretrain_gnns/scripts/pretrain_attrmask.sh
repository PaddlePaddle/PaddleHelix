#!/bin/bash
cd $(dirname $0)
cd ..

source ~/.bashrc
source ./scripts/utils.sh

root_path="$(pwd)/../../.."
export FLAGS_fraction_of_gpu_memory_to_use=0.6
export PYTHONPATH="$root_path/":$PYTHONPATH

pretrain_attrmask() {
	compound_encoder_config="model_configs/pregnn_paper.json"
	# compound_encoder_config="model_configs/pregnn_feat_l8_readavg.json"
	model_config="model_configs/pre_Attrmask.json"
	data_path=$root_path/data/chem_dataset/zinc_standard_agent
	exp_name="pregnn_paper-pre_Attrmask"
	echo $exp_name

	CUDA_VISIBLE_DEVICES=3 paddle2.0 pretrain_attrmask.py \
			--batch_size=256 \
			--num_workers=2 \
			--max_epoch=41 \
			--lr=1e-3 \
			--dropout_rate=0.2 \
			--data_path=$data_path \
			--compound_encoder_config=$compound_encoder_config \
			--model_config=$model_config \
			--init_model=$init_model \
			--model_dir=$root_path/output/pretrain_gnns/$exp_name > log/pretrain-$exp_name.txt 2>&1

	# CUDA_VISIBLE_DEVICES=2,3,4,5 paddle2.0 -m paddle.distributed.launch pretrain_attrmask.py \
	# 		--dist \
	# 		--batch_size=256 \
	# 		--max_epoch=2 \
	# 		--lr=1e-3 \
	# 		--dropout_rate=0.2 \
	# 		--data_path=$data_path \
	# 		--compound_encoder_config=$compound_encoder_config \
	# 		--model_config=$model_config \
	# 		--init_model=$init_model \
	# 		--model_dir=$root_path/output/pretrain_gnns/Attrmask
}

pretrain_attrmask

