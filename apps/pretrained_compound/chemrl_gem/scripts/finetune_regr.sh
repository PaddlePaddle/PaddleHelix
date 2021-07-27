#!/bin/bash
cd $(dirname $0)
cd ..

source ~/.bashrc
source ./scripts/utils.sh

root_path="$(pwd)/../../.."
export PYTHONPATH="$root_path/":$PYTHONPATH

datasets="esol freesolv lipophilicity qm7 qm8 qm9"

compound_encoder_config="model_configs/geognn_l8.json"
model_config="model_configs/down_mlp2.json"

init_model="./pretrain_models/chemrl_gem.pdparams"


for dataset in $datasets; do
	echo $dataset
	data_path="$root_path/data/chem_dataset/$dataset"
	cached_data_path="./cached_data/$dataset"
	if [ ! -f "$cached_data_path.done" ]; then
		rm -r $cached_data_path
		paddle2.0 finetune_regr.py \
				--task=data \
				--num_workers=10 \
				--dataset_name=$dataset \
				--data_path=$data_path \
				--cached_data_path=$cached_data_path \
				--compound_encoder_config=$compound_encoder_config \
				--model_config=$model_config
		if [ $? -ne 0 ]; then
			echo "Generate data failed for $dataset"
			exit 1
		fi
		touch $cached_data_path.done
	fi

	lrs_list="1e-3,1e-3 1e-3,4e-3 4e-3,4e-3"
	drop_list="0.1 0.2"
	if [ "$dataset" == "qm8" ] || [ "$dataset" == "qm9" ]; then
		batch_size=256
	elif [ "$dataset" == "freesolv" ]; then
		batch_size=30
	else
		batch_size=32
	fi
	run_times=4
	for lrs in $lrs_list; do
		IFS=, read -r -a array <<< "$lrs"
		lr=${array[0]}
		head_lr=${array[1]}
		for dropout_rate in $drop_list; do
			for time in $(seq 1 $run_times); do
				CUDA_VISIBLE_DEVICES=0 paddle2.0 finetune_regr.py \
						--batch_size=$batch_size \
						--max_epoch=100 \
						--dataset_name=$dataset \
						--data_path=$data_path \
						--cached_data_path=$cached_data_path \
						--split_type=scaffold \
						--compound_encoder_config=$compound_encoder_config \
						--model_config=$model_config \
						--init_model=$init_model \
						--model_dir=$root_path/output/chemrl_gem/finetune/$dataset \
						--encoder_lr=$lr \
						--head_lr=$head_lr \
						--dropout_rate=$dropout_rate
			done
		done
	done
done

