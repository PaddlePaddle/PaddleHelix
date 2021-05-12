#!/bin/bash
cd $(dirname $0)
cd ..

source ~/.bashrc

root_path="$(pwd)/../../.."

export PYTHONPATH="$root_path/":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python -m vae_train.py \
        --train zinc_processed \
        --vocab data/zinc/vocab.txt \
        --config configs/config.json \
        --save_dir vae_models \
        --num_workers 2 \
        --epoch 50 \
        --batch_size 32 \
        --use_gpu True
        
