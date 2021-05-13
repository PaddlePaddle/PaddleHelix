#!/bin/bash
cd $(dirname $0)
cd ..

source ~/.bashrc

root_path="$(pwd)/../../.."

export PYTHONPATH="$root_path/":$PYTHONPATH

python sample.py \
        --nsample 1000 \
        --vocab data/zinc/vocab.txt \
        --model vae_models/model.iter-441000 \
        --config configs/config.json \
        --output sampling_output.txt 


