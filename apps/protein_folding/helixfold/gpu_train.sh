#!/bin/bash
source ~/.bashrc

cd $(dirname $0)

root_path="$(pwd)"
# python_bin="/opt/compiler/gcc-8.2/lib/ld-linux-x86-64.so.2 --library-path /opt/compiler/gcc-8.2/lib:/usr/lib64:/home/ide/v2/conda/envs/cuda10.1_cudnn7.6.5/lib /home/liulihang/tools/paddle-dev/bin/python"
python_bin="/opt/conda/envs/helixfold/bin/python"
# python_bin="python3"

# export NCCL_DEBUG=INFO
export PYTHONPATH=$root_path:$PYTHONPATH
# export PADDLE_NODE_NUM=$PADDLE_TRAINERS_NUM
export PADDLE_NODE_NUM=1
TM_SCORE_BIN="$root_path/tools/tm_score"
LDDT_SCORE_BIN="$root_path/tools/lddt"
chmod +x $TM_SCORE_BIN
chmod +x $LDDT_SCORE_BIN

# disable C++ enisum, using python enisum
export FLAGS_new_einsum=0

train_af2() {
    start_step=0
    train_step=105
    CUDA_VISIBLE_DEVICES=0 $python_bin train.py \
            ${only_test} \
            --tm_score_bin="$TM_SCORE_BIN" \
            --lddt_score_bin="$LDDT_SCORE_BIN" \
            --data_config=${data_config} \
            --train_config=${train_config} \
            --model_name=${model_name} \
            --init_model=${init_model} \
            --start_step=${start_step} \
            --train_step=${train_step} \
            --precision=${precision} \
            --num_workers 6 \
            --seed 2022 \
            --batch_size=$batch_size \
            --dap_degree=$dap_degree \
            --bp_degree=$bp_degree \
            ${log_step} \
            ${eval_step} \
            ${save_step} \
            --model_dir="./debug_models" \
            --log_dir="./debug_log" \
            # &> ./debug_log/$exp_name.log
}


exp_name="demo"

mkdir -p debug_log debug_models

### 
{
    if [[ "$exp_name" == "demo" ]]; then
        batch_size=1
        dap_degree=1
        bp_degree=1
        train_config="./train_configs/demo.json"
        data_config="./data_configs/demo.json"
        model_name="initial"
        precision="bf16"
        # precision="fp32"
        log_step="--log_step=20"
        eval_step="--eval_step=1000"
        save_step="--save_step=1000"
        # init_model="$root_path/data/af2_pd_params/model_5.pdparams"
        train_af2
    fi
}
