#!/bin/bash
source ~/.bashrc

cd $(dirname $0)

root_path="$(pwd)"
conda activate helixfold
# python_bin="/opt/compiler/gcc-8.2/lib/ld-linux-x86-64.so.2 --library-path /opt/compiler/gcc-8.2/lib:/usr/lib64:/home/ide/v2/conda/envs/cuda10.1_cudnn7.6.5/lib /home/liulihang/tools/paddle-dev/bin/python"
python_bin="/opt/conda/envs/helixfold/bin/python"
# python_bin="python3"

# export NCCL_DEBUG=INFO
export PYTHONPATH=$root_path:$PYTHONPATH
# export PADDLE_NODE_NUM=$PADDLE_TRAINERS_NUM
# export PADDLE_NODE_NUM=1
TM_SCORE_BIN="$root_path/tools/tm_score"
LDDT_SCORE_BIN="$root_path/tools/lddt"
chmod +x $TM_SCORE_BIN
chmod +x $LDDT_SCORE_BIN

# disable C++ enisum, using python enisum
export FLAGS_new_einsum=0

train_af2_single() {
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


train_af2_distributed() {
    start_step=0
    train_step=105
    $python_bin -m paddle.distributed.launch train.py \
            --distributed \
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


exp_name="$1"
# exp_name="demo-initial" # model 1
# exp_name="demo-finetune" # model 1.1.1

mkdir -p debug_log debug_models

### Initial Training_N1C1
{
    if [[ "$exp_name" == "demo_initial_N1C1" ]]; then
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
        train_af2_single
    fi
}

### Finetune_N1C1
{
    if [[ "$exp_name" == "demo_finetune_N1C1" ]]; then
        export FLAGS_allocator_strategy=naive_best_fit
        export FLAGS_fraction_of_gpu_memory_to_use=0.92
        
        batch_size=1
        dap_degree=1
        bp_degree=1
        train_config="./train_configs/demo.json"
        data_config="./data_configs/demo.json"
        model_name="finetune"
        precision="bf16"
        # precision="fp32"
        log_step="--log_step=20"
        eval_step="--eval_step=1000"
        save_step="--save_step=1000"
        # init_model="$root_path/data/af2_pd_params/model_5.pdparams"
        train_af2_single
    fi
}

### Initial Training_N1C8
{
    if [[ "$exp_name" == "demo_initial_N1C8" ]]; then
        export PADDLE_NNODES=1
        export PADDLE_MASTER="127.0.0.1:12538" # set PADDLE_MASTER="xxx.xxx.xxx.xxx:port" according to your network environment
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

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
        train_af2_distributed
    fi
}

### Finetune_N1C8
{
    if [[ "$exp_name" == "demo_finetune_N1C8" ]]; then
        export FLAGS_allocator_strategy=naive_best_fit
        export FLAGS_fraction_of_gpu_memory_to_use=0.92
        export PADDLE_NNODES=1
        export PADDLE_MASTER="127.0.0.1:12538" # set PADDLE_MASTER="xxx.xxx.xxx.xxx:port" according to your network environment
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        
        batch_size=1
        dap_degree=1
        bp_degree=1
        train_config="./train_configs/demo.json"
        data_config="./data_configs/demo.json"
        model_name="finetune"
        precision="bf16"
        # precision="fp32"
        log_step="--log_step=20"
        eval_step="--eval_step=1000"
        save_step="--save_step=1000"
        # init_model="$root_path/data/af2_pd_params/model_5.pdparams"
        train_af2_distributed
    fi
}

### Initial Training_N8C64
{
    if [[ "$exp_name" == "demo_initial_N8C64" ]]; then
        export PADDLE_NNODES=8 # set number of devices
        export PADDLE_MASTER="xxx.xxx.xxx.xxx:port" # set PADDLE_MASTER="xxx.xxx.xxx.xxx:port" according to your network environment
        # export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

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
        train_af2_distributed
    fi
}

### Finetune_N8C64
{
    if [[ "$exp_name" == "demo_finetune_N8C64" ]]; then
        export FLAGS_allocator_strategy=naive_best_fit
        export FLAGS_fraction_of_gpu_memory_to_use=0.92
        export PADDLE_NNODES=8 # set number of devices
        export PADDLE_MASTER="xxx.xxx.xxx.xxx:port" # set PADDLE_MASTER="xxx.xxx.xxx.xxx:port" according to your network environment
        # export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        
        batch_size=1
        dap_degree=1
        bp_degree=1
        train_config="./train_configs/demo.json"
        data_config="./data_configs/demo.json"
        model_name="finetune"
        precision="bf16"
        # precision="fp32"
        log_step="--log_step=20"
        eval_step="--eval_step=1000"
        save_step="--save_step=1000"
        # init_model="$root_path/data/af2_pd_params/model_5.pdparams"
        train_af2_distributed
    fi
}

### Initial Training_N8C64_dp16_bp2_dap2
{
    if [[ "$exp_name" == "demo_initial_N8C64_dp16_bp2_dap2" ]]; then
        export PADDLE_NNODES=8 # set number of devices
        export PADDLE_MASTER="xxx.xxx.xxx.xxx:port" # set PADDLE_MASTER="xxx.xxx.xxx.xxx:port" according to your network environment

        batch_size=1
        dap_degree=2
        bp_degree=2
        train_config="./train_configs/demo.json"
        data_config="./data_configs/demo.json"
        model_name="initial"
        # init_model="$root_path/data/af2_pd_params/model_5.pdparams"
        precision="bf16"
        log_step="--log_step=20"
        eval_step="--eval_step=1000"
        save_step="--save_step=1000"
        train_af2_distributed
    fi
}

# Initial Training_N8C64_dp32_bp2_dap1
{
    if [[ "$exp_name" == "demo_initial_N8C64_dp32_bp1_dap2" ]]; then
        export PADDLE_NNODES=8 # set number of devices
        export PADDLE_MASTER="xxx.xxx.xxx.xxx:port" # set PADDLE_MASTER="xxx.xxx.xxx.xxx:port" according to your network environment

        batch_size=1
        dap_degree=2
        bp_degree=1
        train_config="./train_configs/demo.json"
        data_config="./data_configs/demo.json"
        model_name="initial"
        precision="bf16"
        log_step="--log_step=20"
        eval_step="--eval_step=1000"
        save_step="--save_step=1000"
        train_af2_distributed
    fi
}
