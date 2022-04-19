#!/bin/bash
#set -eu

module rm compiler/rocm/2.9
module load compiler/rocm/4.0.1
module load apps/anaconda3/5.2.0
source activate ~/conda-envs/paddle_20220413

allhost=$1
echo "-------------input params ${SLURM_NODEID}--------------"
echo "${SLURM_NODEID} allhost:$allhost"
echo "SLURM_NODEID:${SLURM_NODEID}"

OLD_IFS="$IFS"
IFS=","
allhost_arr=($allhost)
IFS="$OLD_IFS"
node_num=${#allhost_arr[@]}
echo "node_num="${#allhost_arr[@]}
export PADDLE_NODE_NUM=${node_num}
echo "PADDLE_NODE_NUM="${PADDLE_NODE_NUM}

export PADDLE_WITH_GLOO=0
export NCCL_SOCKET_IFNAME=eno1
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_3
export HIP_VISIBLE_DEVICES=0,1,2,3

DD_RAND_SEED=1

export FLAGS_conv2d_disable_cudnn=True
export MIOPEN_FIND_MODE=3

echo "[INFO]: Rand seed "${DD_RAND_SEED}
echo "[INFO]: PATH="$PATH
echo "[INFO]: PYTHONPATH="$PYTHONPATH

log_dir="log/log_${SLURM_NODEID}"
rm -rf ${log_dir}
if [ ! -d ${log_dir} ]; then
    mkdir ${log_dir}
fi

root_path="$(pwd)/../../"
export DEBUG=1
export PYTHONPATH=$root_path:$PYTHONPATH

TM_SCORE_BIN="/public/home/test_alphafold/data/20220118/af2/tm_score"
LDDT_SCORE_BIN="/public/home/test_alphafold/data/20220118/af2/lddt"

precision="fp32"
data_config="./data_configs/demo.json"
train_config="./train_configs/initial.json"
model_name="initial_model_5_dcu"
start_step=1
batch_size=1
train_step=100000
export MAX_EVAL_SIZE=1000

distributed_args="--run_mode=collective --log_dir=${log_dir}"
python -m paddle.distributed.launch ${distributed_args} \
    --ips="${allhost}" --gpus="0,1" \
    train.py \
    --distributed \
    --tm_score_bin="$TM_SCORE_BIN" \
    --lddt_score_bin="$LDDT_SCORE_BIN" \
    --precision=${precision} \
    --data_config=${data_config} \
    --train_config=${train_config} \
    --model_name=${model_name} \
    --init_model=${init_model} \
    --start_step=${start_step} \
    --batch_size=$batch_size \
    --train_step=${train_step} \
    --num_workers=0 \
    --model_dir="./debug_models" \
    
echo "done"
