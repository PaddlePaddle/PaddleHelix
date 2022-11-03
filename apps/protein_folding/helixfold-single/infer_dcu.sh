#!/bin/bash
#set -eu

module rm compiler/rocm/2.9
module load compiler/rocm/4.0.1
module load apps/anaconda3/5.2.0
source activate ~/conda-envs/paddle_20221102

allhost=$1
echo "-------------input params ${SLURM_NODEID}--------------"
echo "${SLURM_NODEID} allhost:$allhost"
echo "SLURM_NODEID:${SLURM_NODEID}"
#set -eu

# -----------dcu start ------------

#export CUDA_LAUNCH_BLOCKING=1
#export HIP_LAUNCH_BLOCKING=1
#export AMD_OCL_WAIT_COMMAND=1
#export FLAGS_benchmark=1

#export HSA_FORCE_FINE_GRAIN_PCIE=1
#LD_LIBRARY_PATH=/opt/rocm-4.0.1/miopen-2.11/lib:$LD_LIBRARY_PATH

# export GLOG_v=1
# export NCCL_DEBUG=INFO
export PADDLE_WITH_GLOO=0
export NCCL_SOCKET_IFNAME=eno1
export NCCL_IB_DISABLE=0
#export NCCL_P2P_DISABLE=1
export NCCL_IB_HCA=mlx5_0
export HIP_VISIBLE_DEVICES=0,1,2,3

#export NCCL_GRAPH_DUMP_FILE=graph.xml
# -----------dcu start ------------

export FLAGS_call_stack_level=2
export FLAGS_eager_delete_tensor_gb=0
export FLAGS_fraction_of_gpu_memory_to_use=0.98
export FLAGS_sync_nccl_allreduce=1
export FLAGS_infcheck_adamoptimizer=True
export FLAGS_check_nan_inf=0
#export FLAGS_memory_fraction_of_eager_deletion=1
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_allocator_strategy=naive_best_fit
#export FLAGS_allocator_strategy="auto_growth"
#export FLAGS_use_cuda_managed_memory=True # not work,require to upgrade ROCm

e_executor=$(echo ${use_experimental_executor-'True'} | tr '[A-Z]' '[a-z]')

use_fuse=$(echo ${use_fuse-'False'} | tr '[A-Z]' '[a-z]')
if [[ ${use_fuse} == "true" ]]; then
    export FLAGS_fuse_parameter_memory_size=32
    export FLAGS_fuse_parameter_groups_size=50
fi

echo "[INFO]: PATH="$PATH
echo "[INFO]: PYTHONPATH="$PYTHONPATH

log_dir="log_${SLURM_NODEID}"
rm -rf ${log_dir}
rm -rf start_sharding_*
rm -rf main_sharding_*
rm -rf main_*
if [ ! -d ${log_dir} ]; then
    mkdir ${log_dir}
fi

details_dir="details"
rm -rf ${details_dir}
if [ ! -d ${details_dir} ]; then
    mkdir ${details_dir}
fi

program_desc_dir="program_desc"
rm -rf ${program_desc_dir}
if [ ! -d ${program_desc_dir} ]; then
    mkdir ${program_desc_dir}
fi

cd "$(dirname $0)"
root_path="$(pwd)"
export PYTHONPATH=$root_path:$PYTHONPATH

HIP_VISIBLE_DEVICES=0 python helixfold_single_inference.py \
        --init_model=./helixfold-single.pdparams \
        --fasta_file=data/7O9F_B.fasta \
        --output_dir="./output" 

echo "done"
