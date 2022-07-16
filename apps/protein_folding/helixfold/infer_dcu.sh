#!/bin/bash
#set -eu

module rm compiler/rocm/2.9
module load compiler/rocm/4.0.1
module load apps/anaconda3/5.2.0
source activate ~/conda-envs/paddle-liqi

allhost=$1
demo=$2
echo "-------------input params ${SLURM_NODEID}--------------"
echo "${SLURM_NODEID} allhost:$allhost demo:$2"
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

DATA_DIR="$root_path/data"
fasta_file="$DATA_DIR/${demo}.fasta"
OUTPUT_DIR="$root_path/output"

distributed_args="--run_mode=collective --log_dir=${log_dir}"
python -m paddle.distributed.launch ${distributed_args} \
  --ips="${allhost}" --gpus="0,1,2,3" \
  run_paddlefold.py \
  --distributed \
  --dap_degree 4 \
  --fasta_paths=${fasta_file} \
  --data_dir=${DATA_DIR} \
  --bfd_database_path=${DATA_DIR}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
  --uniclust30_database_path=${DATA_DIR}/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
  --uniref90_database_path=${DATA_DIR}/uniref90/uniref90.fasta \
  --mgnify_database_path=${DATA_DIR}/mgnify/mgy_clusters_2018_12.fa \
  --pdb70_database_path=${DATA_DIR}/pdb70/pdb70 \
  --template_mmcif_dir=${DATA_DIR}/pdb_mmcif/mmcif_files \
  --obsolete_pdbs_path=${DATA_DIR}/pdb_mmcif/obsolete.dat \
  --max_template_date=2020-05-14 \
  --model_names=model_5 \
  --output_dir=${OUTPUT_DIR} \
  --preset='full_dbs' \
  --random_seed=0

echo "done"
