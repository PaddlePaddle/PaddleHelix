#!/bin/bash

cd "$(dirname $0)"
root_path="$(pwd)"
# demo=$1

export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH
export PYTHONPATH=$root_path:$PYTHONPATH

# export FLAGS_allocator_strategy=naive_best_fit
# export GLOG_v=6
# export GLOG_vmodule=auto_growth_best_fit_allocator=4,stream_safe_cuda_allocator=4
export FLAGS_free_when_no_cache_hit=1
# export FLAGS_fraction_of_gpu_memory_to_use=0.96
# export FLAGS_benchmark=1
export FLAGS_call_stack_level=2
# export FLAGS_use_system_allocator=1

DATA_DIR="/root/paddlejob/workspace/env_run/output"                  # path to data directory
# fasta_file="$root_path/demo_data/demo_infer/7XJT.fasta"            # 336AA
# fasta_file="$root_path/demo_data/demo_infer/6LTH.fasta"            # 2285AA
# fasta_file="$root_path/demo_data/demo_infer/A0A1L7F979.fasta"      # 4096AA
# fasta_file="$root_path/demo_data/demo_infer/Q9PU36.fasta"          # 5120AA
fasta_file="$root_path/demo_data/demo_infer/F2ULY7.fasta"          # 6600AA 
# fasta_file="$root_path/demo_data/demo_infer/Q03001.fasta"          # 7570AA
# fasta_file="$root_path/demo_data/demo_infer/Q8NF91.fasta"          # 8797AA
# fasta_file="$root_path/demo_data/demo_infer/Q5HPA2.fasta"          # 9439AA
# fasta_file="$root_path/demo_data/demo_infer/Q8WXI7.fasta"          # 14507AA

OUTPUT_DIR="$root_path/demo_data/demo_output"                        # path to outputs directory
log_dir="$root_path/demo_data/casp14_demo/demo_log"                  # path to log directory
MODELS="model_5"

# Use DAP
distributed=true

# 'fp32' or 'bf16'
PRECISION='bf16'

# 'O1' or 'O2'
AMP_LEVEL='O1'

# Enable C++ enisum instead of python enisum
export FLAGS_new_einsum=1

# Enable/Disable bf16 optimization
export FLAGS_use_autotune=1

# Enable LayerNorm optimization
export FLAGS_use_fast_math=1


if [ $distributed == true ]
then
  # Enable unified memory for EXTREMELY LONG SEQUENCE PROTEIN
  # export FLAGS_use_cuda_managed_memory=true

  # Enable DAP for EXTREMELY LONG SEQUENCE PROTEIN
  python_cmd="python -m paddle.distributed.launch --log_dir=${log_dir} --gpus=0,1,2,3,4,5,6,7 "
  distributed_flag="--distributed"
  DAP_DEGREE=8

  # Reduce the size of subbatch_size when the gpu memory is not enough 
  SUBBATCH_SIZE=32
else
  # Enable unified memory for EXTREMELY LONG SEQUENCE PROTEIN
  # export FLAGS_use_cuda_managed_memory=true

  python_cmd="CUDA_VISIBLE_DEVICES=0 python "
  distributed_flag=""
  DAP_DEGREE=1

  # Reduce the size of subbatch_size when the gpu memory is not enough 
  SUBBATCH_SIZE=1
fi

$python_cmd run_helixfold.py \
  ${distributed_flag} \
  --dap_degree=${DAP_DEGREE} \
  --fasta_paths=${fasta_file} \
  --data_dir=${DATA_DIR} \
  --bfd_database_path=${DATA_DIR}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
  --small_bfd_database_path=${DATA_DIR}/small_bfd/bfd-first_non_consensus_sequences.fasta \
  --uniclust30_database_path=${DATA_DIR}/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
  --uniref90_database_path=${DATA_DIR}/uniref90/uniref90.fasta \
  --mgnify_database_path=${DATA_DIR}/mgnify/mgy_clusters_2018_12.fa \
  --pdb70_database_path=${DATA_DIR}/pdb70/pdb70 \
  --template_mmcif_dir=${DATA_DIR}/pdb_mmcif/mmcif_files \
  --obsolete_pdbs_path=${DATA_DIR}/pdb_mmcif/obsolete.dat \
  --max_template_date=2020-05-14 \
  --model_names=${MODELS} \
  --output_dir=${OUTPUT_DIR} \
  --disable_amber_relax \
  --enable_low_memory \
  --seed 2022 \
  --preset='full_dbs' \
  --random_seed=0 \
  --precision=${PRECISION} \
  --amp_level=${AMP_LEVEL} \
  --subbatch_size=${SUBBATCH_SIZE} \
  ${@:2}
