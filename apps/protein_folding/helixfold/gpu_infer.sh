#!/bin/bash

cd "$(dirname $0)"
root_path="$(pwd)"
demo=$1

export PYTHONPATH=$root_path:$PYTHONPATH

#DATA_DIR="$root_path/data"
DATA_DIR="/root/paddlejob/workspace/env_run/alphafold_data"
fasta_file="$root_path/demo_data/casp14_demo/fasta/${demo}.fasta"
OUTPUT_DIR="$root_path/demo_data/casp14_demo/output"
log_dir="$root_path/demo_data/casp14_demo/demo_log"
MODELS="model_1,model_5"
SUBBATCH_SIZE=48
DAP_DEGREE=1

distributed=1

# 'fp32' or 'bf16'
PRECISION='bf16'

# enable dap or unified memory for EXTREMELY LONG SEQUENCE PROTEIN
# DAP_DEGREE=8
# export FLAGS_use_cuda_managed_memory=true

# Reduce the size of subbatch_size when the gpu memory is not enough 
# SUBBATCH_SIZE=1

if [ $distributed == 1 ]
then
  python_cmd="python -m paddle.distributed.launch --log_dir=${log_dir} --gpus=0,1,2,3,4,5,6,7 "
  distributed_flag="--distributed"
else
  python_cmd="CUDA_VISIBLE_DEVICES=0 python "
  distributed_flag=""
fi

$python_cmd \
  run_helixfold.py \
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
  --seed 2022 \
  --preset='reduced_dbs' \
  --random_seed=0 \
  --precision=${PRECISION} \
  --subbatch_size=${SUBBATCH_SIZE} \
  ${@:2}
