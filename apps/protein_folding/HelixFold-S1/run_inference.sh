#!/bin/bash

# ============== User Configuration ==============
inputjson=${1}
outpath=${2}
mkdir -p ${outpath}

# Python and Environment Configuration
PYTHON_BIN="PATH/TO/YOUR/PYTHON"
export LD_LIBRARY_PATH="PATH/TO/YOUR/ENV/lib:$LD_LIBRARY_PATH"
ENV_BIN="PATH/TO/YOUR/ENV/bin"

# Database Paths
DATA_DIR="PATH/TO/DATA"

# Model Configuration
CKPT_DIR="./init_models"

model_name_gen1="allatom_interface_gen_fixrepr_save_intermediate_interface"
init_model_gen1="${CKPT_DIR}/interface_gen100_merge-at-conditioning_fixrepr_atom5A_compress/step_604000.pdparams"

model_name_infer2="allatom_interface"
init_model_infer2="${CKPT_DIR}/interface004_any_interface0.7_lr2e-4_atom5A_compress/step_496000.pdparams"

# CUDA Device
export CUDA_VISIBLE_DEVICES=0

# ============== Run MSA Module ==============
echo ">>> Running MSA module"
${PYTHON_BIN} run_msa_module.py \
    --jackhmmer_binary_path "$ENV_BIN/jackhmmer" \
    --hhblits_binary_path "$ENV_BIN/hhblits" \
    --hhsearch_binary_path "$ENV_BIN/hhsearch" \
    --kalign_binary_path "$ENV_BIN/kalign" \
    --hmmsearch_binary_path "$ENV_BIN/hmmsearch" \
    --hmmbuild_binary_path "$ENV_BIN/hmmbuild" \
    --nhmmer_binary_path "$ENV_BIN/nhmmer" \
    --preset='reduced_dbs' \
    --small_bfd_database_path "$DATA_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta" \
    --uniprot_database_path "$DATA_DIR/uniprot/uniprot.fasta" \
    --pdb_seqres_database_path "$DATA_DIR/pdb_seqres.txt" \
    --uniref90_database_path "$DATA_DIR/uniref90/uniref90.fasta" \
    --mgnify_database_path "$DATA_DIR/mgnify/mgy_clusters_2018_12.fa" \
    --template_mmcif_dir "$DATA_DIR/pdb_mmcif/mmcif_files" \
    --obsolete_pdbs_path "$DATA_DIR/pdb_mmcif/obsolete.dat" \
    --ccd_preprocessed_path "$DATA_DIR/ccd_preprocessed_etkdg.pkl.gz" \
    --rfam_database_path "$DATA_DIR/rna_msa_db/Rfam-14.9_rep_seq.fasta" \
    --species_identifer_map_path "$DATA_DIR/rna_msa_db/taxonomy_AND_taxonomies_with_1_uniprot_2024_07_14.tsv.gz" \
    --max_template_date=2021-09-30 \
    --json_path $inputjson \
    --output_dir ${outpath} 

# ============== Run Gen Module 1 ==============
echo ">>> Running gen module 1"
${PYTHON_BIN} run_inference_module.py \
    --json_path $inputjson \
    --output_dir ${outpath} \
    --model_names "${model_name_gen1}" \
    --init_model "${init_model_gen1}" \
    --ccd_preprocessed_path "$DATA_DIR/ccd_preprocessed_etkdg.pkl.gz"

# ============== Run Inference Module 2 ==============
echo ">>> Running inference module 2"
${PYTHON_BIN} run_inference_S1_module2.py \
    --json_path $inputjson \
    --output_dir ${outpath} \
    --model_names "${model_name_infer2}" \
    --init_model "${init_model_infer2}" \
    --ccd_preprocessed_path "$DATA_DIR/ccd_preprocessed_etkdg.pkl.gz"

echo ">>> Inference completed! Output: ${outpath}"
