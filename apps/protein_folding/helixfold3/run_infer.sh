#!/bin/bash

PYTHON_BIN="/usr/bin/python3"
ENV_BIN="/root/miniconda3/bin"
# MAXIT_SRC="/root/paddlejob/workspace/output/yexianbin/dcu_command/maxit-v11.200-prod-src/"
MAXIT_SRC="/home/rudder_paddle_home/custom_workspace/init_models/paddlehelix/paddlefold/maxit-v11.100-prod-src"
DATA_DIR="./data"
export PATH="$MAXIT_SRC/hin:$PATH"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" inference.py \
    --maxit_binary "$MAXIT_SRC/bin/maxit" \
    --jackhmmer_binary_path "$ENV_BIN/jackhmmer" \
	--hhblits_binary_path "$ENV_BIN/hhblits" \
	--hhsearch_binary_path "$ENV_BIN/hhsearch" \
	--kalign_binary_path "$ENV_BIN/kalign" \
	--hmmsearch_binary_path "$ENV_BIN/hmmsearch" \
	--hmmbuild_binary_path "$ENV_BIN/hmmbuild" \
    --preset='reduced_dbs' \
    --bfd_database_path "$DATA_DIR/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt" \
    --small_bfd_database_path "$DATA_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta" \
    --bfd_database_path "$DATA_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta" \
    --uniclust30_database_path "$DATA_DIR/uniclust30/uniclust30_2018_08/uniclust30_2018_08" \
    --uniprot_database_path "$DATA_DIR/uniprot/uniprot.fasta" \
    --pdb_seqres_database_path "$DATA_DIR/pdb_seqres/pdb_seqres.txt" \
    --uniref90_database_path "$DATA_DIR/uniref90/uniref90.fasta" \
    --mgnify_database_path "$DATA_DIR/mgnify/mgy_clusters_2018_12.fa" \
    --template_mmcif_dir "$DATA_DIR/pdb_mmcif/mmcif_files" \
    --obsolete_pdbs_path "$DATA_DIR/pdb_mmcif/obsolete.dat" \
    --ccd_preprocessed_path "$DATA_DIR/ccd_preprocessed_etkdg.pkl.gz" \
    --rfam_database_path "$DATA_DIR/RNA_MSA_databases/Rfam-14.9_rep_seq.fasta" \
    --max_template_date=2020-05-14 \
    --input_json data/demo_6zcy.json \
    --output_dir ./output \
    --model_name allatom_demo \
    --init_model init_models/helixfold_aa_init_model.pdparams \
    --infer_times 3 \
    --precision "bf16" \
    --no_msa_templ_feats # comment it to enable MSA searching