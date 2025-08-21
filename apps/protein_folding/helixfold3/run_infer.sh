#!/bin/bash

PYTHON_BIN="/usr/bin/python3" # changes to your python
ENV_BIN="/root/miniconda3/bin"  # change to your env
DATA_DIR="./data"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" inference.py \
    --jackhmmer_binary_path "$ENV_BIN/jackhmmer" \
	--hhblits_binary_path "$ENV_BIN/hhblits" \
	--hhsearch_binary_path "$ENV_BIN/hhsearch" \
	--kalign_binary_path "$ENV_BIN/kalign" \
	--hmmsearch_binary_path "$ENV_BIN/hmmsearch" \
	--hmmbuild_binary_path "$ENV_BIN/hmmbuild" \
    --nhmmer_binary_path "$ENV_BIN/nhmmer" \
    --preset='reduced_dbs' \
    --reduced_bfd_database_path "$DATA_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta" \
    --uniprot_database_path "$DATA_DIR/uniprot/uniprot.fasta" \
    --pdb_seqres_database_path "$DATA_DIR/pdb_seqres/pdb_seqres.txt" \
    --uniref90_database_path "$DATA_DIR/uniref90/uniref90.fasta" \
    --mgnify_database_path "$DATA_DIR/mgnify/mgy_clusters_2018_12.fa" \
    --template_mmcif_dir "$DATA_DIR/pdb_mmcif/mmcif_files" \
    --obsolete_pdbs_path "$DATA_DIR/pdb_mmcif/obsolete.dat" \
    --ccd_preprocessed_path "$DATA_DIR/ccd_preprocessed_etkdg.pkl.gz" \
    --rfam_database_path "$DATA_DIR/Rfam-14.9_rep_seq.fasta" \
    --max_template_date=2021-09-30 \
    --input_json data/demo_6zcy.json \
    --output_dir ./output \
    --model_name allatom_demo \
    --init_model "${DATA_DIR}/params/HelixFold3-params-20250714/HelixFold3-20250714.pdparams" \
    --infer_times 1 \
    --diff_batch_size 1 \
    --precision "fp32"