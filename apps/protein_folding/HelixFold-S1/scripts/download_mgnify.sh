#!/bin/bash
# Usage: bash download_mgnify.sh /path/to/download/directory
set -e

if [[ $# -eq 0 ]]; then
    echo "Error: download directory must be provided as an input argument."
    exit 1
fi

if ! command -v aria2c &> /dev/null ; then
    echo "Error: aria2c could not be found. Please install aria2c (sudo apt install aria2)."
    exit 1
fi

DOWNLOAD_DIR="$1"
ROOT_DIR="${DOWNLOAD_DIR}/mgnify"
# Mirror of:
# ftp://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2018_12/mgy_clusters.fa.gz
SOURCE_URL="https://storage.googleapis.com/alphafold-databases/casp14_versions/mgy_clusters_2018_12.fa.gz"
BASENAME=$(basename "${SOURCE_URL}")

mkdir --parents "${ROOT_DIR}"
aria2c "${SOURCE_URL}" --dir="${ROOT_DIR}"
pushd "${ROOT_DIR}"
gunzip "${ROOT_DIR}/${BASENAME}"
popd