#!/bin/bash
# Usage: bash download_pdb_seqres.sh /path/to/download/directory
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
ROOT_DIR="${DOWNLOAD_DIR}/pdb_seqres"
SOURCE_URL="ftp://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt"
BASENAME=$(basename "${SOURCE_URL}")

mkdir --parents "${ROOT_DIR}"
aria2c "${SOURCE_URL}" --dir="${ROOT_DIR}"