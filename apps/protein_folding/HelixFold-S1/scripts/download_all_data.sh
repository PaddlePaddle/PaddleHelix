#!/bin/bash
# Usage: bash download_all_data.sh /path/to/download/directory
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
DOWNLOAD_MODE="${2:-full_dbs}"  # Default mode to full_dbs.
if [[ "${DOWNLOAD_MODE}" != full_dbs && "${DOWNLOAD_MODE}" != reduced_dbs ]]
then
  echo "DOWNLOAD_MODE ${DOWNLOAD_MODE} not recognized."
  exit 1
fi

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

echo "Downloading HelixFold3 checkpoints..."
bash "${SCRIPT_DIR}/download_helixfold3_checkpoints.sh" "${DOWNLOAD_DIR}"

if [[ "${DOWNLOAD_MODE}" = reduced_dbs ]] ; then
  echo "Downloading Small BFD..."
  bash "${SCRIPT_DIR}/download_small_bfd.sh" "${DOWNLOAD_DIR}"
else
  echo "Downloading BFD..."
  bash "${SCRIPT_DIR}/download_bfd.sh" "${DOWNLOAD_DIR}"
fi

echo "Downloading MGnify..."
bash "${SCRIPT_DIR}/download_mgnify.sh" "${DOWNLOAD_DIR}"

echo "Downloading PDB mmCIF files..."
bash "${SCRIPT_DIR}/download_pdb_mmcif.sh" "${DOWNLOAD_DIR}"

echo "Downloading Uniclust30..."
bash "${SCRIPT_DIR}/download_uniclust30.sh" "${DOWNLOAD_DIR}"

echo "Downloading Uniref90..."
bash "${SCRIPT_DIR}/download_uniref90.sh" "${DOWNLOAD_DIR}"

echo "Downloading UniProt..."
bash "${SCRIPT_DIR}/download_uniprot.sh" "${DOWNLOAD_DIR}"

echo "Downloading PDB SeqRes..."
bash "${SCRIPT_DIR}/download_pdb_seqres.sh" "${DOWNLOAD_DIR}"

echo "Downloading RNA MSA..."
bash "${SCRIPT_DIR}/download_rna.sh" "${DOWNLOAD_DIR}"

echo "Downloading CCD pickel..."
bash "${SCRIPT_DIR}/download_ccd_pkl.sh" "${DOWNLOAD_DIR}"

echo "All data downloaded."