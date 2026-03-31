#!/bin/bash
# Usage: bash download_pdb_mmcif.sh /path/to/download/directory
set -e

if [[ $# -eq 0 ]]; then
    echo "Error: download directory must be provided as an input argument."
    exit 1
fi

if ! command -v aria2c &> /dev/null ; then
    echo "Error: aria2c could not be found. Please install aria2c (sudo apt install aria2)."
    exit 1
fi

if ! command -v rsync &> /dev/null ; then
    echo "Error: rsync could not be found. Please install rsync."
    exit 1
fi

DOWNLOAD_DIR="$1"
ROOT_DIR="${DOWNLOAD_DIR}/pdb_mmcif"
RAW_DIR="${ROOT_DIR}/raw"
MMCIF_DIR="${ROOT_DIR}/mmcif_files"

echo "Running rsync to fetch all mmCIF files (note that the rsync progress estimate might be inaccurate)..."
mkdir --parents "${RAW_DIR}"
rsync --recursive --links --perms --times --compress --info=progress2 --delete --port=33444 \
  rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ \
  "${RAW_DIR}"

echo "Unzipping all mmCIF files..."
find "${RAW_DIR}/" -type f -iname "*.gz" -exec gunzip {} +

echo "Flattening all mmCIF files..."
mkdir --parents "${MMCIF_DIR}"
find "${RAW_DIR}" -type d -empty -delete  # Delete empty directories.
for subdir in "${RAW_DIR}"/*; do
  mv "${subdir}/"*.cif "${MMCIF_DIR}"
done

# Delete empty download directory structure.
find "${RAW_DIR}" -type d -empty -delete

aria2c "ftp://ftp.wwpdb.org/pub/pdb/data/status/obsolete.dat" --dir="${ROOT_DIR}"