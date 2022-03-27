import os
import argparse
from concurrent import futures
from functools import partial

from Bio.PDB import PDBList


def download_pdb(prot, pdir):
    pdb = PDBList()
    pdb.retrieve_pdb_file(prot, pdir=pdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./PDB_files",
        help="Where to save downloaded PDB files.",
    )
    parser.add_argument(
        "--prot_list",
        type=str,
        default="proteins_all.txt",
        help="Text file with list of proteins to be downloaded",
    )
    args = parser.parse_args()

    prots = [p.strip() for p in open(args.prot_list)]

    try:
        os.makedirs(args.save_dir)
    except FileExistsError:
        pass

    with futures.ThreadPoolExecutor() as executor:
        executor.map(partial(download_pdb, pdir=args.save_dir), prots)
