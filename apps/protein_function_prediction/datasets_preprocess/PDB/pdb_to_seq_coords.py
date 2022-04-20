import os
import glob
import argparse
from concurrent import futures
from functools import partial
from collections import defaultdict

import numpy as np
from Bio import PDB
from Bio.PDB import MMCIFParser
from Bio.Data.IUPACData import protein_letters_3to1_extended as iupac_3to1_ext
from Bio.Data.SCOPData import protein_letters_3to1 as scop_3to1


_aa3to1_dict = {**iupac_3to1_ext, **scop_3to1}
aa_codes = {
    aa: code
    for code, aa in enumerate(
        [
            "-",
            "D",
            "G",
            "U",
            "L",
            "N",
            "T",
            "K",
            "H",
            "Y",
            "W",
            "C",
            "P",
            "V",
            "S",
            "O",
            "I",
            "E",
            "F",
            "X",
            "Q",
            "A",
            "B",
            "Z",
            "R",
            "M",
        ]
    )
}


def process_structrure(pdb_file_chains, save_dir):
    pdb_file, chain_ids = pdb_file_chains
    prot = os.path.split(pdb_file)[-1].split(".")[0].upper()

    parser = MMCIFParser()
    try:
        model = parser.get_structure(None, pdb_file)[0]
    except PDB.PDBExceptions.PDBConstructionException:
        return

    for c_id in set(chain_ids):
        try:
            chain = model[c_id]
        except KeyError:
            return

        seq = []
        coords = []
        for residue in chain.get_unpacked_list():
            if "CA" in residue:
                xyz = residue["CA"].get_coord()
                if coords and np.allclose(
                    coords[-1], xyz
                ):  # Ignore residue if too close to the previous one.
                    continue
                aa_c = aa_codes.get(_aa3to1_dict.get(residue.get_resname(), "-"), 0)
                seq.append(aa_c)
                coords.append(xyz)
        if seq:
            npz_filename = os.path.join(save_dir, f"{prot}-{chain.id}.npz")
            # if os.path.exists(npz_filename):
            #     print(f'{prot}-{c_id} exists already!')
            #     return
            np.savez_compressed(npz_filename, seq=seq, coords=coords)
            print(f"{npz_filename} saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdb_dir",
        type=str,
        default="./PDB_files",
        help="Protein PDB files directory.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./chain_seqs_coords",
        help="Where to save retrieved chain sequences and coordinates.",
    )

    args = parser.parse_args()

    available_chains = defaultdict(list)

    for prot_chain in open("protein_chains_all.txt"):
        prot, chain = prot_chain.strip().split("-")
        file_name = os.path.join(args.pdb_dir, f"{prot.lower()}.cif")
        if os.path.exists(file_name):
            available_chains[file_name].append(chain)

    try:
        os.makedirs(args.save_dir)
    except FileExistsError:
        pass

    with futures.ProcessPoolExecutor() as executor:
        results = executor.map(
            partial(process_structrure, save_dir=args.save_dir),
            available_chains.items(),
        )

        for _ in results:
            pass  # Just to check exceptions raised in threads.
