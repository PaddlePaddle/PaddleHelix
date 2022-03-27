import csv
import argparse
from itertools import count
import os

import numpy as np


def create_annot_idx_map(annotations):
    return {annot: idx for idx, annot in enumerate(annotations)}


def load_GO_annot(filename):
    """Load GO annotations"""
    onts = ["molecular_function", "biological_process", "cellular_component"]
    gonames = {ont: [] for ont in onts}
    goterm_idx_maps = {ont: [] for ont in onts}
    with open(filename, mode="r") as tsvfile:
        reader = csv.reader(tsvfile, delimiter="\t")

        # molecular function
        next(reader, None)  # skip the headers
        goterm_idx_maps[onts[0]] = create_annot_idx_map(next(reader))
        next(reader, None)  # skip the headers
        gonames[onts[0]] = next(reader)

        # biological process
        next(reader, None)  # skip the headers
        goterm_idx_maps[onts[1]] = create_annot_idx_map(next(reader))
        next(reader, None)  # skip the headers
        gonames[onts[1]] = next(reader)

        # cellular component
        next(reader, None)  # skip the headers
        goterm_idx_maps[onts[2]] = create_annot_idx_map(next(reader))
        next(reader, None)  # skip the headers
        gonames[onts[2]] = next(reader)

        ont_proteins_dict = {ont: {} for ont in onts}

        next(reader, None)  # skip the headers
        counts = {ont: np.zeros(len(goterm_idx_maps[ont]), dtype=float) for ont in onts}
        for row in reader:
            prot, prot_goterms = row[0], row[1:]
            for i in range(3):
                goterm_indices = [
                    goterm_idx_maps[onts[i]][goterm]
                    for goterm in prot_goterms[i].split(",")
                    if goterm != ""
                ]
                ont_proteins_dict[onts[i]][prot] = np.array([goterm_indices])
                counts[onts[i]][goterm_indices] += 1.0
    return ont_proteins_dict, goterm_idx_maps, counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annot_file", type=str, default="./nrPDB-GO_2019.06.18_annot.tsv"
    )
    parser.add_argument("--save_dir", type=str, default="./labels")
    args = parser.parse_args()

    try:
        os.makedirs(args.save_dir)
    except FileExistsError:
        pass
    ont_prots_dict, goterm_idx_maps, counts = load_GO_annot(args.annot_file)
    idx_goterm_maps = {}
    for ont in goterm_idx_maps:
        idx_goterm_maps[ont] = {
            idx: goterm for goterm, idx in goterm_idx_maps[ont].items()
        }

    for ont in ont_prots_dict:
        filename = os.path.join(args.save_dir, ont + ".npz")
        np.savez_compressed(
            filename,
            name=[ont],
            counts=counts[ont],
            idx_goterm_map=idx_goterm_maps[ont],
            **ont_prots_dict[ont],
        )
        print(f"{filename} saved!")
