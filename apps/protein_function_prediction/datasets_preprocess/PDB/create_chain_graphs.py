import os
import glob
import argparse
from concurrent import futures
from functools import partial

import numpy as np
from scipy.spatial import distance_matrix


def generate_contact_graph(chain_file, cmap_thresh, save_dir):
    prot_chain_file = os.path.split(chain_file)[-1]
    save_file_name = os.path.join(save_dir, prot_chain_file)

    # if os.path.exists(save_file_name):
    #     print(f'Chain graph exists: {save_file_name}')
    #     return

    chain_seq_coords = np.load(chain_file)
    coords = chain_seq_coords["coords"]
    dist = distance_matrix(coords, coords)
    np.fill_diagonal(dist, np.inf)  # Remove self-loop
    n2n_edges = np.stack((dist <= cmap_thresh).nonzero(), axis=1)  # Edge list
    n2n_edge_dist = dist[n2n_edges[:, 0], n2n_edges[:, 1]]

    assert np.all(n2n_edge_dist > 0), f"{prot_chain_file}"

    assert np.all(
        n2n_edges[:, 0] != n2n_edges[:, 1]
    ), f"{prot_chain_file} has self-loops."
    assert len(n2n_edges) == len(n2n_edge_dist)

    e2e_conn = (n2n_edges[:, 1][:, None] == n2n_edges[:, 0][None]) & (
        n2n_edges[:, 0][:, None] != n2n_edges[:, 1][None]
    )
    e2e_edges = np.stack(e2e_conn.nonzero(), axis=1)
    assert np.all(
        e2e_edges[:, 0] != e2e_edges[:, 1]
    ), f"{prot_chain_file} line_graph has self-loops"

    # compute polar angle using cosine formula between two vectors.
    v = coords[n2n_edges[:, 1]] - coords[n2n_edges[:, 0]]
    cos_theta = (v[e2e_edges[:, 1]] * v[e2e_edges[:, 0]]).sum(axis=-1) / (
        n2n_edge_dist[e2e_edges[:, 1]] * n2n_edge_dist[e2e_edges[:, 0]]
    )

    assert np.allclose(cos_theta[cos_theta <= -1], -1) and np.allclose(
        cos_theta[cos_theta >= 1], 1
    ), prot_chain_file
    assert len(e2e_edges) == len(cos_theta)

    cos_theta = np.clip(cos_theta, -1, 1)
    polar_ang = np.rad2deg(np.arccos(cos_theta))
    assert np.all(0 <= polar_ang) and np.all(polar_ang <= 180), prot_chain_file

    np.savez_compressed(
        save_file_name,
        seq=chain_seq_coords["seq"],
        coords=coords,
        n2n_edges=n2n_edges,
        n2n_edge_dist=n2n_edge_dist,
        e2e_edges=e2e_edges,
        e2e_polar_ang=polar_ang,
    )
    print(f"{save_file_name} saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cmap_thresh", type=int, default=10, help="Threshold for contact map."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./chain_graphs",
        help="Where to save generated protein chain graphs.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./chain_seqs_coords",
        help="Directory containing protein chain sequences and coordinates",
    )
    args = parser.parse_args()

    args.save_dir += f"/{args.cmap_thresh:02d}"
    try:
        os.makedirs(args.save_dir)
    except FileExistsError:
        pass

    input_data_files = glob.glob(args.input_dir + "/*.npz")

    with futures.ProcessPoolExecutor() as executor:
        results = executor.map(
            partial(
                generate_contact_graph,
                cmap_thresh=args.cmap_thresh,
                save_dir=args.save_dir,
            ),
            input_data_files,
        )

        for r in results:
            pass  # Check raised exceptions from threads.
