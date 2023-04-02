import os
import glob
import argparse
from concurrent import futures
from functools import partial

import numpy as np
import numpy.linalg as LA
from scipy.spatial import distance_matrix


def generate_contact_graph(chain_file, cmap_thresh, save_dir):
    # print('In generate')
    save_dir += f'/{cmap_thresh:02d}'
    prot_chain_file = os.path.split(chain_file)[-1]
    save_file_name = os.path.join(save_dir, prot_chain_file)

    if os.path.exists(save_file_name):
        print(f'Chain graph exists: {save_file_name}')
        return
    
 

    try:
        os.makedirs(save_dir)
    except FileExistsError: pass

    chain_seq_coords = np.load(chain_file)
    coords = chain_seq_coords['coords']
    dist = distance_matrix(coords, coords)
    # temp_idx = np.arange(dist.shape[0])
    # dist[temp_idx, temp_idx]= 2 * cmap_thresh
    np.fill_diagonal(dist, np.inf)# To remove self-loop
    n2n_edges = np.stack((dist <= cmap_thresh).nonzero(), axis=1)
    n2n_edge_dist = dist[n2n_edges[:, 0], n2n_edges[:, 1]]

    # Angles between amino acids
    assert np.all(n2n_edge_dist > 0), prot_chain_file

    assert np.all(n2n_edges[:, 0] != n2n_edges[:, 1])
    assert len(n2n_edges) == len(n2n_edge_dist)

    # compute polar angle using cosine formula between two vectors.
    v = chain_seq_coords['ortho_vecs']
    cos_theta = ((v[n2n_edges[:, 1]] * v[n2n_edges[:, 0]]).sum(axis=-1) /
                    (LA.norm(v[n2n_edges[:, 1]]) * LA.norm(v[n2n_edges[:, 0]])))
    
    # assert np.all(-1 < cos_theta) and np.all(cos_theta < 1)
    assert np.allclose(cos_theta[cos_theta <= -1], -1) and np.allclose(cos_theta[cos_theta >= 1], 1), prot_chain_file
    assert len(n2n_edges) == len(cos_theta)

    cos_theta = np.clip(cos_theta, -1, 1)
    inter_ang = np.rad2deg(np.arccos(cos_theta))
    # polar_ang = np.clip(polar_ang, 0, 180)
    assert np.all(0 <= inter_ang) and np.all(inter_ang <= 180), prot_chain_file

    np.savez_compressed(
        save_file_name,
        seq=chain_seq_coords['seq'],
        coords=coords,
        n_coords=chain_seq_coords['n_coords'],
        c_coords=chain_seq_coords['c_coords'],
        o_coords=chain_seq_coords['o_coords'],
        cb_coords=chain_seq_coords['cb_coords'],
        n2n_edges=n2n_edges,
        n2n_edge_dist=n2n_edge_dist,
        local_sys=chain_seq_coords['local_sys'],
        pos_in_chain=chain_seq_coords['pos_in_chain'],
        inter_ang=inter_ang
    )
    print(f'{save_file_name} saved!')


if __name__ == '__main__':
    data_source = '.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmap_thresh', type=int, default=10, help='threshold for contact map')
    parser.add_argument('--save_dir', type=str, default=f'{data_source}/data/afs/GEOM-2/data/chain_graphs')
    parser.add_argument('--input_dir', type=str, default=f'{data_source}/data/afs/GEOM-2/data/chain_seqs_coords')
    args = parser.parse_args()

    # print('running main')
    try:
        os.makedirs(args.save_dir)
    except FileExistsError: pass

    input_data_files = glob.glob(args.input_dir+'/*')

    assert len(input_data_files)


    with futures.ProcessPoolExecutor() as executor:
        results = executor.map(partial(generate_contact_graph, cmap_thresh=args.cmap_thresh, save_dir=args.save_dir), input_data_files)
        
        for r in results: pass # Simply for retrieving raised exception from threads.
