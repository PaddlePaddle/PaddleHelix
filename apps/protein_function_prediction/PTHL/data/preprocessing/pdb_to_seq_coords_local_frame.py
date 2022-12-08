import os 
import glob 
from concurrent import futures
from functools import partial
from collections import defaultdict

import numpy as np
from numpy import linalg as LA
from Bio import PDB
from Bio.PDB import MMCIFParser
from Bio.PDB.vectors import rotaxis
from Bio.Data.IUPACData import protein_letters_3to1_extended as iupac_3to1_ext
from Bio.Data.SCOPData import protein_letters_3to1 as scop_3to1


_aa3to1_dict = {**iupac_3to1_ext, **scop_3to1}
aa_codes = { aa: code for code, aa in 
            enumerate(['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
                    'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M'])}

prot_map = {i.split(',')[0]: i.split(',')[1] for i in open('/home/ssd2/v_kazadiarnold/data_splits/old_new.txt')}


def get_C_beta_coords(residue):
    if residue.has_id('CB'): 
        return residue['CB'].get_coord()
    n = residue["N"].get_vector()
    c = residue["C"].get_vector()
    ca = residue["CA"].get_vector()
    n = n - ca
    c = c - ca
    rot = rotaxis(-np.pi * 120.0/180.0, c)
    # apply rotation to ca-n vector
    cb_at_origin = n.left_multiply(rot)
    # put on top of ca atom
    cb = list(cb_at_origin + ca)
    cb = np.array(cb)

    return cb

def process_structrure(pdb_file_chains, save_dir):
    pdb_file, chain_ids = pdb_file_chains
    # print(pdb_file, chain_ids)
    prot = os.path.split(pdb_file)[-1].split('.')[0].upper()
    
    parser = MMCIFParser()
    try:
        model = parser.get_structure(None, pdb_file)[0]
    except PDB.PDBExceptions.PDBConstructionException:
        return 

    for c_id in set(chain_ids):
    # for chain in parser.get_structure(None, pdb_file)[0]:
        try:
            chain = model[c_id]
        except KeyError:
            return 

        # In order to retrieve torsion angles 
        # chain.atom_to_internal_coordinates()

        seq = []
        coords = []
        n_coords = []
        c_coords = []
        o_coords = []
        cb_coords = []
        local_sys = []
        ortho_vecs = []
        pos_in_chain =[]

        tors_ang = defaultdict(list)

        for residue in chain.get_unpacked_list():
            if residue.has_id('CA'):
                xyz = residue['CA'].get_coord()
                if coords and np.allclose(coords[-1], xyz): 
                    continue 

                try:
                    c = residue['C'].get_coord()
                    n = residue['N'].get_coord()
                    o = residue['O'].get_coord()
                except KeyError as exc:
                    # print(exc, file=open(f'errors/{prot}-{chain.id}-{"_".join(map(str, residue.get_id()))}.txt', 'w'))
                    continue

                coords.append(xyz)
                n_coords.append(n)
                c_coords.append(c)
                o_coords.append(o)
                aa_c = aa_codes.get(_aa3to1_dict.get(residue.get_resname(), '-'), 0)
                seq.append(aa_c)

                # Get position in chain
                pos_in_chain.append(residue.get_id()[1])

                # Generate local coordinate system
                c_a = xyz
                x_axis = n - c_a
                z_axis = np.cross(x_axis, c - c_a)
                y_axis = np.cross(z_axis, x_axis)
                x_axis = x_axis / LA.norm(x_axis)
                y_axis = y_axis / LA.norm(y_axis)
                z_axis = z_axis / LA.norm(z_axis)
                l_sys = np.stack([x_axis, y_axis, z_axis], axis=-1)
                assert np.allclose(LA.norm(l_sys, axis=-1), 1), LA.norm(l_sys, axis=-1)
                atol = 1e-6
                assert np.allclose(l_sys[:, 0] @ l_sys[:, 1], 0, rtol=0, atol=atol) and np.allclose(l_sys[:, 0] @ l_sys[:, 2], 0, rtol=0, atol=atol) \
                        and np.allclose(l_sys[:, 1] @ l_sys[:, 2], 0, rtol=0, atol=atol), f'{l_sys}\n{l_sys[:, 0] @ l_sys[:, 1]}, {l_sys[:, 0] @ l_sys[:, 2]}, {l_sys[:, 1] @ l_sys[:, 2]}'
                
                # assert np.allclose(x_axis @ y_axis, 0) and np.allclose(x_axis @ z_axis, 0) \
                #         and np.allclose(y_axis @ z_axis, 0), print(x_axis, y_axis, z_axis, x_axis @ y_axis, x_axis @ z_axis, y_axis @ z_axis)
                local_sys.append(l_sys)

                #Normal vector involving C-beta 
                c_b = get_C_beta_coords(residue)
                assert len(c_b) == 3, c_b
                c_a_axis = c_a - c_b
                n_axis = n - c_b  
                cross_vec = np.cross(c_a_axis, n_axis)
                ortho_vecs.append(cross_vec)
                cb_coords.append(c_b)

                # Torsion angles
                # if residue.internal_coord:
                #     tors_ang['psi'].append(residue.internal_coord.get_angle("psi"))
                #     tors_ang['phi'].append(residue.internal_coord.get_angle("phi"))
                #     tors_ang['omega'].append(residue.internal_coord.get_angle("omega"))  # or "omg"
                # else:
                #     print('No torsions!', file=open(f'errors/{prot}-{chain.id}-{"_".join(map(str, residue.get_id()))}.txt', 'w'))
                #     tors_ang['psi'].append(np.nan)
                #     tors_ang['phi'].append(np.nan)
                #     tors_ang['omega'].append(np.nan)  # or "omg"

                #TODO: Get residue's secondary structure annotation.
                
        if seq:
            npz_filename = os.path.join(save_dir, f'{prot}-{chain.id}.npz')
            if os.path.exists(npz_filename):
                print(f'{prot}-{c_id} exists already!')
                return 
            np.savez_compressed(
                npz_filename,
                seq=seq,
                coords=coords,
                n_coords=n_coords,
                c_coords=c_coords,
                o_coords=o_coords,
                cb_coords=cb_coords,
                local_sys=local_sys,
                pos_in_chain=pos_in_chain,
                ortho_vecs=ortho_vecs,
                **tors_ang,
            )
            print(f'{npz_filename} saved!')


if __name__ == '__main__':
    data_source = '.'
    pdb_dir = f'{data_source}/datasets/PDB'
    # pdb_files = glob.glob(pdb_dir+'/*.cif')
    save_dir = f'{data_source}/data/afs/GEOM-2/data/chain_seqs_coords'

    selected_chains = defaultdict(list)
    # pdb_files = list()

    for prot_chain in open(f'{data_source}/data_splits/protein_chains_all.txt'):
        prot, chain = prot_chain.strip().split('-')
        prot = prot_map.get(prot, prot) # Replace old name by new name, if applicable
        file_name = os.path.join(pdb_dir, f'{prot.lower()}.cif')
        if os.path.exists(file_name):
            selected_chains[file_name].append(chain)

    try:
        os.makedirs(save_dir)
    except FileExistsError: pass 

    with futures.ProcessPoolExecutor() as executor:
        results = executor.map(partial(process_structrure, save_dir=save_dir), selected_chains.items())

        for _ in results: pass # Just to check exceptions raised in threads.
