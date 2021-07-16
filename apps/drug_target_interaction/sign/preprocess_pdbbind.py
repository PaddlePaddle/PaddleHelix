# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocessing code for the protein-ligand complex.
"""

import os
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import openbabel
from openbabel import pybel
from featurizer import Featurizer
from scipy.spatial import distance_matrix

def pocket_atom_num_from_mol2(name, path):
    n = 0
    with open('%s/%s/%s_pocket.mol2' % (path, name, name)) as f:
        for line in f:
            if '<TRIPOS>ATOM' in line:
                break
        for line in f:
            cont = line.split()
            if '<TRIPOS>BOND' in line or cont[7] == 'HOH':
                break
            n += int(cont[5][0] != 'H')
    return n

def pocket_atom_num_from_pdb(name, path):
    n = 0
    with open('%s/%s/%s_pocket.pdb' % (path, name, name)) as f:
        for line in f:
            if 'REMARK' in line:
                break
        for line in f:
            cont = line.split()
            # break
            if cont[0] == 'CONECT':
                break
            n += int(cont[-1] != 'H' and cont[0] == 'ATOM')
    return n

## function -- feature
def gen_feature(path, name, featurizer):
    charge_idx = featurizer.FEATURE_NAMES.index('partialcharge')
    ligand = next(pybel.readfile('mol2', '%s/%s/%s_ligand.mol2' % (path, name, name)))
    ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)
    pocket = next(pybel.readfile('mol2' ,'%s/%s/%s_pocket.mol2' % (path, name, name)))
    pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)
    node_num = pocket_atom_num_from_mol2(name, path)
    pocket_coords = pocket_coords[:node_num]
    pocket_features = pocket_features[:node_num]
    try:
        assert (ligand_features[:, charge_idx] != 0).any()
        assert (pocket_features[:, charge_idx] != 0).any()
        assert (ligand_features[:, :9].sum(1) != 0).all()
    except:
        print(name)
    lig_atoms, pock_atoms = [], []
    for i, atom in enumerate(ligand):
        if atom.atomicnum > 1:
            lig_atoms.append(atom.atomicnum)
    for i, atom in enumerate(pocket):
        if atom.atomicnum > 1:
            pock_atoms.append(atom.atomicnum)
    for x in pock_atoms[node_num:]:
        assert x == 8
    pock_atoms = pock_atoms[:node_num]
    assert len(lig_atoms)==len(ligand_features) and len(pock_atoms)==len(pocket_features)
    
    ligand_edges = gen_pocket_graph(ligand)
    pocket_edges = gen_pocket_graph(pocket)
    return {'lig_co': ligand_coords, 'lig_fea': ligand_features, 'lig_atoms': lig_atoms, 'lig_eg': ligand_edges, 'pock_co': pocket_coords, 'pock_fea': pocket_features, 'pock_atoms': pock_atoms, 'pock_eg': pocket_edges}

## function -- pocket graph
def gen_pocket_graph(pocket):
    edge_l = []
    idx_map = [-1]*(len(pocket.atoms)+1)
    idx_new = 0
    for atom in pocket:
        edges = []
        a1_sym = atom.atomicnum
        a1 = atom.idx
        if a1_sym == 1:
            continue
        idx_map[a1] = idx_new
        idx_new += 1
        for natom in openbabel.OBAtomAtomIter(atom.OBAtom):
            if natom.GetAtomicNum() == 1:
                continue
            a2 = natom.GetIdx()
            bond = openbabel.OBAtom.GetBond(natom,atom.OBAtom)
            bond_t = bond.GetBondOrder()
            edges.append((a1,a2,bond_t))
        edge_l += edges
    edge_l_new = []
    for a1,a2,t in edge_l:
        a1_, a2_ = idx_map[a1], idx_map[a2]
        assert((a1_!=-1)&(a2_!=-1))
        edge_l_new.append((a1_,a2_,t))
    return edge_l_new

def dist_filter(dist_matrix, theta): 
    pos = np.where(dist_matrix<=theta)
    ligand_list, pocket_list = pos
    return ligand_list, pocket_list

def pairwise_atomic_types(path, processed_dict, atom_types, atom_types_):
    keys = [(i,j) for i in atom_types_ for j in atom_types]
    for name in tqdm(os.listdir(path)):
        if len(name) != 4:
            continue
        ligand = next(pybel.readfile('mol2', '%s/%s/%s_ligand.mol2' % (path, name, name)))
        pocket = next(pybel.readfile('pdb' ,'%s/%s/%s_protein.pdb' % (path, name, name)))
        coords_lig = np.vstack([atom.coords for atom in ligand])
        coords_poc = np.vstack([atom.coords for atom in pocket])
        atom_map_lig = [atom.atomicnum for atom in ligand]
        atom_map_poc = [atom.atomicnum for atom in pocket]
        dm = distance_matrix(coords_lig, coords_poc)
        # print(coords_lig.shape, coords_poc.shape, dm.shape)
        ligs, pocks = dist_filter(dm, 12)
        # print(len(ligs),len(pocks))
        
        fea_dict = {k: 0 for k in keys}
        for x, y in zip(ligs, pocks):
            x, y = atom_map_lig[x], atom_map_poc[y]
            if x not in atom_types or y not in atom_types_: continue
            fea_dict[(y, x)] += 1
            
        processed_dict[name]['type_pair'] = list(fea_dict.values())

    return processed_dict

def load_pk_data(data_path):
    res = dict()
    with open(data_path) as f:
        for line in f:
            if '#' in line:
                continue
            cont = line.strip().split()
            if len(cont) < 5:
                continue
            code, pk = cont[0], cont[3]
            res[code] = float(pk)
    return res

def get_lig_atom_types(feat):
    pos = np.where(feat[:,:9]>0)
    src_list, dst_list = pos
    return dst_list
def get_pock_atom_types(feat):
    pos = np.where(feat[:,18:27]>0)
    src_list, dst_list = pos
    return dst_list

def cons_spatial_gragh(dist_matrix, theta=5):
    pos = np.where((dist_matrix<=theta)&(dist_matrix!=0))
    src_list, dst_list = pos
    dist_list = dist_matrix[pos]
    edges = [(x,y) for x,y in zip(src_list, dst_list)]
    return edges, dist_list

def cons_mol_graph(edges, feas):
    size = feas.shape[0]
    edges = [(x,y) for x,y,t in edges]
    return size, feas, edges

def pocket_subgraph(node_map, edge_list, pock_dist):
    edge_l = []
    dist_l = []
    node_l = set()
    for coord, dist in zip(edge_list, np.concatenate([pock_dist, pock_dist])):
        x,y = coord
        if x in node_map and y in node_map:
            x, y = node_map[x], node_map[y]
            edge_l.append((x,y))
            dist_l.append(dist)
            node_l.add(x)
            node_l.add(y)
    dist_l = np.array(dist_l)
    return edge_l, dist_l

def edge_ligand_pocket(dist_matrix, lig_size, theta=4, keep_pock=False, reset_idx=True):
    
    pos = np.where(dist_matrix<=theta)
    ligand_list, pocket_list = pos
    if keep_pock:
        node_list = range(dist_matrix.shape[1])
    else:
        node_list = sorted(list(set(pocket_list)))
    node_map = {node_list[i]:i+lig_size for i in range(len(node_list))}
    
    dist_list = dist_matrix[pos]
    if reset_idx:
        edge_list = [(x,node_map[y]) for x,y in zip(ligand_list, pocket_list)]
    else:
        edge_list = [(x,y) for x,y in zip(ligand_list, pocket_list)]
    
    edge_list += [(y,x) for x,y in edge_list]
    dist_list = np.concatenate([dist_list, dist_list])
    
    return dist_list, edge_list, node_map

def add_identity_fea(lig_fea, pock_fea, comb=1):
    if comb == 1:
        lig_fea = np.hstack([lig_fea, [[1]]*len(lig_fea)])
        pock_fea = np.hstack([pock_fea, [[-1]]*len(pock_fea)])
    elif comb == 2:
        lig_fea = np.hstack([lig_fea, [[1,0]]*len(lig_fea)])
        pock_fea = np.hstack([pock_fea, [[0,1]]*len(pock_fea)])
    else:
        lig_fea = np.hstack([lig_fea, [[0]*lig_fea.shape[1]]*len(lig_fea)])
        if len(pock_fea) > 0:
            pock_fea = np.hstack([[[0]*pock_fea.shape[1]]*len(pock_fea), pock_fea])
    
    return lig_fea, pock_fea

def cons_lig_pock_graph_with_spatial_context(ligand, pocket, add_fea=2, theta=5, keep_pock=False, pocket_spatial=True):
    lig_fea, lig_coord, lig_atoms_raw, lig_edge = ligand
    pock_fea, pock_coord, pock_atoms_raw, pock_edge = pocket
    
    # inter-relation between ligand and pocket
    lig_size = lig_fea.shape[0]
    dm = distance_matrix(lig_coord, pock_coord)
    lig_pock_dist, lig_pock_edge, node_map = edge_ligand_pocket(dm, lig_size, theta=theta, keep_pock=keep_pock)

    # construct ligand graph & pocket graph
    lig_size, lig_fea, lig_edge = cons_mol_graph(lig_edge, lig_fea)
    pock_size, pock_fea, pock_edge = cons_mol_graph(pock_edge, pock_fea)
    
    # construct spatial context graph based on distance
    dm = distance_matrix(lig_coord, lig_coord)
    edges, lig_dist = cons_spatial_gragh(dm, theta=theta)
    if pocket_spatial:
        dm_pock = distance_matrix(pock_coord, pock_coord)
        edges_pock, pock_dist = cons_spatial_gragh(dm_pock, theta=theta)
    lig_edge = edges
    pock_edge = edges_pock
    
    # map new pocket graph
    pock_size = len(node_map)
    pock_fea = pock_fea[sorted(node_map.keys())]
    pock_edge, pock_dist = pocket_subgraph(node_map, pock_edge, pock_dist)
    pock_coord_ = pock_coord[sorted(node_map.keys())]
    
    # construct ligand-pocket graph
    size = lig_size + pock_size
    lig_fea, pock_fea = add_identity_fea(lig_fea, pock_fea, comb=add_fea)

    feas = np.vstack([lig_fea, pock_fea]) if len(pock_fea) > 0 else lig_fea
    edges = lig_edge + lig_pock_edge + pock_edge
    lig_atoms = get_lig_atom_types(feas)
    pock_atoms = get_pock_atom_types(feas)
    assert len(lig_atoms) ==  lig_size and len(pock_atoms) == pock_size
    
    atoms = np.concatenate([lig_atoms, pock_atoms]) if len(pock_fea) > 0 else lig_atoms
    
    lig_atoms_raw = np.array(lig_atoms_raw)
    pock_atoms_raw = np.array(pock_atoms_raw)
    pock_atoms_raw = pock_atoms_raw[sorted(node_map.keys())]
    atoms_raw = np.concatenate([lig_atoms_raw, pock_atoms_raw]) if len(pock_atoms_raw) > 0 else lig_atoms_raw
     
    coords = np.vstack([lig_coord, pock_coord_]) if len(pock_fea) > 0 else lig_coord
    if len(pock_fea) > 0:
        assert size==max(node_map.values())+1
    assert feas.shape[0]==coords.shape[0]
    return lig_size, coords, feas, atoms

def random_split(dataset_size, split_ratio=0.9, seed=0, shuffle=True):
    """random splitter"""
    np.random.seed(seed)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(split_ratio * dataset_size)
    train_idx, valid_idx = indices[:split], indices[split:]
    return train_idx, valid_idx


def process_dataset(core_path, refined_path, dataset_name, output_path, cutoff):
    core_set_list = [x for x in os.listdir(core_path) if len(x) == 4]
    refined_set_list = [x for x in os.listdir(refined_path) if len(x) == 4]
    path = refined_path

    # atomic sets for long-range interactions
    atom_types = [6,7,8,9,15,16,17,35,53]
    atom_types_ = [6,7,8,16]

    # atomic feature generation
    featurizer = Featurizer(save_molecule_codes=False)
    processed_dict = {}
    for name in tqdm(os.listdir(path)):
        if len(name) != 4:
            continue
        processed_dict[name] = gen_feature(path, name, featurizer)

    # interaction features
    processed_dict = pairwise_atomic_types(path, processed_dict, atom_types, atom_types_)
    # load pka (binding affinity) data
    pk_dict = load_pk_data(path+'index/INDEX_general_PL_data.2016')
    data_dict = processed_dict
    for k,v in processed_dict.items():
        v['pk'] = pk_dict[k]
        data_dict[k] = v

    refined_id, refined_data, refined_pk = [], [], []
    core_id, core_data, core_pk = [], [], []

    for k, v in tqdm(data_dict.items()):
        ligand = (v['lig_fea'], v['lig_co'], v['lig_atoms'], v['lig_eg'])
        pocket = (v['pock_fea'], v['pock_co'], v['pock_atoms'], v['pock_eg'])
        graph = cons_lig_pock_graph_with_spatial_context(ligand, pocket, add_fea=3, theta=cutoff, keep_pock=False, pocket_spatial=True)
        cofeat, pk = v['type_pair'], v['pk']
        graph = list(graph) + [cofeat]
        if k in core_set_list:
            core_id.append(k)
            core_data.append(graph)
            core_pk.append(pk)
            continue
        refined_id.append(k)
        refined_data.append(graph)
        refined_pk.append(pk)

    # split train and valid
    train_idxs, valid_idxs = random_split(len(refined_data), split_ratio=0.9, seed=2020, shuffle=True)
    train_g = [refined_data[i] for i in train_idxs]
    train_y = [refined_pk[i] for i in train_idxs]
    valid_g = [refined_data[i] for i in valid_idxs]
    valid_y = [refined_pk[i] for i in valid_idxs]
    train = (train_g, train_y)
    valid = (valid_g, valid_y)
    test = (core_data, core_pk)

    with open(os.path.join(output_path, dataset_name + '_train.pkl'), 'wb') as f:
        pickle.dump(train, f)
    with open(os.path.join(output_path, dataset_name + '_val.pkl'), 'wb') as f:
        pickle.dump(valid, f)
    with open(os.path.join(output_path, dataset_name + '_test.pkl'), 'wb') as f:
        pickle.dump(test, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path_core', type=str)
    parser.add_argument('--data_path_refined', type=str)
    parser.add_argument('--output_path', type=str, default='./data/')
    parser.add_argument('--dataset_name', type=str, default='pdbbind2016')
    parser.add_argument('--cutoff', type=float, default=5.)
    args = parser.parse_args()
    process_dataset(args.data_path_core, args.data_path_refined, args.dataset_name, args.output_path, args.cutoff)
