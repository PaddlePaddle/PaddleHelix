# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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
This file implements the data preprocessing for PDBbind dataset.
"""

import os
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import openbabel
from openbabel import pybel
from tfbio_data import Featurizer
from scipy.spatial import distance_matrix

core_set_list = '3ao4, 3gv9, 1uto, 1ps3, 4ddk, 4jsz, 3g2z, 3dxg, 3l7b, 3gr2, 3kgp, 3fcq, 3lka, 3zt2,3udh, 3g31, 4llx, 4u4s, 4owm, 5aba, 2xdl, 4kz6, 2ymd, 3aru, 1bcu, 3zsx, 4ddh,4mrw, 4eky, 4mrz, 4abg, 5a7b, 3dx1, 4bkt, 2v00, 4cig, 3n7a, 3d6q, 2hb1, 3twp,4agn, 1c5z, 3nq9, 4msn, 2w66, 3kwa, 3g2n, 4cr9, 4ih5, 4de2, 3ozt, 3f3a, 1a30,3ivg, 3u9q, 3rsx, 3pxf, 2wbg, 3rr4, 4w9c, 3mss, 4agp, 4mgd, 1vso, 4jxs, 1q8t,3acw, 4lzs, 3r88, 4ciw, 2w4x, 2brb, 1p1q, 3d4z, 1bzc, 1nc3, 4agq, 4w9l, 2yge,5c1w, 2r9w, 3gy4, 3syr, 3zso, 2br1, 1s38, 3b27, 4gkm, 4m0z, 1w4o, 3ueu, 4ih7, 4jfs, 3ozs, 3bv9, 1gpk, 1syi, 2cbv, 1ydr, 4de3, 3coz, 2wca, 3u5j, 4dli, 1z9g, 3arv,3n86,  5c28,  4j28,  3jvr,  1o5b,  2y5h,  3qqs,  3wz8,  4dld,  3ehy,  3uev,  3ebp,  1o0h,1q8u,  4de1,  4msc,  4w9i,  3ary,  3coy,  3f3c,  2fxs,  4kzq,  2qnq,  1nc1,  2wvt,  1yc1,3bgz, 4wiv, 3k5v, 4eor, 3uew, 2wnc, 2zb1, 2qbr, 3arq, 2j78, 4ea2, 1r5y, 4m0y,1gpn,  2weg,  4kzu,  4mme,  3cj4,  3uo4,  3wtj,  3jvs,  1k1i,  2yfe,  4k77,  2xj7,  2iwx,4f09,  4djv,  4w9h,  4ogj,  1p1n,  3dx2,  2xnb,  3n76,  3pyy,  2zcr,  3oe5,  3jya,  3gbb,3uex, 4f9w, 2wer, 1lpg, 3zdg, 1z95, 1pxn, 3arp, 3f3d, 3tsk, 2j7h, 2xii, 4cra, 4gfm,1oyt, 3p5o, 3gc5, 2vvn, 1qf1, 1ydt, 3pww, 1owh, 2zy1, 3up2, 4j21, 2xys, 2qbq,3oe4, 3rlr, 2xb8, 2c3i, 4e5w, 3f3e, 1u1b, 3qgy, 3ryj, 4j3l, 3prs, 4pcs, 4hge, 1o3f,2qe4,  3uuo,  3cyx,  3e92,  3fur,  2cet,  5tmn,  3ag9,  3kr8,  3nx7,  3fv2,  4eo8,  3e5a,1nvq, 2v7a, 4x6p, 1h23, 4e6q, 2al5, 2qbp, 2zda, 3b68, 2xbv, 3b1m, 2fvd, 2vw5,2wn9, 3ejr, 4qd6, 3u8k, 3ge7, 4crc, 4ivb, 2vkm, 2wtv, 3b5r, 2zcq, 3e93, 4k18,2p4y,  3dd0,  3nw9,  3ui7,  3uri,  1qkt,  1h22,  3gnw,  1sqa,  4jia,  3b65,  3fv1,  4qac,2yki,  3g0w,  4ivd,  4ty7,  2pog,  4gr0,  1eby,  1z6e,  1e66,  4ivc,  4twp,  4rfm,  1y6r,3u8n, 4tmn, 2p15, 3myg, 4gid, 3utu, 5c2h, 1mq6, 5dwr, 4f2w, 2x00, 3o9i, 4f3c'
core_set_list = core_set_list.replace(' ', '').split(',')
featurizer = Featurizer(save_molecule_codes=False)
charge_idx = featurizer.FEATURE_NAMES.index('partialcharge')

def cons_spatial_gragh(dist_matrix, theta=5):
    """
    Construct the spatial graph based on the cutoff theta.
    """
    pos = np.where((dist_matrix<=theta)&(dist_matrix!=0))
    src_list, dst_list = pos
    dist_list = dist_matrix[pos]
    edges = [(x,y) for x,y in zip(src_list, dst_list)]
    return edges


def cons_mol_graph(edges, feas):
    """
    Construct the molecular graph based on bond between atoms.
    """
    size = feas.shape[0]
    edges = [(x,y) for x,y,t in edges]
    return size, feas, edges


def pocket_subgraph(node_map, edge_list):
    """
    Extract the subgraph of protein-pocket from the edge_list.
    """
    edge_l = []
    node_l = set()
    for x,y in edge_list:
        if x in node_map and y in node_map:
            x, y = node_map[x], node_map[y]
            edge_l.append((x,y))
            node_l.add(x)
            node_l.add(y)
    return edge_l


def edge_ligand_pocket(dist_matrix, lig_size, theta=4, keep_pock=False, reset_idx=True):
    """
    Extract the edges between the ligand and protein-pocket.
    """
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
    return dist_list, edge_list, node_map


def add_identity_fea(lig_fea, pock_fea, comb=1):
    """
    extend the atom features into features of protein-ligand complex.
    """
    if comb == 1:
        lig_fea = np.hstack([lig_fea, [[1]]*len(lig_fea)])
        pock_fea = np.hstack([pock_fea, [[-1]]*len(pock_fea)])
    elif comb == 2:
        lig_fea = np.hstack([lig_fea, [[1,0]]*len(lig_fea)])
        pock_fea = np.hstack([pock_fea, [[0,1]]*len(pock_fea)])
    else:
        lig_fea = np.hstack([lig_fea, [[0]*lig_fea.shape[1]]*len(lig_fea)])
        pock_fea = np.hstack([[[0]*pock_fea.shape[1]]*len(pock_fea), pock_fea])
    return lig_fea, pock_fea


def cons_lig_pock_graph_with_spatial_context(ligand, pocket, add_fea=2, theta=5, keep_pock=False, pocket_spatial=True):
    """
    construct the protein-ligand complex graph.
    """
    lig_fea, lig_coord, lig_edge = ligand
    pock_fea, pock_coord, pock_edge = pocket
    
    # inter-relation between ligand and pocket
    lig_size = lig_fea.shape[0]
    dm = distance_matrix(lig_coord, pock_coord)
    dist_list, lig_pock_edge, node_map = edge_ligand_pocket(dm, lig_size, theta=theta, keep_pock=keep_pock)
    
    # construct ligand graph & pocket graph
    lig_size, lig_fea, lig_edge = cons_mol_graph(lig_edge, lig_fea)
    pock_size, pock_fea, pock_edge = cons_mol_graph(pock_edge, pock_fea)
    
    # construct spatial context graph based on distance
    dm = distance_matrix(lig_coord, lig_coord)
    edges = cons_spatial_gragh(dm, theta=theta)
    if pocket_spatial:
        dm_pock = distance_matrix(pock_coord, pock_coord)
        edges_pock= cons_spatial_gragh(dm_pock, theta=theta)
    lig_edge = edges
    pock_edge = edges_pock
    
    # map new pocket graph
    pock_size = len(node_map)
    pock_fea = pock_fea[sorted(node_map.keys())]
    pock_edge = pocket_subgraph(node_map, pock_edge)
    pock_coord_ = pock_coord[sorted(node_map.keys())]
    
    # construct ligand-pocket graph
    size = lig_size + pock_size
    lig_fea, pock_fea = add_identity_fea(lig_fea, pock_fea, comb=add_fea)

    feas = np.vstack([lig_fea, pock_fea])
    edges = lig_edge + lig_pock_edge + pock_edge
    coords = np.vstack([lig_coord, pock_coord_])
    assert size==max(node_map.values())+1
    assert feas.shape[0]==coords.shape[0]

    return size, feas, edges, coords


## function -- feature generation for protein and ligand
def gen_feature(path, name):
    """
    feature generation for protein and ligand
    """
    ligand = next(pybel.readfile('mol2', '%s/%s/%s_ligand.mol2' % (path, name, name)))
    pocket = next(pybel.readfile('mol2' ,'%s/%s/%s_pocket.mol2' % (path, name, name)))
    ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)
    assert (ligand_features[:, charge_idx] != 0).any()
    pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)
    assert (pocket_features[:, charge_idx] != 0).any()
    ligand_edges = gen_pocket_graph(ligand)
    pocket_edges = gen_pocket_graph(pocket)
    return {'lig_co': ligand_coords, 'lig_fea': ligand_features, 'lig_eg': ligand_edges, 'pock_co': pocket_coords, 'pock_fea': pocket_features, 'pock_eg': pocket_edges}


## function -- generate molecular graph
def gen_pocket_graph(pocket):
    """
    generate molecular graph 
    """
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


## function -- load affinity values from dataset
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


def process_dataset(path, dataset_name, output_path, cutoff):
    """
    main function
    """
    refine_set_processed = {}
    for name in tqdm(os.listdir(path)):
        if len(name) != 4:
            continue
        refine_set_processed[name] = gen_feature(path, name)

    # load pk (binding affinity) data
    pk_data_path = os.path.join(path, 'index/INDEX_refined_data.2016')
    refined_pk_dict = load_pk_data(pk_data_path)
    # add pk into the dict
    refined_dict = refine_set_processed
    for k,v in refine_set_processed.items():
        v['pk'] = refined_pk_dict[k]
        refined_dict[k] = v
    
    train_drug, train_pk = [], []
    test_drug, test_pk = [], []

    for k, v in tqdm(refined_dict.items()):
        ligand = (v['lig_fea'], v['lig_co'], v['lig_eg'])
        pocket = (v['pock_fea'], v['pock_co'], v['pock_eg'])
        graph = cons_lig_pock_graph_with_spatial_context(ligand, pocket, add_fea=3, theta=cutoff, keep_pock=False, pocket_spatial=True)

        pk = v['pk']
        if k not in core_set_list:
            train_drug.append(graph)
            train_pk.append(pk)
        else:
            test_drug.append(graph)
            test_pk.append(pk)
            
    print(len(train_drug), len(test_drug), train_drug[0][1].shape)

    # save
    train_data = (train_drug, train_pk)
    test_data = (test_drug, test_pk)
    output_train_path = os.path.join(output_path, '%s_train.pickle' % dataset_name)
    output_test_path = os.path.join(output_path, '%s_test.pickle' % dataset_name)
    with open(output_train_path, 'wb') as f:
        pickle.dump(train_data, f)
    with open(output_test_path, 'wb') as f:
        pickle.dump(test_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../refined-set/')
    parser.add_argument('--output_path', type=str, default='../data/')
    parser.add_argument('--dataset_name', type=str, default='v2016_LPHIN3f5t_Sp')
    parser.add_argument('--cutoff', type=float, default=5.)
    args = parser.parse_args()

    process_dataset(args.data_path, args.dataset_name, args.output_path, args.cutoff)
