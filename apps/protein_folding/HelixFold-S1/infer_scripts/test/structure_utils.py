import os
import sys
import gzip
import pickle
import time
from collections import defaultdict

import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from scipy.spatial.distance import cdist

def load_ccd_dict(ccd_preprocessed_path):
    assert os.path.exists(ccd_preprocessed_path),\
              (f'[CCD] ccd_preprocessed_path: {ccd_preprocessed_path} not exist.')
    st_1 = time.time()
    if 'pkl.gz' in ccd_preprocessed_path:
        with gzip.open(ccd_preprocessed_path, "rb") as fp:
            ccd_preprocessed_dict = pickle.load(fp)
    elif '.pkl' in ccd_preprocessed_path:
        with open(ccd_preprocessed_path, "rb") as fp:
            ccd_preprocessed_dict = pickle.load(fp)
    print(f'[CCD] load ccd dataset done. use {time.time()-st_1}s;'\
                    f'Has length of {len(ccd_preprocessed_dict)}')
    
    return ccd_preprocessed_dict


def find_bond_index(residue, ccd, atom_index_mapping):
    ret = []
    for cov_b in ccd['coval_bonds']:
        try:
            bond_l = atom_index_mapping[residue[cov_b[0]]]
            bond_r = atom_index_mapping[residue[cov_b[1]]]
            ret.append([bond_l, bond_r])
        except KeyError as e:
            # print(e.__str__(), "not found")
            continue
    return ret


def get_ccd_bond(structure, unique_ccd_info):
    atoms = list(structure.get_atoms())
    atom_index_mapping = {atom: idx for idx, atom in enumerate(atoms)}
    mask = np.zeros([len(atoms), len(atoms)], dtype=bool)

    for r in structure.get_residues():
        if r.resname not in unique_ccd_info:
            continue    # 跳过不在 CCD 中的残基
        cur_ccd = unique_ccd_info[r.resname]
        bd_i_list = find_bond_index(r, cur_ccd, atom_index_mapping)
        for bd_i in bd_i_list:
            mask[bd_i[0], bd_i[1]] = True
            mask[bd_i[1], bd_i[0]] = True
    return mask

def get_bond_matrix(structure):
    atoms = list(structure.get_atoms())
    atom_index_mapping = {atom: idx for idx, atom in enumerate(atoms)}
    bond_mask = np.zeros([len(atoms), len(atoms)], dtype=bool)

    # 遍历每条链
    for chain in structure.get_chains():
        residues = list(chain.get_residues())
        
        for i in range(len(residues)-1):
            current = residues[i]
            next_res = residues[i+1]
            
            # 处理肽键（蛋白质）
            if is_aa(current) and is_aa(next_res):
                try:
                    c_atom = current["C"]
                    n_atom = next_res["N"]
                    bond_mask[atom_index_mapping[c_atom], atom_index_mapping[n_atom]] = True
                    bond_mask[atom_index_mapping[n_atom], atom_index_mapping[c_atom]] = True
                except KeyError:
                    pass
                
            # 处理磷酸二酯键（核酸）
            elif current.resname in ['DA', 'DC', 'DG', 'DT', 'A', 'C', 'G', 'U']:
                try:
                    o3_prime = current["O3'"]
                    p_next = next_res["P"]
                    bond_mask[atom_index_mapping[o3_prime], atom_index_mapping[p_next]] = True
                    bond_mask[atom_index_mapping[p_next], atom_index_mapping[o3_prime]] = True
                except KeyError:
                    pass
    return bond_mask

def check_clashes(cif_file, clash_cutoff=1.8):
    """
    计算一个 cif 结构文件中原子冲突的比例。一个原子与其他非共价结合的任意原子的距离在 clash_cutoff 之内(单位是 Å)，则认为此原子存在冲突。
    排除正常的共价结合：
        CCD 内部的共价键，肽键，磷酸二酯键
    未排除的共价结合：
        自定义小分子内部的共价键
    """
    parser = MMCIFParser()
    if cif_file.endswith('.gz'):
        with gzip.open(cif_file, 'rt') as f:
            structure = parser.get_structure("protein", f)
    else:
        with open(cif_file, 'r') as f:
            structure = parser.get_structure("protein", f)
    atoms = list(structure.get_atoms())
    coords = np.array([atom.get_coord() for atom in atoms])
    
    # 计算所有原子间的距离矩阵
    dist_matrix = cdist(coords, coords)
    np.fill_diagonal(dist_matrix, np.inf)  # 忽略自身

    # 排除共价键干扰（注意仅处理 CCD 内部的共价键，和肽键+磷酸二酯键，手动指定的连接键不考虑，自定义小分子内部的键暂不考虑）
    # 1. 找到结构文件中 unique ccd
    unique_ccd = set([r.resname for r in structure.get_residues()])

    # 2.
    ccd_dict = load_ccd_dict("./infer_scripts/demo_data/ccd_preprocessed_etkdg.pkl.gz")
    unique_ccd_info = {k: ccd_dict[k] for k in unique_ccd if k in ccd_dict}

    # 3. 排除 CCD 内部共价键
    mask = np.zeros_like(dist_matrix, dtype=bool)
    # 3.1 排除 CCD 内部共价键
    mask |= get_ccd_bond(structure, unique_ccd_info)
    # 3.2 排除肽键、磷酸二酯键
    mask |= get_bond_matrix(structure)

    # 4. 标记冲突（距离 < 1.5Å）
    clashes = dist_matrix < clash_cutoff
    clashes[mask] = False
    
    atom_clash = clashes.max(axis=0)
    clash_ratio = atom_clash.mean()
    
    return clash_ratio, clashes

# 示例调用
if __name__ == "__main__":
    cif_path = sys.argv[1]
    ratio, clash_flags = check_clashes(cif_path)
    # ratio, clash_flags = check_clashes("./infer_scripts/testcases/clash/1hbo-assembly1_nowater.cif.gz")
    print(f"冲突原子比例: {ratio:.3%}")
    if ratio > 0.01:
        print("结构冲突原子比例大于1%！需要人工检查！")
    else:
        print("结构冲突原子比例小于1%，检查通过！")
