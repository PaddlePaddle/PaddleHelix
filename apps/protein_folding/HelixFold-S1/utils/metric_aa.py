#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

"""Metrics."""

import sys
import os
from os.path import join, basename, dirname, exists
from typing import Any, Tuple, Union, List, Optional
import numpy as np
import copy
import subprocess
import pickle
from Bio import PDB
import paddle
import paddle.distributed as dist
import json
import re
from collections import defaultdict
from scipy import stats
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from rdkit import Chem  
from rdkit.Chem import AllChem  
from rdkit.Chem import rdMolAlign

from posebusters import PoseBusters
from pathlib import Path

import matplotlib.pyplot as plt
# import seaborn as sns

from helixfold.common import protein
from helixfold.common import all_atom_pdb_save
from helixfold.model.utils import get_confidence_metrics
from helixfold.model import lddt
from utils.utils import tree_map, tree_flatten, tree_filter
from ppfleetx.distributed.protein_folding import dp
from helixfold.data.utils import atom_level_keys, map_to_continuous_indices, build_interface_info_with_random_residues_module2 as build_interface_info_with_random_residues
from helixfold.model.chain_align_np_aa import get_transform
from scipy.spatial import distance_matrix

from .ensemble_analyze import generate_ensemble_report

def dist_all_reduce(x, return_num=False, distributed=False):
    x_num = len(x)
    x_sum = 0 if x_num == 0 else np.sum(x)
    if distributed:
        if os.environ.get('DCU_MODE_SOFTMAX_CLIP', '0') == '1':
            import paddle.distributed as dist
            x_num = int(dist.all_reduce(paddle.to_tensor(x_num, dtype='int64')))
            x_sum = int(dist.all_reduce(paddle.to_tensor(x_sum, dtype='float32')))
        else:
            x_num = int(dp.all_reduce(paddle.to_tensor(x_num, dtype='int64')))
            x_sum = float(dp.all_reduce(paddle.to_tensor(x_sum, dtype='float32')))
    x_mean = 0 if x_num == 0 else x_sum / x_num
    if return_num:
        return x_mean, x_num
    else:
        return x_mean


def dist_all_gather(x, distributed=False):
    if distributed:
        import paddle.distributed as dist
        value_list = []
        dist.all_gather_object(value_list, x)
        return value_list
    else:
        return [x]


def merge_complex_to_monomer(input_pdb, output_pdb, new_chain_id='A'):
    """
    Merge a complex PDB file into a monomer PDB file by
    adding all residues from all chains into a single chain A.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', input_pdb)

    # Create a new structure
    new_structure = PDB.Structure.Structure('new_structure')
    new_model = PDB.Model.Model(0)
    new_chain = PDB.Chain.Chain(new_chain_id)

    # Variable to keep track of residue numbering
    new_residue_id = 1

    # Iterate over all chains in all models in the input structure
    offset = 0
    for model in structure:
        for chain in model:
            max_residue_id = 0
            for residue in chain:
                # Copy the residue and set its new id
                new_residue = residue.copy()
                new_residue.id = (' ', residue.id[1] + offset, ' ')
                new_chain.add(new_residue)
                max_residue_id = max(max_residue_id, residue.id[1])
            offset += max_residue_id + 1

    # Add the new chain to the new model and the new model to the new structure
    new_model.add(new_chain)
    new_structure.add(new_model)

    # Write the new structure to a PDB file
    io = PDB.PDBIO()
    io.set_structure(new_structure)
    io.save(output_pdb)


def calc_ligand_rmsd(coords1, coords2):
    """
    Calculate RMSD between predicted ligand and experimental ligand.
    v1, v2 are numpy arrays of shape (N, 3). N is the number of valid atoms.
    """
    assert coords1.shape == coords2.shape  

    distance_squared = np.sum((coords1 - coords2) ** 2, axis=1)  
    rmsd = np.sqrt(np.mean(distance_squared))  
      
    return rmsd


def create_mol(atomic_nums, bonds, atom_poses):
    """
    atomic_nums: (N,)
    bonds: (M, 2)
    atom_poses: (N, 3)
    """
    N = len(atomic_nums)

    ## create mol with atoms and bonds
    mol = Chem.RWMol()  
    for num in atomic_nums:
        mol.AddAtom(Chem.Atom(num))
    used_bonds = set()
    for i, j in bonds:
        if (i, j) in used_bonds or (j, i) in used_bonds:
            continue
        mol.AddBond(i, j)
        used_bonds.add((i, j))
    mol = mol.GetMol()  
    
    ## add conformation
    conformer = Chem.Conformer(len(mol.GetAtoms()))  
    for i, (x, y, z) in enumerate(atom_poses):  
        conformer.SetAtomPosition(
                i, (float(x), float(y), float(z)))  
    mol.AddConformer(conformer)
    
    return mol


def get_mol_coords(mol):
    """get_mol_coords"""
    atom_poses = []
    conformer = mol.GetConformer()
    for atom_index in range(mol.GetNumAtoms()):
        x, y, z = conformer.GetAtomPosition(atom_index)
        atom_poses.append([x, y, z])
    return np.array(atom_poses)


def set_mol_coords(mol, atom_poses):
    """set_mol_coords"""
    # TODO: some mol has `H` in the end which can't be removed by 
    #  Chem.RemoveHs()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    symbols = [x for x in symbols if x != 'H']
    assert len(symbols) == len(atom_poses)
    conformer = mol.GetConformer()
    for i, (x, y, z) in enumerate(atom_poses):  
        conformer.SetAtomPosition(
                i, (float(x), float(y), float(z)))  


def calc_ligand_obrms(atomic_nums, bonds, pred_coords, ref_coords):
    """
    atomic_nums: (N,)
    bonds: (M, 2)
    pred_coords: (N, 3)
    ref_coords: (N, 3)
    """
    pred_mol = create_mol(atomic_nums, bonds, pred_coords)
    ref_mol = create_mol(atomic_nums, bonds, ref_coords)
    rmsd = rdMolAlign.CalcRMS(pred_mol, ref_mol)
    return rmsd, pred_mol, ref_mol


def calc_ligand_obrms_modi(pred_atomic_nums, pred_bonds, pred_coords, atomic_nums, bonds, ref_coords):
    """
    atomic_nums: (N,)
    bonds: (M, 2)
    pred_coords: (N, 3)
    ref_coords: (N, 3)
    """
    pred_mol = create_mol(pred_atomic_nums, pred_bonds, pred_coords)
    ref_mol = create_mol(atomic_nums, bonds, ref_coords)
    rmsd = rdMolAlign.CalcRMS(pred_mol, ref_mol)
    return rmsd, pred_mol, ref_mol


def calc_posebusters(pred_mol, ref_mol, protein_pdb_file):
    """
    calc_posebusters
    """
    buster = PoseBusters(config="redock")
    df = buster.bust([pred_mol], ref_mol, protein_pdb_file)
    column_names = df.columns.values
    data_array = df.to_numpy()[0].astype('str')
    bust_scores = {f'bust-{k}': float(v == 'True') for k, v in zip(column_names, data_array)}
    return df, bust_scores


def get_tm_scores(tm_score_bin, pred_pdb_file, exp_pdb_file, is_complex=False):
    """
    Get TM-score, RMSD, GDT-TS and GDT-HA scores.
    is_complex: whether the input pdb is a complex.
    """
    assert exists(pred_pdb_file), pred_pdb_file
    assert exists(exp_pdb_file), exp_pdb_file
    if is_complex:
        cmd = f'{tm_score_bin} -c -ter 0 {pred_pdb_file} {exp_pdb_file} > {pred_pdb_file}.tmscore'
    else:
        cmd = f'{tm_score_bin} {pred_pdb_file} {exp_pdb_file} > {pred_pdb_file}.tmscore'
    # print(f"cmd: {cmd}")
    # s = os.popen(cmd).readlines()
    res = {}
    # for line in s:
    os.system(cmd)
    for line in open(f'{pred_pdb_file}.tmscore','r'):
        line = line.strip()
        if line[:8] == "TM-score":
            res['TM-score'] = float(line.split()[2])
        elif line[:6] == "GDT-TS":
            res['GDT-TS'] = float(line.split()[1])
        elif line[:6] == "GDT-HA":
            res['GDT-HA'] = float(line.split()[1])
        elif line[:4] == 'RMSD':
            res['RMSD'] = float(line.split('=')[1].strip())
    return res


def get_tm_score_align(tm_score_bin, pred_pdb_file, exp_pdb_file, is_complex=False):
    """
    Get TM-score, RMSD, GDT-TS and GDT-HA scores.
    is_complex: whether the input pdb is a complex.
    """
    assert exists(pred_pdb_file), pred_pdb_file
    assert exists(exp_pdb_file), exp_pdb_file
    if is_complex:
        cmd = f'{tm_score_bin} -c -ter 0 {pred_pdb_file} {exp_pdb_file} > {pred_pdb_file}.tmscore'
    else:
        cmd = f'{tm_score_bin} {pred_pdb_file} {exp_pdb_file} > {pred_pdb_file}.tmscore'
    res = {}
    os.system(cmd)
    with open(f'{pred_pdb_file}.tmscore','r') as f:
        for line in f:
            line = line.strip()
            if 'rotation matrix' in line:
                break
        next(f)
        mat = []
        for i in range(3):
            line = next(f)
            mat.append([float(x) for x in line.strip().split()[1:]])
        mat = np.array(mat)
        t = mat[:, 0]
        R = mat[:, 1:]
    return R.T, t


def get_lddt_scores(lddt_score_bin, pred_pdb_file, exp_pdb_file, is_complex=False):
    """get_lddt_scores"""
    def _get_lddt(cmd):
        s = os.popen(cmd).readlines()
        for line in s:
            segs = line.strip().split(':')
            if len(segs) == 2 and segs[0] == "Global LDDT score":
                return float(segs[1].strip())
        return np.nan

    assert exists(pred_pdb_file), pred_pdb_file
    assert exists(exp_pdb_file), exp_pdb_file
    # merge all chains to one monomer and save to a new pdb
    if is_complex:
        try:
            new_pred_pdb_file = pred_pdb_file.replace('.pdb', '.c2m.pdb')
            merge_complex_to_monomer(pred_pdb_file, new_pred_pdb_file)
            pred_pdb_file = new_pred_pdb_file
            new_exp_pdb_file = exp_pdb_file.replace('.pdb', '.c2m.pdb')
            merge_complex_to_monomer(exp_pdb_file, new_exp_pdb_file)
            exp_pdb_file = new_exp_pdb_file
        except Exception as e:
            print(f"[ERROR] failed to merge complex to monomer, {e}")
            return {'LDDT': 0, 'LDDTa': 0}
    res = {}
    # get lddt score
    cmd = f'{lddt_score_bin} {pred_pdb_file} {exp_pdb_file}'
    res['LDDT'] = _get_lddt(cmd)
    # get lddta score
    cmd = f'{lddt_score_bin} -c {pred_pdb_file} {exp_pdb_file}'
    res['LDDTa'] = _get_lddt(cmd)
    return res

def get_dockq_scores_all_inter(pred_pdb_file, exp_pdb_file, raw_output_path=None):
    """ new dockq tool which compute avg metric for all interfaces. """
    run_dockq_cmd = f'{sys.executable} -m DockQ {pred_pdb_file} {exp_pdb_file}'
    try: 
        run_output = subprocess.check_output(run_dockq_cmd, shell=True)
    except Exception as e:
        print('[ERROR] failed to run DockQ', e)
        # print(run_output)
        return {'DockQ': 0, 'Fnat': 0, 'Fnonnat': 0, 'iRMS': 0, 'LRMS': 0}

    if not raw_output_path is None:
        with open(raw_output_path, "w") as raw_out: raw_out.write(run_output.decode('utf8'))

    res = defaultdict(list)
    for line in run_output.decode('utf8').split("\n"):
        for metric in ['DockQ', 'Fnat', 'Fnonnat', 'iRMS', 
                        'LRMS', "clashes", "F1", "DockQ_F1"]:
            """
            line example:
                Native chains: A, C
                Model chains: A, C
                DockQ: 0.809
                irms: 0.765
                Lrms: 2.153
                fnat: 0.692
                fnonnat: 0.357
                clashes: 0.000
                F1: 0.667
                DockQ_F1: 0.800
            """
            metric_val = re.findall(f'{metric.lower()}: (\d+\.\d+)', line.lower())
            if len(metric_val) > 0:
                res[metric].append(float(metric_val[0]))

    for metric in res:
        res[metric] = np.mean(res[metric])
    return res
    
def get_dockq_scores(dockq_score_dir, pred_pdb_file, exp_pdb_file, chain_info_list, raw_output_path=None):
    """ tbd."""
    export_cmd = f"export PATH=./EMBOSS-6.6.0/emboss:$PATH"
    ## add no_needle to fix dcu issue in locating needle
    run_dockq_cmd = f'{sys.executable} DockQ.py {pred_pdb_file} {exp_pdb_file} -no_needle'
    dockq_group = []
    for c_idx, chain_info in enumerate(chain_info_list):
        if chain_info.get("is_ligand", False):
            # dockq_group.append(chain_info["id"])
            # use relative pos of chain_id to identify grouped chain
            dockq_group.append(protein.PDB_CHAIN_IDS[c_idx+1]) 
    if len(dockq_group) > 0:
        run_dockq_cmd += f" -native_chain1 {' '.join(dockq_group)}"
    run_cmd = f"cd {dockq_score_dir} && {export_cmd} && {run_dockq_cmd}"
    try: 
        run_output = subprocess.check_output(run_cmd, shell=True)
    except Exception as e:
        print('[ERROR] failed to run DockQ', e)
        return None

    if not raw_output_path is None:
        with open(raw_output_path, "w") as raw_out: raw_out.write(run_output.decode('utf8'))

    res = {}
    dockq_score = 0
    for line in run_output.decode('utf8').split("\n"):
        for metric in ['DockQ', 'Fnat', 'Fnonnat', 'iRMS', 'LRMS']:
            if line.startswith(metric):
                res[metric] = float(line.split(' ')[1])
    return res


def merge_pdb_files(pdb_files, output_file):
    """merge pdb files into one pdb file"""
    with open(output_file, 'w') as outfile:
        for pdb_file in pdb_files:
            with open(pdb_file, 'r') as infile:
                for line in infile:
                    if line.strip() == 'END':
                        continue
                    outfile.write(line)
                outfile.write('TER\n')
        outfile.write('END\n')


def get_common_feat_for_pdb_save(batch, batch_i):
    """get common_feat from batch data"""
    common_feat = {k: batch['feat'][k][batch_i] 
                if k in batch['feat'] else batch['label'][k][batch_i]
                for k in all_atom_pdb_save.required_keys_for_saving}
    common_feat['all_ccd_ids'] = str(common_feat['all_ccd_ids']).split()
    common_feat['all_atom_ids'] = str(common_feat['all_atom_ids']).split()
    for feat_key in common_feat:
        if isinstance(common_feat[feat_key], paddle.Tensor):
            common_feat[feat_key] = common_feat[feat_key].numpy()
    return common_feat


def save_train_allatom_results(batch, results, out_dir):
    """save allatom results from training"""
    os.makedirs(out_dir, exist_ok=True)
    diff_results = results['diffusion_module']

    ## save structures
    x_names = ['y', 'x_noisy', 'x_denoised', 'x_gt', 'x_gt_aligned']
    for x_name in x_names:
        x = diff_results[x_name].numpy()
        batch_size, diff_batch_size = x.shape[:2]
        for batch_i in range(batch_size):
            for diff_i in range(diff_batch_size):
                pdb_file = os.path.join(out_dir, f'batch{batch_i}-diff{diff_i}-{x_name}.pdb')
                common_feat = get_common_feat_for_pdb_save(batch, batch_i)
                all_atom_pdb_save.prediction_to_pdb(x[batch_i, diff_i], common_feat, pdb_file)

    ## save values
    v_names = ['t_hat', 'loss_mse', 'loss_mse_w', 
            'loss_huber', 'loss_smooth_lddt',
            'loss_rmsd_frz-all', 'loss_rmsd_frz-protein',
            'loss_rmsd_frz-dna', 'loss_rmsd_frz-rna',
            'loss_rmsd_frz-ligand']
    for v_name in v_names:
        if v_name not in diff_results:
            continue
        np.savetxt(join(out_dir, f'{v_name}.txt'), 
                diff_results[v_name].numpy().T, fmt='%f')
    open(join(out_dir, 'prot_name.txt'), 'w').write('\n'.join(batch['name']))

    ## save type ratio
    with open(join(out_dir, 'type_ratio.txt'), 'w') as f:
        mask = batch['feat']['seq_mask'].numpy() == 1
        for k in ['is_protein', 'is_dna', 'is_rna', 'is_ligand']:
            string = f'{k} '
            counts = (batch['feat'][k] * mask).numpy().sum(1)
            string += ' '.join([str(x) for x in counts])
            f.write(string + '\n')


def merge_pdb_files(pdb_files, output_file):
    """merge pdb files into one pdb file"""
    with open(output_file, 'w') as outfile:
        for pdb_file in pdb_files:
            with open(pdb_file, 'r') as infile:
                for line in infile:
                    if line.strip() == 'END':
                        continue
                    outfile.write(line)
                outfile.write('TER\n')
        outfile.write('END\n')


def save_allatom_results(
        batch, results, batch_i, diff_i,
        xt_traj_file=None, x0_traj_file=None,
        pred_pdb_file=None, pred_pdb_masked_file=None, 
        exp_pdb_file=None, exp_pdb_masked_file=None,
        save_trajectory=False):

    chain_info_list = json.loads(batch['chain_info_list'][batch_i])
    diff_results = results['diffusion_module']
    # 1 feat extraction
    common_feat = {k: batch['feat'][k][batch_i]
            for k in all_atom_pdb_save.required_keys_for_saving if k in batch['feat']}
    common_feat.update(
        {k: batch['label'][k][batch_i]
            for k in  all_atom_pdb_save.required_keys_for_saving if k in batch['label']}
    )
    common_feat['all_ccd_ids'] = str(common_feat['all_ccd_ids']).split()
    common_feat['all_atom_ids'] = str(common_feat['all_atom_ids']).split()
    pred_dict = {
        "pos": diff_results['final_atom_positions'][:, diff_i].numpy(),
        "mask": diff_results['final_atom_mask'][:, diff_i].numpy(),
    }
    exp_dict = {
        "pos": batch['label']['all_atom_pos'].numpy(),
        "mask": batch['label']['all_atom_pos_mask'].numpy(),
    }
    n_token = batch['label']['residue_index'].shape[1]
    token_mask = batch['label']['all_centra_token_indice_mask'][batch_i].numpy().astype('bool')
    token_mask = token_mask[:n_token]
    atom_mask = np.logical_and(pred_dict["mask"] > 0, 
                exp_dict["mask"] > 0)[batch_i]  # [N_atom]
    atom_mask = np.logical_and(atom_mask, 
            token_mask[batch['feat']['ref_token2atom_idx'][batch_i].numpy()])
    # token_mask = paddle.geometric.segment_mean(data=atom_mask.astype('int32'), 
    #         segment_ids=ref_token2atom_idx.astype('int32'))  # [N_token]
    for feat_key in common_feat:
        if isinstance(common_feat[feat_key], paddle.Tensor):
            common_feat[feat_key] = common_feat[feat_key].numpy()

    def apply_mask(key, val, a_mask=atom_mask, t_mask=token_mask):
        """ apply mask to val """
        val = np.array(val)
        if key in atom_level_keys:
            if key in ['ref_token2atom_idx']:
                return map_to_continuous_indices(val[a_mask])
            return val[a_mask]
        elif key == 'pae':
            return val[:n_token, :n_token][t_mask:, :][:, t_mask]
        else:
            return val[:n_token][t_mask]

    # tensor to numpy
    for feat_key in common_feat:
        if isinstance(common_feat[feat_key], paddle.Tensor):
            common_feat[feat_key] = common_feat[feat_key].numpy()

    # 2.0 save prediction
    all_atom_pdb_save.prediction_to_pdb(pred_dict["pos"][batch_i], 
            common_feat, pred_pdb_file)

    # 2.1 save ground truth
    all_atom_pdb_save.prediction_to_pdb(exp_dict["pos"][batch_i], 
            common_feat, exp_pdb_file)

    try:
        common_feat_masked = {k: apply_mask(k, v) for k, v in common_feat.items()}
        # 3.0 save prediction masked 
        all_atom_pdb_save.prediction_to_pdb(
            pred_dict["pos"][batch_i][atom_mask], 
            common_feat_masked, 
            pred_pdb_masked_file)

        # 3.1 save ground truth masked
        all_atom_pdb_save.prediction_to_pdb(
            exp_dict["pos"][batch_i][atom_mask], 
            common_feat_masked, 
            exp_pdb_masked_file)

        if batch['feat']['interface_source'][batch_i] == 'gt':
            interface_info = batch['feat']['interface_info_sample'][batch_i].numpy()
        elif batch['feat']['interface_source'][batch_i] == 'gen':
            repeat_j = 0
            if 'gen_interfaces' in results['interface_head'] \
                and len(results['interface_head']['gen_interfaces']) > batch_i \
                and len(results['interface_head']['gen_interfaces'][batch_i]) > repeat_j:
                interface_info = results['interface_head']['gen_interfaces'][batch_i][repeat_j].interface_map()
            else:
                interface_info = None
        else:
            interface_info = None

        # check if batch contains interface info
        if 'interface_info' in batch['feat']:
            print("[DEBUG]: interface_info found in  batch['feat'] ")
            # compute interface atom mask
            interface_info = batch['feat']['interface_info'][batch_i].numpy()
            if len(interface_info.shape) > 2:
                # if multiple interface info are given, using the first one
                interface_info = interface_info[:,:,0]
            if interface_info.sum() > 0:
                print("[DEBUG]: interface_info.sum() > 0 ")
                interface_token_mask = np.logical_or(np.max(interface_info, axis=0), np.max(interface_info, axis=1))
                interface_token_mask = np.logical_and(interface_token_mask, token_mask)
                interface_atom_mask = interface_token_mask[batch['feat']['ref_token2atom_idx'][batch_i].numpy()]
                interface_atom_mask = np.logical_and(interface_atom_mask, atom_mask)

                interface_common_feat_masked = {k: apply_mask(k, v, interface_atom_mask, interface_token_mask) for k, v in common_feat.items()}

                # 3.3 save interface prediction masked
                pred_pdb_interface_masked_file = pred_pdb_masked_file.replace('.pdb', '_interface.pdb')
                all_atom_pdb_save.prediction_to_pdb(
                    pred_dict["pos"][batch_i][interface_atom_mask], 
                    interface_common_feat_masked, 
                    pred_pdb_interface_masked_file)
                print(f"[DEBUG]: saved  pred interface struct to {pred_pdb_interface_masked_file}")

                # 3.4 save interface ground truth masked
                exp_pdb_interface_masked_file = exp_pdb_masked_file.replace('.pdb', '_interface.pdb')
                all_atom_pdb_save.prediction_to_pdb(
                    exp_dict["pos"][batch_i][interface_atom_mask], 
                    interface_common_feat_masked, 
                    exp_pdb_interface_masked_file)
                print(f"[DEBUG]: saved  exp interface struct to {exp_pdb_interface_masked_file}")
            else:
                print("[DEBUG]: interface_info is all zero ")    
        else:
            print("[DEBUG]: interface_info not found in  batch['feat'] ")

    except Exception as e:
        import traceback
        print(f'Error saving masked pdbs: {e}')
        traceback.print_exc(file=sys.stdout)

    # 4 save traj
    def _save_trajectory(atom_pos_list, pdb_file):
        """ atom_pos_list: (T, N_atom, 3) """
        tmp_files = []
        for ti, atom_pos in enumerate(atom_pos_list):
            tmp_file = f'{pdb_file}.tmp_{ti:03d}'
            pred_dict.update({
                "pos": atom_pos
            })
            all_atom_pdb_save.prediction_to_pdb(pred_dict["pos"], common_feat, tmp_file)
            tmp_files.append(tmp_file)
        merge_pdb_files(tmp_files, pdb_file)
        os.system(f'rm {" ".join(tmp_files)}')

    if save_trajectory:
        _save_trajectory(diff_results["x0_list"][batch_i, diff_i], x0_traj_file)
        _save_trajectory(diff_results["xt_list"][batch_i, diff_i], xt_traj_file)

def draw_interface_heatmap(interface_map, asym_id, output_file, title=''):

    N = interface_map.shape[0]
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw the heatmap
    sns.heatmap(interface_map, ax=ax, cmap='Blues', cbar_kws={'label': 'Interface Probability'})
    
    # Identify unique chain boundaries
    chain_boundaries = np.where(np.diff(asym_id))[0] + 1
    chain_boundaries = np.concatenate(([0], chain_boundaries, [N]))
    
    # Draw bold lines at chain boundaries
    for boundary in chain_boundaries[:-1]:
        ax.axhline(boundary, color='black', linewidth=2)
        ax.axvline(boundary, color='black', linewidth=2)
 
    # Set titles and labels
    ax.set_title(title)
    ax.set_xlabel('Token Index')
    ax.set_ylabel('Token Index')
    
    # Show the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return output_file

def save_interface_map(batch, results, batch_i, output_dir, base_name):

    asym_id = batch['feat']['asym_id'][batch_i].numpy() # [N_token,]
    # save ground truth 
    if 'interface_info_full' in batch['feat']:    
        gt_interface_map = batch['feat']['interface_info_full'][batch_i].numpy() # [N_token, N_token]
        file_name = f'{output_dir}/{base_name}-interface-gt.png'
        draw_interface_heatmap(gt_interface_map, asym_id, file_name, 'Ground-truth Interface Map')

    if 'interface_info_mix' in batch['feat']:    
        gt_interface_map = batch['feat']['interface_info_mix'][batch_i].numpy() # [N_token, N_token]
        file_name = f'{output_dir}/{base_name}-interface-gt-mixed.png'
        draw_interface_heatmap(gt_interface_map, asym_id, file_name, 'Mixed Homogeneous Interface Map')

    # save predicted probability and sample results at each step
    if 'interface_head' in results and 'intermediate_interface_infos' in results['interface_head']:
        pred_interface_prob_in_steps = \
            results['interface_head']['intermediate_interface_infos']['pred_interface_prob_in_steps'][batch_i]
        interface_info_in_steps = \
            results['interface_head']['intermediate_interface_infos']['interface_info_in_steps'][batch_i]

        for repeat_j in range(len(pred_interface_prob_in_steps)):
            for step_i in range(len(pred_interface_prob_in_steps[repeat_j])):
                # save predicted probability at step i
                file_name = f'{output_dir}/{base_name}-interface-prob-repeat{repeat_j}-step{step_i}.png'
                draw_interface_heatmap(pred_interface_prob_in_steps[repeat_j][step_i], asym_id, file_name, 
                            f'Predicted Interface Probability at Repeat {repeat_j}, Step {step_i}')

                # save sampled results at step i
                file_name = f'{output_dir}/{base_name}-sampled-repeat{repeat_j}-step{step_i+1}.png'
                draw_interface_heatmap(interface_info_in_steps[repeat_j][step_i], asym_id, file_name, 
                            f'Sampled Interface at Repeat {repeat_j}, Step {step_i}')

    return None


def save_interface_prob(batch, results, batch_i, output_dir, base_name):

    if 'intermediate_interface_infos' not in results['interface_head'] \
        or len(results['interface_head']['intermediate_interface_infos']['pred_interface_prob_in_steps']) <= batch_i:
        return None

    pred_interface_prob_repeats = results['interface_head']['intermediate_interface_infos']['pred_interface_prob_in_steps'][batch_i]
    if len(pred_interface_prob_repeats) == 0:
        return None

    repeat_i = 0
    pred_interface_prob_step0 = pred_interface_prob_repeats[repeat_i][0] 
    gt_interface_info = batch['feat']['interface_info_full'][batch_i].numpy()

    # save pred_interface_prob_step0 to pkl file
    file_path = f'{output_dir}/{base_name}-interface-prob-step0.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(pred_interface_prob_step0, f)

    # save gt_interface_info to pkl file
    file_path = f'{output_dir}/{base_name}-interface-gt.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(gt_interface_info, f)

    return None


def save_interface_sampled_results(batch, results, batch_i, output_dir, base_name):
    """ Save the generated interface info to json file. 
        A sample of the json:
        {
            "n_token": 10,
            "nodes": [
                {"index": [0, 1], "match_annotation": 1},
                {"index": [1, 2], "match_annotation": 0},
                {"index": [2, 3], "match_annotation": -1}
            ],
            "probability": 0.85,
            "precision": 0.3333333333333333
        }
    """
    
    if 'gen_interfaces' not in results['interface_head']:
        return None
    
    file_path = f'{output_dir}/{base_name}-gen-interfaces.json'

    if len(results['interface_head']['gen_interfaces']) > batch_i:
        gen_interfaces = \
            results['interface_head']['gen_interfaces'][batch_i]
            
        with open(file_path, 'w') as f:
            json_list = [obj.to_dict() for obj in gen_interfaces]
            json.dump(json_list, f, indent=4)
            # for gen_interface in gen_interfaces:
            #     f.write(gen_interface.to_json())
            #     f.write('\n')

    return file_path

def save_interface_sampled_indices(batch, results, batch_i, output_dir, base_name):
    """ Save the indices of generated interface info to pkl file. 
        The generated interface info indices is in shape of [gen_repeats, interface_size, 2]. """
    
    if 'gen_interface_indices' not in results['interface_head']:
        return None
    
    file_path = f'{output_dir}/{base_name}-gen-interface-indices.pkl'

    gen_interface_indices = \
        results['interface_head']['gen_interface_indices'][batch_i]
        
    if isinstance(gen_interface_indices, paddle.Tensor):
        gen_interface_indices = gen_interface_indices.astype('int64').numpy() # [gen_repeats, N_token, N_token]
    
    with open(file_path, 'wb') as f:
        pickle.dump(gen_interface_indices, f)

    return file_path

def get_protein_ca_pocket(lig_gt_pos, pro_gt_ca_pos, theta=10):
    """tbd"""
    # find protein ca positions that should be kept in the pocket.
    dis_matrix = distance_matrix(lig_gt_pos, pro_gt_ca_pos)
    pro_ca_list = dis_matrix.min(axis=0) <= theta
    pro_ca_index = np.where(pro_ca_list)[0]

    return pro_ca_index

def get_primary_ca_chain(prot_ca_index, asym_id):
    """
        # prot_ca_index: index
        # asym_id: N_atom
        get the primary protein chain of ca index, which is the chain with the most asym id.
    """
    ca_asym_id = asym_id[prot_ca_index]
    # NOTE： when there are multiple chains with the same ca number of asym id, 
    # we choose the first one.
    uniq_asym_count = np.bincount(ca_asym_id)
    pri_asym_id = np.argmax(uniq_asym_count)
    pri_ca_idx = np.where(ca_asym_id == pri_asym_id)[0] 
    return prot_ca_index[pri_ca_idx]


def get_pocket_align_rmsd(batch, results, batch_i, diff_i, 
        pred_pdb_file, exp_pdb_file, use_tm_align=False, tm_score_bin=None,
        use_perm_align=False, perm_align_source=None):
    """
    pocket alignment RMSD
    """
    chain_info_list = json.loads(batch['chain_info_list'][batch_i])
    diff_results = results['diffusion_module']

    if use_perm_align:
        # TODO: otherwise will cause IndexError like posebuster_v1 7nlv
        batch = copy.deepcopy(batch)
        from helixfold.model import chain_align_np_aa
        if perm_align_source == 'label_full':
            aligned_label = chain_align_np_aa.multi_chain_permutation_align(
                    batch['feat'], batch['label_full'], 
                    tree_map(lambda x: x[:, diff_i], diff_results))
        else:
            aligned_label = chain_align_np_aa.multi_chain_permutation_align(
                    batch['feat'], batch['label'], 
                    tree_map(lambda x: x[:, diff_i], diff_results))
        batch['label'].update(aligned_label)

    ## NOTE: add_key for target ligand, because it may has a lot of other ligand.
    chain_ids = np.array(batch['feat']['all_chain_ids'][batch_i].split()) # N_atom
    is_assigned_ligand_chainIDs = [ 
            item["id"] for item in chain_info_list if item.get("is_assigned_ligand", False)]
    assert len(is_assigned_ligand_chainIDs) == 1, f"get_pocket_align_rmsd is not support multi assigned ligands"
    is_assigned_ligand_chainID = is_assigned_ligand_chainIDs[0]
    is_assigned_ligand_mask = np.array(chain_ids == is_assigned_ligand_chainID) # N_atom
    
    ref_token2atom_idx = batch['feat']['ref_token2atom_idx'][batch_i].numpy()
    is_protein = batch['feat']['is_protein'][batch_i].numpy().astype('bool')
    is_ligand = batch['feat']['is_ligand'][batch_i].numpy().astype('bool')
    
    ## N_token -> N_atom 
    perm_asym_id = batch['feat']['perm_asym_id'][batch_i].numpy()
    perm_is_protein = is_protein[ref_token2atom_idx]
    perm_is_ligand = is_ligand[ref_token2atom_idx]
    perm_is_ligand = np.logical_and(perm_is_ligand, is_assigned_ligand_mask) # select the assigned ligand.
    
    pred_pos = diff_results['final_atom_positions'][batch_i, diff_i].numpy()
    pred_pos_mask = diff_results['final_atom_mask'][batch_i, diff_i].numpy().astype('bool')
    gt_pos = batch['label']['all_atom_pos'][batch_i].numpy()
    gt_pos_mask = batch['label']['all_atom_pos_mask'][batch_i].numpy().astype('bool')
    centra_token = batch['label']['all_centra_token_indice'][batch_i].numpy()
    centra_token_mask = batch['label']['all_centra_token_indice_mask'][batch_i].numpy().astype('bool')
    
    ## get gt protein ca pos and ligand pos
    protein_gt_ca_pos = gt_pos[centra_token][centra_token_mask & is_protein]
    lig_gt_pos = gt_pos[gt_pos_mask & perm_is_ligand]
    prot_ca_index = get_protein_ca_pocket(lig_gt_pos, protein_gt_ca_pos)
    if len(prot_ca_index) < 3:
        ## after permutation, the lig_pos and prot_pos seems to diverse in some case
        return {'ligand_obrms': 1000000}

    ## get primary protein chain
    perm_ca_asym_id = perm_asym_id[centra_token][centra_token_mask & is_protein]
    prot_ca_index = get_primary_ca_chain(prot_ca_index, perm_ca_asym_id)
    if len(prot_ca_index) < 3:
        ## after permutation, the lig_pos and prot_pos seems to diverse in some case
        return {'ligand_obrms': 1000000}
    
    ## get gt protein ca pos and pred protein ca pos
    protein_gt_ca_pos_select = protein_gt_ca_pos[prot_ca_index]
    protein_pred_ca_pos_select = pred_pos[centra_token][centra_token_mask & is_protein][prot_ca_index]
    
    ## align pocket
    if use_tm_align:
        r, x = get_tm_score_align(tm_score_bin, exp_pdb_file, pred_pdb_file, is_complex=True)
    else:
        r, x = get_transform(protein_gt_ca_pos_select, protein_pred_ca_pos_select)
    align_gt_pos = np.matmul(gt_pos, r) + x
    common_feat = get_common_feat_for_pdb_save(batch, batch_i)
    protein_pdb_file = exp_pdb_file.replace('.pdb', '.align2pred.pdb')
    all_atom_pdb_save.prediction_to_pdb(align_gt_pos, 
            common_feat, protein_pdb_file)
    protein_pdb_file = pred_pdb_file.replace('.pdb', '.inalign.pdb')
    all_atom_pdb_save.prediction_to_pdb(pred_pos, 
            common_feat, protein_pdb_file)
    
    ## get aligned all entity
    align_prot_gt_pos = align_gt_pos[perm_is_protein]
    align_ligand_gt_pos = align_gt_pos[perm_is_ligand]
    ligand_pred_pos = pred_pos[perm_is_ligand]
    
    ## cal ligand rmsd
    rmsd = calc_ligand_rmsd(ligand_pred_pos, align_ligand_gt_pos)
    ## cal ligand obrms
    # TODO: get blocked at "7pa4"
    if batch['name'][batch_i] == '7pa4':
        obrms = rmsd
        bust_scores = {}
    else:
        ref_element = batch['feat']['ref_element'][batch_i].numpy()[perm_is_ligand]
        atomic_nums = list(map(int, ref_element))
        bond_mat = batch['feat']['token_bonds'][batch_i].numpy()[ref_token2atom_idx][:, ref_token2atom_idx]
        ligand_bond_mat = bond_mat[perm_is_ligand][:, perm_is_ligand]
        index1, index2 = np.nonzero(ligand_bond_mat)
        bonds = [(int(i), int(j)) for i, j in zip(index1, index2)]
        obrms = calc_ligand_obrms(atomic_nums, bonds, 
                ligand_pred_pos, align_ligand_gt_pos)
        ## save protein pdb without ligand
        common_feat = get_common_feat_for_pdb_save(batch, batch_i)
        for k, v in common_feat.items():
            if len(v) == len(is_assigned_ligand_mask):
                cur_mask = ~is_assigned_ligand_mask
            else:
                cur_mask = batch['feat']['asym_id'][batch_i].numpy() != \
                        perm_asym_id[np.nonzero(is_assigned_ligand_mask)[0][0]]
            common_feat[k] = np.array(v)[cur_mask]
        common_feat['ref_token2atom_idx'] = map_to_continuous_indices(
                common_feat['ref_token2atom_idx'])
        protein_pdb_file = pred_pdb_file.replace('.pdb', '.noligand.pdb')
        all_atom_pdb_save.prediction_to_pdb(pred_pos[~is_assigned_ligand_mask], 
                common_feat, protein_pdb_file)
        ## calc posebusters
        try:
            supplier = Chem.SDMolSupplier(batch['ligand_structure_file'][batch_i])
            sdf_mol = Chem.RemoveHs(supplier[0])
            pred_mol = copy.deepcopy(sdf_mol)
            set_mol_coords(pred_mol, ligand_pred_pos)
            ref_mol = copy.deepcopy(sdf_mol)
            set_mol_coords(ref_mol, align_ligand_gt_pos)
            df, bust_scores = calc_posebusters(pred_mol, ref_mol, protein_pdb_file)
            df.to_csv(join(dirname(protein_pdb_file), f'bust-{batch["name"][batch_i]}.csv'))
        except Exception as e:
            bust_scores = dict()
            print(e)
    ret = {
        "ligand_rmsd": rmsd,
        "ligand_rmsd_2A": float(rmsd <= 2),
        "ligand_obrms": obrms,
        "ligand_obrms_2A": float(obrms <= 2),
    }
    ret.update(bust_scores)
    return ret

def get_interface_token_pair_distance(batch, results, batch_i, diff_i):
    '''
        Compute the max and mean distances between predicted positions of token pairs 
        specified in the interface_info feature.
    '''

    if 'interface_info' not in batch['feat']:
        return dict()

    interface_info = batch['feat']['interface_info'][batch_i].numpy()
    if len(interface_info.shape) > 2:
        # if multiple interface info are given, using the first one
        interface_info = interface_info[:,:,0]
    if interface_info.sum() <= 0:
        return dict()

    diff_results = results['diffusion_module']
    all_centra_token_indice = batch['label']['all_centra_token_indice'][batch_i].numpy()
    pred_token_pos = diff_results['final_atom_positions'][batch_i, diff_i][all_centra_token_indice].numpy()

    dist = pairwise_distances(pred_token_pos, pred_token_pos)

    pred_interface_distances = dist * interface_info

    ret = {
        'interface_max_dist': pred_interface_distances.max(),
        'interface_mean_dist': pred_interface_distances.sum() / interface_info.sum()
    }

    return ret

def get_interface_token_pair_recall_rate(batch, results, batch_i, diff_i, dist_thres, dist_type='heavy_atom'):
    '''
        Compute the recall rate of the token pairs specified by the interface_info feature:
        interface_recall = TP/(TP+FP), 
        where TP is the number of token pairs in the intersection of the given interface_info
        and the predicted cross-interface token pairs. TP+FP is the token pairs in the given
        interface_info.
    '''
    if 'interface_info' not in batch['feat']:
        return dict()

    interface_info = batch['feat']['interface_info'][batch_i].numpy()
    if len(interface_info.shape) > 2:
        # if multiple interface info are given, using the first one
        interface_info = interface_info[:,:,0]
    if interface_info.sum() <= 0: # not token pair in the interface_info feature
        return dict()

    diff_results = results['diffusion_module']
    
    pred_pos = diff_results['final_atom_positions'][batch_i, diff_i].numpy()
    all_centra_token_indice = batch['label']['all_centra_token_indice'][batch_i].numpy()
    pred_token_pos = diff_results['final_atom_positions'][batch_i, diff_i][all_centra_token_indice].numpy()

    asym_id = batch['feat']['asym_id'][batch_i].numpy()
    centra_token_mask = batch['label']['all_centra_token_indice_mask'][batch_i].numpy().astype('bool')

    # compute the predicted interface matrix from the predicted token positions
    pred_interface = build_interface_info_with_random_residues(asym_id, 
        atom_pos=pred_pos, 
        atom_mask=batch['label']['all_atom_pos_mask'][batch_i].numpy(),
        atom_to_token_mapping=batch['feat']['ref_token2atom_idx'][batch_i].numpy(),
        token_pos=pred_token_pos,
        token_pos_mask=centra_token_mask, top_n=-1, m=-1, 
        dist_thres=dist_thres, 
        dist_type=dist_type,
        seed=None)
    pred_interface = pred_interface[0].interface_map()

    interface_recall = (pred_interface * interface_info).sum() / interface_info.sum()

    ret = {
        'interface_recall': interface_recall
    }
    
    return ret


def get_interface_token_pair_precision(batch, results, batch_i, diff_i, dist_thres, dist_type,
                                            chain_info_list, output_dir, 
                                            save_interface_from_pred_struct=False):
    '''
        Compute the precision of the interface derived from the predicted structure:
        precision = TP/(TP+FP), 
        where TP is the number of token pairs in the intersection of the given interface_info
        and the predicted cross-interface token pairs. TP+FP is the token pairs in the 
        interface_info derived from the predicted structure.
    '''
    if 'interface_info_full' not in batch['feat'] \
        or len(batch['feat']['interface_info_full']) < batch_i:
        print(f"protein {batch['name'][batch_i]} has no interface_info_full feature.")
        return dict()

    gt_interface_info = batch['feat']['interface_info_full'][batch_i].numpy()
    if gt_interface_info.sum() <= 0: # not token pair in the interface_info feature
        print(f"protein {batch['name'][batch_i]} has no interface in the ground truth structure.")
        return dict()

    diff_results = results['diffusion_module']
    
    pred_pos = diff_results['final_atom_positions'][batch_i, diff_i].numpy()
    all_centra_token_indice = batch['label']['all_centra_token_indice'][batch_i].numpy()
    pred_token_pos = diff_results['final_atom_positions'][batch_i, diff_i][all_centra_token_indice].numpy()

    asym_id = batch['feat']['asym_id'][batch_i].numpy()
    entity_id = batch['feat']['entity_id'][batch_i].numpy()
    centra_token_mask = batch['label']['all_centra_token_indice_mask'][batch_i].numpy().astype('bool')

    # compute the predicted interface matrix from the predicted token positions
    pred_interface = build_interface_info_with_random_residues(asym_id,
        atom_pos=pred_pos, 
        atom_mask=batch['label']['all_atom_pos_mask'][batch_i].numpy(),
        atom_to_token_mapping=batch['feat']['ref_token2atom_idx'][batch_i].numpy(),
        token_pos=pred_token_pos,
        token_pos_mask=centra_token_mask, top_n=-1, m=-1, 
        dist_thres=dist_thres, dist_type=dist_type,
        seed=None)
    pred_interface = pred_interface[0].interface_map()

    interface_precision = (pred_interface * gt_interface_info).sum() / (pred_interface.sum() + 1e-10)
    ret = {
        'structure_interface_precision': interface_precision
    }

    if 'confidence_head' in results and 'pae' in results['confidence_head'].keys():
        pae_matrix = results['confidence_head']['pae'][batch_i].detach().cpu().numpy().squeeze()
        pred_interface_pae_avg = (pae_matrix * pred_interface).sum() / (pred_interface.sum() + 1e-10)
        ret['pred_interface_neg_avgPAE'] = -pred_interface_pae_avg

    if save_interface_from_pred_struct:
        prot_name = batch['name'][batch_i]
        chain_ids = "_".join([chain_info["id"] for chain_info in chain_info_list])
        base_name = f'{prot_name}_{chain_ids}'

        file_name = f'{output_dir}/{base_name}-diff{diff_i}-interface.npz'
        np.savez_compressed(file_name, pred_interface=pred_interface.astype(bool), 
                                       gt_interface=gt_interface_info.astype(bool))

        # file_name = f'{output_dir}/{base_name}-diff{diff_i}-interface-gt.png'
        # draw_interface_heatmap(gt_interface_info, asym_id, file_name, 
        #                       title='Ground-truth Interface Map')

        # file_name = f'{output_dir}/{base_name}-diff{diff_i}-interface-pred-struct-derived.png' 
        # draw_interface_heatmap(pred_interface, asym_id, file_name, 
        #                       title='Predicted structure derived interface map.')

        tp_interface = pred_interface * gt_interface_info
        fp_interface = pred_interface * (1 - gt_interface_info)

        if 'confidence_head' in results and 'pae' in results['confidence_head'].keys():
            pae_matrix = results['confidence_head']['pae'][batch_i].detach().cpu().numpy().squeeze()
            tp_pae_matrix = pae_matrix * tp_interface
            fp_pae_matrix = pae_matrix * fp_interface

            tp_pae_items = pae_matrix[tp_interface > 0].flatten()
            fp_pae_items = pae_matrix[fp_interface > 0].flatten()

            # file_name = f'{output_dir}/{base_name}-diff{diff_i}-interface-pae-box-chart.png'
            # plot_box_chart(tp_pae_items, fp_pae_items, title='PAE of Interface Map.', y_label='PAE', 
            #         label1='True Positives', label2='False Positives', 
            #         save_path=file_name)

            # file_name = f'{output_dir}/{base_name}-diff{diff_i}-interface-pred-struct-derived-pae-tp.png'
            # draw_interface_heatmap(tp_pae_matrix, asym_id, file_name, vmin=0.0, vmax=30.0,
            #                       title='PAE of True Positive Interface Map.')

            # file_name = f'{output_dir}/{base_name}-diff{diff_i}-interface-pred-struct-derived-pae-fp.png'
            # draw_interface_heatmap(fp_pae_matrix, asym_id, file_name, vmin=0.0, vmax=30.0,
            #                       title='PAE of False Positive Interface Map.')
        
    return ret

def draw_interface_heatmap(interface_map, asym_id, output_file, vmin=0.0, vmax=1.0,  title=''):

    N = interface_map.shape[0]
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw the heatmap
    sns.heatmap(interface_map, ax=ax, cmap='Blues', cbar_kws={'label': 'Interface Probability'},
                vmin=vmin, vmax=vmax)
    
    # Identify unique chain boundaries
    chain_boundaries = np.where(np.diff(asym_id))[0] + 1
    chain_boundaries = np.concatenate(([0], chain_boundaries, [N]))
    
    # Draw bold lines at chain boundaries
    for boundary in chain_boundaries[:-1]:
        ax.axhline(boundary, color='black', linewidth=2)
        ax.axvline(boundary, color='black', linewidth=2)
 
    # Set titles and labels
    ax.set_title(title)
    ax.set_xlabel('Token Index')
    ax.set_ylabel('Token Index')
    
    # Show the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return output_file

def plot_box_chart(data1, data2, title='', y_label='Value', 
                    label1='Group 1', label2='Group 2', save_path=None):
    """
    Plots a box chart for two sets of data and optionally saves the figure.
 
    Parameters:
    - data1: First set of data (list or array)
    - data2: Second set of data (list or array)
    - title: Title of the plot
    - y_label: Label for the y-axis
    - label1: Label for the first set of data
    - label2: Label for the second set of data
    - save_path: Path to save the figure (if None, the figure is not saved)
    """
    # Create a figure and axis
    fig, ax = plt.subplots()
 
    # Plot the box chart
    ax.boxplot([data1, data2], labels=[label1, label2])
 
    # Add a title and axis labels
    ax.set_title(title)
    ax.set_xlabel('Groups')
    ax.set_ylabel(y_label)
 
    # Save the figure if a save_path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    return fig

def save_interface_map(batch, results, batch_i, output_dir, base_name):

    asym_id = batch['feat']['asym_id'][batch_i].numpy() # [N_token,]
    # save ground truth 
    if 'interface_info_full' in batch['feat']:    
        gt_interface_map = batch['feat']['interface_info_full'][batch_i].numpy() # [N_token, N_token]
        file_name = f'{output_dir}/{base_name}-interface-gt.png'
        draw_interface_heatmap(gt_interface_map, asym_id, file_name, title='Ground-truth Interface Map')

    return None

def save_local_conf_scores(batch, results, batch_i, output_dir, chain_info_list):
    prot_name = batch['name'][batch_i]
    chain_ids = "_".join([chain_info["id"] for chain_info in chain_info_list])
    base_name = f'{prot_name}_{chain_ids}'

    # save PAE map
    if 'confidence_head' in results and 'pae' in results['confidence_head'].keys():
        pae_matrix = results['confidence_head']['pae'][batch_i].detach().cpu().numpy().squeeze().astype(np.float16)
        file_name = f'{output_dir}/{base_name}-pae.npz'
        np.savez_compressed(file_name, pae_matrix=pae_matrix)

    # save pLDDT
    if 'confidence_head' in results and 'atom_plddts' in results['confidence_head'].keys():
        atom_plddts = results['confidence_head']['atom_plddts'][batch_i].detach().cpu().numpy().squeeze().astype(np.float16)

        file_name = f'{output_dir}/{base_name}-atom-plddts.npz'
        np.savez_compressed(file_name, atom_plddts=atom_plddts)

    # save asym_id
    asym_id = batch['feat']['asym_id'][batch_i].numpy().astype(np.uint8)
    file_name = f'{output_dir}/{base_name}-asym-id.npz'
    np.savez_compressed(file_name, asym_id=asym_id)

    # save is_ligand
    is_ligand = batch['feat']['is_ligand'][batch_i].numpy().astype(bool)
    file_name = f'{output_dir}/{base_name}-is-ligand.npz'
    np.savez_compressed(file_name, is_ligand=is_ligand)

    # save is_ligand_aa
    is_ligand_aa = batch['feat']['is_ligand_aa'][batch_i].numpy().astype(bool)

    file_name = f'{output_dir}/{base_name}-is-ligand-aa.npz'
    np.savez_compressed(file_name, is_ligand_aa=is_ligand_aa)

    return


def get_chemical_validity(
        pred_atom_pos, 
        gt_atom_pos, 
        atom_mask,
        all_centra_token_indice,
        all_centra_token_indice_mask):
    """
    Chemical validity
    Args:
        pred_atom_pos: (N_atom, 3)
        gt_atom_pos: (N_atom, 3)
        atom_mask: (N_atom,)
        all_centra_token_indice: (N_token,)
        all_centra_token_indice_mask: (N_token,)
    """
    def _get_bond_len(pred_dists, gt_dists, bond_len_thres):
        # skip self bond
        gt_dists += np.eye(gt_dists.shape[0]) * 1e10
        indices1, indices2 = np.where(gt_dists < bond_len_thres)
        pred_bond_len = pred_dists[indices1, indices2]
        if len(pred_bond_len) > 0:
            return np.mean(pred_bond_len < bond_len_thres)
        return 1.0
    
    def _get_atom_clash(pred_dists, atom_clash_thres):
        # skip self clash
        return np.mean(np.sum(pred_dists < atom_clash_thres, 1) - 1)

    ret = {}

    valid_pred_atom_pos = pred_atom_pos[atom_mask == 1]
    valid_gt_atom_pos = gt_atom_pos[atom_mask == 1]
    pred_dists = pairwise_distances(valid_pred_atom_pos, valid_pred_atom_pos)
    gt_dists = pairwise_distances(valid_gt_atom_pos, valid_gt_atom_pos)
    
    # bond_len
    ret['chem_valid-bond_len2A'] = _get_bond_len(
            pred_dists, gt_dists, bond_len_thres=2.0)
    
    # atom clash
    ret['chem_valid-atom_clash1A_per_atom'] = _get_atom_clash(
            pred_dists, atom_clash_thres=1.0)

    ## neighboring residue connection
    pred_token_pos = pred_atom_pos[all_centra_token_indice][all_centra_token_indice_mask == 1]
    gt_token_pos = gt_atom_pos[all_centra_token_indice][all_centra_token_indice_mask == 1]
    pred_token_dists = pairwise_distances(pred_token_pos, pred_token_pos)
    gt_token_dists = pairwise_distances(gt_token_pos, gt_token_pos)
    ret['chem_valid-nei_res_conn4.3A'] = _get_bond_len(
            pred_token_dists, gt_token_dists, bond_len_thres=4.3)
    return ret


class AllAtomEval(object):
    """
    Utilize exe `cal_score` to get TMScore, GDT-TS and GDT-HA
    """
    def __init__(self, 
            data_config=None,
            output_dir=None,
            tm_score_bin=None,
            lddt_score_bin=None,
            dockq_score_dir=None):
        self.data_config = data_config
        self.eval_type = data_config.eval_type
        self.output_dir = output_dir
        self.tm_score_bin = tm_score_bin
        self.lddt_score_bin = lddt_score_bin
        self.dockq_score_dir = dockq_score_dir

        os.makedirs(self.output_dir, exist_ok=True)

        self.name_list = []
        self.scores_list = []

    def _cal_scores(self, batch, results, batch_i, diff_i,
            pred_pdb_file, exp_pdb_file, chain_info_list):
        """calculate tm_score, gdt-ha, gdt-ts and lddt"""
        if self.eval_type == 'protein_protein':
            pred_pdb_file = os.path.abspath(pred_pdb_file)
            exp_pdb_file = os.path.abspath(exp_pdb_file)
            cur_scores = dict()
            dockq_result = get_dockq_scores(self.dockq_score_dir, pred_pdb_file, 
                                          exp_pdb_file, chain_info_list)
            if dockq_result is not None:
                cur_scores.update(dockq_result)
            cur_scores.update(get_tm_scores(self.tm_score_bin, pred_pdb_file, 
                                            exp_pdb_file, is_complex=True))
            cur_scores.update(get_lddt_scores(self.lddt_score_bin, pred_pdb_file, 
                                              exp_pdb_file, is_complex=True))
        elif self.eval_type == 'dockq_all_inter':
            pred_pdb_file = os.path.abspath(pred_pdb_file)
            exp_pdb_file = os.path.abspath(exp_pdb_file)
            cur_scores = get_dockq_scores_all_inter(pred_pdb_file, exp_pdb_file)
            cur_scores.update(get_tm_scores(self.tm_score_bin, pred_pdb_file, 
                                            exp_pdb_file, is_complex=True))
            cur_scores.update(get_lddt_scores(self.lddt_score_bin, pred_pdb_file, 
                                              exp_pdb_file, is_complex=True))
        elif self.eval_type == 'protein':
            cur_scores = get_tm_scores(self.tm_score_bin, pred_pdb_file, exp_pdb_file)
            cur_scores.update(get_lddt_scores(self.lddt_score_bin, pred_pdb_file, exp_pdb_file))
        elif self.eval_type == 'protein_ligand':
            align_args = [
                ['notm_noperm', {'use_tm_align': False, 'use_perm_align': False}],
                ['tm_noperm', {'use_tm_align': True, 'tm_score_bin': self.tm_score_bin, 
                        'use_perm_align': False}],
                ['notm_perm', {'use_tm_align': False, 'use_perm_align': True}],
                ['notm_permfull', {'use_tm_align': False, 'use_perm_align': True, 
                        'perm_align_source': 'label_full'}],
            ]
            best_cur_scores = None
            best_type = None
            for type_name, args in align_args:
                cur_scores = get_pocket_align_rmsd(batch, results, batch_i, diff_i, 
                        pred_pdb_file, exp_pdb_file, **args)
                print(type_name, cur_scores['ligand_obrms'])
                if best_cur_scores is None or \
                        cur_scores['ligand_obrms'] < best_cur_scores['ligand_obrms']:
                    best_cur_scores = cur_scores
                    best_type = type_name
            cur_scores = best_cur_scores
            cur_scores.update({f'align-{k[0]}': 0.0 for k in align_args})
            cur_scores[f'align-{best_type}'] = 1.0
        else:
            raise ValueError(f'eval type {self.eval_type} not supported yet')

        use_interface_info = self.data_config.get("use_interface_info", False)
        save_interface_from_pred_struct = self.data_config.get("save_interface_from_pred_struct", False)
        interface_config = self.data_config.get("interface_config", {"dist_thres": 5}) 
        dist_thres = interface_config['dist_thres']
        dist_type = interface_config.get('dist_type', 'heavy_atom')
        if use_interface_info:
            # compute interface distance metrics
            cur_scores.update(get_interface_token_pair_distance(batch, results, batch_i, diff_i))
            # compute interface recalls
            cur_scores.update(get_interface_token_pair_recall_rate(batch, results, batch_i, diff_i, dist_thres, dist_type))
        
        cur_scores.update(get_interface_token_pair_precision(batch, results, batch_i, diff_i, dist_thres, dist_type,
                                                                chain_info_list, self.output_dir, 
                                                                save_interface_from_pred_struct))

        save_local_conf_scores_ = self.data_config.get("save_local_conf_scores", False)
        # save local conf scores
        if save_local_conf_scores_:
            save_local_conf_scores(batch, results, batch_i, self.output_dir, chain_info_list)

        return cur_scores

    def _cal_inmemory_scores(self, batch, results, batch_i, diff_i):
        """
        get scores that can be directly calculated from
        inmemory batch and results
        """
        diff_results = results['diffusion_module']
        gt_atom_pos = batch['label']['all_atom_pos'][batch_i: batch_i + 1]
        atom_mask = batch['label']['all_atom_pos_mask'][batch_i: batch_i + 1][..., None]
        pred_atom_pos = diff_results['final_atom_positions'][batch_i: batch_i + 1, diff_i]
        
        ## lddt
        score_dict = {}
        score_dict['approxLDDT_15A'] = float(lddt.lddt(
                pred_atom_pos, gt_atom_pos, atom_mask, cutoff=15.)[0])
        score_dict['approxLDDT_30A'] = float(lddt.lddt(
                pred_atom_pos, gt_atom_pos, atom_mask, cutoff=30.)[0])
        
        ## lddt for rna
        is_rna_aa = batch['feat']['is_rna'][batch_i][batch['feat']['ref_token2atom_idx'][batch_i]]
        is_rna_aa = paddle.cast(is_rna_aa, dtype=atom_mask.dtype)
        rna_atom_mask = atom_mask * is_rna_aa[None, :, None]
        score_dict['approxLDDT_30A_rna'] = float(lddt.lddt(
                pred_atom_pos, gt_atom_pos, rna_atom_mask, cutoff=30.)[0])
        
        ## chemical validity
        score_dict.update(get_chemical_validity(
                pred_atom_pos.squeeze([0]), 
                gt_atom_pos.squeeze([0]), 
                atom_mask.squeeze([0, 2]),
                batch['label']['all_centra_token_indice'][batch_i],
                batch['label']['all_centra_token_indice_mask'][batch_i]))
        
        ## NOTE: interface lddt (ilddt for Protein-Nucleic)
        is_protein_aa = batch['feat']['is_protein'][batch_i][batch['feat']['ref_token2atom_idx'][batch_i]]
        is_protein_aa = paddle.cast(is_protein_aa, dtype=atom_mask.dtype)
        protein_atom_mask = atom_mask * is_protein_aa[None, :, None]
        protein_rna_interface_mask = lddt.get_interface_mask(gt_atom_pos, atom_mask, 
                                                            protein_atom_mask, rna_atom_mask, cutoff=30.)
        score_dict['iLDDT_30A_interface_30A_rna'] = float(lddt.lddt(
                                                pred_atom_pos, gt_atom_pos, atom_mask, 
                                                pair_mask=protein_rna_interface_mask, cutoff=30.)[0])

        is_dna_aa = batch['feat']['is_dna'][batch_i][batch['feat']['ref_token2atom_idx'][batch_i]]
        is_dna_aa = paddle.cast(is_dna_aa, dtype=atom_mask.dtype)
        dna_atom_mask = atom_mask * is_dna_aa[None, :, None]
        protein_dna_interface_mask = lddt.get_interface_mask(gt_atom_pos, atom_mask, 
                                                            protein_atom_mask, dna_atom_mask, cutoff=30.)
        score_dict['iLDDT_30A_interface_30A_dna'] = float(lddt.lddt(
                                                pred_atom_pos, gt_atom_pos, atom_mask, 
                                                pair_mask=protein_dna_interface_mask, cutoff=30.)[0])
        
        return score_dict

    def _cal_confidence_scores(self, batch, results, batch_i, diff_i):
        if not 'confidence_head' in results:
            return {}
        conf_results = results['confidence_head']
        confidence_score_names = [
                'mean_plddt', 'ptm', 'iptm', 'has_clash',
                'actifpTM', 'actifpTM_interfaceMask',
                'ranking_confidence', 'ligand_iptm', 'ligand_iptm_mean',
                'ligand_mean_plddt'
                ]
        score_dict = {k: float(conf_results[k][batch_i, diff_i]) 
                for k in confidence_score_names if k in conf_results}

        ## chain_pair_iptm for protein_ligand
        if self.eval_type == 'protein_ligand':
            # get ligand asym_id
            chain_info_list = json.loads(batch['chain_info_list'][batch_i])
            ligand_chain_ids = [item["id"] for item in chain_info_list 
                    if item.get("is_assigned_ligand", False)]
            assert len(ligand_chain_ids) == 1
            ligand_chain_id = ligand_chain_ids[0]
            chain_ids = np.array(batch['feat']['all_chain_ids'][batch_i].split())
            ligand_mask = np.array(chain_ids == ligand_chain_id)
            perm_asym_id = batch['feat']['perm_asym_id'][batch_i].numpy()
            ligand_asym_id = perm_asym_id[np.nonzero(ligand_mask)[0][0]]

            # get chain_pair_iptm
            cp_asym_ids = conf_results['chain_pair_asym_ids'][batch_i, diff_i].numpy()
            cp_iptm = conf_results['chain_pair_iptm'][batch_i, diff_i].numpy()
            cp_mask = conf_results['chain_pair_mask'][batch_i, diff_i].numpy()
            index = np.nonzero(cp_asym_ids == ligand_asym_id)[0][0]
            mask = np.zeros_like(cp_mask)
            mask[index, :] = 1.0
            mask[:, index] = 1.0
            mask[index, index] = 0.0
            mask *= cp_mask
            score = (cp_iptm * mask).sum() / (mask.sum() + 1e-8)
            score_dict['chain_pair_iptm-ligand'] = score

        return score_dict
    
    def _save_score_txt(self, score_dict, name, out_file):
        print('[Eval Scores]', name, score_dict)
        with open(out_file, 'w') as f:
            out_str = f'{name}\t'
            out_str += '\t'.join([f'{k}:{v}' for k, v in score_dict.items()])
            f.write(out_str + '\n')

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def add(self, batch, results):
        """tbd"""
        features = batch['feat']
        batch_size, diff_batch_size = results['diffusion_module']['final_atom_positions'].shape[:2]
        for batch_i in range(batch_size):
            prot_name = batch['name'][batch_i]
            if 'msa' in features:
                msa_depth, seq_len = features['msa'].shape[1:3] # [batch, msa_depth, seq_len]
            else:
                msa_depth, seq_len = 1, features['restype'].shape[1]
            chain_info_list = json.loads(batch['chain_info_list'][batch_i])
            chain_ids = "_".join([chain_info["id"] for chain_info in chain_info_list])
            for diff_i in range(diff_batch_size):
                save_name = f'{prot_name}_{chain_ids}-diff{diff_i}'
                pred_pdb_file = f'{self.output_dir}/{save_name}-pred.pdb'
                pred_pdb_masked_file = f'{self.output_dir}/{save_name}-pred.masked.pdb'
                exp_pdb_file = f'{self.output_dir}/{save_name}-exp.pdb'
                exp_pdb_masked_file = f'{self.output_dir}/{save_name}-exp.masked.pdb'
                # TODO: check jsonline of {"protein_name": "7yx8", "chains": [{"id": "A"}, {"id": "D"}]}
                try:
                    save_allatom_results(
                            batch, results, batch_i, diff_i, 
                            x0_traj_file=f'{self.output_dir}/{save_name}-pred-x0_traj.pdb',
                            xt_traj_file=f'{self.output_dir}/{save_name}-pred-xt_traj.pdb',
                            pred_pdb_file=pred_pdb_file,
                            pred_pdb_masked_file=pred_pdb_masked_file,
                            exp_pdb_file=exp_pdb_file,
                            exp_pdb_masked_file=exp_pdb_masked_file)
                except Exception as e:
                    import traceback
                    print(f'Error saving {save_name}: {e}')
                    traceback.print_exc(file=sys.stdout)
                    continue
                if "pdb_struct_file" in batch:  
                    exp_pdb_masked_file = batch["pdb_struct_file"][batch_i] # use assigned pdb file as ground truth
                    pred_pdb_masked_file = pred_pdb_file # pred_pdb_masked_file not exists when exp_pdb is not provided
                score_dict = {'msa_depth': msa_depth, 'seq_len': seq_len}

                if exists(pred_pdb_masked_file) and exists(exp_pdb_masked_file):
                    score_dict.update(self._cal_scores(batch, results, batch_i, diff_i,
                            pred_pdb_masked_file, exp_pdb_masked_file, chain_info_list))

                score_dict.update(self._cal_inmemory_scores(batch, results, batch_i, diff_i))
                score_dict.update(self._cal_confidence_scores(batch, results, batch_i, diff_i))
                self.name_list.append(f'{prot_name}_{chain_ids}')
                self.scores_list.append(score_dict)
                self._save_score_txt(score_dict, f'{prot_name}_{chain_ids}',
                        out_file=f'{self.output_dir}/{save_name}-scores.txt')

            # save generated interface info
            base_name = f'{prot_name}_{chain_ids}'
            interface_output_dir = os.path.join(self.output_dir, 'gen_interfaces')
            if not os.path.exists(interface_output_dir):
                os.makedirs(interface_output_dir)
            save_interface_sampled_results(batch, results, batch_i, interface_output_dir, base_name)
            # save_interface_sampled_indices(batch, results, batch_i, interface_output_dir, base_name)

            # save interface map
            interface_config = self.data_config.get("interface_config", {"save_interface_map": False})
            if interface_config.get('save_interface_map', False) and 'interface_head' in results:
                save_interface_map(batch, results, batch_i, self.output_dir, base_name)

            if interface_config.get('save_interface_prob', False) and 'interface_head' in results:
                save_interface_prob(batch, results, batch_i, self.output_dir, base_name)

    def get_result(self):
        """get_result"""
        return {k: np.array(v) for k, v in self.score_dict.items()}

    def get_result_after_gather(self, all_name_list, all_scores_list):
        """
        all_scores_list: a list of scores of all samples, each 
            item corresponds to a sample.
        """
        def _get_pR(scores_list, key1, key2):
            left_list = [s for s in scores_list 
                    if key1 in s and key2 in s]
            array1 = np.array([s[key1] for s in left_list])
            array2 = np.array([s[key2] for s in left_list])
            if len(array1) < 2:
                return np.nan
            return stats.pearsonr(array1, array2)[0]

        assert len(all_name_list) == len(all_scores_list), (
                f'{len(all_name_list)} != {len(all_scores_list)}')
        ret = {'sample_num': len([x for x in all_scores_list 
                                  if self.data_config.target_score_name in x])}

        ## calculate mean of basic scores
        for key in all_scores_list[0].keys():
            ret[key] = np.mean([s[key] for s in all_scores_list if key in s])

        ## calculate pearson correlation between scores
        ret['pR-lddt15-lddt'] = _get_pR(
                all_scores_list, 'approxLDDT_15A', 'mean_plddt')
        ret['pR-lddt30-lddt'] = _get_pR(
                all_scores_list, 'approxLDDT_30A', 'mean_plddt')
        ret['pR-lddt15-ranking_confidence'] = _get_pR(
                all_scores_list, 'approxLDDT_15A', 'ranking_confidence')
        if self.eval_type == 'protein_protein':
            ret['pR-DockQ-ranking_confidence'] = _get_pR(
                    all_scores_list, 'DockQ', 'ranking_confidence')
        elif self.eval_type == 'dockq_all_inter':
            ret['pR-DockQ-ranking_confidence'] = _get_pR(
                    all_scores_list, 'DockQ', 'ranking_confidence')
        elif self.eval_type == 'protein':
            ret['pR-TM-ranking_confidence'] = _get_pR(
                    all_scores_list, 'TM-score', 'ranking_confidence')
        elif self.eval_type == 'protein_ligand':
            ret['pR-obrms-ranking_confidence'] = _get_pR(
                    all_scores_list, 'ligand_obrms_2A', 'ranking_confidence')
        else:
            raise ValueError(f'eval type {self.eval_type} not supported yet')

        ## generate ensemble report
        generate_ensemble_report(
                all_name_list, 
                all_scores_list, 
                self.data_config.target_score_name,
                out_file=f'{self.output_dir}/ensemble-report.txt')
        return ret


class ResultsCollect(object):
    def __init__(self, 
            data_config=None,
            output_dir=None, 
            tm_score_bin=None,
            lddt_score_bin=None,
            dockq_score_dir=None,
            distributed=False):
        
        self.data_config = data_config
        self.distributed = distributed

        self.res_dict_list = []
        if not self.data_config is None:
            self.aa_eval = AllAtomEval(
                    data_config,
                    output_dir,
                    tm_score_bin, 
                    lddt_score_bin, 
                    dockq_score_dir)

    def add(self, batch, results, extra_dict):
        """
        batch, results: 
        extra_dict: {key: float, ...}
        """
        res_dict = self._extract_loss_dict(results)     # {key: float, ...}
        res_dict.update(extra_dict)
        self.res_dict_list.append(res_dict)
        if not self.data_config is None:
            self.aa_eval.add(batch, results)
    
    def get_result(self):
        res = {}
        # get results in res_dict_list
        all_res_dict_list = dist_all_gather(
                self.res_dict_list, 
                distributed=self.distributed)
        all_res_dict_list = [d for l in all_res_dict_list for d in l]
        if len(all_res_dict_list) > 0:
            for k in sorted(list(all_res_dict_list[0].keys())):
                res[k] = np.mean([d[k] for d in all_res_dict_list if k in d])

        # get aa_eval results
        if not self.data_config is None:
            all_name_list = dist_all_gather(
                    self.aa_eval.name_list, 
                    distributed=self.distributed)
            all_name_list = [d for l in all_name_list for d in l]
            all_scores_list = dist_all_gather(
                    self.aa_eval.scores_list, 
                    distributed=self.distributed)
            all_scores_list = [d for l in all_scores_list for d in l]
            res.update(self.aa_eval.get_result_after_gather(
                    all_name_list, all_scores_list))

        ## if exists "_exists" key
        reweight_res = {}
        for k, v in res.items():
            if not k.endswith('_exist'):
                continue
            value_key = k[:-len('_exist')]
            if value_key in res:
                reweight_res[f'{value_key}_reweight'] = res[value_key] / (v + 1e-8)
        res.update(reweight_res)
        return res

    def _extract_loss_dict(self, results):
        """extract value with 'loss' or 'fape' in key"""
        def _calc_tensor_mean(x):
            if x.dtype == paddle.bfloat16:
                x = x.cast("float32")
            if len(x.shape) == 0:
                return x.item()
            else:
                return x.numpy().mean()

        res = tree_flatten(results)
        res = tree_filter(lambda k: 'loss' in k or 'fape' in k, None, res)
        res = tree_map(lambda x: _calc_tensor_mean(x), res)
        return res