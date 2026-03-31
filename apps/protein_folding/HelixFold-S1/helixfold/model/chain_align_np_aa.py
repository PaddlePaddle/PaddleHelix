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

"""Modules for the multi chain permutation align."""

import os
import numpy as np
import paddle
from helixfold.common import residue_constants

from rdkit import Chem
from rdkit.Chem.rdMolAlign import GetBestAlignmentTransform

MIN_NUM_FOR_ANCHOR_CHAIN = 100


def multi_chain_permutation_align(batch, label, out):
    """
    Args:
        batch: dict of cropped features
        label: dict of raw labels
    """
    feat_names = ['perm_asym_id', 'perm_atom_index']
    label_names = ['all_atom_pos', 'all_atom_pos_mask', 
                'perm_entity_id', 'perm_asym_id']
    out_names = ['final_atom_positions', 'final_atom_mask']
    batch = {k: batch[k].numpy() for k in feat_names}
    label = {k: label[k].numpy() for k in label_names}
    out = {k: out[k].numpy() for k in out_names}

    best_labels = []
    bs = out['final_atom_positions'].shape[0]
    for i in range(bs):
        new_label = chain_align_per_sample(batch, label, out, i)
        best_labels.append(new_label)

    aligned_label = merge_aligned_labels(best_labels)
    aligned_label = tree_map(paddle.to_tensor, aligned_label)
    return aligned_label


def make_asym_to_entity_dict(gt_asym_id, gt_entity_id):
    """
    asym_to_entity_dict: {
        asym_id: entity_id
    }
    """
    res = {}
    uniq_asym_id = [int(x) for x in np.unique(gt_asym_id)]
    for aid in uniq_asym_id:
        eid = int(gt_entity_id[gt_asym_id == aid][0])
        res[aid] = eid
    return res


def make_gt_ca_pos_dict(gt_ca_pos, gt_ca_mask, gt_asym_id, asym_to_entity_dict):
    """
    gt_ca_pos_dict: {
        entity_id: [(m1, 3), (m2, 3), ...]
    }
    gt_ca_mask_dict: {
        entity_id: [(m1), (m2), ...]
    }
    gt_residx_offset_dict: {
        entity_id: [int, int, ...]
    }, recording offset in the concated ground-truth chains
    """
    uniq_asym_id = [int(x) for x in np.unique(gt_asym_id)]
    gt_ca_pos_dict = {}
    gt_ca_mask_dict = {}
    gt_residx_offset_dict = {}
    full_index = np.arange(gt_asym_id.shape[0])
    for aid in uniq_asym_id:
        eid = asym_to_entity_dict[aid]
        if not eid in gt_ca_pos_dict:
            gt_ca_pos_dict[eid] = []
            gt_ca_mask_dict[eid] = []
            gt_residx_offset_dict[eid] = []
        flag = (gt_asym_id == aid)
        gt_ca_pos_dict[eid].append(gt_ca_pos[flag])
        gt_ca_mask_dict[eid].append(gt_ca_mask[flag])
        gt_residx_offset_dict[eid].append(int(full_index[flag][0]))
    return gt_ca_pos_dict, gt_ca_mask_dict, gt_residx_offset_dict


def make_pred_ca_pos_list(pred_ca_pos, pred_ca_mask, pred_asym_id, pred_residue_index):
    """
    pred_ca_pos_list: [(n1, 3), (n2, 3), ...]
    pred_ca_mask_list: [(n1), (n2), ...]
    pred_asym_id_list: [int, ...]
    pred_residue_index_list: [(n1), (n2), ...]
    """
    uniq_asym_id = [int(x) for x in np.unique(pred_asym_id)]
    pred_ca_pos_list = []
    pred_ca_mask_list = []
    pred_asym_id_list = []
    pred_residue_index_list = []
    for aid in uniq_asym_id:
        flag = (pred_asym_id == aid)
        pred_ca_pos_list.append(pred_ca_pos[flag])
        pred_ca_mask_list.append(pred_ca_mask[flag])
        pred_asym_id_list.append(aid)
        pred_residue_index_list.append(pred_residue_index[flag])
    return pred_ca_pos_list, pred_asym_id_list, pred_residue_index_list


def chain_align_per_sample(batch_feat, batch_label, out, batch_i):
    """
    chain align for the batch_i sample.
    we assume that the residues of the same asym_id 
    is organized continuously and the asym_id=0 is
    put in the end.

    Required keys:
        for batch_feat: 
            'perm_asym_id': (B, N,)
            'perm_atom_index': (B, N,)
        for batch_label:
            'all_atom_pos': (B, M, 3)
            'all_atom_pos_mask': (B, M)
            'perm_entity_id': (B, M)
            'perm_asym_id': (B, M)
        for out:
            'final_atom_positions': (B, N, 3)
            'final_atom_mask': (B, N)
        M is the full length, N is the length of the cropped sequence
    """
    ## gt labels
    gt_ca_pos = batch_label['all_atom_pos'][batch_i] # (M, 3)
    gt_ca_mask = batch_label['all_atom_pos_mask'][batch_i] # (M)
    gt_entity_id = batch_label['perm_entity_id'][batch_i] # (M)
    gt_asym_id = batch_label['perm_asym_id'][batch_i] # (M)
    asym_to_entity_dict = make_asym_to_entity_dict(gt_asym_id, gt_entity_id)
    gt_ca_pos_dict, gt_ca_mask_dict, gt_residx_offset_dict = \
            make_gt_ca_pos_dict(
                gt_ca_pos, gt_ca_mask, gt_asym_id, asym_to_entity_dict)

    ## pred features, keep the valid ones
    pred_asym_id = batch_feat['perm_asym_id'][batch_i]              # (N)
    pred_seq_mask = pred_asym_id > 0            # asym_id = 0 is the padding value
    pred_asym_id = pred_asym_id[pred_seq_mask]
    pred_residue_index = batch_feat['perm_atom_index'][batch_i][pred_seq_mask]        # (N)
    pred_ca_pos = out['final_atom_positions'][batch_i][pred_seq_mask]    # (N, 3)
    pred_ca_mask = out['final_atom_mask'][batch_i][pred_seq_mask]        # (N)
    pred_ca_pos_list, pred_asym_id_list, pred_residue_index_list = \
            make_pred_ca_pos_list(
                pred_ca_pos, pred_ca_mask, pred_asym_id, pred_residue_index)

    # pred_entity_id = batch_feat['entity_id'][batch_i]              # (N)
    anchor_eid, anchor_sid = select_anchor_chain(
            gt_ca_mask_dict, pred_asym_id_list, pred_residue_index_list, asym_to_entity_dict,
            min_num_for_anchor_chain=MIN_NUM_FOR_ANCHOR_CHAIN)
    assert anchor_eid is not None and anchor_sid is not None

    best_rmsd, best_perm_list = None, None
    for pred_i in range(len(pred_asym_id_list)):
        ## permute the pred_chain that has the same entity_id as the anchor_gt_chain
        aid = pred_asym_id_list[pred_i]
        eid = asym_to_entity_dict[aid]
        if eid != anchor_eid:
            continue
        
        ## get the pos of pred_chain and anchor_gt_chain to be aligned
        pred_pos = pred_ca_pos_list[pred_i]
        pred_res_idx = pred_residue_index_list[pred_i]
        anchor_gt_pos = gt_ca_pos_dict[eid][anchor_sid][pred_res_idx]
        anchor_gt_mask = gt_ca_mask_dict[eid][anchor_sid][pred_res_idx] == 1
        ## skip candidate that has less than 3 valid atoms.
        if np.sum(anchor_gt_mask) < 3:
            continue

        ## align all gt_chains
        r, x = get_transform(anchor_gt_pos[anchor_gt_mask], pred_pos[anchor_gt_mask])
        aligned_gt_ca_pos_dict = tree_map(
                lambda poses: [np.matmul(pos, r) + x for pos in poses], gt_ca_pos_dict)

        ## greedily find chain matching
        rmsd, perm_list = find_optimal_perm(
                pred_ca_pos_list, pred_asym_id_list, pred_residue_index_list,
                aligned_gt_ca_pos_dict, gt_ca_mask_dict,
                asym_to_entity_dict)
        if best_rmsd is None or rmsd < best_rmsd:
            best_rmsd = rmsd
            best_perm_list = perm_list
    
    asym_perm, seq_perm = perm_list_to_seq_perm(
            best_perm_list, gt_residx_offset_dict, pred_residue_index_list)
    # print('best_perm_list', best_perm_list)
    aligned_label = {}
    N = batch_feat['perm_asym_id'][batch_i].shape[0]
    for name in ['all_atom_pos', 'all_atom_pos_mask']:
        # [B, N, ...]
        new_label = batch_label[name][batch_i][seq_perm]
        pad_shape = [N - new_label.shape[0]] + list(new_label.shape[1:])
        pad_value = np.zeros(pad_shape, new_label.dtype)
        aligned_label[name] = np.concatenate([new_label, pad_value], 0)
        
    return aligned_label


def select_anchor_chain(
        gt_ca_mask_dict, 
        pred_asym_id_list, 
        pred_residue_index_list, 
        asym_to_entity_dict,
        min_num_for_anchor_chain):
    """
    find entity_id with minimal sym_num in ground truth.
    if minimal sym_num the same, find the maximal seq_len after cropping.
    chains with seq_len < min_num_for_anchor_chain is not considered.
    """
    select_eid, select_sid = None, None
    select_sym_num = 1000000
    select_seq_len = 0
    for pred_i in range(len(pred_asym_id_list)):
        aid = pred_asym_id_list[pred_i]
        residue_index = pred_residue_index_list[pred_i]
        eid = asym_to_entity_dict[aid]
        sym_num = len(gt_ca_mask_dict[eid])
        for sid, gt_ca_mask in enumerate(gt_ca_mask_dict[eid]):
            seq_len = np.sum(gt_ca_mask[residue_index])
            if not min_num_for_anchor_chain is None and \
                    seq_len < min_num_for_anchor_chain:
                continue
            if (sym_num == select_sym_num and seq_len > select_seq_len) \
                    or sym_num < select_sym_num:
                select_eid = eid
                select_sid = sid
                select_sym_num = sym_num
                select_seq_len = seq_len
    
    # if all chains are shorter than min_num_for_anchor_chain
    #   then research again with min_num_for_anchor_chain=3
    if select_eid is None:
        return select_anchor_chain(
                gt_ca_mask_dict, 
                pred_asym_id_list, 
                pred_residue_index_list, 
                asym_to_entity_dict,
                min_num_for_anchor_chain=3)
    return select_eid, select_sid


def get_transform(a_gt_pos, a_pred_pos):
    """
    get_transform
    """
    assert a_gt_pos.shape[0] >= 3
    ac_gt = np.mean(a_gt_pos, axis=0)
    ac_pred = np.mean(a_pred_pos, axis=0)

    # Apply http://en.wikipedia.org/wiki/Kabsch_algorithm
    P = a_gt_pos - ac_gt
    Q = a_pred_pos - ac_pred

    # [N, 3]^T * [N, 3] => covariance [3, 3]
    C = np.matmul(np.transpose(P, [1, 0]), Q)
    U, S, V = np.linalg.svd(C)
    d = (np.linalg.det(U) * np.linalg.det(V)) < 0.0
    if d:
        U[:, -1] = -U[:, -1]

    R = np.matmul(U, V)
    T = ac_pred - np.matmul(ac_gt, R)
    return R, T


def calc_rmsd(v1, v2):
    """
    v1: [*, 3]
    v2: [*, 3]
    output: [*]
    """
    d = np.sqrt(np.sum(np.square(v1 - v2), -1))
    return d


def find_optimal_perm(
        pred_ca_pos_list, pred_asym_id_list, pred_residue_index_list,
        aligned_gt_ca_pos_dict, gt_ca_mask_dict,
        asym_to_entity_dict):
    """
    each pred_chain find its nearest gt_chain
    """
    total_rmsd = 0
    perm_list = []
    gt_visited_dict = tree_map(lambda x: [False] * len(x), gt_ca_mask_dict)
    for pred_i in range(len(pred_ca_pos_list)):
        aid = pred_asym_id_list[pred_i]
        eid = asym_to_entity_dict[aid]        
        pred_pos = pred_ca_pos_list[pred_i]
        pred_res_idx = pred_residue_index_list[pred_i]

        best_rmsd, best_perm = None, None
        for gt_id, gt_pos in enumerate(aligned_gt_ca_pos_dict[eid]):
            if gt_visited_dict[eid][gt_id]:
                continue
            gt_pos = gt_pos[pred_res_idx]
            gt_mask = gt_ca_mask_dict[eid][gt_id][pred_res_idx] == 1
            pos_dim = 3
            if os.environ.get('DCU_MODE_SOFTMAX_CLIP', '0') == '1':
                gt_mask = np.tile(gt_mask[:, None], [1, pos_dim]) # repeat dimension 1 to match pred_pos/ gt_pos
            if np.sum(gt_mask) == 0:
                rmsd = 1e9
            else:
                pred_pos, gt_pos = np.reshape(pred_pos, (-1, pos_dim)), np.reshape(gt_pos, (-1, pos_dim)) # avoid shape missmatch
                rmsd = float(calc_rmsd(
                        np.mean(pred_pos[gt_mask], 0), 
                        np.mean(gt_pos[gt_mask], 0)))
            if best_rmsd is None or rmsd < best_rmsd:
                best_rmsd = rmsd
                best_perm = (eid, gt_id)

        total_rmsd += best_rmsd
        perm_list.append(best_perm)
        gt_visited_dict[best_perm[0]][best_perm[1]] = True
    return total_rmsd, perm_list


def perm_list_to_seq_perm(perm_list, gt_residx_offset_dict, pred_residue_index_list):
    """
    perm_list_to_seq_perm
    """
    asym_perm = []
    seq_perm = []
    for pred_i, (eid, gt_id) in enumerate(perm_list):
        cur_offset = gt_residx_offset_dict[eid][gt_id]
        cur_residue_index = cur_offset + pred_residue_index_list[pred_i]
        asym_perm.append(cur_offset)
        seq_perm.append(cur_residue_index)
    asym_perm = np.array(asym_perm, 'int64')
    seq_perm = np.concatenate(seq_perm)
    return asym_perm, seq_perm


def asym_perm_to_seq_perm(asym_id, chains, perm):
    """
    asym_perm_to_seq_perm
    """
    seq_perm = []
    ids = np.arange(asym_id.shape[0])
    for i in perm:
        seq_perm.append(ids[asym_id == chains[i]])

    seq_perm = np.concatenate(seq_perm)
    return seq_perm


def merge_aligned_labels(label_lst):
    """
    merge_aligned_labels
    """
    merge = dict()
    for k in label_lst[0].keys():
        merge[k] = np.stack([i[k] for i in label_lst])

    return merge


def tree_map_with_list(f, obj):
    """
    tree_map_with_list
    """
    if type(obj) is dict:
        res = {}
        for k in obj:
            res[k] = tree_map_with_list(f, obj[k])
        return res
    
    if type(obj) is list:
        res = []
        for i in obj:
            res.append(tree_map_with_list(f, i))
        return res
    
    return f(obj)


def tree_map(f, d):
    """
    tree_map
    """
    new_d = {}
    for k in d:
        if type(d[k]) is dict:
            new_d[k] = tree_map(f, d[k])
        else:
            new_d[k] = f(d[k])
    return new_d


"""
permutation for ligands
"""

def create_mol(atomic_nums, bonds, atom_poses=None):
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
    if not atom_poses is None:
        mol_set_coordinates(mol, atom_poses)
    return mol


def mol_set_coordinates(mol, atom_poses):
    """
    mol: RDKit Mol object
    atom_poses: (N, 3)
    """
    conformer = Chem.Conformer(len(mol.GetAtoms()))  
    for i, (x, y, z) in enumerate(atom_poses):  
        conformer.SetAtomPosition(
                i, (float(x), float(y), float(z)))  
    mol.AddConformer(conformer)


import signal

def handler(signum, frame):
    """timeout handler"""
    raise TimeoutError("The operation timed out!")
signal.signal(signal.SIGALRM, handler)


def ligand_permutation_align(batch, pred_atom_pos, gt_atom_pos, atom_mask):
    """
    Assume pred_atom_pos and gt_atom_pos are aligned.
    Assume the batch_dim is tiled such that all the samples are actually from 
        the same sample. So we only use the 1st sample to create the meta_data.

    Args:
        pred_atom_pos: (B, N, 3)
        gt_atom_pos: (B, N, 3), it should be globally aligned with pred_atom_pos
        atom_mask: (B, N)
    """
    def _create_meta_data(batch, batch_i):
        meta_data = {}
        ref_token2atom_idx = batch['ref_token2atom_idx'][batch_i]
        cur_mask = atom_mask[batch_i]
        is_ligand = batch['is_ligand'][batch_i]
        is_ligand_aa = (is_ligand[ref_token2atom_idx] * cur_mask).astype('bool')

        ## get each ligand
        perm_asym_id = batch['perm_asym_id'][batch_i]
        uniq_asym_ids = np.unique(perm_asym_id[is_ligand_aa])
        for a_id in uniq_asym_ids:
            a_mask = (perm_asym_id == a_id)
            ref_element = batch['ref_element'][batch_i][a_mask]
            atomic_nums = list(map(int, ref_element))
            bond_mat = batch['token_bonds'][batch_i][ref_token2atom_idx][:, ref_token2atom_idx]
            ligand_bond_mat = bond_mat[a_mask][:, a_mask]
            index1, index2 = np.nonzero(ligand_bond_mat)
            bonds = [(int(i), int(j)) for i, j in zip(index1, index2)]
            
            probe_mol = create_mol(atomic_nums, bonds)
            ref_mol = create_mol(atomic_nums, bonds)
            meta_data[a_id] = {
                'a_mask': a_mask,
                'probe_mol': probe_mol,
                'ref_mol': ref_mol,
            }
        meta_data['uniq_asym_ids'] = uniq_asym_ids
        return meta_data

    # TODO: this function will block in some case
    return gt_atom_pos

    if batch['is_ligand'].sum() == 0:
        return gt_atom_pos

    ## assume the batch_dim is tiled
    assert paddle.all(batch['ref_token2atom_idx'] == batch['ref_token2atom_idx'][0:1])

    feat_names = ['ref_token2atom_idx', 'perm_asym_id', 'ref_element',
            'token_bonds', 'is_ligand']
    batch = {k: batch[k].numpy() for k in feat_names}
    pred_atom_pos = pred_atom_pos.numpy()
    gt_atom_pos = gt_atom_pos.numpy()
    atom_mask = atom_mask.numpy()

    new_gt_atom_pos = gt_atom_pos.copy()

    # Set an alarm for 3 seconds
    signal.alarm(3)
    try:
        B, N = pred_atom_pos.shape[0:2]
        meta_data0 = _create_meta_data(batch, 0)
        global_index = np.arange(N, dtype='int64')
        rmsd_func = lambda x, y: np.sqrt(np.sum((x - y) ** 2, -1).mean())
        for batch_i in range(B):
            for a_id in meta_data0['uniq_asym_ids']:
                a_mask = meta_data0[a_id]['a_mask']
                probe_mol = meta_data0[a_id]['probe_mol']
                ref_mol = meta_data0[a_id]['ref_mol']
                mol_set_coordinates(probe_mol, gt_atom_pos[batch_i][a_mask])
                mol_set_coordinates(ref_mol, pred_atom_pos[batch_i][a_mask])
                # TODO: GetBestAlignmentTransform will apply transfomation which is not wanted
                rmsd, transformation, atom_pairs = GetBestAlignmentTransform(
                        probe_mol, ref_mol, maxIters=1000)
                left_index = np.array([i[0] for i in atom_pairs])
                right_index = np.array([i[1] for i in atom_pairs])
                if np.all(left_index == right_index):
                    continue
                global_left_index = global_index[a_mask][left_index]
                global_right_index = global_index[a_mask][right_index]
                new_gt_atom_pos[batch_i, global_left_index] = gt_atom_pos[batch_i, global_right_index]
                rmsd1 = rmsd_func(gt_atom_pos[batch_i][a_mask], pred_atom_pos[batch_i][a_mask])
                rmsd2 = rmsd_func(new_gt_atom_pos[batch_i][a_mask], pred_atom_pos[batch_i][a_mask])
                ## the alignment may result in a worse rmsd
                if rmsd1 < rmsd2:
                    new_gt_atom_pos[batch_i, a_mask] = gt_atom_pos[batch_i, a_mask]
    except TimeoutError as e:
        print('[Error] ligand_permutation_align timeout', e)
    finally:
        signal.alarm(0)

    new_gt_atom_pos = paddle.to_tensor(new_gt_atom_pos)
    return new_gt_atom_pos

