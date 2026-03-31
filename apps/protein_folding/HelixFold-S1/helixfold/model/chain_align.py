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
import paddle.nn as nn
from helixfold.common import residue_constants
from helixfold.model import utils


def multi_chain_permutation_align(batch, label, out):
    """
    Args:
        batch: dict of cropped features
        label: dict of raw labels
    """
    with paddle.no_grad():
        best_labels = []
        bs = out['final_atom_positions'].shape[0]
        for i in range(bs):
            new_label = chain_align_per_sample(batch, label, out, i)
            best_labels.append(new_label)

        aligned_label = merge_aligned_labels(best_labels)
    return aligned_label


def make_asym_to_entity_dict(gt_asym_id, gt_entity_id):
    """
    asym_to_entity_dict: {
        asym_id: entity_id
    }
    """
    res = {}
    uniq_asym_id = [int(x) for x in paddle.unique(gt_asym_id)]
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
    uniq_asym_id = [int(x) for x in paddle.unique(gt_asym_id)]
    gt_ca_pos_dict = {}
    gt_ca_mask_dict = {}
    gt_residx_offset_dict = {}
    full_index = paddle.arange(gt_asym_id.shape[0])
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
    uniq_asym_id = [int(x) for x in paddle.unique(pred_asym_id)]
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
    """
    ca_idx = residue_constants.atom_order['CA']

    ## gt labels
    gt_ca_pos = batch_label['all_atom_positions'][batch_i][:, ca_idx] # (M, 3)
    gt_ca_mask = batch_label['all_atom_mask'][batch_i][:, ca_idx] # (M)
    gt_entity_id = batch_label['entity_id'][batch_i] # (M)
    gt_asym_id = batch_label['asym_id'][batch_i] # (M)
    asym_to_entity_dict = make_asym_to_entity_dict(gt_asym_id, gt_entity_id)
    gt_ca_pos_dict, gt_ca_mask_dict, gt_residx_offset_dict = \
            make_gt_ca_pos_dict(
                gt_ca_pos, gt_ca_mask, gt_asym_id, asym_to_entity_dict)

    ## pred features, keep the valid ones
    pred_asym_id = batch_feat['asym_id'][batch_i]              # (N)
    pred_seq_mask = pred_asym_id > 0            # asym_id = 0 is the padding value
    pred_asym_id = pred_asym_id[pred_seq_mask]
    pred_residue_index = batch_feat['residue_index'][batch_i][pred_seq_mask]        # (N)
    pred_ca_pos = out['final_atom_positions'][batch_i, :, ca_idx][pred_seq_mask]    # (N, 3)
    pred_ca_mask = out['final_atom_mask'][batch_i, :, ca_idx][pred_seq_mask]        # (N)
    pred_ca_pos_list, pred_asym_id_list, pred_residue_index_list = \
            make_pred_ca_pos_list(
                pred_ca_pos, pred_ca_mask, pred_asym_id, pred_residue_index)

    pred_entity_id = batch_feat['entity_id'][batch_i]              # (N)
    anchor_eid, anchor_sid = select_anchor_chain(
            gt_ca_mask_dict, pred_asym_id_list, pred_residue_index_list, asym_to_entity_dict)

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
        ## skip candidate that has less than 3 valid residue.
        ## `select_anchor_chain` will guarantee at least one
        ## candidate meet the condision
        if paddle.sum(anchor_gt_mask) < 3:
            continue

        ## align all gt_chains
        r, x = get_transform(anchor_gt_pos[anchor_gt_mask], pred_pos[anchor_gt_mask])
        aligned_gt_ca_pos_dict = tree_map(
                lambda poses: [paddle.matmul(pos, r) + x for pos in poses], gt_ca_pos_dict)

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
    print(best_perm_list)
    aligned_label = {}
    N_res = batch_feat['asym_id'][batch_i].shape[0]
    for i in batch_label.keys():
        if i == 'resolution':
            # [B, S]
            # aligned_label[i] = batch_label[i][batch_i][asym_perm]
            aligned_label[i] = batch_label[i][batch_i]
        else:
            # [B, N, ...]
            new_label = batch_label[i][batch_i][seq_perm]
            pad_shape = [N_res - new_label.shape[0]] + new_label.shape[1:]
            pad_value = paddle.zeros(pad_shape, new_label.dtype)
            aligned_label[i] = paddle.concat([new_label, pad_value], 0)
        
    # TODO:
    # assert paddle.mean(batch_feat['aatype'][batch_i] == aligned_label['aatype_index']) > 0.9, \
    #         f'the new aligned_label should have the identical aatype as the original batch_feat'
    return aligned_label


def select_anchor_chain(
        gt_ca_mask_dict, pred_asym_id_list, pred_residue_index_list, asym_to_entity_dict):
    """
    find entity_id with minimal sym_num in ground truth.
    if minimal sym_num the same, find the maximal seq_len after cropping.
    if seq_len < 3 is not considered.
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
            seq_len = paddle.sum(gt_ca_mask[residue_index])
            if seq_len < 3:
                continue
            if (sym_num == select_sym_num and seq_len > select_seq_len) \
                    or sym_num < select_sym_num:
                select_eid = eid
                select_sid = sid
                select_sym_num = sym_num
                select_seq_len = seq_len
    return select_eid, select_sid


def get_transform(a_gt_pos, a_pred_pos):
    """
    get_transform
    """
    assert a_gt_pos.shape[0] >= 3
    ac_gt = paddle.mean(a_gt_pos, axis=0)
    ac_pred = paddle.mean(a_pred_pos, axis=0)

    # Apply http://en.wikipedia.org/wiki/Kabsch_algorithm
    P = a_gt_pos - ac_gt
    Q = a_pred_pos - ac_pred

    # [N, 3]^T * [N, 3] => covariance [3, 3]
    C = paddle.matmul(paddle.transpose(P, [1, 0]), Q)
    ## TODO: paddle dcu doesn't have svd op
    if os.environ.get('DCU_MODE_SVD', '0') == '1':
        U, S, V = np.linalg.svd(C.numpy())
        d = (np.linalg.det(U) * np.linalg.det(V)) < 0.0
        if d:
            U[:, -1] = -U[:, -1]
        U = paddle.to_tensor(U)
        V = paddle.to_tensor(V)
    else:
        U, S, V = paddle.linalg.svd(C)

        # Decide whether required to correct rotation matrix
        # to ensure a right-handed coordinate system
        d = (paddle.linalg.det(U) * paddle.linalg.det(V)) < 0.0
        if d:
            U[:, -1] = -U[:, -1]

    R = paddle.matmul(U, V)
    T = ac_pred - paddle.matmul(ac_gt, R)
    return R, T


def calc_rmsd(v1, v2):
    """
    v1: [*, 3]
    v2: [*, 3]
    output: [*]
    """
    d = paddle.sqrt(paddle.sum(paddle.square(v1 - v2), -1))
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
                gt_mask = paddle.tile(gt_mask.unsqueeze(1), [1, pos_dim]) # repeat dimension 1 to match pred_pos/ gt_pos
            if paddle.sum(gt_mask) == 0:
                rmsd = 1e9
            else:
                pred_pos, gt_pos = paddle.reshape(pred_pos, (-1, pos_dim)), paddle.reshape(gt_pos, (-1, pos_dim)) # avoid shape missmatch
                if os.environ.get('DCU_MODE_SOFTMAX_CLIP', '0') == '1':
                    rmsd = float(calc_rmsd(
                            paddle.mean(paddle.masked_select(pred_pos, gt_mask).reshape((-1, pos_dim)), 0), 
                            paddle.mean(paddle.masked_select(gt_pos, gt_mask).reshape((-1, pos_dim)), 0)))
                else:
                    rmsd = float(calc_rmsd(
                            paddle.mean(pred_pos[gt_mask], 0), 
                            paddle.mean(gt_pos[gt_mask], 0)))
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
    asym_perm = paddle.to_tensor(asym_perm, 'int64')
    seq_perm = paddle.concat(seq_perm)
    return asym_perm, seq_perm


def asym_perm_to_seq_perm(asym_id, chains, perm):
    """
    asym_perm_to_seq_perm
    """
    seq_perm = []
    ids = paddle.arange(asym_id.shape[0])
    for i in perm:
        seq_perm.append(ids[asym_id == chains[i]])

    seq_perm = paddle.concat(seq_perm)
    return seq_perm


def merge_aligned_labels(label_lst):
    """
    merge_aligned_labels
    """
    merge = dict()
    for k in label_lst[0].keys():
        merge[k] = paddle.stack([i[k] for i in label_lst])

    return merge


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