"""
constraint utils
"""
import os
import paddle
import numpy as np
import logging
from typing import Union
from helixfold.data.utils import token_completion, dist_to_bins, get_bin_masks

logger = logging.getLogger(__file__)

def get_constraint_info(feat, 
                        all_chain_ids, chain_ids, 
                        constraint_infos, max_pair_num=10, max_dist=20):
    """
    convert constraint info from user_input_json to model feat
    Args:
        all_chain_ids: N_token, np.ndarray, such as: ['1-1', '1-2', ...]
        chain_ids: N_token, np.ndarray, such as: ['1-1', '1-2', ...]
        constrains_infos: list of dict, such as: [
        {
            "type": "distance",
            "level": "residue",
            "left_entity": "1-1-123",
            "right_entity": "3-1-106",
            "distance": "3"
        }
    ]
    return: dict of numpy.ndarray
    """
    batch_size, n_token = feat['asym_id'].shape
    assert batch_size == 1, "Only support batch size 1 for now"
    logger.info(f"n_token: {n_token}")

    constrains_info = np.zeros([batch_size, n_token, n_token, max_dist], dtype=np.float32)
    constrains_token_level = -1 * np.ones((batch_size, max_pair_num, 2), dtype=np.float32)

    constrains_info_chain = np.zeros([batch_size, n_token, n_token, max_dist], dtype=np.float32)
    constrains_chain_level = -1 * np.ones((batch_size, max_pair_num, 2), dtype=np.float32)

    for cid, const_info in enumerate(constraint_infos):
        if const_info["level"] in ["residue"] and const_info["type"] in ["distance"]:
            dist = int(const_info["distance"])            
            for batch_i in range(batch_size):
                # map to token ids
                src_ind = map_input_json_constraint_to_token_ids(const_info['left_entity'], 
                                                                asym_ids=feat['asym_id'][batch_i].numpy(), 
                                                                ent_ids=feat['entity_id'][batch_i].numpy(),
                                                                res_ids=feat['residue_index'][batch_i].numpy(),
                                                                raw_chain_ids_atom=all_chain_ids,
                                                                raw_chain_ids_token=chain_ids)
                tgt_ind = map_input_json_constraint_to_token_ids(const_info['right_entity'], 
                                                                asym_ids=feat['asym_id'][batch_i].numpy(), 
                                                                ent_ids=feat['entity_id'][batch_i].numpy(),
                                                                res_ids=feat['residue_index'][batch_i].numpy(),
                                                                raw_chain_ids_atom=all_chain_ids,
                                                                raw_chain_ids_token=chain_ids)
                const_dist = dist_to_bins([dist], buckets=np.arange(1, max_dist + 1))

                logger.info(f'is_ligand: {feat["is_ligand"].shape}, src_ind: {src_ind}, tgt_ind: {tgt_ind}')
                constrains_token_level[batch_i, cid, 0] = src_ind
                constrains_token_level[batch_i, cid, 1] = tgt_ind
                src_ind_completed = token_completion(src_ind, asym_id=feat['asym_id'][batch_i], 
                                            is_ligand=feat['is_ligand'][batch_i], restype=feat['restype'][batch_i],
                                            residue_index=feat['residue_index'][batch_i]
                                            )
                tgt_ind_completed = token_completion(tgt_ind, asym_id=feat['asym_id'][batch_i], 
                                            is_ligand=feat['is_ligand'][batch_i], restype=feat['restype'][batch_i],
                                            residue_index=feat['residue_index'][batch_i]
                                            )
                const_dist_masks = get_bin_masks(const_dist, bucket_size=max_dist)
                logger.info(f'src_ind_completed: {src_ind_completed}, tgt_ind_completed: {tgt_ind_completed}')
                logger.info(f'const_dist: {const_dist} const_mask: {const_dist_masks}')

                old_mask = constrains_info[batch_i, src_ind_completed[0], tgt_ind_completed[0]]
                new_mask = ((const_dist_masks[0] + old_mask) > 0).astype(np.float32)        # 逻辑或，如果用户重复定义同一对，留最近的那个约束
                constrains_info[0][np.ix_(src_ind_completed, tgt_ind_completed)] = new_mask
                constrains_info[0][np.ix_(tgt_ind_completed, src_ind_completed)] = new_mask
                logger.info(f'constrains_info: {constrains_info.mean()}')
        else: 
            raise NotImplementedError

    return {'constrains_token_info': constrains_info, 
            'constrains_token_pairs': constrains_token_level, 
            'constrains_chain_info': constrains_info_chain, 
            'constrains_chain_pairs': constrains_chain_level}


def map_input_json_constraint_to_token_ids(constrain_info, asym_ids, ent_ids, res_ids,
                                            raw_chain_ids_atom, raw_chain_ids_token) -> int:
    """
    Args:
        constrain_info: str, such as: "1-1-123", format: "entity_id-sym_chain_id-res_id"

        asym_ids, N_token, such as: [1, 2, 3...]
        ent_ids, N_token,  such as: [1, 2, 3...]
        res_ids, N_token,  such as: [0, 1, 2...]
        raw_chain_ids_atom, N_atom, np.ndarray, such as: ['1-1', '1-2', ...]
        raw_chain_ids_token, N_token, np.ndarray, such as: ['1-1', '1-2', ...]
    Returns:
        token-level ids, int index starts from 0
    """
    user_asymid, tok_offset = constrain_info.rsplit('-', 1)
    in_ent_count = (raw_chain_ids_token == user_asymid)
    in_tok_offset = in_ent_count & (res_ids == int(tok_offset) - 1)  # N_token

    if os.environ.get('DEBUG', False) != "0":
        logger.info(f'In map_input_json_constraint_to_token_ids .........')
        logger.info(f'user_asymid, user_tok_offset: {user_asymid}, {tok_offset}')
        logger.info(f'asym_id: {asym_ids}')
        logger.info(f'ent_id: {ent_ids}')
        logger.info(f'res_id: {res_ids}')
        logger.info(f'raw_chain_ids_token: {raw_chain_ids_token}')
        logger.info(f'in_ent_count: {in_ent_count}')
        logger.info(f'in_tok_offset: {in_tok_offset}')
        logger.info(f'in_tok_offset.nonzero()[0]: {in_tok_offset.nonzero()[0]}')
        logger.info(f'End map_input_json_constraint_to_token_ids .........')

    return np.argwhere(in_tok_offset)[0, 0] # 取出第一个非零元素的索引（token 级别）
    