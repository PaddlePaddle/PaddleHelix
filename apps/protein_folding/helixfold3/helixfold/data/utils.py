#   Copyright (c) 2024 PaddleHelix Authors. All Rights Reserved.
#
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# International License (the "License");  you may not use this file  except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
	Utility functions for feature processing.
"""

import re
import functools
from functools import reduce
from typing import Mapping, List

import numpy as np
from helixfold.common import residue_constants

ATOM_LEVEL_KEYS = [
    'perm_entity_id', 'perm_asym_id', 'all_chain_ids', 'all_ccd_ids', 'all_atom_ids', 'perm_atom_index', 

    'ref_pos', 'ref_mask', 'ref_element', 'ref_charge', 'ref_atom_name_chars',
    'ref_space_uid', 'ref_token2atom_idx', 

    'label_ccd_ids', 'label_atom_ids', 'all_atom_pos',
    'all_atom_pos_mask',

	'atom_plddts',
]

## results key need to be save.
DISPLAY_DIM = frozenset(["SINGLE", "NUM_CHAIN", "NUM_ATOM", "NUM_TOKEN", 
        "NUM_CHAIN, NUM_CHAIN", "NUM_ATOM, NUM_ATOM", "NUM_TOKEN, NUM_TOKEN"])

DISPLAY_RESULTS_KEYS = {
    'atom_chain_ids': "NUM_ATOM",
    'atom_plddts': "NUM_ATOM",
    'pae': "NUM_TOKEN, NUM_TOKEN",
    'token_chain_ids': "NUM_TOKEN",
    'token_res_ids': "NUM_TOKEN",
    'chain_plddt': "NUM_CHAIN", 
    'chain_ptm': "NUM_CHAIN", 
    'chain_pair_iptm': "NUM_CHAIN, NUM_CHAIN", 
    'chain_pair_pae_min': "NUM_CHAIN, NUM_CHAIN",
    'iptm': "SINGLE",
    'ptm': "SINGLE",
    'has_clash': "SINGLE", 
    'mean_plddt': "SINGLE",
    'ranking_confidence': "SINGLE",
}


def find_two_closest_atoms(frame_coords: np.ndarray,
						   	frame_coords_mask: np.ndarray
							) -> Mapping[str, np.ndarray]:
	"""Find the two closest atoms to the reference bi atom, Only for ligand.
	
		Args: 
			frame_coords: the coordinates of all atoms; (N,3)
			frame_coords_mask: the mask of the frame atoms; (N,1)
		Returns:
		 	dict, record relative index or mask of the frame atoms in one residue.
	"""
	diff = frame_coords[:, np.newaxis, :] - frame_coords[np.newaxis, :, :]
	dist_matrix = np.linalg.norm(diff, axis=2)
	np.fill_diagonal(dist_matrix, np.inf)
	closest_indices = np.argsort(dist_matrix, axis=1)[:, :2]

	atom_nums = frame_coords.shape[0]
	frame_mask = np.zeros(atom_nums, dtype=np.int32)
	ai_mask = np.zeros(atom_nums, dtype=np.int32) ## record the index of the frame atoms in one residue
	bi_mask = np.zeros(atom_nums, dtype=np.int32)
	ci_mask = np.zeros(atom_nums, dtype=np.int32)
	for bi in range(frame_coords.shape[0]):
		ai = closest_indices[bi, 0]
		ci = closest_indices[bi, 1]
		if not is_frame_atoms_collinear(frame_coords[ai], 
										frame_coords[bi], 
										frame_coords[ci]):
			ai_mask[bi] = ai
			bi_mask[bi] = bi
			ci_mask[bi] = ci
			frame_mask[bi] = 1

	return {
		"ai_indice": ai_mask, # N_atom = N_token
		"bi_indice": bi_mask,
		"ci_indice": ci_mask,
		"frame_indice_mask": frame_mask * frame_coords_mask,
		"frame_atom_offset": np.array([atom_nums], dtype=np.int32),
	}


def is_frame_atoms_collinear(ai: np.ndarray, 
							 bi: np.ndarray, 
							 ci: np.ndarray, 
							 threshold: float = 25) -> bool:
	"""Check if the three atoms are collinear.
	
		Args:
			ai: the coordinates of the first atom; (3,)
			bi: the coordinates of the second atom; (3,)
			ci: the coordinates of the third atom; (3,)
			threshold: the threshold of the angle; (default: 25)
		Returns:
			bool, True if the three atoms are collinear, False otherwise.
	"""
	## calculate the angle between the two vectors
	vec_ab = bi - ai
	vec_bc = ci - bi

	## calculate the Norm-2 of the two vectors
	norm_ab = np.linalg.norm(vec_ab)
	norm_bc = np.linalg.norm(vec_bc)
	if norm_ab == 0 or norm_bc == 0:
		return True

	cos_theta = np.dot(vec_ab, vec_bc) / (norm_ab * norm_bc)
	cos_theta = np.clip(cos_theta, -1, 1)
	theta = np.arccos(cos_theta) * 180 / np.pi
	return theta < threshold


def get_pae_frame_mask(atom_ids_list: List, 
						atom_positions_list: List,
						residue_name_3: str,
						residue_is_standard: bool,
						residue_is_missing: bool,
						ref_atom_ids_index: Mapping[str, int]) -> Mapping[str, np.ndarray]:
	"""
		function to get the mask of the PAE frame.
			# _atom_ids_list, _atom_positions_list is the ground truth. pos/atom_ids
			# N_atom
		returns:
			dict, frame ai,bi,ci indice and indice mask
	"""
	assert len(atom_ids_list) == len(atom_positions_list)
	total_nums = len(ref_atom_ids_index)
	assert total_nums > 0, f'TODO filter - Got CCD <{residue_name_3}>: 0 atom nums.'
	frame_atom_pos = np.zeros([total_nums, 3], dtype=np.float32)
	frame_atom_pos_mask = np.zeros([total_nums], dtype=np.int32)
	frame_mask = np.zeros([total_nums], dtype=np.int32) 
	ai_mask = np.zeros([total_nums], dtype=np.int32) 
	bi_mask = np.zeros([total_nums], dtype=np.int32) 
	ci_mask = np.zeros([total_nums], dtype=np.int32)

	# N_token 
	res = { 'ai_indice':  np.array([0], dtype=np.int32),
			'bi_indice':  np.array([0], dtype=np.int32),
			'ci_indice':  np.array([0], dtype=np.int32),
			'frame_indice_mask':  np.array([0], dtype=np.int32),
			'frame_atom_offset': np.array([total_nums], dtype=np.int32)}	
	
	## NOTE: if reisude is missing, return the invalid mask for frame.
	if residue_is_missing:
		if not residue_is_standard:
			# N_atom
			res['ai_indice'] = ai_mask
			res['bi_indice'] = bi_mask
			res['ci_indice'] = ci_mask
			res['frame_indice_mask'] = frame_mask
		return res

	for at_id, at_pos in zip(atom_ids_list, atom_positions_list):
		if at_id in ref_atom_ids_index: 
			adjust_idx = ref_atom_ids_index[at_id]
			frame_atom_pos[adjust_idx] = at_pos
			frame_atom_pos_mask[adjust_idx] = 1

			if residue_name_3 in residue_constants.PROTEIN_LIST:
				if at_id in residue_constants.PROTEIN_FRAME_ATOM:
					if at_id == 'N':
						ai_mask[adjust_idx] = 1
					elif at_id == 'CA':
						bi_mask[adjust_idx] = 1
					elif at_id == 'C':
						ci_mask[adjust_idx] = 1
			elif residue_name_3 in residue_constants.DNA_RNA_LIST:
				if at_id in residue_constants.DNA_RNA_FRAME_ATOM:
					if at_id == "C1'":
						ai_mask[adjust_idx] = 1
					elif at_id == "C3'":
						bi_mask[adjust_idx] = 1
					elif at_id == "C4'":
						ci_mask[adjust_idx] = 1
			else:
				## ligand/ion/non-standard token is need to be post processed.
				ai_mask[adjust_idx] = 1
				bi_mask[adjust_idx] = 1
				ci_mask[adjust_idx] = 1
		else:
			## NOTE: To filter the atom_ids not in the ccd_dict.
			pass
	
	frame_atom_nums = np.sum(reduce(np.logical_or, [ai_mask, bi_mask, ci_mask]))
	if frame_atom_nums < 3:
		## NOTE: if frame atom mask is less than 3, the frame is marked as invalid. such as Zn, Na, Cl, etc.
		if not residue_is_standard:
			# N_token = atoms. non-standard residue.
			res['ai_indice'] = np.zeros_like(ai_mask)
			res['bi_indice'] = np.zeros_like(bi_mask)
			res['ci_indice'] = np.zeros_like(ci_mask)
			res['frame_indice_mask'] = np.zeros_like(frame_mask)
		return res
	elif residue_is_standard and frame_atom_nums > 3:
		## NOTE: if the frame atom nums is more than 3 and residue is standard, the frame is marked as invalid
		## some standard ccd may have more than one frame atoms in ai, bi, ci. this missleading frame is not valid.
		return res

	if residue_is_standard:
		res['ai_indice'] = np.where(ai_mask)[0]
		res['bi_indice'] = np.where(bi_mask)[0]
		res['ci_indice'] = np.where(ci_mask)[0]
		res['frame_indice_mask'] = np.array([1], dtype=np.int32)
	else:
		# N_token = atoms. non-standard residue.
		## ligand is need to be post processed.
		res = find_two_closest_atoms(frame_atom_pos, frame_atom_pos_mask)

	return res 


def map_to_continuous_indices(arr: np.ndarray) -> np.ndarray:
    """Map the index array to continuous indices.
    
    	Args:
    		arr: the index array; (N,)
    	Returns:
    		the continuous index array; (N,)
		Example:
			input: [3, 3, 3, 3, 74, 74, 74, ... , n-2, n-1, n-1, n, n, n]
			output: [0, 0, 0, 0, 1, 1, 1, ....., m-2, m-1, m-1, m, m, m]
    """
    if arr.shape[0] == 0: return arr
    index_map = {arr[0]:0}
    counter_idx = 0
    for i in range(1, len(arr)):
        assert arr[i] >= arr[i-1], \
            f"not an ascending array at pos {i} i: {arr[i]} i-1: {arr[i-1]}"
        if not arr[i] == arr[i-1]:
            counter_idx += 1
            index_map[arr[i]] = counter_idx
    for i in range(len(arr)):
        arr[i] = index_map[arr[i]]
    return arr


@functools.lru_cache(maxsize=256)
def int_id_to_str_id(num: int) -> str:
    """Encodes a number as a string, using reverse spreadsheet style naming.

    Args:
        num: A positive integer.

    Returns:
        A string that encodes the positive integer using reverse spreadsheet style,
        naming e.g. 1 = A, 2 = B, ..., 27 = AA, 28 = BA, 29 = CA, ... This is the
        usual way to encode chain IDs in mmCIF files.
    """
    if num <= 0:
        raise ValueError(f'Only positive integers allowed, got {num}.')

    num = num - 1  # 1-based indexing.
    output = []
    while num >= 0:
        output.append(chr(num % 26 + ord('A')))
        num = num // 26 - 1
    return ''.join(output)


@functools.lru_cache(maxsize=256)
def str_id_to_int_id(str_id: str) -> int:
    """Encodes an mmCIF-style string chain ID as an integer.

    The integer IDs are one based so this function is the inverse of
    int_id_to_str_id.

    Args:
        str_id: A string chain ID consisting only of upper case letters A-Z.

    Returns:
        An integer that can be used to order mmCIF chain IDs in the standard
        (reverse spreadsheet style) ordering.
    """
    if not re.match('^[A-Z]+$', str_id):
        raise ValueError(f'String ID must be upper case letters, got {str_id}.')

    offset = ord('A') - 1
    output = 0
    for i, c in enumerate(str_id):
        output += (ord(c) - offset) * int(26**i)
    return output


@np.vectorize
def user_asymid_to_weight(user_asymid: str) -> tuple:
    """Convert user asymid to weight and decorate with np.vectorize.
    
    Args:
        user_asymid: str, such as: "A-11"
    
    Returns:
        tuple: (int, int), such as: (1, 11)
    """
    parts = user_asymid.split('-')
    assert len(parts) == 2, f"Invalid user_asymid: {user_asymid}"
    return (str_id_to_int_id(parts[0]), int(parts[1]))


def sorted_results_by_chain_order(results, ori_chain_ids, ori_chain_ids_token):
	"""Reorder the inference results by the original chain ids.
		
	Args:
		results: dict, inference results.
		ori_chain_ids: np.ndarray, original chain ids for atom.
		ori_chain_ids_token: np.ndarray, original chain ids for token.
	Returns:
		reordered_results: dict, reordered inference results.
	"""
	seen = set()
	# Maintain the order of unique chain IDs
	ori_chain_level = [x for x in ori_chain_ids_token if not (x in seen or seen.add(x))]
	ent_weights_chain, sym_weights_chain = user_asymid_to_weight(ori_chain_level)
	ent_weights_token, sym_weights_token = user_asymid_to_weight(ori_chain_ids_token)
	ent_weights_atom, sym_weights_atom = user_asymid_to_weight(ori_chain_ids)
	
	sorted_indices_chain = np.lexsort((sym_weights_chain, ent_weights_chain))
	sorted_indices_token = np.lexsort((sym_weights_token, ent_weights_token))
	sorted_indices_atom = np.lexsort((sym_weights_atom, ent_weights_atom))
	
	reordered_results = {}
	for key in list(results.keys()):
		assert DISPLAY_RESULTS_KEYS[key] in DISPLAY_DIM, \
			f"key {key} not in DISPLAY_DIM, Got {DISPLAY_RESULTS_KEYS[key]}, " \
			f"Expected keys are: {DISPLAY_DIM}"
		_res = results.pop(key)
		if DISPLAY_RESULTS_KEYS[key] == 'SINGLE':
			reordered_results[key] = _res
		elif DISPLAY_RESULTS_KEYS[key] == 'NUM_CHAIN':
			reordered_results[key] = np.take(_res, sorted_indices_chain, axis=0) 
		elif DISPLAY_RESULTS_KEYS[key] == 'NUM_TOKEN':
			reordered_results[key] = np.take(_res, sorted_indices_token, axis=0) 
		elif DISPLAY_RESULTS_KEYS[key] == 'NUM_ATOM':
			reordered_results[key] = np.take(_res, sorted_indices_atom, axis=0) 
		elif DISPLAY_RESULTS_KEYS[key] == 'NUM_CHAIN, NUM_CHAIN':
			reordered_results[key] = np.take(_res, sorted_indices_chain, axis=0) 
			reordered_results[key] = np.take(reordered_results[key], sorted_indices_chain, axis=1) 
		elif DISPLAY_RESULTS_KEYS[key] == 'NUM_TOKEN, NUM_TOKEN':
			reordered_results[key] = np.take(_res, sorted_indices_token, axis=0) 
			reordered_results[key] = np.take(reordered_results[key], sorted_indices_token, axis=1)
		else:
			raise ValueError(f"key {key} not supported in _reorder_results yet.")

	return reordered_results