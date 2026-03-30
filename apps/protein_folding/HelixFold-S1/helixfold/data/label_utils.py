''' tbd. '''
import sys
import pickle
sys.path.append('..')
import gzip
import os, io
import numpy as np
from functools import reduce
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List
from Bio.PDB import PDBParser

import helixfold.data.mmcif_parsing_paddle as mmcif_parsing
from helixfold.data.parsers import parse_ccd_fasta
from helixfold.common import residue_constants

LABEL_KEYS = [
	"label_ccd_ids",  # [N_atom, ]
	"label_atom_ids", # [N_atom,]
	"all_atom_pos", # [N_atom, 3]
	"all_atom_pos_mask", # [N_atom]
	"all_centra_token_indice",  # [N_token, ]
	"all_centra_token_indice_mask",  # [N_token,]
	"all_token_to_atom_nums", # [N_token,]
	"pseudo_beta", # [N_token,]
	"pseudo_beta_mask", # [N_token, ]
	"frame_ai_indice", # [N_token, ]
	"frame_bi_indice", # [N_token, ]
	"frame_ci_indice", # [N_token, ]
	"frame_mask", # [N_token, ]
]


NEED_OFFSET_KEYS = [
	"all_centra_token_indice", # [N_token, ]
	"frame_ai_indice", # [N_token, ]
	"frame_bi_indice", # [N_token, ]
	"frame_ci_indice",  # [N_token, ]
]


class Error(Exception):
  """Base class for exceptions."""


class CaDistanceError(Error):
	"""An error indicating that a CA atom distance exceeds a threshold."""


def find_two_closest_atoms(frame_coords: np.ndarray,
							frame_coords_mask: np.ndarray):
	"""
		Find the two closest atoms to the reference bi atom, Only for ligand
		Args: 
			frame_coords: the coordinates of all atoms; (N,3)
			frame_coords_mask: the mask of the frame atoms; (N,1)
		Returns:
		 	dict, # N_atom = N_token, indice and indice mask 
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


def is_frame_atoms_collinear(ai, bi, ci, threshold=25):
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


def get_pae_frame_mask(atom_ids_list: list, 
						atom_positions_list: list,
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


def pad_pdb_atom_pos_to_ccd_pos(residue_name_3: str, 
								positions: List[float],
								atom_ids: List[str],
								bfactors: List[float],
								confidence_threshold: float = 0.5, 
								ccd_dict: dict = None) -> Mapping[str, np.ndarray]:
	"""
	NOTE: use to padding each residue-level pos. Only for distill protein pdb data.
	ccd_dict: key is ccd id, below list all the keys:
		id: str
		smiles: List[str]
		atom_ids: List[str] # [OXT]
		atom_symbol: List[str] # [O]
		charge: List[int]
		leave_atom_flag: List[str]
		position: List[List[float]] # [[-0.456 0.028  -0.001], [-0.376 1.240  0.001]]
		coval_bonds: List[List[str]] # (C   OXT SING)
		raw_string: Any
			
	return:
		padding residue(atom pos)
			atom_pos, atom_mask, centra_token_indice, centra_token_indice_mask
			and so on.
	"""
	assert ccd_dict is not None

	_residue_is_missing = False  ## NOTE: we assert the residue_name is not missing.
	_residue_name = residue_name_3
	_residue_is_standard = True if _residue_name in residue_constants.STANDARD_LIST \
									else False
	_atom_positions_list = positions
	_atom_ids_list = atom_ids

	refs = ccd_dict[_residue_name]  # O(1)
	ref_atom_ids_index = { 
		name: i for i, name in enumerate(refs['atom_ids'])
	}
	total_nums = len(ref_atom_ids_index)
	assert total_nums > 0, f'TODO filter - Got CCD <{_residue_name}>: 0 atom nums.'

	bfactors_res = 0
	padding_atom_pos = np.zeros([total_nums, 3], dtype=np.float32)
	padding_atom_mask = np.zeros([total_nums], dtype=np.int32)
	## centra_token_indice is used to indicate centra atom idx
	centra_token_indice = np.zeros([total_nums], dtype=np.int32) # [N_atom] 0-1, indicate whether is centra-token
	pseudo_token_indice = np.zeros([total_nums], dtype=np.int32)  # [N_atom] 0-1, indicate whether is pseudo-token

	for at_id, at_pos, bfa in zip(_atom_ids_list, _atom_positions_list, bfactors):
		## NOTE: we filter the atom_id not in reference ccd's atom_ids
		if at_id in ref_atom_ids_index: 
			adjust_idx = ref_atom_ids_index[at_id]
			padding_atom_pos[adjust_idx] = at_pos
			padding_atom_mask[adjust_idx] = 1
			bfactors_res = bfa
			if _residue_is_standard:
				if at_id in residue_constants.CENTRA_TOKEN:
					centra_token_indice[adjust_idx] = 1
					if _residue_name.upper() == "GLY":
						pseudo_token_indice[adjust_idx] = 1
				elif at_id in residue_constants.PSEUDO_TOKEN:
					if at_id == 'CB': 
						# is protein-residue, only protein residue has CB in CCD.
						pseudo_token_indice[adjust_idx] = 1
					elif at_id == 'P' and _residue_name.upper() in residue_constants.DNA_RNA_LIST: 
						# is dna/rna nucleotides 
						pseudo_token_indice[adjust_idx] = 1
			else:
				centra_token_indice[adjust_idx] = 1
				pseudo_token_indice[adjust_idx] = 1
		else:
			# print(f'[WARNING] filter <{at_id}> not in ccd <{_residue_name}>.')
			pass

	## Get Frame atom mask
	frame_indice = get_pae_frame_mask(atom_ids_list=_atom_ids_list, 
						atom_positions_list=_atom_positions_list,
						residue_name_3=_residue_name,
						residue_is_standard=_residue_is_standard,
						residue_is_missing=_residue_is_missing,
						ref_atom_ids_index=ref_atom_ids_index)

	## NOTE: residue-level factor, indicate the confidence of prediction
	## if low confidence, this residue will drop.
	high_confidence = bfactors_res > confidence_threshold
	if(not high_confidence):
		padding_atom_mask = np.zeros_like(padding_atom_mask, dtype=np.int32)
		centra_token_indice = centra_token_indice * padding_atom_mask
		pseudo_token_indice = pseudo_token_indice * padding_atom_mask
		frame_indice['frame_indice_mask'] = np.zeros_like(frame_indice['frame_indice_mask'])
		frame_indice['ai_indice'] = np.zeros_like(frame_indice['ai_indice'])
		frame_indice['bi_indice'] = np.zeros_like(frame_indice['bi_indice'])
		frame_indice['ci_indice'] = np.zeros_like(frame_indice['ci_indice'])

	pad_feats = {
		**frame_indice,
		'ccd_ids': np.array([_residue_name] * total_nums , dtype=object),  # N_atom
		'atom_ids': refs['atom_ids'], # [N_atom]
		'atom_pos': padding_atom_pos, # [total_nums, 3]
		'atom_mask': padding_atom_mask, # [total_nums]
		'res_b_factors': np.array([bfactors_res], dtype=np.float32), # [N_token]
		'token_to_atom_nums': np.array([total_nums], dtype=np.int32) \
										 if _residue_is_standard else np.ones([total_nums], dtype=np.int32)
	}

	## N_atom -> N_token : centra_token_indice, centra_token_indice_mask
	if _residue_is_standard:
		if not _residue_is_missing:
			centra_token_indice = np.where(centra_token_indice == 1)[0]
			assert len(centra_token_indice) <= 1, f"residue should be has only one centra-token, Got {len(centra_token_indice)}"
			if len(centra_token_indice) == 1:
				pad_feats['centra_token_indice'] = centra_token_indice #[N_token]
				pad_feats['centra_token_indice_mask'] = np.array([1], dtype=np.int32)
			else:
				pad_feats['centra_token_indice'] = np.array([0], dtype=np.int32) #[N_token]
				pad_feats['centra_token_indice_mask'] = np.array([0], dtype=np.int32)
			
			pseudo_token_indice = np.where(pseudo_token_indice == 1)[0]
			assert len(pseudo_token_indice) <= 1, f"residue should be has only one pesudo-token. Got {len(pseudo_token_indice)}"
			if len(pseudo_token_indice) == 1:
				pad_feats['pseudo_token_indice'] = pseudo_token_indice
				pad_feats['pseudo_token_indice_mask'] = np.array([1], dtype=np.int32)
			else:
				pad_feats['pseudo_token_indice'] = np.array([0], dtype=np.int32)
				pad_feats['pseudo_token_indice_mask'] = np.array([0], dtype=np.int32)
		else:
			pad_feats['centra_token_indice'] = np.array([0], dtype=np.int32) #[N_token]
			pad_feats['centra_token_indice_mask'] = np.array([0], dtype=np.int32)

			pad_feats['pseudo_token_indice'] = np.array([0], dtype=np.int32)
			pad_feats['pseudo_token_indice_mask'] = np.array([0], dtype=np.int32)
	else:
		# if is non-standard, token_nums == atom_nums
		pad_feats['centra_token_indice'] = np.zeros_like(centra_token_indice, dtype=np.int32)
		pad_feats['centra_token_indice_mask'] = centra_token_indice

		pad_feats['pseudo_token_indice'] = np.zeros_like(pseudo_token_indice, dtype=np.int32)
		pad_feats['pseudo_token_indice_mask'] = pseudo_token_indice

	return pad_feats


def from_distill_pdb_string(pdb_str: str, 
							ccd_preprocessed_dict: dict, 
							confidence_threshold: float = 0.5, 
							chain_id: Optional[str] = None) -> dict:
	"""Takes a PDB string and constructs a Protein object.

	WARNING: All non-standard residue types will be converted into UNK. All
		non-standard atoms will be ignored.

	Args:
		confidence_threshold: the residue-level bfactor cutoff of distill pdb;
		pdb_str: The contents of the pdb file
		ccd_preprocessed_dict: The preprocessed ccd dictionary
		chain_id: If chain_id is specified (e.g. A), then only that chain
		is parsed. Otherwise all chains are parsed.

	Returns:
		A new `Protein` parsed from the pdb contents.
	"""
	pdb_fh = io.StringIO(pdb_str)
	parser = PDBParser(QUIET=True)
	structure = parser.get_structure('none', pdb_fh)
	models = list(structure.get_models())
	if len(models) != 1:
		raise ValueError(
			f'Only single model PDBs are supported. Found {len(models)} models.')
	model = models[0]

	all_bfactors = np.empty((0,), dtype=np.float32) # N_token
	all_chain_ids = np.empty((0,), dtype=object) # N_token
	all_residue_index = np.empty((0,), dtype=np.int32) # N_token

	all_ccd_ids = np.empty((0,), dtype=object) 
	all_atom_ids = np.empty((0,), dtype=object)
	all_atom_pos = np.empty((0, 3), dtype=np.float32)
	all_atom_mask = np.empty((0,), dtype=np.int32)
	all_centra_token_indice = np.empty((0,), dtype=np.int32)
	all_centra_token_indice_mask = np.empty((0,), dtype=np.int32)
	all_token_to_atom_nums = np.empty((0,), dtype=np.int32)
	all_pseudo_token_indice = np.empty((0,), dtype=np.int32)
	all_pseudo_token_indice_mask = np.empty((0,), dtype=np.int32)

	frame_ai_indice = np.empty((0,), dtype=np.int32) # Ntoken
	frame_bi_indice = np.empty((0,), dtype=np.int32) # Ntoken
	frame_ci_indice = np.empty((0,), dtype=np.int32) # Ntoken
	frame_mask = np.empty((0,), dtype=np.int32) # Ntoken
	
	for chain in model:
		if chain_id is not None and chain.id != chain_id:
			continue

		frame_indice_offset = 0
		for res in chain:
			if res.id[2] != ' ':
				raise ValueError(
					f'PDB contains an insertion code at chain {chain.id} and residue '
					f'index {res.id[1]}. These are not supported.')
			res_shortname = residue_constants.restype_3to1.get(res.resname, 'X')
			restype_idx = residue_constants.AF3_restype_order[res.resname]
			_res_pos, _res_atomid, _res_bfactors = [], [], []
			for atom in res:
				_res_pos.append(atom.coord)
				_res_atomid.append(atom.name)
				_res_bfactors.append(atom.bfactor)
			
			pad_feats = pad_pdb_atom_pos_to_ccd_pos(res.resname, _res_pos, 
													_res_atomid, _res_bfactors, 
													confidence_threshold,
													ccd_preprocessed_dict)

			pad_feats['ai_indice'] = pad_feats['ai_indice'] + frame_indice_offset
			pad_feats['bi_indice'] = pad_feats['bi_indice'] + frame_indice_offset
			pad_feats['ci_indice'] = pad_feats['ci_indice'] + frame_indice_offset
			frame_indice_offset += pad_feats['frame_atom_offset']		

			all_atom_ids = np.concatenate((all_atom_ids, pad_feats['atom_ids']))
			all_ccd_ids = np.concatenate((all_ccd_ids, pad_feats['ccd_ids']))

			all_atom_pos = np.concatenate((all_atom_pos, pad_feats['atom_pos']))
			all_atom_mask = np.concatenate((all_atom_mask, pad_feats['atom_mask']))
			all_centra_token_indice = np.concatenate((all_centra_token_indice, 
								pad_feats['centra_token_indice']))
			all_centra_token_indice_mask = np.concatenate((all_centra_token_indice_mask, 
								pad_feats['centra_token_indice_mask']))
			all_token_to_atom_nums = np.concatenate((all_token_to_atom_nums,
								pad_feats['token_to_atom_nums']))
			all_pseudo_token_indice = np.concatenate((all_pseudo_token_indice, 
								pad_feats['pseudo_token_indice']))
			all_pseudo_token_indice_mask = np.concatenate((all_pseudo_token_indice_mask, 
								pad_feats['pseudo_token_indice_mask']))

			all_bfactors = np.concatenate((all_bfactors, pad_feats['res_b_factors']))
			all_chain_ids = np.concatenate((all_chain_ids, np.array([chain.id], dtype=object)))
			all_residue_index = np.concatenate((all_residue_index, np.array([res.id[1]], dtype=np.int32)))

			frame_ai_indice = np.concatenate((frame_ai_indice, pad_feats['ai_indice']))
			frame_bi_indice = np.concatenate((frame_bi_indice, pad_feats['bi_indice']))
			frame_ci_indice = np.concatenate((frame_ci_indice, pad_feats['ci_indice']))
			frame_mask = np.concatenate((frame_mask, pad_feats['frame_indice_mask']))

	# print(all_centra_token_indice)
	cumsum_array = np.cumsum(all_token_to_atom_nums)
	all_centra_token_indice = all_centra_token_indice + np.insert(cumsum_array[:-1], 0, 0)
	all_pseudo_token_indice = all_pseudo_token_indice + np.insert(cumsum_array[:-1], 0, 0)
	assert all_atom_pos.shape[0] == all_atom_mask.shape[0] == all_atom_ids.shape[0]
	assert all_centra_token_indice.shape[0] == all_centra_token_indice_mask.shape[0]
	assert all_pseudo_token_indice.shape[0] == all_pseudo_token_indice_mask.shape[0]
	assert frame_ai_indice.shape[0] == frame_bi_indice.shape[0] == frame_ci_indice.shape[0] == frame_mask.shape[0] # Ntoken
	assert np.max(frame_ai_indice) < all_atom_pos.shape[0] and \
			np.max(frame_bi_indice) < all_atom_pos.shape[0] and \
			np.max(frame_ci_indice) < all_atom_pos.shape[0]

	# Chain IDs are usually characters so map these to ints.
	unique_chain_ids = np.unique(all_chain_ids)
	chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
	all_chain_ids = np.array([chain_id_mapping[cid] for cid in all_chain_ids])

	return {
		"label_ccd_ids": all_ccd_ids, # [N_atom, ]
		"label_atom_ids": all_atom_ids,	# [N_atom,]
		"all_atom_pos": all_atom_pos, # [N_atom, 3]
		"all_atom_pos_mask": all_atom_mask,  # [N_atom]
		"all_centra_token_indice": all_centra_token_indice, # [N_token, ]
		"all_centra_token_indice_mask": all_centra_token_indice_mask, # [N_token,]
		"all_token_to_atom_nums": all_token_to_atom_nums, # [N_token,]
		"pseudo_beta": all_atom_pos[all_pseudo_token_indice], # [N_token,]
		"pseudo_beta_mask": all_pseudo_token_indice_mask, # [N_token, ]

		## some distill features for check with origin AF2-multimer
		"release_date": np.array(['None'], dtype=object),
		"resolution": np.array([0.]).astype(np.float32),
		"all_b_factors": all_bfactors, # N_token
		"all_chain_ids": all_chain_ids, # N_token
		"all_residue_index": all_residue_index, # N_token

		## for pae loss computation.
		"frame_ai_indice": frame_ai_indice, # [N_token, ]
		"frame_bi_indice": frame_bi_indice, # [N_token, ]
		"frame_ci_indice": frame_ci_indice, # [N_token, ]
		"frame_mask": frame_mask, # [N_token, ]
	}


def from_distill_afdb_np_feats(af2_np_structure: dict,
						ccd_preprocessed_dict: dict, 
						confidence_threshold: float = 0.5,) -> dict:
	"""
		The function is used to map af2-afdb structure to hf3 label feats.
		af2_np_structure: af2-afdb structure np.ndarray, has keys,
			aatype_index, # Num_residue
			all_atom_positions,  # Num_residue, 37, 3
			all_atom_mask,  # Num_residue, 37, 3
			resolution,
			confidence_list # Num_residue
			ccd_ids_list, # Num_residue
		ccd_preprocessed_dict: The preprocessed ccd dictionary
		confidence_threshold: the residue-level bfactor cutoff of distill pdb;
	"""
	## atom_order_index, the mapping from af2 atom_idx_mask to atom_id, such as 0 -> CA. 1 -> CB.
	ccd_ids_list = af2_np_structure['ccd_ids_list']
	bfactors = af2_np_structure['confidence_list']
	af2_atom_order_index = {v: k for k, v in residue_constants.atom_order.items()}

	all_bfactors = np.empty((0,), dtype=np.float32) # N_token
	all_ccd_ids = np.empty((0,), dtype=object) 
	all_atom_ids = np.empty((0,), dtype=object)
	all_atom_pos = np.empty((0, 3), dtype=np.float32)
	all_atom_mask = np.empty((0,), dtype=np.int32)
	all_centra_token_indice = np.empty((0,), dtype=np.int32)
	all_centra_token_indice_mask = np.empty((0,), dtype=np.int32)
	all_token_to_atom_nums = np.empty((0,), dtype=np.int32)
	all_pseudo_token_indice = np.empty((0,), dtype=np.int32)
	all_pseudo_token_indice_mask = np.empty((0,), dtype=np.int32)

	frame_ai_indice = np.empty((0,), dtype=np.int32) # Ntoken
	frame_bi_indice = np.empty((0,), dtype=np.int32) # Ntoken
	frame_ci_indice = np.empty((0,), dtype=np.int32) # Ntoken
	frame_mask = np.empty((0,), dtype=np.int32) # Ntoken

	frame_indice_offset = 0
	for idx, (atom_pos, atom_mask) in enumerate(zip(af2_np_structure['all_atom_positions'], \
														af2_np_structure['all_atom_mask'])):
		_res_name_3 = ccd_ids_list[idx]
		_res_pos = atom_pos[atom_mask == 1]
		_res_bfactors = [bfactors[idx]] * len(_res_pos)
		_res_atom_order = np.where(atom_mask == 1)[0]
		_res_atomid = [af2_atom_order_index[order_idx] for order_idx in _res_atom_order]
		
		pad_feats = pad_pdb_atom_pos_to_ccd_pos(_res_name_3, _res_pos, 
												_res_atomid, _res_bfactors, 
												confidence_threshold,
												ccd_preprocessed_dict)

		pad_feats['ai_indice'] = pad_feats['ai_indice'] + frame_indice_offset
		pad_feats['bi_indice'] = pad_feats['bi_indice'] + frame_indice_offset
		pad_feats['ci_indice'] = pad_feats['ci_indice'] + frame_indice_offset
		frame_indice_offset += pad_feats['frame_atom_offset']	

		all_atom_ids = np.concatenate((all_atom_ids, pad_feats['atom_ids']))
		all_ccd_ids = np.concatenate((all_ccd_ids, pad_feats['ccd_ids']))

		all_atom_pos = np.concatenate((all_atom_pos, pad_feats['atom_pos']))
		all_atom_mask = np.concatenate((all_atom_mask, pad_feats['atom_mask']))
		all_centra_token_indice = np.concatenate((all_centra_token_indice, 
							pad_feats['centra_token_indice']))
		all_centra_token_indice_mask = np.concatenate((all_centra_token_indice_mask, 
							pad_feats['centra_token_indice_mask']))
		all_token_to_atom_nums = np.concatenate((all_token_to_atom_nums,
							pad_feats['token_to_atom_nums']))
		all_pseudo_token_indice = np.concatenate((all_pseudo_token_indice, 
							pad_feats['pseudo_token_indice']))
		all_pseudo_token_indice_mask = np.concatenate((all_pseudo_token_indice_mask, 
							pad_feats['pseudo_token_indice_mask']))

		all_bfactors = np.concatenate((all_bfactors, pad_feats['res_b_factors']))
		frame_ai_indice = np.concatenate((frame_ai_indice, pad_feats['ai_indice']))
		frame_bi_indice = np.concatenate((frame_bi_indice, pad_feats['bi_indice']))
		frame_ci_indice = np.concatenate((frame_ci_indice, pad_feats['ci_indice']))
		frame_mask = np.concatenate((frame_mask, pad_feats['frame_indice_mask']))


	cumsum_array = np.cumsum(all_token_to_atom_nums)
	all_centra_token_indice = all_centra_token_indice + np.insert(cumsum_array[:-1], 0, 0)
	all_pseudo_token_indice = all_pseudo_token_indice + np.insert(cumsum_array[:-1], 0, 0)
	assert all_atom_pos.shape[0] == all_atom_mask.shape[0] == all_atom_ids.shape[0]
	assert all_centra_token_indice.shape[0] == all_centra_token_indice_mask.shape[0]
	assert all_pseudo_token_indice.shape[0] == all_pseudo_token_indice_mask.shape[0]
	assert frame_ai_indice.shape[0] == frame_bi_indice.shape[0] == frame_ci_indice.shape[0] == frame_mask.shape[0] # Ntoken
	assert np.max(frame_ai_indice) < all_atom_pos.shape[0] and \
			np.max(frame_bi_indice) < all_atom_pos.shape[0] and \
			np.max(frame_ci_indice) < all_atom_pos.shape[0]

	return {
		"label_ccd_ids": all_ccd_ids, # [N_atom, ]
		"label_atom_ids": all_atom_ids,	# [N_atom,]
		"all_atom_pos": all_atom_pos, # [N_atom, 3]
		"all_atom_pos_mask": all_atom_mask,  # [N_atom]
		"all_centra_token_indice": all_centra_token_indice, # [N_token, ]
		"all_centra_token_indice_mask": all_centra_token_indice_mask, # [N_token,]
		"all_token_to_atom_nums": all_token_to_atom_nums, # [N_token,]
		"pseudo_beta": all_atom_pos[all_pseudo_token_indice], # [N_token,]
		"pseudo_beta_mask": all_pseudo_token_indice_mask, # [N_token, ]

		## some distill features for check with origin AF2-multimer
		"release_date": np.array(['None'], dtype=object),
		"resolution": af2_np_structure['resolution'],
		"all_b_factors": all_bfactors, # N_token

		## for pae loss computation.
		"frame_ai_indice": frame_ai_indice, # [N_token, ]
		"frame_bi_indice": frame_bi_indice, # [N_token, ]
		"frame_ci_indice": frame_ci_indice, # [N_token, ]
		"frame_mask": frame_mask, # [N_token, ]
	}


def pad_atom_pos_to_ccd_pos(res_at_position: mmcif_parsing.ResidueAtPosition,
							ccd_dict: dict) -> Mapping[str, np.ndarray]:
	"""
		NOTE: use to padding each residue-level pos.
		ccd_dict: key is ccd id, below list all the keys:
			id: str
			smiles: List[str]
			atom_ids: List[str] # [OXT]
			atom_symbol: List[str] # [O]
			charge: List[int]
			leave_atom_flag: List[str]
			position: List[List[float]] # [[-0.456 0.028  -0.001], [-0.376 1.240  0.001]]
			coval_bonds: List[List[str]] # (C   OXT SING)
			raw_string: Any
		
		res_at_position: mmcif_parsing.ResidueAtPosition, 
			can read atom_pos, atom_id, is_miss_info from .cif
		
		return:
			padding residue(atom pos)
				atom_pos, atom_mask, centra_token_indice, centra_token_indice_mask
	"""
	_residue_is_missing = res_at_position.is_missing
	_residue_name = res_at_position.residue_name
	_residue_is_standard = True if _residue_name in residue_constants.STANDARD_LIST \
									else False
	_atom_positions_list = res_at_position.positions
	_atom_ids_list = res_at_position.atom_ids

	refs = ccd_dict[_residue_name]  # O(1)
	ref_atom_ids_index = { 
		name: i for i, name in enumerate(refs['atom_ids'])
	}
	total_nums = len(ref_atom_ids_index)
	assert total_nums > 0, f'TODO filter - Got CCD <{_residue_name}>: 0 atom nums.'

	padding_atom_pos = np.zeros([total_nums, 3], dtype=np.float32)
	padding_atom_mask = np.zeros([total_nums], dtype=np.int32)
	## centra_token_indice is used to indicate centra atom idx
	centra_token_indice = np.zeros([total_nums], dtype=np.int32) # [N_atom] 0-1, indicate whether is centra-token
	pseudo_token_indice = np.zeros([total_nums], dtype=np.int32)  # [N_atom] 0-1, indicate whether is pseudo-token

	for at_id, at_pos in zip(_atom_ids_list, _atom_positions_list):
		## NOTE: we filter the atom_id not in reference ccd's atom_ids
		if at_id in ref_atom_ids_index: 
			adjust_idx = ref_atom_ids_index[at_id]
			padding_atom_pos[adjust_idx] = at_pos
			padding_atom_mask[adjust_idx] = 1
			if _residue_is_standard:
				if at_id in residue_constants.CENTRA_TOKEN:
					centra_token_indice[adjust_idx] = 1
					if _residue_name.upper() == "GLY":
						pseudo_token_indice[adjust_idx] = 1
				elif at_id in residue_constants.PSEUDO_TOKEN:
					if at_id == 'CB': 
						# is protein-residue, only protein residue has CB in CCD.
						pseudo_token_indice[adjust_idx] = 1
					elif at_id == 'P' and _residue_name.upper() in residue_constants.DNA_RNA_LIST: 
						# is dna/rna nucleotides 
						pseudo_token_indice[adjust_idx] = 1
			else:
				centra_token_indice[adjust_idx] = 1
				pseudo_token_indice[adjust_idx] = 1
		else:
			# print(f'[WARNING] filter <{at_id}> not in ccd <{_residue_name}>.')
			pass

	## Get Frame atom mask
	frame_indice = get_pae_frame_mask(atom_ids_list=_atom_ids_list, 
						atom_positions_list=_atom_positions_list,
						residue_name_3=_residue_name,
						residue_is_standard=_residue_is_standard,
						residue_is_missing=_residue_is_missing,
						ref_atom_ids_index=ref_atom_ids_index)

	pad_feats = {
		**frame_indice,
		'residue_is_standard': _residue_is_standard,
		'ccd_ids': np.array([_residue_name] * total_nums , dtype=object),  # N_atom
		'atom_ids': refs['atom_ids'], # [N_atom]
		'atom_pos': padding_atom_pos, # [total_nums, 3]
		'atom_mask': padding_atom_mask, # [total_nums]
  		'centra_token_indice': centra_token_indice, #[N_atom] 
		'pseudo_token_indice': pseudo_token_indice, #[N_atom] 
		'token_to_atom_nums': np.array([total_nums], dtype=np.int32) \
										 if _residue_is_standard else np.ones([total_nums], dtype=np.int32)
	}

	## N_atom -> N_token : centra_token_indice, centra_token_indice_mask
	if _residue_is_standard:
		if not _residue_is_missing:
			centra_token_indice = np.where(centra_token_indice == 1)[0]
			assert len(centra_token_indice) <= 1, f"residue should be has only one centra-token, Got {len(centra_token_indice)}"
			if len(centra_token_indice) == 1:
				pad_feats['centra_token_indice'] = centra_token_indice #[N_token]
				pad_feats['centra_token_indice_mask'] = np.array([1], dtype=np.int32)
			else:
				pad_feats['centra_token_indice'] = np.array([0], dtype=np.int32) #[N_token]
				pad_feats['centra_token_indice_mask'] = np.array([0], dtype=np.int32)
			
			pseudo_token_indice = np.where(pseudo_token_indice == 1)[0]
			assert len(pseudo_token_indice) <= 1, f"residue should be has only one pesudo-token. Got {len(pseudo_token_indice)}"
			if len(pseudo_token_indice) == 1:
				pad_feats['pseudo_token_indice'] = pseudo_token_indice
				pad_feats['pseudo_token_indice_mask'] = np.array([1], dtype=np.int32)
			else:
				pad_feats['pseudo_token_indice'] = np.array([0], dtype=np.int32)
				pad_feats['pseudo_token_indice_mask'] = np.array([0], dtype=np.int32)
		else:
			pad_feats['centra_token_indice'] = np.array([0], dtype=np.int32) #[N_token]
			pad_feats['centra_token_indice_mask'] = np.array([0], dtype=np.int32)

			pad_feats['pseudo_token_indice'] = np.array([0], dtype=np.int32)
			pad_feats['pseudo_token_indice_mask'] = np.array([0], dtype=np.int32)
	else:
		# if is non-standard, token_nums == atom_nums
		pad_feats['centra_token_indice'] = np.zeros_like(centra_token_indice, dtype=np.int32)
		pad_feats['centra_token_indice_mask'] = centra_token_indice

		pad_feats['pseudo_token_indice'] = np.zeros_like(pseudo_token_indice, dtype=np.int32)
		pad_feats['pseudo_token_indice_mask'] = pseudo_token_indice

	return pad_feats


def get_atom_positions(
	mmcif_object: mmcif_parsing.MmcifObject,
	mmcif_chain_id: str,
	ccd_dict: dict) -> Mapping[str, np.ndarray]:
	"""Gets atom positions and mask from a list of mmcif_object Residues."""
	chain = mmcif_object.seqres_to_structure[mmcif_chain_id]
	num_res = len(chain)
	
	all_ccd_ids = np.empty((0,), dtype=object)
	all_atom_ids = np.empty((0,), dtype=object)
	all_atom_pos = np.empty((0, 3), dtype=np.float32)
	all_atom_mask = np.empty((0,), dtype=np.int32)
	all_centra_token_indice = np.empty((0,), dtype=np.int32)
	all_centra_token_indice_mask = np.empty((0,), dtype=np.int32)
	all_token_to_atom_nums = np.empty((0,), dtype=np.int32)
	all_pseudo_token_indice = np.empty((0,), dtype=np.int32)
	all_pseudo_token_indice_mask = np.empty((0,), dtype=np.int32)

	frame_ai_indice = np.empty((0,), dtype=np.int32) # Ntoken
	frame_bi_indice = np.empty((0,), dtype=np.int32) # Ntoken
	frame_ci_indice = np.empty((0,), dtype=np.int32) # Ntoken
	frame_mask = np.empty((0,), dtype=np.int32) # Ntoken
	
	frame_indice_offset = 0
	for idx, res_index in enumerate(chain.keys()):
		res_at_position = chain[res_index]
		# print(res_index, res_at_position.residue_name)
		pad_feats = pad_atom_pos_to_ccd_pos(res_at_position, ccd_dict)
		
		# # if res_at_position.is_missing:
		# for k, v in pad_feats.items():
		# 	if k.startswith('pseudo'):
		# 		print(k, v.shape)
		pad_feats['ai_indice'] = pad_feats['ai_indice'] + frame_indice_offset
		pad_feats['bi_indice'] = pad_feats['bi_indice'] + frame_indice_offset
		pad_feats['ci_indice'] = pad_feats['ci_indice'] + frame_indice_offset
		frame_indice_offset += pad_feats['frame_atom_offset']			
		
		all_atom_ids = np.concatenate((all_atom_ids, pad_feats['atom_ids']))
		all_ccd_ids = np.concatenate((all_ccd_ids, pad_feats['ccd_ids']))

		all_atom_pos = np.concatenate((all_atom_pos, pad_feats['atom_pos']))
		all_atom_mask = np.concatenate((all_atom_mask, pad_feats['atom_mask']))
		all_centra_token_indice = np.concatenate((all_centra_token_indice, 
													pad_feats['centra_token_indice']))
		all_centra_token_indice_mask = np.concatenate((all_centra_token_indice_mask, 
													pad_feats['centra_token_indice_mask']))
		all_token_to_atom_nums = np.concatenate((all_token_to_atom_nums,
													pad_feats['token_to_atom_nums']))
		all_pseudo_token_indice = np.concatenate((all_pseudo_token_indice, 
													pad_feats['pseudo_token_indice']))
		all_pseudo_token_indice_mask = np.concatenate((all_pseudo_token_indice_mask, 
													pad_feats['pseudo_token_indice_mask']))
		frame_ai_indice = np.concatenate((frame_ai_indice, pad_feats['ai_indice']))
		frame_bi_indice = np.concatenate((frame_bi_indice, pad_feats['bi_indice']))
		frame_ci_indice = np.concatenate((frame_ci_indice, pad_feats['ci_indice']))
		frame_mask = np.concatenate((frame_mask, pad_feats['frame_indice_mask']))
		
	
	# print(all_centra_token_indice)
	cumsum_array = np.cumsum(all_token_to_atom_nums)
	offset = np.insert(cumsum_array[:-1], 0, 0)
	all_centra_token_indice = all_centra_token_indice + offset
	all_pseudo_token_indice = all_pseudo_token_indice + offset
	assert all_atom_pos.shape[0] == all_atom_mask.shape[0] == all_atom_ids.shape[0]
	assert all_centra_token_indice.shape[0] == all_centra_token_indice_mask.shape[0]
	assert all_pseudo_token_indice.shape[0] == all_pseudo_token_indice_mask.shape[0]
	assert frame_ai_indice.shape[0] == frame_bi_indice.shape[0] == frame_ci_indice.shape[0] == frame_mask.shape[0] # Ntoken
	assert np.max(frame_ai_indice) < all_atom_pos.shape[0] and \
			np.max(frame_bi_indice) < all_atom_pos.shape[0] and \
			np.max(frame_ci_indice) < all_atom_pos.shape[0]

	return {
		"label_ccd_ids": all_ccd_ids, # [N_atom, ]
		"label_atom_ids": all_atom_ids,	# [N_atom,]
		"all_atom_pos": all_atom_pos, # [N_atom, 3]
		"all_atom_pos_mask": all_atom_mask,  # [N_atom]
		"all_centra_token_indice": all_centra_token_indice, # [N_token, ]
		"all_centra_token_indice_mask": all_centra_token_indice_mask, # [N_token,]
		"all_token_to_atom_nums": all_token_to_atom_nums, # [N_token,]
		"pseudo_beta": all_atom_pos[all_pseudo_token_indice], # [N_token,]
		"pseudo_beta_mask": all_pseudo_token_indice_mask, # [N_token, ]

		## for pae loss computation.
		"frame_ai_indice": frame_ai_indice, # [N_token, ]
		"frame_bi_indice": frame_bi_indice, # [N_token, ]
		"frame_ci_indice": frame_ci_indice, # [N_token, ]
		"frame_mask": frame_mask, # [N_token, ]
	}


def load_chain(mmcif_object, ccd_preprocessed_dict, mmcif_chain_id='A'):
	"""Load chain info."""
	positions_infos = get_atom_positions(mmcif_object, mmcif_chain_id, ccd_preprocessed_dict)

	return {
		**positions_infos
	}


def load_meta_info(mmcif_object_unit):
	"""Load basic info."""
	resolution = np.array([mmcif_object_unit.header['resolution']], dtype=np.float32)
	
	if 'release_date' in mmcif_object_unit.header:
		release_date =  np.array([mmcif_object_unit.header['release_date']], dtype=object)
	else:
		release_date = np.array(['2024-05-20'], dtype=object)

	return {
		'release_date': release_date,
		'resolution': resolution, # [,]
	}


def get_position_label(
	mmcif_object_unit: mmcif_parsing.MmcifObject,
	mmcif_object_assembly: mmcif_parsing.MmcifObject = None,
	ccd_preprocessed_dict: dict = None,
	check_release_date: bool = True,
 	mmcif_chain_id: str = None):
	'''
		NOTE: mmcif_parsing.parse, `is_assembly_filter = False` becaues time-expensive filter.
		mmcif_object_unit:  mmcif_parsing.MmcifObject, for meta info parsing. such as bond_info, release_date and resolution.
		mmcif_object_assembly: mmcif_parsing.MmcifObject, for structure info parsing.
		mmcif_chain_id: chain id in mmcif file, str
	'''
	## TODO: The maximum number of polymer chains in a considered structure is 300 for training and 1000 for evaluation
	## train or eval flag ?
	assert mmcif_object_unit is not None or mmcif_object_assembly is not None
	file_id = mmcif_object_unit.file_id if not mmcif_object_unit is None else mmcif_object_assembly.file_id
	label_dict = {}

	## Read meta info, such as resolution and release_date
	meta_dict = {}
	if mmcif_object_unit:
		meta_dict = load_meta_info(mmcif_object_unit)

	## Read structure info. if assembly is initialized, to parse the assembly to get all label;
	if mmcif_object_assembly is not None:
		protein_chain_d = load_chain(mmcif_object_assembly, ccd_preprocessed_dict, mmcif_chain_id)
	else:
		protein_chain_d = load_chain(mmcif_object_unit, ccd_preprocessed_dict, mmcif_chain_id)
	
	protein_chain_d.update(meta_dict)
	if 'resolution' in protein_chain_d and protein_chain_d['resolution'] > 9:
		print(f'=> Skip [{file_id}] due to worse resolution {protein_chain_d["resolution"]}')
		return None
	if check_release_date and 'release_date' in protein_chain_d and protein_chain_d['release_date'] > "2021-09-30":
		print(f"=> Skip [{file_id}] due to data cutoff(2021-09-30). Got [{protein_chain_d['release_date']}]")
		return None

	label_dict[mmcif_chain_id] = protein_chain_d
	return label_dict

if __name__ == '__main__':
	# python helixfold/data/label_utils.py 
	import gzip
	import time
	import pickle

	def _read_cif(protein_struct_file):
		if protein_struct_file.endswith('.cif.gz'):
			with gzip.open(protein_struct_file, 'r') as f:
				cif_string = f.read().decode('utf8')
		else:
			cif_string = "".join(open(protein_struct_file, 'r').readlines())
		return cif_string

	# protein_struct_file_unit = 'data_assembly/mmcif/7lt0.cif.gz'
	# protein_struct_file_assembly = 'data_assembly/mmcif_assembly1/7lt0-assembly1.cif.gz'
	# protein_struct_file_unit = '/root/paddlejob/workspace/output/yexianbin/data/mmCIF_files/8jix.cif.gz' ## MET no pesudo-beta
	# protein_struct_file_assembly = '/root/paddlejob/workspace/output/yexianbin/data/cif_bioassembly/raw/all/8jix-assembly1.cif.gz'

	protein_struct_file_unit = 'data_assembly/mmcif/7etr.cif.gz' # B is glycan.
	protein_struct_file_assembly = 'data_assembly/mmcif_assembly1/1b3y-assembly1.cif.gz'  # B is glycan.

	# protein_struct_file_unit = 'data_preprocess/raw_data/debug/7z1k.cif' # non-standard residue.
	# protein_struct_file_unit = 'data_preprocess/raw_data/debug/7z1k-revised.cif' # non-standard residue. revised, add 2 standard residue.
	# # protein_struct_file_assembly = 'data_assembly/mmcif_assembly1/7z1k-assembly1.cif.gz'  # non-standard residue.

	# protein_struct_file_unit = 'data_preprocess/raw_data/debug/7km8.cif' # has protein + ligand + ion

	ccd_preprocessed_path = "data_assembly/ccd_preprocessed_etkdg.pkl.gz"
	st_1 = time.time()
	if 'pkl.gz' in ccd_preprocessed_path:
		with gzip.open(ccd_preprocessed_path, "rb") as fp:
			ccd_preprocessed_dict = pickle.load(fp)
	elif '.pkl' in ccd_preprocessed_path:
		with open(ccd_preprocessed_path, "rb") as fp:
			ccd_preprocessed_dict = pickle.load(fp)
	print(f'[CCD] load ccd dataset done. use {time.time()-st_1}s')

	cif_unit = _read_cif(protein_struct_file_unit)
	# cif_assem = _read_cif(protein_struct_file_assembly)
	parse_result_unit = mmcif_parsing.parse(file_id=protein_struct_file_unit, mmcif_string=cif_unit, 
															is_assembly_filter=False)
	mmcif_object_unit = parse_result_unit.mmcif_object
	# parse_result_assem = mmcif_parsing.parse(file_id=protein_struct_file_assembly, mmcif_string=cif_assem, 
	# 														is_assembly_filter=False)
	# mmcif_object_assem = parse_result_assem.mmcif_object


	mmcif_chain_id='C'
	# res = get_position_label(mmcif_object_unit=None, mmcif_object_assembly=mmcif_object_assem, ccd_preprocessed_dict=ccd_preprocessed_dict, 
	# 									mmcif_chain_id='A')
	res = get_position_label(mmcif_object_unit=mmcif_object_unit, mmcif_object_assembly=None, ccd_preprocessed_dict=ccd_preprocessed_dict, 
										mmcif_chain_id=mmcif_chain_id, check_release_date=False)
	
	res = res[mmcif_chain_id]
	all_atom_pos = res['all_atom_pos']
	all_atom_pos_mask = res['all_atom_pos_mask']
	frame_ai_indice = res['frame_ai_indice']
	frame_bi_indice = res['frame_bi_indice']
	frame_ci_indice = res['frame_ci_indice']
	frame_mask = res['frame_mask'].astype(bool)
	print(frame_ai_indice.shape, frame_bi_indice.shape, frame_ci_indice.shape)
	print(f'==> {all_atom_pos.shape}')
	print(np.max(frame_ai_indice), np.max(frame_bi_indice), np.max(frame_ci_indice))


	import pdb;pdb.set_trace()