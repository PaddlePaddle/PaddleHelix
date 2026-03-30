"""Functions for building the features for the HelixFold-3 protmodel."""

import contextlib
from functools import partial
import tempfile
from typing import Tuple, Union, List, Optional
import numpy as np
import tempfile

from helixfold.common import residue_constants
from helixfold.data import parsers
from helixfold.data import templates_quat_affine as quat_affine
from helixfold.data import pipeline_conf_bonds, pipeline_token_feature

# Internal import (7716).

MAX_TEMPLATE_NUM = 4

af2_keep_keys_in_hf3 = set([
'msa', 'num_alignments', 'template_aatype', 'template_all_atom_masks', 'template_all_atom_positions', 
'deletion_matrix', 'deletion_mean','num_templates', 'cluster_bias_mask'
])

AF2_NEED_TO_PADDING_KEYS = set([ 
  'template_aatype', 'template_all_atom_masks', 'template_all_atom_positions', 
  'msa', 'deletion_matrix', 'deletion_mean', 'profile', 'has_deletion', 'deletion_value','template_domain_names'
  # 'template_pseudo_beta_mask',
  # 'template_backbone_frame_mask', 'template_distogram', 'template_unit_vector',
])


AF2_PADDING_FEATS = {
  'msa': residue_constants.AF3_restype_order['-'],
  'template_aatype': residue_constants.AF3_restype_order['-'],
}


AF2_PADDING_DIM = {
  # msa
  'msa': [1],
  'deletion_matrix': [1],
  'deletion_mean': [0],
  'profile': [0],
  'has_deletion': [1],
  'deletion_value': [1],

  # template
  'template_aatype': [1],
  'template_all_atom_masks': [1],
  'template_all_atom_positions': [1],
  'template_pseudo_beta_mask': [1],
  'template_backbone_frame_mask': [1],
  'template_distogram': [1, 2],
  'template_unit_vector': [1, 2],
  'template_domain_names': [1],
}


@contextlib.contextmanager
def temp_fasta_file(fasta_str: str):
  with tempfile.NamedTemporaryFile('w', suffix='.fasta') as fasta_file:
    fasta_file.write(fasta_str)
    fasta_file.seek(0)
    yield fasta_file.name


def repeat_elements_along_axis(array: np.ndarray, axis: int, repeat_indices: list, repeat_counts: list) -> np.ndarray:
    """
    Repeat specific elements along the specified axis of the array a specified number of times.

    Args:
        array (np.ndarray): The numpy array to repeat elements for.
        axis (int): The axis along which to repeat elements.
        repeat_indices (list): The list of indices of the elements to repeat along the specified axis.
        repeat_counts (list): The list of repetition counts corresponding to each index in repeat_indices.

    Returns:
        np.ndarray: The resulting numpy array with repeated elements along the specified axis.
    """
    # Validate inputs
    if len(repeat_indices) != len(repeat_counts):
        raise ValueError("repeat_indices and repeat_counts must have the same length")
    
    # Calculate the new shape after repeating elements
    new_shape = list(array.shape)
    for idx, count in zip(repeat_indices, repeat_counts):
        new_shape[axis] += (count - 1)
    
    # Initialize the result array
    result = np.empty(new_shape, dtype=array.dtype)
    
    # Prepare slicing templates
    index_expansion = (slice(None),) * axis
    result_index = [slice(None)] * array.ndim
    current_pos = 0

    for i in range(array.shape[axis]):
        if i in repeat_indices:
            repetitions = repeat_counts[repeat_indices.index(i)]
        else:
            repetitions = 1
        result_index[axis] = slice(current_pos, current_pos + repetitions)
        
        # Slice the array correctly along the axis
        slice_tuple = index_expansion + (i,)
        repeated_slice = np.repeat(np.expand_dims(array[slice_tuple], axis), repetitions, axis=axis)
        
        result[tuple(result_index)] = repeated_slice
        current_pos += repetitions

    return result


def copy_extra_feats_from_af2(raw_af2_feats, to_feats, prefix: str = ''):
  extra_keys_from_af2 = [
    'num_alignments', 'num_templates', 'cluster_bias_mask']

  if len(prefix) > 0:
    prefix = prefix + '_'

  for extra_key in extra_keys_from_af2:
    if extra_key in raw_af2_feats:
      to_feats[prefix + extra_key] = raw_af2_feats.pop(extra_key)

  return to_feats


def get_ccd_insert_index_and_nums(ccd_preprocessed_dict, ccd_list: np.ndarray, extra_mol_feats=None) -> Tuple[int, int]:
    """Get the insert index and value for a residue.
    Args:
      ccd_list: list of ccd sequence, ['GLY', 'ALA', 'THR']
      extra_mol_feats: list of USER-defined ccd mapping; such as:
      {
        "UNK-1": {position:}
        "UNK-2": {position:}
      }
    Returns:
      insert index and value.
    """
    index_nums = []
    for idx, ccd_id in enumerate(ccd_list):
        if ccd_id not in residue_constants.STANDARD_LIST:
            if ccd_id not in ccd_preprocessed_dict:
              assert not extra_mol_feats is None and ccd_id in extra_mol_feats
              _ccd_feats = extra_mol_feats[ccd_id]
            else:
              _ccd_feats = ccd_preprocessed_dict[ccd_id]
            num_atoms = len(_ccd_feats['position'])
            assert num_atoms > 0, f'TODO filter - Got CCD <{ccd_id}>: 0 atom nums.'
            index_nums.append((idx, num_atoms))
    return index_nums


def pad_to_length(array, axis, target_length, value=0):
  ## Padding Features in the specified dimension
    pad_width = [(0, 0)] * array.ndim  
    pad_amount = target_length - array.shape[axis]
    if pad_amount > 0:
        pad_width[axis] = (0, pad_amount)
    return np.pad(array, pad_width, mode='constant', constant_values=value)


def make_hhblits_profile(msa: np.ndarray):
    """Compute the HHblits MSA profile.
       msa shape: (N_msa, N_token)
       return: 
          hhblits_profile, shape (N_token, 22)
    """
    def _one_hot(depth, indices):
        """tbd."""
        res = np.eye(depth)[indices.reshape(-1)]
        return res.reshape(list(indices.shape) + [depth])

    # Compute the profile for every residue (over all MSA sequences).
    hhblits_profile = np.mean(_one_hot(len(residue_constants.AF3_restype_order), 
                                            msa), axis=0).astype('float32')
    return hhblits_profile


def make_pseudo_beta(protein, prefix=''):
  """Create pseudo-beta (alpha for glycine) position and mask."""
  def _pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
      """Create pseudo beta features."""
      is_gly = np.equal(aatype, residue_constants.restype_order['G'])
      ca_idx = residue_constants.atom_order['CA']
      cb_idx = residue_constants.atom_order['CB']
      pseudo_beta = np.where(
          np.tile(is_gly[..., None].astype("int32"),
                  [1,] * len(is_gly.shape) + [3,]).astype("bool"),
          all_atom_positions[..., ca_idx, :],
          all_atom_positions[..., cb_idx, :])

      if all_atom_masks is not None:
          pseudo_beta_mask = np.where(is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
          pseudo_beta_mask = pseudo_beta_mask.astype(np.float32)
          return pseudo_beta, pseudo_beta_mask

      return pseudo_beta

  assert prefix in ['', 'template_']
  pseudo_beta, pseudo_beta_mask = _pseudo_beta_fn(
      protein['template_aatype' if prefix else 'all_atom_aatype'],
      protein[prefix + 'all_atom_positions'],
      protein['template_all_atom_masks' if prefix else 'all_atom_mask'])

  protein[prefix + 'pseudo_beta'] = pseudo_beta
  protein[prefix + 'pseudo_beta_mask'] = pseudo_beta_mask
  return protein

def make_template_inter_mask(protein):
  # make template inter mask by template_domain_names
  from collections import Counter
  assert 'template_domain_names' in protein
  binary_array = np.zeros_like(protein['template_domain_names'], dtype=np.float32)
  mask_2d = np.zeros((binary_array.shape[0], protein['asym_id'].shape[0], protein['asym_id'].shape[0]), dtype=np.float32)
  # binary_array 根据前缀相同且后缀不同的生成链间 mask
  for index_template in range(binary_array.shape[0]):
    prefixes, suffixes = [], []
    for index in range(protein['template_domain_names'][index_template].shape[0]):
      value = protein['template_domain_names'][index_template][index]
      if value != 0:
        prefixes.append(value.decode().split('_')[0])
        suffixes.append(value.decode().split('_')[-1])
      else:
        prefixes.append(index)
        suffixes.append(index)
    prefix_counts = Counter(prefixes)
    # for most_common_prefix, count in prefix_counts.most_common():
    most_common_prefix, count = prefix_counts.most_common(1)[0]
    if count > 1:
      binary_array[index_template] = np.where(np.array(prefixes) == most_common_prefix, 1, 0)
    
    used_references = set()
    for index_chain in range(len(suffixes)):
      if suffixes[index_chain] not in used_references and prefixes[index_chain] == most_common_prefix:
        used_references.add(suffixes[index_chain])
      else:
        binary_array[index_template][index_chain] = 0

    outer_product = np.outer(binary_array[index_template], binary_array[index_template])

    for i in range(protein['asym_id'].shape[0]):
      for j in range(protein['asym_id'].shape[0]):
        if not protein['is_protein'][i] or not protein['is_protein'][j]:
          continue
        if protein['asym_id'][i] == protein['asym_id'][j] or outer_product[protein['asym_id'][i]-1, protein['asym_id'][j]-1] == 1:
          mask_2d[index_template, i, j] = 1

  protein['template_inter_mask'] = mask_2d
  protein.pop('template_domain_names') 
  return protein

def dgram_from_positions(positions, num_bins, min_bin, max_bin):
    lower_breaks = np.linspace(min_bin, max_bin, num_bins)
    lower_breaks = np.square(lower_breaks)
    upper_breaks = np.concatenate([lower_breaks[1:], np.array([1e8], dtype=np.float32)])

    def _squared_difference(x, y):
        return np.square(x - y)

    dist2 = np.sum(
        _squared_difference(
            np.expand_dims(positions, axis=-2),
            np.expand_dims(positions, axis=-3)),
        axis=-1, keepdims=True)

    dgram = ((dist2 > lower_breaks.astype(dist2.dtype)).astype(np.float32) *
             (dist2 < upper_breaks.astype(dist2.dtype)).astype(np.float32))
    
    return dgram


def make_template_further_feature(protein):

  def _np_unstack(a, axis=0):
      return np.moveaxis(a, axis, 0)

  dtype = np.float32

  n, ca, c = [residue_constants.atom_order[a]
              for a in ('N', 'CA', 'C')]
  rot, trans = quat_affine.make_transform_from_reference(
      n_xyz=protein['template_all_atom_positions'][..., n, :],
      ca_xyz=protein['template_all_atom_positions'][..., ca, :],
      c_xyz=protein['template_all_atom_positions'][..., c, :])
  affines = quat_affine.QuatAffine(
      quaternion=quat_affine.rot_to_quat(rot),
      translation=trans,
      rotation=rot)

  points = [np.expand_dims(x, axis=-2) for x in
            _np_unstack(affines.translation, axis=-1)]
  affine_vec = affines.invert_point(points, extra_dims=1)
  inv_distance_scalar = 1.0 / np.sqrt(
      1e-6 + sum([np.square(x) for x in affine_vec]))

  # NOTE: Backbone affine mask: whether the residue has C, CA, N
  # (the template mask defined above only considers pseudo CB).
  template_mask = (
      protein['template_all_atom_masks'][..., n] *
      protein['template_all_atom_masks'][..., ca] *
      protein['template_all_atom_masks'][..., c])
  protein['template_backbone_frame_mask'] = template_mask  # [N_template, N_token]

  template_mask_2d = template_mask[..., None] * template_mask[..., None, :]
  inv_distance_scalar *= template_mask_2d

  unit_vector = [(x * inv_distance_scalar)[..., None] for x in affine_vec]
  unit_vector = [x.astype(dtype) for x in unit_vector]
  unit_vector = np.concatenate(unit_vector, axis=-1)
  protein['template_unit_vector'] = unit_vector

  return protein


def feats_pad_and_concatenate(array_list: List[np.ndarray], padding_dim: Union[tuple, list], value=0):
    """
    Optimally pads the array to the target_shape by copying its values to a preallocated array of the target shape.
    
    Args:
        array_list list[np.ndarray]: The input array to be padded.
        padding_dim: list[int]: the dimensional 
        value (scalar): The value to use for padding.

    Returns:
        np.ndarray: The padded array with the specified target shape.
    """
    def _check_and_get_padding_shape(array_list):
        padding_shape = []
        num_dims_set = set()
        for array in array_list:
            num_dims_set.add(len(array.shape))
            if not padding_shape:
                padding_shape = list(array.shape)
            else:
                for i in range(len(array.shape)):
                  if i not in padding_dim:
                    padding_shape[i] = max(padding_shape[i], array.shape[i])
                  else:
                    padding_shape[i] = padding_shape[i] + array.shape[i]
        assert len(num_dims_set) == 1, f'Padding needs arrays of the same dimensional length, but got {num_dims_set}'
        assert all([pdim < len(padding_shape) for pdim in padding_dim]), 'Padding dim should no larger than dimensional length'
        return padding_shape

    target_shape = _check_and_get_padding_shape(array_list)
    padded_array = np.full(target_shape, value, dtype=array_list[0].dtype)

    padding_offset = [0] * len(target_shape)
    for array in array_list:
      _slices = []
      for idx, dim in enumerate(array.shape):
        _slices.append(slice(padding_offset[idx], dim + padding_offset[idx]))
        if idx in padding_dim:
          padding_offset[idx] += dim

      slices = tuple(_slices)
      padded_array[slices] = array

    return padded_array


def msa_features_transform(raw_feats: dict, ccd_preprocessed_dict: dict, 
                                                extra_feats: Optional[dict]=None) -> dict:
  """
    Function for AF2 msa features -> AF3 msa features, include DNA/RNA/protein/ligand.
  """
  ccd_list = raw_feats['ccd_seqs'] 
  assert len(ccd_list) > 0
  if 'msa' not in raw_feats:
    raw_feats['msa'] = np.ones([1, len(ccd_list)], dtype=np.int32) * residue_constants.AF3_restype_order['-']
    raw_feats['deletion_matrix'] = np.zeros([1, len(ccd_list)], dtype=np.float32)
    raw_feats['deletion_mean'] = np.zeros([len(ccd_list), ], dtype=np.float32)

  ###### 1. get basic features for raw msa from AF2-multimer output.
  raw_feats['profile'] = make_hhblits_profile(raw_feats['msa'])
  raw_feats['has_deletion'] = np.clip(raw_feats['deletion_matrix'], np.array(0), np.array(1))
  raw_feats['deletion_value'] = np.arctan(raw_feats['deletion_matrix'] / 3.) * (2. / np.pi)

  #### 2. insert token for non-standard list
  ## Raw msa is followed by AF2-multimer output, and `msa` length is the same as `ccd_seqs` length.
  assert len(ccd_list) == raw_feats['msa'].shape[-1] 
  index_nums = get_ccd_insert_index_and_nums(ccd_preprocessed_dict, ccd_list, extra_feats)

  if len(index_nums) > 0:
    repeat_indices, repeat_counts = map(list, zip(*index_nums))
    partial_repeat = partial(repeat_elements_along_axis, repeat_indices=repeat_indices, repeat_counts=repeat_counts)

    raw_feats['msa'] = partial_repeat(raw_feats['msa'], axis=1)
    raw_feats['deletion_matrix'] = partial_repeat(raw_feats['deletion_matrix'], axis=1)
    raw_feats['deletion_mean'] = partial_repeat(raw_feats['deletion_mean'], axis=0)
    raw_feats['has_deletion'] = partial_repeat(raw_feats['has_deletion'], axis=1)
    raw_feats['deletion_value'] = partial_repeat(raw_feats['deletion_value'], axis=1)
    raw_feats['profile'] = partial_repeat(raw_feats['profile'], axis=0)

  return raw_feats


def template_features_transform(raw_feats: dict, ccd_preprocessed_dict: dict, chain_type: str,
                                                extra_feats: Optional[list]=None) -> dict:
  """
    Function for AF2 Template features -> AF3 Template features, include DNA/RNA/protein/ligand.
  """
  ccd_list = raw_feats['ccd_seqs'] 
  assert len(ccd_list) > 0
  ## Raw template is followed by AF2-multimer output, and length is the same as `ccd_seqs` length.
  if 'template_aatype' not in raw_feats:
    num_residues = len(ccd_list)
    raw_feats["template_aatype"] = np.ones((MAX_TEMPLATE_NUM, num_residues), dtype=np.int32) * residue_constants.AF3_restype_order['-'] 
    raw_feats["template_all_atom_masks"] = np.zeros((MAX_TEMPLATE_NUM, num_residues, 37), dtype=np.float32)  
    raw_feats["template_all_atom_positions"] = np.zeros((MAX_TEMPLATE_NUM, num_residues, 37, 3), dtype=np.float32)
    
    if chain_type == 'protein':
      raw_feats["template_domain_names"] = np.zeros((MAX_TEMPLATE_NUM, raw_feats['assembly_num_chains']), dtype=np.float32)
    elif chain_type in ['dna', 'rna']:
      raw_feats["template_domain_names"] = np.zeros((MAX_TEMPLATE_NUM, 1), dtype=np.float32)
    else:  
      raw_feats["template_domain_names"] = np.zeros((MAX_TEMPLATE_NUM, num_residues), dtype=np.float32)
      #'ligand'

  #### 2. insert token for non-standard list
  assert len(ccd_list) == raw_feats['template_aatype'].shape[-1] 
  index_nums = get_ccd_insert_index_and_nums(ccd_preprocessed_dict, ccd_list, extra_feats)

  if len(index_nums) > 0:
    repeat_indices, repeat_counts = map(list, zip(*index_nums))
    partial_repeat = partial(repeat_elements_along_axis, repeat_indices=repeat_indices, repeat_counts=repeat_counts)

    raw_feats['template_aatype'] = partial_repeat(raw_feats['template_aatype'], axis=1)
    raw_feats['template_all_atom_masks'] = partial_repeat(raw_feats['template_all_atom_masks'], axis=1)
    raw_feats['template_all_atom_positions'] = partial_repeat(raw_feats['template_all_atom_positions'], axis=1)

  return raw_feats


def assembly_all_feats(raw_feats, to_feats, 
                        ordered_chain_types=('protein', 'dna', 'rna', 'ligand')):

  assembly_feats = {
    k: [] for k in AF2_NEED_TO_PADDING_KEYS
  }
  for chain_type in ordered_chain_types:
    if chain_type not in raw_feats:
      continue
    for pad_key in AF2_NEED_TO_PADDING_KEYS:
      assembly_feats[pad_key].append(raw_feats[chain_type][pad_key])

  ### for padding:
  for pad_key, feats_list in assembly_feats.items():
    padding_dim = AF2_PADDING_DIM[pad_key]
    padding_value = AF2_PADDING_FEATS.get(pad_key, 0)
    to_feats[pad_key] = feats_pad_and_concatenate(feats_list, padding_dim, value=padding_value)

  ## NOTE: MSA first row should be equal the restype, from `seq_token.`
  to_feats['msa'][0, :] = to_feats['seq_token']['restype'][:]

  return to_feats


def post_convert(ccd_preprocessed_dict, all_chain_feats_dict, 
                      ordered_chain_types=('protein', 'dna', 'rna', 'ligand')):
  """
    The post convert for all chain feats and logic is protein/rna/dna -> raw to get basic_feats -> insert token -> padding
    all_chain_feats_dict, should has keys: 
          seq_token、conf_bond、protein/dna/rna/ligand(MSA feats / ccd_seqs / extra_feats).
  """

  if not all_chain_feats_dict:
    return {}

  each_type_feats = {}
  for chain_type in ordered_chain_types:
    if chain_type in all_chain_feats_dict:
      feats = all_chain_feats_dict.pop(chain_type)
      token_length = all_chain_feats_dict['seq_token'][f'is_{chain_type}'].sum()

      extra_feats = None
      if 'extra_feats' in feats:
        extra_feats = feats.pop('extra_feats')
      
      ## NOTE: below is basic features convert and insert non-standard token.
      feats = msa_features_transform(feats, ccd_preprocessed_dict, extra_feats=extra_feats)
      feats = template_features_transform(feats, ccd_preprocessed_dict, chain_type, extra_feats=extra_feats)
      assert token_length == feats['msa'].shape[-1] == feats['template_aatype'].shape[-1]
      each_type_feats[chain_type] = feats

  all_chain_feats_dict = assembly_all_feats(each_type_feats, all_chain_feats_dict, 
                                            ordered_chain_types=ordered_chain_types)

  ## flatten：
  results = {}
  for k, v in all_chain_feats_dict.items():
    if isinstance(v, dict):
      results.update(v)
    else:
      results[k] = v

  return results
