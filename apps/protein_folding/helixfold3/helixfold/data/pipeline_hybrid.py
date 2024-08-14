"""Functions for building the features for the HelixFold-3 protmodel."""

from functools import partial
from typing import  Tuple, Union, List, Optional
import numpy as np
from helixfold.common import residue_constants
from helixfold.data import templates_quat_affine as quat_affine


MAX_TEMPLATE_NUM = 4

HF2_NEED_TO_PADDING_KEYS = set([ 
  'template_aatype', 'template_all_atom_masks', 'template_all_atom_positions', 
  'msa', 'deletion_matrix', 'deletion_mean', 'profile', 'has_deletion', 'deletion_value',
])


HF2_PADDING_FEATS = {
  'msa': residue_constants.HF3_restype_order['-'],
  'template_aatype': residue_constants.HF3_restype_order['-'],
}


HF2_PADDING_DIM = {
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
}



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
    hhblits_profile = np.mean(_one_hot(len(residue_constants.HF3_restype_order), 
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
    Function for HF2 msa features -> HF3 msa features, include DNA/RNA/protein/ligand.
  """
  ccd_list = raw_feats['ccd_seqs'] 
  assert len(ccd_list) > 0
  if 'msa' not in raw_feats:
    raw_feats['msa'] = np.ones([1, len(ccd_list)], dtype=np.int32) * residue_constants.HF3_restype_order['-']
    raw_feats['deletion_matrix'] = np.zeros([1, len(ccd_list)], dtype=np.float32)
    raw_feats['deletion_mean'] = np.zeros([len(ccd_list), ], dtype=np.float32)

  ###### 1. get basic features for raw msa from HF2-multimer output.
  raw_feats['profile'] = make_hhblits_profile(raw_feats['msa'])
  raw_feats['has_deletion'] = np.clip(raw_feats['deletion_matrix'], np.array(0), np.array(1))
  raw_feats['deletion_value'] = np.arctan(raw_feats['deletion_matrix'] / 3.) * (2. / np.pi)

  #### 2. insert token for non-standard list
  ## Raw msa is followed by HF2-multimer output, and `msa` length is the same as `ccd_seqs` length.
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


def template_features_transform(raw_feats: dict, ccd_preprocessed_dict: dict,
                                                extra_feats: Optional[list]=None) -> dict:
  """
    Function for HF2 Template features -> HF3 Template features, include DNA/RNA/protein/ligand.
  """
  ccd_list = raw_feats['ccd_seqs'] 
  assert len(ccd_list) > 0
  ## Raw template is followed by HF2-multimer output, and length is the same as `ccd_seqs` length.
  if 'template_aatype' not in raw_feats:
    num_residues = len(ccd_list)
    raw_feats["template_aatype"] = np.ones((MAX_TEMPLATE_NUM, num_residues), dtype=np.int32) * residue_constants.HF3_restype_order['-'] 
    raw_feats["template_all_atom_masks"] = np.zeros((MAX_TEMPLATE_NUM, num_residues, 37), dtype=np.float32)  
    raw_feats["template_all_atom_positions"] = np.zeros((MAX_TEMPLATE_NUM, num_residues, 37, 3), dtype=np.float32)

  # # get `template_pseudo_beta_mask` and template_pseudo_beta
  # raw_feats = make_pseudo_beta(raw_feats, prefix='template_')
  # raw_feats = make_template_further_feature(raw_feats)

  #### 2. insert token for non-standard list
  assert len(ccd_list) == raw_feats['template_aatype'].shape[-1] 
  index_nums = get_ccd_insert_index_and_nums(ccd_preprocessed_dict, ccd_list, extra_feats)

  if len(index_nums) > 0:
    repeat_indices, repeat_counts = map(list, zip(*index_nums))
    partial_repeat = partial(repeat_elements_along_axis, repeat_indices=repeat_indices, repeat_counts=repeat_counts)

    raw_feats['template_aatype'] = partial_repeat(raw_feats['template_aatype'], axis=1)
    raw_feats['template_all_atom_masks'] = partial_repeat(raw_feats['template_all_atom_masks'], axis=1)
    raw_feats['template_all_atom_positions'] = partial_repeat(raw_feats['template_all_atom_positions'], axis=1)
    # raw_feats['template_pseudo_beta_mask'] = partial_repeat(raw_feats['template_pseudo_beta_mask'], axis=1)
    # raw_feats['template_backbone_frame_mask'] = partial_repeat(raw_feats['template_backbone_frame_mask'], axis=1)
    # raw_feats['template_distogram'] = partial_repeat(raw_feats['template_distogram'], axis=1)
    # raw_feats['template_distogram'] = partial_repeat(raw_feats['template_distogram'], axis=2)
    # raw_feats['template_unit_vector'] = partial_repeat(raw_feats['template_unit_vector'], axis=1)
    # raw_feats['template_unit_vector'] = partial_repeat(raw_feats['template_unit_vector'], axis=2)

  return raw_feats


def assembly_all_feats(raw_feats, to_feats, token_nums):

  assembly_feats = {
    k: [] for k in HF2_NEED_TO_PADDING_KEYS
  }
  for chain_type in ['protein', 'dna', 'rna', 'ligand']:
    if chain_type not in raw_feats:
      continue
    for pad_key in HF2_NEED_TO_PADDING_KEYS:
      assembly_feats[pad_key].append(raw_feats[chain_type][pad_key])

  ### for padding:
  for pad_key, feats_list in assembly_feats.items():
    padding_dim = HF2_PADDING_DIM[pad_key]
    padding_value = HF2_PADDING_FEATS.get(pad_key, 0)
    to_feats[pad_key] = feats_pad_and_concatenate(feats_list, padding_dim, value=padding_value)

  ## NOTE: MSA first row should be equal the restype, from `seq_token.`
  to_feats['msa'][0, :] = to_feats['seq_token']['restype'][:]

  return to_feats


def _post_convert(ccd_preprocessed_dict, all_chain_feats_dict):
  """
    all_chain_feats_dict, should has keys: 
          seq_token、conf_bond、protein/dna/rna/ligand.
  """
  # ## logic is protein/rna/dna -> raw to get basic_feats -> insert token -> padding(ligand or others, that is concat ??).

  if not all_chain_feats_dict:
    return {}

  each_type_feats = {}
  for chain_type in ['protein', 'dna', 'rna', 'ligand']:
    if chain_type in all_chain_feats_dict:
      feats = all_chain_feats_dict.pop(chain_type)
      # copy_extra_feats_from_hf2(feats, all_chain_feats_dict, prefix=chain_type)
      token_length = all_chain_feats_dict['seq_token'][f'is_{chain_type}'].sum()

      extra_feats = None
      if 'extra_feats' in feats:
        extra_feats = feats.pop('extra_feats')
      
      ## NOTE: below is basic features convert and insert non-standard token.
      feats = msa_features_transform(feats, ccd_preprocessed_dict, extra_feats=extra_feats)
      feats = template_features_transform(feats, ccd_preprocessed_dict, extra_feats=extra_feats)
      assert token_length == feats['msa'].shape[-1] == feats['template_aatype'].shape[-1]
      each_type_feats[chain_type] = feats

  token_nums = len(all_chain_feats_dict['seq_token']['token_index'])
  all_chain_feats_dict = assembly_all_feats(each_type_feats, all_chain_feats_dict, token_nums)

  ## flatten：
  results = {}
  for k, v in all_chain_feats_dict.items():
    if isinstance(v, dict):
      results.update(v)
    else:
      results[k] = v

  return results

