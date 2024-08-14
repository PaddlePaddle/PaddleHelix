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

"""Protein data type."""


from typing import Any, Mapping, Union
from helixfold.common import residue_constants
import numpy as np
import paddle
import itertools
import os

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

# Generate all possible single-character and two-character chain identifiers
all_chain_ids = [ch for ch in PDB_CHAIN_IDS] + [''.join(pair) for pair in itertools.product(PDB_CHAIN_IDS, repeat=2)]
PDB_MAX_CHAINS_TWO_CHAR = len(all_chain_ids)  # 62 + 62^2 = 62 + 3844 = 3906

required_keys_for_saving = [
  'all_ccd_ids', 'all_atom_ids', 
  'ref_token2atom_idx', 'restype', 
  'residue_index', 'asym_id',
  'all_atom_pos_mask',
]


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
  chain_end = 'TER'
  return (f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} '
          f'{chain_name:>1}{residue_index:>4}')



def prediction_to_pdb(pred_atom_pos: Union[np.ndarray, paddle.Tensor], 
                      FeatsDict: FeatureDict, pdb_file_path: str) -> str:
  """
   convert prediction position, to pdb file.
    - prediction_atom_pos: np.ndarray | paddle.Tensor
    - FeatsDict: Feats
    - pdb_file_path: path to save pdb_file
  """
  if isinstance(pred_atom_pos, paddle.Tensor):
    pred_atom_pos = pred_atom_pos.numpy()

  restypes_mapping = residue_constants.HF3_restype_order
  res_idxto3 = {v : k for k, v in restypes_mapping.items()}
  # atom_types = ccd_constants.CCD_ATOMID_MAP

  pdb_lines = []

  ccd_ids = FeatsDict["all_ccd_ids"] # N_atom
  atom_ids = FeatsDict["all_atom_ids"] # N_atom
  atom_positions = pred_atom_pos  # N_atom
  atom_mask = FeatsDict["all_atom_pos_mask"]  # N_atom

  ref_token2atom_idx = FeatsDict["ref_token2atom_idx"] # N_token
  aatype = FeatsDict["restype"] # N_token
  residue_index = FeatsDict["residue_index"].astype(np.int32) # N_token
  chain_index = FeatsDict["asym_id"].astype(np.int32) # N_token
  b_factors = FeatsDict['atom_plddts'] # N_atom

  aatype = aatype[ref_token2atom_idx]  # N_token -> N_atom
  residue_index = residue_index[ref_token2atom_idx] # N_token -> N_atom
  chain_index = chain_index[ref_token2atom_idx] # N_token -> N_atom
  
  if np.any(aatype >= residue_constants.HF3_restype_nums):
    raise ValueError('Invalid aatypes.')

  def get_chain_id(index):
      if index < PDB_MAX_CHAINS_TWO_CHAR:
          return all_chain_ids[index]
      else:
          raise ValueError("Index exceeds the maximum number of supported chains")

  # Construct a mapping from chain integer indices to chain ID strings.
  chain_ids = {}
  for i in np.unique(chain_index):  # np.unique gives sorted output.
      if i >= PDB_MAX_CHAINS_TWO_CHAR:
          raise ValueError(
              f'The PDB format supports at most {PDB_MAX_CHAINS_TWO_CHAR} chains.')
      chain_ids[i] = get_chain_id(i)

  pdb_lines.append('MODEL     1')
  atom_index = 1
  last_chain_index = chain_index[0]
  # Add all atom sites.
  for i in range(aatype.shape[0]):
    # Close the previous chain if in a multichain PDB.
    if last_chain_index != chain_index[i]:
      pdb_lines.append(_chain_end(
          atom_index, ccd_ids[i - 1], chain_ids[chain_index[i - 1]],
          residue_index[i - 1]))
      last_chain_index = chain_index[i]
      atom_index += 1  # Atom index increases at the TER symbol.

    res_name_3 = ccd_ids[i]
    atom_name, pos, mask, b_factor = atom_ids[i], atom_positions[i], \
                              atom_mask[i], b_factors[i]

    if mask < 0.5:
      continue

    record_type = 'ATOM' if res_name_3 in residue_constants.STANDARD_LIST else 'HETATM'
    name = atom_name if len(atom_name) == 4 else f' {atom_name}'
    alt_loc = ''
    insertion_code = ''
    occupancy = 1.00
    element = atom_name[0]
    if record_type == "HETATM" and res_name_3.lower() in \
                        map(str.lower, residue_constants.ATOM_ELEMENT.keys()):
      element = atom_name
    charge = ''
    # PDB is a columnar format, every space matters here!
    atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                f'{res_name_3:>3} {chain_ids[chain_index[i]]:>1}'
                f'{residue_index[i]:>4}{insertion_code:>1}   '
                f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                f'{element:>2}{charge:>2}')
    pdb_lines.append(atom_line)
    atom_index += 1

  # Close the final chain.
  pdb_lines.append(_chain_end(atom_index, ccd_ids[-1],
                              chain_ids[chain_index[-1]], residue_index[-1]))
  pdb_lines.append('ENDMDL')
  pdb_lines.append('END')

  # Pad all lines to 80 characters.
  pdb_lines = [line.ljust(80) for line in pdb_lines]
  _pdb_file = '\n'.join(pdb_lines) + '\n'  # Add terminating newline.

  with open(pdb_file_path, 'w') as f:
    f.write(_pdb_file)

  return pdb_file_path



def prediction_to_mmcif(pred_atom_pos: Union[np.ndarray, paddle.Tensor], 
                      FeatsDict: FeatureDict, 
                      maxit_binary: str,
                      mmcif_path: str) -> str:
  """
   convert prediction position, to pdb file.
    - prediction_atom_pos: np.ndarray | paddle.Tensor
    - FeatsDict: Feats
    - maxit_binary: path to maxit_binary, use to convert pdb to cif
    - mmcif_path: path to save *.cif
  """
  assert maxit_binary is not None and os.path.exists(maxit_binary), (
      f'maxit_binary: {maxit_binary} not exists. '
      f'link: https://sw-tools.rcsb.org/apps/MAXIT/source.html')
  assert mmcif_path.endswith('.cif'), f'mmcif_path should endswith .cif; got {mmcif_path}'

  pdb_path = mmcif_path.replace('.cif', '.pdb')
  pdb_path = prediction_to_pdb(pred_atom_pos, FeatsDict, pdb_path)
  msg = os.system(f'{maxit_binary} -i {pdb_path} -o 1 -output {mmcif_path}')
  if msg != 0:
    print(f'convert pdb to cif failed, error message: {msg}')
  return mmcif_path