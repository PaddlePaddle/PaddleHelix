# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for minimization."""
import io
import collections
import numbers
from alphafold_paddle.common import residue_constants
from Bio import PDB
import numpy as np
from simtk.openmm import app as openmm_app
from simtk.openmm.app.internal.pdbstructure import PdbStructure


def overwrite_pdb_coordinates(pdb_str: str, pos) -> str:
  pdb_file = io.StringIO(pdb_str)
  structure = PdbStructure(pdb_file)
  topology = openmm_app.PDBFile(structure).getTopology()
  with io.StringIO() as f:
    openmm_app.PDBFile.writeFile(topology, pos, f)
    return f.getvalue()


def overwrite_b_factors(pdb_str: str, bfactors: np.ndarray) -> str:
  """Overwrites the B-factors in pdb_str with contents of bfactors array.

  Args:
    pdb_str: An input PDB string.
    bfactors: A numpy array with shape [1, n_residues, 37]. We assume that the
      B-factors are per residue; i.e. that the nonzero entries are identical in
      [0, i, :].

  Returns:
    A new PDB string with the B-factors replaced.
  """
  if bfactors.shape[-1] != residue_constants.atom_type_num:
    raise ValueError(
        f'Invalid final dimension size for bfactors: {bfactors.shape[-1]}.')

  parser = PDB.PDBParser(QUIET=True)
  handle = io.StringIO(pdb_str)
  structure = parser.get_structure('', handle)

  curr_resid = ('', '', '')
  idx = -1
  for atom in structure.get_atoms():
    atom_resid = atom.parent.get_id()
    if atom_resid != curr_resid:
      idx += 1
      if idx >= bfactors.shape[0]:
        raise ValueError('Index into bfactors exceeds number of residues. '
                         'B-factors shape: {shape}, idx: {idx}.')
    curr_resid = atom_resid
    atom.bfactor = bfactors[idx, residue_constants.atom_order['CA']]

  new_pdb = io.StringIO()
  pdb_io = PDB.PDBIO()
  pdb_io.set_structure(structure)
  pdb_io.save(new_pdb)
  return new_pdb.getvalue()


def assert_equal_nonterminal_atom_types(
    atom_mask: np.ndarray, ref_atom_mask: np.ndarray):
  """Checks that pre- and post-minimized proteins have same atom set."""
  # Ignore any terminal OXT atoms which may have been added by minimization.
  oxt = residue_constants.atom_order['OXT']
  no_oxt_mask = np.ones(shape=atom_mask.shape, dtype=np.bool)
  no_oxt_mask[..., oxt] = False
  np.testing.assert_almost_equal(ref_atom_mask[no_oxt_mask],
                                 atom_mask[no_oxt_mask])


def mask_mean_np(mask, value, axis=None, drop_mask_channel=False, eps=1e-10):
  """Masked mean with numpy."""
  if drop_mask_channel:
    mask = mask[..., 0]

  mask_shape = mask.shape
  value_shape = value.shape

  assert len(mask_shape) == len(value_shape)

  if isinstance(axis, numbers.Integral):
    axis = [axis]
  elif axis is None:
    axis = list(range(len(mask_shape)))
  assert isinstance(axis, collections.Iterable), (
      'axis needs to be either an iterable, integer or "None"')

  broadcast_factor = 1.
  for axis_ in axis:
    value_size = value_shape[axis_]
    mask_size = mask_shape[axis_]
    if mask_size == 1:
      broadcast_factor *= value_size
    else:
      assert mask_size == value_size

  axis = tuple(axis)
  return (np.sum(mask * value, axis=axis) /
          (np.sum(mask, axis=axis) * broadcast_factor + eps))