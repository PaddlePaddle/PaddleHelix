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

"""Rdkit utils for data preprocess."""

import os
import copy
import collections
import logging
from typing import Dict, Tuple, List

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

DEBUG = os.getenv("DEBUG", "0") == "1"
logger = logging.getLogger(__file__)

class RdkitConvertError(Exception):
	"""Base class for exceptions in Rdkit conversion."""
	def __init__(self, message):
		self.message = message
	
	def __str__(self) -> str:
		return self.message


class SMILESParseError(RdkitConvertError):
	"""Exception raised for invalid smiles."""
	def __init__(self, message):
		self.message = message
		super().__init__(self.message)


class ConformerGenerationError(RdkitConvertError):
	"""Exception raised for ETKDGv3 Conformer error."""
	def __init__(self, message):
		self.message = message
		super().__init__(self.message)


class RdkitConstants:
	"""Constants variable for rdkit in HF3 project"""

	ALLOWED_LIGAND_BONDS_TYPE = {
		rdkit.Chem.rdchem.BondType.SINGLE: ("SING", 1), 
		rdkit.Chem.rdchem.BondType.DOUBLE: ("DOUB", 2), 
		rdkit.Chem.rdchem.BondType.TRIPLE: ("TRIP", 3),
		rdkit.Chem.rdchem.BondType.QUADRUPLE: ("QUAD", 4), 
		rdkit.Chem.rdchem.BondType.AROMATIC: ("AROM", 12),
	}

	ALLOWED_LIGAND_BONDS_TYPE_MAP = {
		k: v for k, v in ALLOWED_LIGAND_BONDS_TYPE.values()
	}

	INVERSE_ALLOWED_LIGAND_BONDS_TYPE_MAP = {
		v: k for k, v in  ALLOWED_LIGAND_BONDS_TYPE.values()
	}

	ALLOWED_MODIFIED_TYPE = ('protein', 'dna', 'rna')


def generate_ETKDGv3_conformer(mol: Chem.Mol) -> Chem.Mol:
	"""use ETKDGv3 for ccd conformer generation"""
	mol = copy.deepcopy(mol)
	try:
		ps = AllChem.ETKDGv3()
		id = AllChem.EmbedMolecule(mol, ps)
		if id == -1:
			raise RuntimeError('rdkit coords could not be generated')
		ETKDG_atom_pos = mol.GetConformers()[0].GetPositions().astype('float32')
		return mol
	except Exception as e:
		logger.error(f'Failed to generate ETKDG_conformer')
	return None


def set_atom_ids_to_mol(mol: Chem.Mol, extra_infos: Dict = None, atom_nums_map: Dict = None) -> Tuple[Chem.Mol, Dict, Dict]:
	"""setup atom_ids for mol, and return the mol with atom_ids set.

	Args:
		mol: Chem.Mol, the molecule to setup atom_ids
		extra_infos: dict, the extra infos for atom_ids, the key is the atom index, the value is the atom name
		atom_nums_map: dict, the atom nums map, the key is the atom symbol, the value is the atom nums

	Returns:
		mol: Chem.Mol, the molecule with atom_ids set
		_atom_nums_map: dict, the updated atom nums map
		idx_to_name: dict, the atom index to name map, the key is the atom index, the value is the atom name
	"""
	if extra_infos is not None:
		copy_extra_infos = copy.deepcopy(extra_infos)
		atom_nums = len(mol.GetAtoms())
		index_keys = list(copy_extra_infos.keys())
		for key in index_keys:
			if key < 0:
				copy_extra_infos[atom_nums + key] = copy_extra_infos.pop(key)

	if atom_nums_map is None:
		_atom_nums_map = collections.defaultdict(int)  # atom_symbol to appear count
	else:
		_atom_nums_map = copy.deepcopy(atom_nums_map)
	
	idx_to_name = {}
	for atom in mol.GetAtoms():
		idx = atom.GetIdx()
		symbol = atom.GetSymbol().upper()
		_atom_nums_map[symbol] += 1

		if extra_infos is not None and idx in copy_extra_infos:
			atom_name = copy_extra_infos[idx]
		else:
			atom_name = f"{symbol}{_atom_nums_map[symbol]}"
		
		atom.SetProp("_TriposAtomName", atom_name)
		idx_to_name[idx] = atom_name

	return mol, _atom_nums_map, idx_to_name


def make_basic_feature_fromMol(mol: Chem.Mol, reset_atom_ids: bool = True) -> Dict[str, List]:
	"""Make basic feature from Mol.
	
	Args:
		mol: Chem.Mol, the molecule to extract features from
		reset_atom_ids: bool, whether to reset the atom ids name to default format
	Returns:
		features: dict, containing the extracted features
	Raises:
		Assertion error: 
			If the input molecule has different atom size with the atom_ids, charges, positions.
	"""
	if reset_atom_ids: 
		mol, _, _ = set_atom_ids_to_mol(mol)

	atom_symbol = [atom.GetSymbol() for atom in mol.GetAtoms()]
	charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
	atom_ids = [atom.GetProp("_TriposAtomName") 
			if atom.HasProp("_TriposAtomName") else '' for atom in mol.GetAtoms()]
	position = mol.GetConformers()[0].GetPositions().astype('float32')
	bonds = []
	for bond in mol.GetBonds():
		_atom_id1 = bond.GetBeginAtomIdx() 
		_atom_id2 = bond.GetEndAtomIdx()
		formal_id1 = mol.GetAtomWithIdx(_atom_id1).GetProp("_TriposAtomName")
		formal_id2 = mol.GetAtomWithIdx(_atom_id2).GetProp("_TriposAtomName")
		## TODO: mmcif only support the following bond types: SING, DOUB, TRIP, QUAD, ARON.
		## Rdkit has some bond types that are not supported by mmcif, so we need to convert them to the supported ones.
		_bond_type, _ = RdkitConstants.ALLOWED_LIGAND_BONDS_TYPE.get(bond.GetBondType(), ("SING", 1))
		bonds.append((formal_id1, formal_id2, _bond_type))

	assert len(atom_symbol) == len(charges) == len(atom_ids) == len(position), \
					f'Got different atom basic info from Chem.Mol,' \
					f'{len(atom_symbol)}, {len(charges)}, {len(atom_ids)}, {len(position)}'
	return {
		"atom_symbol": atom_symbol,
		"charge": charges,
		"atom_ids": atom_ids,
		"coval_bonds": bonds,
		"position": position,
	}


def smiles_to_rdMol(smiles: str) -> Chem.Mol:
	"""convert smiles to rdkit mol.
	
	Args:
		smiles: str, the smiles string
	Returns:
		mol: Chem.Mol, the rdkit mol
	"""
	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		raise SMILESParseError(f"Invalid SMILES: {smiles};{smiles}")
	
	mol = Chem.AddHs(mol)
	optimal_mol = generate_ETKDGv3_conformer(mol)
	if optimal_mol is None:
		raise ConformerGenerationError(f"Conformer Generation Error: {smiles};{smiles}")
	optimal_mol_wo_H = Chem.RemoveAllHs(optimal_mol, sanitize=False)
	return optimal_mol_wo_H

