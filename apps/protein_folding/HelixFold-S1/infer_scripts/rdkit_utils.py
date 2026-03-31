import os
import copy
import collections
import json
import logging
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import CombineMols

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


class SideChainConnectionError(RdkitConvertError):
	"""Exception raised for side chain connection error."""
	def __init__(self, message):
		self.message = message
		super().__init__(self.message)


class RdkitConstants:
	"""
		Constants variable for rdkit in HF3 project
	"""

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
	DNA_SMILES_BACKBONE = "O=P(O)(O)OCC1OC*CC1(O)"
	DNA_NAME_MAPPING = {
		0: "OP1",
		1: "P",
		2: "OP2",
		3: "OP3",
		4: "O5'",
		5: "C5'",
		6: "C4'",
		7: "O4'",
		8: "C1'",
		-1: "O3'",
		-2: "C3'",
		-3: "C2'",
	}
	RNA_SMILES_BACKBONE = "O=P(O)(O)OCC1OC*C(O)C1(O)"
	RNA_NAME_MAPPING = {
		0: "OP1",
		1: "P",
		2: "OP2",
		3: "OP3",
		4: "O5'",
		5: "C5'",
		6: "C4'",
		7: "O4'",
		8: "C1'",
		-1: "O3'",
		-2: "C3'",
		-3: "O2'",
		-4: "C2'",
	}
	PROT_SMILES_BACKBONE = "NC*C(=O)O"
	PROT_NAME_MAPPING = {
		0: "N",
		1: "CA",
		-1: "OXT",
		-2: "O",
		-3: "C",
	}
	MODIFIED_BACKBONE_NAME_MAPPING = {
		'dna': {
			'backbone': DNA_SMILES_BACKBONE, 
			'name': DNA_NAME_MAPPING,
		},
		'rna': {
			'backbone': RNA_SMILES_BACKBONE,
			'name': RNA_NAME_MAPPING,
		},
		'protein': {
			'backbone': PROT_SMILES_BACKBONE,
			'name': PROT_NAME_MAPPING,
		},
	}


def validate_R_smiles(smiles_sidechain, connect_idx):
	"""
		Function to check if a sidechain and connecting index is valid.
	"""
	mol_sidechain = Chem.MolFromSmiles(smiles_sidechain)
	if mol_sidechain is None:
		raise SMILESParseError(f"Invalid R_smiles: {smiles_sidechain};非法的R_smiles: {smiles_sidechain}")
	
	max_atoms = len(mol_sidechain.GetAtoms())
	if connect_idx is None or connect_idx >= max_atoms or connect_idx < 0:
		raise SMILESParseError(f"connect_idx {connect_idx} not in R_smiles"\
								f"(0-{max_atoms - 1});连接位点{connect_idx}不在R_smiles中({0}-{max_atoms-1})")


def generate_ETKDGv3_conformer(mol: Chem.Mol, max_attempts: int = 10) -> Chem.Mol:
	"""use ETKDGv3 for ccd conformer generation"""
	mol = Chem.Mol(mol)
	try:
		for _ in range(max_attempts):
			ps = AllChem.ETKDGv3()
			ps.useRandomCoords = True
			id = AllChem.EmbedMolecule(mol, ps)
			if id == -1:
				continue
			ETKDG_atom_pos = mol.GetConformers()[0].GetPositions().astype('float32')
			return mol
		raise RuntimeError(f'rdkit coords could not be generated after {max_attempts} attempts.')
	except Exception as e:
		logger.error(f'Failed to generate ETKDG_conformer: {e}')
	return None


def is_valid_connection(mol_sidechain, connecting_atom):
	"""
		Function to check if a connection atom is valid.
	"""
	explicit_valence = connecting_atom.GetExplicitValence()
	implicit_valence = connecting_atom.GetImplicitValence()
	formal_charge = connecting_atom.GetFormalCharge()
	explicit_hs = connecting_atom.GetNumExplicitHs()
	implicit_hs = connecting_atom.GetNumImplicitHs()

	if DEBUG:
		logger.debug(f"Connecting atom information:")
		logger.debug(f"Atom ID: {connecting_atom.GetIdx()}")
		logger.debug(f"Atom type: {connecting_atom.GetSymbol()}")
		logger.debug(f"Explicit valence: {explicit_valence}")
		logger.debug(f"Implicit valence: {implicit_valence}")
		logger.debug(f"Formal charge: {formal_charge}")
		logger.debug(f"Explicit Hs: {explicit_hs}")
		logger.debug(f"Implicit Hs: {implicit_hs}")

	if explicit_hs > 0:
		logger.warning(f"Converting {explicit_hs} explicit Hs to implicit.")
		connecting_atom.SetNumExplicitHs(0)
		connecting_atom.UpdatePropertyCache()

	implicit_hydrogens = max(0, explicit_hs + formal_charge)
	connect_count = max(implicit_valence, implicit_hydrogens)
	logger.info(f"implicit_hydrogens: {implicit_hydrogens} | implicit_valence: {implicit_valence}")
	if connect_count <= 0:
		return False

	return True


def set_atom_ids_to_mol(mol: Chem.Mol, extra_infos: dict = None, atom_nums_map: dict = None):
	"""
		setup atom_ids for mol, and return the mol with atom_ids set.
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


def sidechain_smiles_to_rdMol(smiles_backbone, smiles_sidechain, connect_idx, backbone_infos: dict = None):
	"""
		Args: 
			smiles_backbone: str, SMILES string of the backbone molecule
			smiles_sidechain: str, SMILES string of the sidechain molecule
			connect_idx: int, index of the atom in the sidechain that will be connected to the backbone
			backbone_infos: dict, mapping from atom idxs to atom names
		Returns:
			mol: Chem.Mol, the generated molecule with the specified atom name
		Example:
			smiles_backbone = "O=P(O)(O)OCC1OC*CC1(O)"
			smiles_sidechain = "[CH](=O)[OH]"
			connect_idx = 0
		Raises:
			SideChainConnectionError: If the input SMILES strings cannot be parsed or the specified connect index is invalid.
			ConformerGenerationError: If the connected molecule cannot be generated ETKDGv3 conformer.
	"""
	 # 1. Generate backbone molecules. Here, "*" is used only as a placeholder, so we load with sanitize=False
	mol_backbone = Chem.MolFromSmiles(smiles_backbone, sanitize=False)
	backbone_star_idx = [atom.GetIdx() for atom in mol_backbone.GetAtoms() 
							if atom.GetSymbol() == '*'][0]
	backbone_connecting_atom = mol_backbone.GetAtomWithIdx(backbone_star_idx).GetNeighbors()[0]
	backbone_idx = backbone_connecting_atom.GetIdx()
	
	# Remove "*" and reload molecules for final processing
	mol_backbone = Chem.MolFromSmiles(smiles_backbone.replace('*', ''))
	mol_sidechain = Chem.MolFromSmiles(smiles_sidechain)
	
	## setup the standard atom_ids to backbone
	mol_backbone, _atom_nums_map, idx_to_name1 = set_atom_ids_to_mol(mol=mol_backbone, extra_infos=backbone_infos)
	mol_sidechain, _, idx_to_name2 = set_atom_ids_to_mol(mol=mol_sidechain, atom_nums_map=_atom_nums_map)
	if len(set(idx_to_name1.values()) & set(idx_to_name2.values())) != 0: 
		logger.error(f"idx_to_name1: {idx_to_name1}, idx_to_name2: {idx_to_name2} are overlapped")
		raise SideChainConnectionError(f"atom_names are overlapped during sidechain_smiles_to_rdMol")

	# 2. Convert to editable molecules and check connectivity
	editable_backbone = Chem.RWMol(mol_backbone)
	editable_sidechain = Chem.RWMol(mol_sidechain)
	connecting_atom = editable_sidechain.GetAtomWithIdx(connect_idx)

	if not is_valid_connection(mol_sidechain, connecting_atom):
		raise SideChainConnectionError(f"Connecting atom in R_smiles "\
						f"{connecting_atom.GetSymbol()}({connect_idx}) is fully saturated;"\
							f"R_smiles连接的原子{connecting_atom.GetSymbol()}({connect_idx})已经饱和")

	# 3. Combine molecules and add a single bond
	combined_mol = CombineMols(editable_backbone, editable_sidechain)
	rw_mol = Chem.RWMol(combined_mol)
	rw_mol.AddBond(backbone_idx, connect_idx + editable_backbone.GetNumAtoms(), Chem.BondType.SINGLE)
	mol = rw_mol.GetMol()

	# 4. Validate the combined molecular structure
	try:
		Chem.SanitizeMol(mol)
		# Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
	except Exception as e:
		raise SideChainConnectionError(f"Invalid molecular structure after combining and Sanitization;连接后的分子结构不合法")

	# 5. Return the combined molecule with EDKDG conformer
	optimal_mol = Chem.AddHs(mol)
	optimal_mol = generate_ETKDGv3_conformer(optimal_mol)
	if optimal_mol is None:
		raise ConformerGenerationError(f"R_smiles Conformer Generation Error: {smiles_sidechain};R_smiles构象生成失败: {smiles_sidechain}")
	mol = Chem.RemoveAllHs(optimal_mol, sanitize=False)

	return mol


def make_basic_feature_fromMol(mol: Chem.Mol, reset_atom_ids: bool = True):
	"""
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


def smiles_to_rdMol(smiles):
	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		raise SMILESParseError(f"Invalid SMILES: {smiles};非法的SMILES: {smiles}")
	
	mol = Chem.AddHs(mol)
	optimal_mol = generate_ETKDGv3_conformer(mol)
	if optimal_mol is None:
		raise ConformerGenerationError(f"Conformer Generation Error: {smiles};构象生成失败: {smiles}")
	optimal_mol_wo_H = Chem.RemoveAllHs(optimal_mol, sanitize=False)
	return optimal_mol_wo_H


if __name__ == '__main__':

	input_json = json.load(open("demo_data/ep_sidechain_modified.json"))
	for e in input_json['entities']:
		chain_type = e['type']
		if chain_type in RdkitConstants.ALLOWED_MODIFIED_TYPE and 'modification' in e:
			smiles_backbone = RdkitConstants.MODIFIED_BACKBONE_NAME_MAPPING[chain_type]['backbone']
			backbone_name_mapping = RdkitConstants.MODIFIED_BACKBONE_NAME_MAPPING[chain_type]['name']
			for modi_bean in e['modification']:
				if 'R_smiles' in modi_bean and len(modi_bean['R_smiles']) > 0:
					R_smiles = modi_bean['R_smiles']
					R_connect_idx = modi_bean.get('R_connect_idx', None)
					validate_R_smiles(R_smiles, R_connect_idx)
					sidechain_smiles_to_rdMol(smiles_backbone=smiles_backbone, 
											smiles_sidechain=R_smiles, 
											connect_idx=R_connect_idx,
											backbone_infos=backbone_name_mapping)
	
	print('[DEMO] all modified pass!')
