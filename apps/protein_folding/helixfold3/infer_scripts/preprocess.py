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
    NOTE: now is only support standard dna/rna/protein seqs
    convert online server json to HF3 input/json;
    keys:
        'seqs': ccd_seqs,
        'msa_seqs': msa_seqs,
        'count': count,
        'extra_mol_infos': {}， for which seqs has the modify residue type or smiles.
"""
import collections
import copy
import os
import json
import sys
import subprocess
import tempfile
import itertools
from absl import logging
from typing import List, Optional, Tuple, Union, Mapping, Literal, Callable, Any
from dataclasses import dataclass, field
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from helixfold.common import residue_constants
from helixfold.data.pipeline_conf_bonds import CovalentBond, parse_covalent_bond_input
from helixfold.data.tools import utils

from openbabel import openbabel


## NOTE: this mapping is only useful for standard dna/rna/protein sequence input.
# protein, rna, dna, ligand, non_polymer (ion and non_polymer is also the ligand.)
ALLOWED_ENTITY_TYPE = list(residue_constants.CHAIN_type_order.keys()) 
PROTEIN_1to3_with_x = residue_constants.PROTEIN_1to3_with_x
DNA_1to2_with_x = residue_constants.DNA_RNA_1to2_with_x_and_gap['dna']
RNA_1to2_with_x = residue_constants.DNA_RNA_1to2_with_x_and_gap['rna']
POLYMER_STANDARD_RESI_ATOMS = residue_constants.residue_atoms

## FROM rdchem.BondType.values
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

USER_LIG_IDS = 'abcdefghijklmnopqrstuvwxyz0123456789'
USER_LIG_IDS_3 = [''.join(pair) for pair in itertools.product(USER_LIG_IDS, repeat=3)]

ERROR_CODES = {
    1: 'Invalid ligand generate.',
    2: 'Invalid entity convert.',
    3: 'Unknown error.'
}

OBABEL_BIN = os.getenv('OBABEL_BIN')
if not os.path.exists(OBABEL_BIN):
    raise FileNotFoundError(f'Cannot find obabel binary at {OBABEL_BIN}.')


def read_json(path):
    if path.endswith('.json.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    else:
        with open(path, 'r') as f:
            return json.load(f)


def alphabet2digit(alphabet):
    return sum((ord(a) - 65) * (26 ** e) for e, a in enumerate(reversed(alphabet)))


def digit2alphabet(digit):
    mod, remainder = divmod(digit, 26)
    alphabet = chr(65 + remainder)
    while mod:
        mod, remainder = divmod(mod, 26)
        alphabet = chr(65 + remainder) + alphabet
    return alphabet


def make_basic_info_fromMol(mol: Chem.Mol):
    ## make basic atom_name to Mol
    _atom_nums_map = collections.defaultdict(int)  # atom_symbol to appear count.
    idx_to_name = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        symbol = symbol.upper()
        _atom_nums_map[symbol] += 1
        atom_name = f"{symbol}{_atom_nums_map[symbol]}"
        atom.SetProp("_TriposAtomName", atom_name)
        idx_to_name[idx] = atom_name

    atom_symbol = [atom.GetSymbol() for atom in mol.GetAtoms()]
    charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    atom_ids = [atom.GetProp("_TriposAtomName") if atom.HasProp("_TriposAtomName") else '' for atom in mol.GetAtoms()]
    position = mol.GetConformers()[0].GetPositions().astype('float32')
    bonds = []
    for bond in mol.GetBonds():
        _atom_id1 = bond.GetBeginAtomIdx() 
        _atom_id2 = bond.GetEndAtomIdx()
        ## Rdkit has some bond types that are not supported by mmcif, so we need to convert them to the supported ones.
        _bond_type, _ = ALLOWED_LIGAND_BONDS_TYPE.get(bond.GetBondType(), ("SING", 1))
        bonds.append((idx_to_name[_atom_id1], idx_to_name[_atom_id2], _bond_type))

    assert len(atom_symbol) == len(charges) == len(atom_ids) == len(position), \
                    f'Got different atom basic info from Chem.Mol, {len(atom_symbol)}, {len(charges)}, {len(atom_ids)}, {len(position)}'
    return {
        "atom_symbol": atom_symbol,
        "charge": charges,
        "atom_ids": atom_ids,
        "coval_bonds": bonds,
        "position": position,
    }


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
        print(f'Failed to generate ETKDG_conformer: {e}')
    return None


def smiles_to_ETKDGMol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    mol = Chem.AddHs(mol)
    optimal_mol = generate_ETKDGv3_conformer(mol)
    optimal_mol_wo_H = Chem.RemoveAllHs(optimal_mol, sanitize=False)
    return optimal_mol_wo_H


class Mol2MolObabel:
    def __init__(self):
        self.obabel_bin = os.getenv('OBABEL_BIN')
        if not (self.obabel_bin and os.path.isfile(self.obabel_bin)):
            raise FileNotFoundError(f'Cannot find obabel binary at {self.obabel_bin}.')
        
        # Get the supported formats
        self.supported_formats: Tuple[str] = self._get_supported_formats()

    def _get_supported_formats(self) -> Tuple[str]:
        """
        Retrieves the list of supported formats from obabel and filters out write-only formats.
        
        Returns:
            tuple: A tuple of supported input formats.
        """
        obabel_cmd = f"{self.obabel_bin} -L formats"
        ret = subprocess.run(obabel_cmd, shell=True, capture_output=True, text=True)
        formats = [line.split()[0] for line in ret.stdout.splitlines() if '[Write-only]' not in line]
        formats.append('smiles')
        
        return tuple(formats)
    
    def _load_mol(self, mol2_file:str, ret:Optional[subprocess.CompletedProcess]=None) -> Chem.Mol:
        mol = Chem.MolFromMol2File(mol2_file, sanitize=False)
        if isinstance(ret, subprocess.CompletedProcess) and '3D coordinate generation failed' in ret.stderr:
            mol = generate_ETKDGv3_conformer(mol)
        optimal_mol_wo_H = Chem.RemoveAllHs(mol, sanitize=False)

        return optimal_mol_wo_H

    def _perform_conversion(self, input_type: str, input_value: str, generate_3d: bool=True) -> Chem.Mol:
        if input_type == 'mol2' and input_value.endswith('.mol2'):
            return self._load_mol(mol2_file=input_value)
        
        save_path=os.path.join('ligands',f'{os.path.basename(input_value)[:-(len(input_type)+1)] if input_type != "smiles" else "UNK"}.mol2')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with utils.timing(f'converting {input_type} to mol2: {input_value}'):
            if input_type == 'smiles':
                obabel_cmd = f"{self.obabel_bin} -:'{input_value}' -omol2 -O{save_path} {'--gen3d' if generate_3d else ''}"
                if len(input_value)>60:
                    logging.warning(f'This takes a while ...')
            else:
                obabel_cmd = f"{self.obabel_bin} -i {input_type} {input_value} -omol2 -O{save_path} {'--gen3d' if generate_3d else ''}"
            logging.debug(f'Launching command: `{obabel_cmd}`')
            ret = subprocess.run(obabel_cmd, shell=True, capture_output=True, text=True)
            return self._load_mol(mol2_file=save_path, ret=ret)
            
    def _convert_to_mol(self, input_type: str, input_value: str, generate_3d: bool=True) -> Chem.Mol:
        if input_type not in self.supported_formats:
            raise ValueError(f'Unsupported small molecule input: {input_type}. \nSupported formats: \n{self.supported_formats}\n')

        if input_type != 'smiles' and not os.path.isfile(input_value):
            raise FileNotFoundError(f'Cannot find the {input_type.upper()} file at {input_value}.')
        
        return self._perform_conversion(input_type, input_value, generate_3d)

    __call__: Callable[[str, str, bool], Chem.Mol] = _convert_to_mol

@dataclass
class Entity:
    dtype: Literal['protein', 'dna', 'rna', 'ligand', 'bond','non_polymer', 'ion']
    seqs: str
    msa_seqs: Union[str, List[CovalentBond]] = ''
    count: int = 1
    extra_mol_infos: dict[str, Any] = field(default_factory=dict)

def polymer_convert(items)-> Entity:
    """
        "type": "protein",                          
        "sequence": "GPDSMEEVVVPEEPPKLVSALATYVQQERLCTMFLSIANKLLPLKP",  
        "count": 1
    """
    dtype = items['type']
    one_letter_seqs = items['sequence']
    count = items['count']

    msa_seqs = one_letter_seqs
    ccd_seqs = []
    for resi_name_1 in one_letter_seqs:
        if dtype == 'protein':
            ccd_seqs.append(f"({PROTEIN_1to3_with_x[resi_name_1]})")
        elif dtype == 'dna':
            ccd_seqs.append(f"({DNA_1to2_with_x[resi_name_1]})")
        elif dtype == 'rna':
            ccd_seqs.append(f"({RNA_1to2_with_x[resi_name_1]})")
        else:
            raise ValueError(f'not support for the {dtype} in polymer_convert')
    ccd_seqs = ''.join(ccd_seqs) ## (GLY)(ALA).....

    return Entity(dtype=dtype, seqs=ccd_seqs, msa_seqs=msa_seqs,count=count)


def covalent_bond_convert(items: Mapping[str, Union[int, str]]) -> Entity:
    """
        "type": "bond",
        "bond": "A,ASN,74,ND2,B,UNK-,"
    """
    dtype = items['type']
    bond = parse_covalent_bond_input(items['bond'])

    return Entity(dtype=dtype, seqs='', msa_seqs=bond)


def ligand_convert(items: Mapping[str, Union[int, str]]) -> Entity:
    """
        "type": "ligand",
        "ccd": "ATP", or "smiles": "CCccc(O)ccc",
        "count": 1
    """
    dtype = items['type']
    count = items['count']
    converter=Mol2MolObabel()
    
    msa_seqs = ""
    _ccd_seqs = []
    ccd_to_extra_mol_infos = {}
    if 'ccd' in items:
        _ccd_seqs.append(f"({items['ccd']})")

    
    elif any(f in items for f in converter.supported_formats):
        for k in converter.supported_formats:
            if k in items:
                break

        ligand_name="UNK-"
        _ccd_seqs.append(f"({ligand_name})")
        # mol_wo_h = smiles_to_ETKDGMol(items['smiles'])
        
        mol_wo_h = converter(k, items[k], items.get('use_3d', True))
        _extra_mol_infos = make_basic_info_fromMol(mol_wo_h)
        ccd_to_extra_mol_infos = {
            ligand_name: _extra_mol_infos
        }
    else:
        raise ValueError(f'not support for the {dtype} in ligand_convert, please check the input. \nSupported input: {converter.supported_formats}')
    ccd_seqs = ''.join(_ccd_seqs) ## (GLY)(ALA).....

    # repeat_ccds, repeat_fasta = [ccd_seqs], [msa_seqs]
    return Entity(dtype='ligand', seqs=ccd_seqs,msa_seqs=msa_seqs,count=count,extra_mol_infos=ccd_to_extra_mol_infos)



def entities_rename_and_filter(items):
    ligand_mapping = {
        'ion': 'ligand'
    }
    items['type'] = ligand_mapping.get(items['type'], items['type'])
    if items['type'] not in ALLOWED_ENTITY_TYPE and items['type'] != 'bond':
        raise ValueError(f'{items["type"]} is not allowed, will be ignored.')
    return items


def modify_name_convert(entities: list[Entity]):
    cur_idx = 0
    for entity in entities:
        # dtype(protein, dna, rna, ligand): no_chains,  msa_seqs, seqs

        ## rename UNK- to UNK-1, 2, 3, 4...
        extra_mol_infos=copy.deepcopy(entity.extra_mol_infos)
        for k in extra_mol_infos:
            user_name_3 = USER_LIG_IDS_3[cur_idx]
            entity.seqs = entity.seqs.replace('UNK-', user_name_3)
            entity.extra_mol_infos[user_name_3] = entity.extra_mol_infos.pop('UNK-')
            cur_idx += 1

    return entities


def online_json_to_entity(json_path: str, out_dir: str)-> list[Entity]:
    obj = read_json(json_path)
    entities = copy.deepcopy(obj['entities'])

    os.makedirs(out_dir, exist_ok=True)
    error_ids = []
    success_entity: list[Entity] = []
    for idx, items in enumerate(entities):
        try: 
            items = entities_rename_and_filter(items)
        except Exception as e:
            print(f'Failed to convert entity {idx}: {items}, {e}')
            error_ids.append((idx, ERROR_CODES[2]))
            continue
        
        try:
            if items['type'] == 'ligand':
                json_obj = ligand_convert(items)
            elif items['type'] == 'bond':
                json_obj = covalent_bond_convert(items)
            else:
                json_obj = polymer_convert(items)
            success_entity.append(json_obj)
        except Exception as e:
            if items['type'] == 'ligand':
                print(f'Failed to convert ligand entity {idx}: {items}, {e}')
                error_ids.append((idx, ERROR_CODES[1]))
            else:
                print(f'Failed to convert polymer entity {idx}: {items}, {e}')
                error_ids.append((idx, ERROR_CODES[3]))

    if len(error_ids) > 0:
        raise RuntimeError(f'[Error] Failed to convert {len(error_ids)}/{len(entities)} entities')    
    
    return modify_name_convert(success_entity)