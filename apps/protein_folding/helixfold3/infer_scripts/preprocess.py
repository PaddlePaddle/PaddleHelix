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
sys.path.append('../')
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from helixfold.common import residue_constants


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

OBABEL_BIN = os.getenv('OBABEL_BIN', '/root/paddlejob/workspace/output/yangpan/program_files/anaconda3/bin/conda')
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


def smiles_toMol_obabel(smiles):
    """
        generate mol from smiles using obabel;
    """    
    with tempfile.NamedTemporaryFile(suffix=".mol2") as temp_file:
        print(f"[OBABEL] Temporary file created: {temp_file.name}")
        obabel_cmd = f"{OBABEL_BIN} -:'{smiles}' -omol2 -O{temp_file.name} --gen3d"
        ret = subprocess.run(obabel_cmd, shell=True, capture_output=True, text=True)
        mol = Chem.MolFromMol2File(temp_file.name, sanitize=False)
        if '3D coordinate generation failed' in ret.stderr:
            mol = generate_ETKDGv3_conformer(mol)
        optimal_mol_wo_H = Chem.RemoveAllHs(mol, sanitize=False)
    return optimal_mol_wo_H


def polymer_convert(items):
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

    # repeat_ccds, repeat_fasta = [ccd_seqs], [msa_seqs]
    return {
        dtype: {
            'seqs': ccd_seqs,
            'msa_seqs': msa_seqs,
            'count': count,
            'extra_mol_infos': {}
        }
    }


def ligand_convert(items):
    """
        "type": "ligand",
        "ccd": "ATP", or "smiles": "CCccc(O)ccc",
        "count": 1
    """
    dtype = items['type']
    count = items['count']
    
    msa_seqs = ""
    _ccd_seqs = []
    ccd_to_extra_mol_infos = {}
    if 'ccd' in items:
        _ccd_seqs.append(f"({items['ccd']})")
    elif 'smiles' in items:
        _ccd_seqs.append(f"(UNK-)")
        # mol_wo_h = smiles_to_ETKDGMol(items['smiles'])
        mol_wo_h = smiles_toMol_obabel(items['smiles'])
        _extra_mol_infos = make_basic_info_fromMol(mol_wo_h)
        ccd_to_extra_mol_infos = {
            "UNK-": _extra_mol_infos
        }
    else:
        raise ValueError(f'not support for the {dtype} in ligand_convert')
    ccd_seqs = ''.join(_ccd_seqs) ## (GLY)(ALA).....

    # repeat_ccds, repeat_fasta = [ccd_seqs], [msa_seqs]
    return {
        'ligand': {
            'seqs': ccd_seqs,
            'msa_seqs': msa_seqs,
            'count': count,
            'extra_mol_infos': ccd_to_extra_mol_infos,
        }
    }


def entities_rename_and_filter(items):
    ligand_mapping = {
        'ion': 'ligand'
    }
    items['type'] = ligand_mapping.get(items['type'], items['type'])
    if items['type'] not in ALLOWED_ENTITY_TYPE:
        raise ValueError(f'{items["type"]} is not allowed, will be ignored.')
    return items


def modify_name_convert(entities: list):
    cur_idx = 0
    for entity_items in entities:
        # dtype(protein, dna, rna, ligand): no_chains,  msa_seqs, seqs
        dtype = list(entity_items.keys())[0]
        items = list(entity_items.values())[0]
        entity_count = items['count']
        msa_seqs = items['msa_seqs']
        extra_mol_infos = items.get('extra_mol_infos', {}) ## dict, 「extra-add, ccd_id」: ccd_features.

        extra_ccd_ids = list(extra_mol_infos.keys())
        ## rename UNK- to UNK-1, 2, 3, 4...
        for k in extra_ccd_ids:
            user_name_3 = USER_LIG_IDS_3[cur_idx]
            items['seqs'] = items['seqs'].replace('UNK-', user_name_3)
            extra_mol_infos[user_name_3] = extra_mol_infos.pop('UNK-')
            cur_idx += 1

    return entities


def online_json_to_entity(json_path, out_dir):
    obj = read_json(json_path)
    entities = copy.deepcopy(obj['entities'])

    os.makedirs(out_dir, exist_ok=True)
    error_ids = []
    success_entity = []
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
    
    success_entity = modify_name_convert(success_entity)
    return success_entity