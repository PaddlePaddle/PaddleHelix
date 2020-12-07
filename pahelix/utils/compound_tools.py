#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tools for compound features.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# allowable node and edge features
# allowable_features = {
#     'possible_atomic_num_list': list(range(1, 119)),
#     'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
#     'possible_chirality_list': [
#         Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
#         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
#         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
#         Chem.rdchem.ChiralType.CHI_OTHER
#     ],
#     'possible_hybridization_list': [
#         Chem.rdchem.HybridizationType.S,
#         Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
#         Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
#         Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
#     ],
#     'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
#     'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
#     'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'possible_bonds': [
#         Chem.rdchem.BondType.SINGLE,
#         Chem.rdchem.BondType.DOUBLE,
#         Chem.rdchem.BondType.TRIPLE,
#         Chem.rdchem.BondType.AROMATIC
#     ],
#     'possible_bond_dirs': [ # only for double bond stereo information
#         Chem.rdchem.BondDir.NONE,
#         Chem.rdchem.BondDir.ENDUPRIGHT,
#         Chem.rdchem.BondDir.ENDDOWNRIGHT
#     ]
# }
        

# COMPOUND_ATOMIC_NUMS = list(range(1, 119))
# COMPOUND_FORMAL_CHARGES = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
# COMPOUND_CHIRAL_TYPES = [
#         Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
#         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
#         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
#         Chem.rdchem.ChiralType.CHI_OTHER]
# COMPOUND_HYBRIDIZATION_TYPES = [
#         Chem.rdchem.HybridizationType.S,
#         Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
#         Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
#         Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
# ]
# COMPOUND_NUMHS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# COMPOUND_IMPLICIT_VALENCE = [0, 1, 2, 3, 4, 5, 6]
# COMPOUND_DEGREES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# COMPOUND_BOND_TYPES = [
#         Chem.rdchem.BondType.SINGLE,
#         Chem.rdchem.BondType.DOUBLE,
#         Chem.rdchem.BondType.TRIPLE,
#         Chem.rdchem.BondType.AROMATIC
# ]
# COMPOUND_BOND_DIRS = [ # only for double bond stereo information
#         Chem.rdchem.BondDir.NONE,
#         Chem.rdchem.BondDir.ENDUPRIGHT,
#         Chem.rdchem.BondDir.ENDDOWNRIGHT
# ]


class CompoundConstants(object):
    """docstring for CompoundConstants"""
    atom_num_list = list(range(1, 119))
    formal_charge_list = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    chiral_type_list = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER]
    hybridization_type_list = [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED]
    numH_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    implicit_valence_list = [0, 1, 2, 3, 4, 5, 6]
    degree_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    bond_type_list = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC]
    bond_dir_list = [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT]


def mol_to_graph_data(mol, add_self_loop=True):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    atom_types = []
    chirality_tags = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            return None
        atom_types.append(CompoundConstants.atom_num_list.index(
                atom.GetAtomicNum()))
        chirality_tags.append(CompoundConstants.chiral_type_list.index(
                atom.GetChiralTag()))
    atom_types = np.array(atom_types)
    chirality_tags = np.array(chirality_tags)

    # bonds
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges = []
        bond_types = []
        bond_directions = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = CompoundConstants.bond_type_list.index(
                    bond.GetBondType())
            bond_direction = CompoundConstants.bond_dir_list.index(
                    bond.GetBondDir())
            # i->j and j->i
            edges += [(i, j), (j, i)]
            bond_types += [bond_type, bond_type]
            bond_directions += [bond_direction, bond_direction]

        edges = np.array(edges)
        bond_types = np.array(bond_types)
        bond_directions = np.array(bond_directions)

    else:   # mol has no bonds
        edges = np.zeros((0, 2)).astype("int")
        bond_types = np.zeros((0,)).astype("int")
        bond_directions = np.zeros((0,)).astype("int")

    if add_self_loop:
        num_node = len(atom_types)
        if num_node == 0:
            return None

        self_edges = []
        for i in range(num_node):
            self_edges.append((i, i))
        self_edges = np.array(self_edges, dtype="int64")
        self_bond_types = np.full((num_node,), 
                len(CompoundConstants.bond_type_list), dtype='int64')
        self_bond_directions = np.full((num_node,), 
                Chem.rdchem.BondDir.NONE, dtype='int64')

        edges = np.concatenate([edges, self_edges], 0)
        bond_types = np.concatenate([bond_types, self_bond_types], 0)
        bond_directions = np.concatenate([bond_directions, self_bond_directions], 0)

    data = {
        'atom_type': atom_types,         # (N,)
        'chirality_tag': chirality_tags,    # (N,)
        'edges': edges,            # (E, 2)
        'bond_type': bond_types,    # (E,)
        'bond_direction': bond_directions,  # (E,)
    }
    return data


def smiles_to_graph_data(smiles, add_self_loop=True):
    """tbd"""
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return None
    data = mol_to_graph_data(mol, add_self_loop)
    return data


# def graph_data_obj_to_mol_simple(data_x, data_edge_index, data_edge_attr):
#     """
#     Convert pytorch geometric data obj to rdkit mol object. NB: Uses simplified
#     atom and bond features, and represent as indices.
#     :param: data_x:
#     :param: data_edge_index:
#     :param: data_edge_attr
#     :return:
#     """
#     mol = Chem.RWMol()

#     # atoms
#     atom_features = data_x.cpu().numpy()
#     num_atoms = atom_features.shape[0]
#     for i in range(num_atoms):
#         atomic_num_idx, chirality_tag_idx = atom_features[i]
#         atomic_num = allowable_features['possible_atomic_num_list'][atomic_num_idx]
#         chirality_tag = allowable_features['possible_chirality_list'][chirality_tag_idx]
#         atom = Chem.Atom(atomic_num)
#         atom.SetChiralTag(chirality_tag)
#         mol.AddAtom(atom)

#     # bonds
#     edge_index = data_edge_index.cpu().numpy()
#     edge_attr = data_edge_attr.cpu().numpy()
#     num_bonds = edge_index.shape[1]
#     for j in range(0, num_bonds, 2):
#         begin_idx = int(edge_index[0, j])
#         end_idx = int(edge_index[1, j])
#         bond_type_idx, bond_dir_idx = edge_attr[j]
#         bond_type = allowable_features['possible_bonds'][bond_type_idx]
#         bond_dir = allowable_features['possible_bond_dirs'][bond_dir_idx]
#         mol.AddBond(begin_idx, end_idx, bond_type)
#         # set bond direction
#         new_bond = mol.GetBondBetweenAtoms(begin_idx, end_idx)
#         new_bond.SetBondDir(bond_dir)

#     # Chem.SanitizeMol(mol) # fails for COC1=CC2=C(NC(=N2)[S@@](=O)CC2=NC=C(
#     # C)C(OC)=C2C)C=C1, when aromatic bond is possible
#     # when we do not have aromatic bonds
#     # Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)

#     return mol


def get_gasteiger_partial_charges(mol, n_iter=12):
    """
    Calculates list of gasteiger partial charges for each atom in mol object.
    :param mol: rdkit mol object
    :param n_iter: number of iterations. Default 12
    :return: list of computed partial charges for each atom.
    """
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol, nIter=n_iter,
                                                  throwOnParamFailure=True)
    partial_charges = [float(a.GetProp('_GasteigerCharge')) for a in
                       mol.GetAtoms()]
    return partial_charges


def create_standardized_mol_id(smiles):
    """

    :param smiles:
    :return: inchi
    """
    if check_smiles_validity(smiles):
        # remove stereochemistry
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles),
                                     isomericSmiles=False)
        mol = AllChem.MolFromSmiles(smiles)
        if not mol is None: # to catch weird issue with O=C1O[al]2oc(=O)c3ccc(cn3)c3ccccc3c3cccc(c3)c3ccccc3c3cc(C(F)(F)F)c(cc3o2)-c2ccccc2-c2cccc(c2)-c2ccccc2-c2cccnc21
            if '.' in smiles: # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol)
                largest_mol = get_largest_mol(mol_species_list)
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
        else:
            return
    else:
        return


def check_smiles_validity(smiles):
    """tbd"""
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except:
        return False


def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively
    :param mol:
    :return:
    """
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list


def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one
    :param mol_list:
    :return:
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]
