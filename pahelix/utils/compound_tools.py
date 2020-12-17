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
| Tools for compound features.
| Adapted from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/loader.py
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


class CompoundConstants(object):
    """
    Constants of atom and bond properties.
    """
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
    atomic_numeric_feat_dim = len(numH_list) + len(implicit_valence_list) + \
        len(degree_list) + 1   # 1 is for aromatic


def atom_numeric_feat(n, allowable, to_one_hot=True):
    """
    Restrict the numeric feature to [0, `max_n`].
    """
    assert len(allowable) > 0
    n = min(n, len(allowable) - 1)
    n = allowable[n]

    if to_one_hot:
        feat = np.zeros(len(allowable))
        feat[n] = 1.0
        return feat
    else:
        return n


def mol_to_graph_data(mol, add_self_loop=True):
    """
    | Converts rdkit mol object to graph data which is a dict of numpy ndarray. 
    
    | NB: Uses simplified atom and bond features, and represent as indices.

    Args: 
        mol: rdkit mol object.
        add_self_loop: whether to add self loop or not.

    Returns:
        a dict of numpy ndarray for the graph data. It consists of atom attibutes, edge attibutes and edge index.
    """
    # atoms
    atom_types = []
    chirality_tags = []
    # Number of sigma electrons excluding electrons bonded to hydrogens
    atom_Hs = []
    # Number of directly-bonded neighbors, aka degree
    atom_degrees = []
    # TODO: what's the difference between implicit and explicit valence?
    atom_implicit_valence = []
    atom_is_aromatic = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            return None
        atom_types.append(CompoundConstants.atom_num_list.index(
                atom.GetAtomicNum()))
        chirality_tags.append(CompoundConstants.chiral_type_list.index(
                atom.GetChiralTag()))
        atom_Hs.append(atom_numeric_feat(
            atom.GetTotalNumHs(), CompoundConstants.numH_list))
        atom_degrees.append(atom_numeric_feat(
            atom.GetDegree(), CompoundConstants.degree_list))
        atom_implicit_valence.append(atom_numeric_feat(
            atom.GetIsAromatic(), CompoundConstants.implicit_valence_list))
        atom_is_aromatic.append(int(atom.GetIsAromatic()))

    atom_types = np.array(atom_types)
    chirality_tags = np.array(chirality_tags)
    atom_Hs = np.array(atom_Hs)
    atom_degrees = np.array(atom_degrees)
    atom_is_aromatic = np.array(atom_is_aromatic)
    atom_implicit_valence = np.array(atom_implicit_valence)

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
        'atom_Hs': atom_Hs,             # (N, 9)
        'atom_degrees': atom_degrees,  # (N, 11)
        'atom_implicit_valence': atom_implicit_valence,  # (N, 7)
        'atom_is_aromatic': atom_is_aromatic,  # (N,)
        'edges': edges,            # (E, 2)
        'bond_type': bond_types,    # (E,)
        'bond_direction': bond_directions,  # (E,)
    }
    return data


def smiles_to_graph_data(smiles, add_self_loop=True):
    """
    Convert smiles to graph data.
    """
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return None
    data = mol_to_graph_data(mol, add_self_loop)
    return data


def get_gasteiger_partial_charges(mol, n_iter=12):
    """
    Calculates list of gasteiger partial charges for each atom in mol object.
    
    Args: 
        mol: rdkit mol object
        n_iter(int): number of iterations. Default 12
    
    Returns: 
        list of computed partial charges for each atom.
    """
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol, nIter=n_iter,
                                                  throwOnParamFailure=True)
    partial_charges = [float(a.GetProp('_GasteigerCharge')) for a in
                       mol.GetAtoms()]
    return partial_charges


def create_standardized_mol_id(smiles):
    """

    Args:
        smiles: smiles sequence
    
    Returns: 
        inchi
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
    """
    Check whether the smile can't be converted to rdkit mol object.
    """
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except Exception as e:
        return False


def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively
    
    Args:
        mol: rdkit mol object.
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

    Args: 
        mol_list(list): a list of rdkit mol object.
    
    Returns:
        the largest mol.
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]
