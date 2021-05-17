#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem

def get_gasteiger_partial_charges(mol, n_iter=12):
    """
    Calculates list of gasteiger partial charges for each atom in mol object.

    Args: 
        mol: rdkit mol object.
        n_iter(int): number of iterations. Default 12.

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
        smiles: smiles sequence.

    Returns: 
        inchi.
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
    list of mol objects or a list containing a single object respectively.

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
    picks the first one.

    Args: 
        mol_list(list): a list of rdkit mol object.

    Returns:
        the largest mol.
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]

def rdchem_enum_to_list(values):
    """values = {0: rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED, 
            1: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, 
            2: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, 
            3: rdkit.Chem.rdchem.ChiralType.CHI_OTHER}
    """
    return [values[i] for i in range(len(values))]


class CompoundKit(object):
    """
    CompoundKit
    """
    atom_vocab_dict = {
        "atomic_num": list(range(1, 119)),
        "chiral_tag": rdchem_enum_to_list(rdchem.ChiralType.values),
        "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "explicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "hybridization": rdchem_enum_to_list(rdchem.HybridizationType.values),
        "implicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "is_aromatic": [0, 1],
        "total_numHs": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    }
    bond_vocab_dict = {
        "bond_dir": rdchem_enum_to_list(rdchem.BondDir.values),
        "bond_type": rdchem_enum_to_list(rdchem.BondType.values),
        "is_in_ring": [0, 1],
    }

    ### functional groups
    day_light_fg_smarts_list = [
        # C
        "[CX4]",
        "[$([CX2](=C)=C)]",
        "[$([CX3]=[CX3])]",
        "[$([CX2]#C)]",
        # C & O
        "[CX3]=[OX1]",
        "[$([CX3]=[OX1]),$([CX3+]-[OX1-])]",
        "[CX3](=[OX1])C",
        "[OX1]=CN",
        "[CX3](=[OX1])O",
        "[CX3](=[OX1])[F,Cl,Br,I]",
        "[CX3H1](=O)[#6]",
        "[CX3](=[OX1])[OX2][CX3](=[OX1])",
        "[NX3][CX3](=[OX1])[#6]",
        "[NX3][CX3]=[NX3+]",
        "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]",
        "[NX3][CX3](=[OX1])[OX2H0]",
        "[NX3,NX4+][CX3](=[OX1])[OX2H,OX1-]",
        "[CX3](=O)[O-]",
        "[CX3](=[OX1])(O)O",
        "[CX3](=[OX1])([OX2])[OX2H,OX1H0-1]",
        "C[OX2][CX3](=[OX1])[OX2]C",
        "[CX3](=O)[OX2H1]",
        "[CX3](=O)[OX1H0-,OX2H1]",
        "[NX3][CX2]#[NX1]",
        "[#6][CX3](=O)[OX2H0][#6]",
        "[#6][CX3](=O)[#6]",
        "[OD2]([#6])[#6]",
        # H
        "[H]",
        "[!#1]",
        "[H+]",
        "[+H]",
        "[!H]",
        # N
        "[NX3;H2,H1;!$(NC=O)]",
        "[NX3][CX3]=[CX3]",
        "[NX3;H2;!$(NC=[!#6]);!$(NC#[!#6])][#6]",
        "[NX3;H2,H1;!$(NC=O)].[NX3;H2,H1;!$(NC=O)]",
        "[NX3][$(C=C),$(cc)]",
        "[NX3,NX4+][CX4H]([*])[CX3](=[OX1])[O,N]",
        "[NX3H2,NH3X4+][CX4H]([*])[CX3](=[OX1])[NX3,NX4+][CX4H]([*])[CX3](=[OX1])[OX2H,OX1-]",
        "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([*])[CX3](=[OX1])[OX2H,OX1-,N]",
        "[CH3X4]",
        "[CH2X4][CH2X4][CH2X4][NHX3][CH0X3](=[NH2X3+,NHX2+0])[NH2X3]",
        "[CH2X4][CX3](=[OX1])[NX3H2]",
        "[CH2X4][CX3](=[OX1])[OH0-,OH]",
        "[CH2X4][SX2H,SX1H0-]",
        "[CH2X4][CH2X4][CX3](=[OX1])[OH0-,OH]",
        "[$([$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H2][CX3](=[OX1])[OX2H,OX1-,N])]",
        "[CH2X4][#6X3]1:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]:\
[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]1",
        "[CHX4]([CH3X4])[CH2X4][CH3X4]",
        "[CH2X4][CHX4]([CH3X4])[CH3X4]",
        "[CH2X4][CH2X4][CH2X4][CH2X4][NX4+,NX3+0]",
        "[CH2X4][CH2X4][SX2][CH3X4]",
        "[CH2X4][cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1",
        "[$([NX3H,NX4H2+]),$([NX3](C)(C)(C))]1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[OX2H,OX1-,N]",
        "[CH2X4][OX2H]",
        "[NX3][CX3]=[SX1]",
        "[CHX4]([CH3X4])[OX2H]",
        "[CH2X4][cX3]1[cX3H][nX3H][cX3]2[cX3H][cX3H][cX3H][cX3H][cX3]12",
        "[CH2X4][cX3]1[cX3H][cX3H][cX3]([OHX2,OH0X1-])[cX3H][cX3H]1",
        "[CHX4]([CH3X4])[CH3X4]",
        "N[CX4H2][CX3](=[OX1])[O,N]",
        "N1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[O,N]",
        "[$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]",
        "[$([NX1-]=[NX2+]=[NX1-]),$([NX1]#[NX2+]-[NX1-2])]",
        "[#7]",
        "[NX2]=N",
        "[NX2]=[NX2]",
        "[$([NX2]=[NX3+]([O-])[#6]),$([NX2]=[NX3+0](=[O])[#6])]",
        "[$([#6]=[N+]=[N-]),$([#6-]-[N+]#[N])]",
        "[$([nr5]:[nr5,or5,sr5]),$([nr5]:[cr5]:[nr5,or5,sr5])]",
        "[NX3][NX3]",
        "[NX3][NX2]=[*]",
        "[CX3;$([C]([#6])[#6]),$([CH][#6])]=[NX2][#6]",
        "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]",
        "[NX3+]=[CX3]",
        "[CX3](=[OX1])[NX3H][CX3](=[OX1])",
        "[CX3](=[OX1])[NX3H0]([#6])[CX3](=[OX1])",
        "[CX3](=[OX1])[NX3H0]([NX3H0]([CX3](=[OX1]))[CX3](=[OX1]))[CX3](=[OX1])",
        "[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]",
        "[$([OX1]=[NX3](=[OX1])[OX1-]),$([OX1]=[NX3+]([OX1-])[OX1-])]",
        "[NX1]#[CX2]",
        "[CX1-]#[NX2+]",
        "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
        "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8].[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
        "[NX2]=[OX1]",
        "[$([#7+][OX1-]),$([#7v5]=[OX1]);!$([#7](~[O])~[O]);!$([#7]=[#7])]",
        # O
        "[OX2H]",
        "[#6][OX2H]",
        "[OX2H][CX3]=[OX1]",
        "[OX2H]P",
        "[OX2H][#6X3]=[#6]",
        "[OX2H][cX3]:[c]",
        "[OX2H][$(C=C),$(cc)]",
        "[$([OH]-*=[!#6])]",
        "[OX2,OX1-][OX2,OX1-]",
        # P
        "[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),\
$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)]),$([P+]([OX1-])([$([OX2H]),$([OX1-])\
,$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]",
        "[$(P(=[OX1])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),\
$([OX1-]),$([OX2][#6]),$([OX2]P)]),$([P+]([OX1-])([OX2][#6])([$([OX2H]),$([OX1-]),\
$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)])]",
        # S
        "[S-][CX3](=S)[#6]",
        "[#6X3](=[SX1])([!N])[!N]",
        "[SX2]",
        "[#16X2H]",
        "[#16!H0]",
        "[#16X2H0]",
        "[#16X2H0][!#16]",
        "[#16X2H0][#16X2H0]",
        "[#16X2H0][!#16].[#16X2H0][!#16]",
        "[$([#16X3](=[OX1])[OX2H0]),$([#16X3+]([OX1-])[OX2H0])]",
        "[$([#16X3](=[OX1])[OX2H,OX1H0-]),$([#16X3+]([OX1-])[OX2H,OX1H0-])]",
        "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",
        "[$([#16X4](=[OX1])(=[OX1])([#6])[#6]),$([#16X4+2]([OX1-])([OX1-])([#6])[#6])]",
        "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H,OX1H0-])]",
        "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H0]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H0])]",
        "[$([#16X4]([NX3])(=[OX1])(=[OX1])[#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[#6])]",
        "[SX4](C)(C)(=O)=N",
        "[$([SX4](=[OX1])(=[OX1])([!O])[NX3]),$([SX4+2]([OX1-])([OX1-])([!O])[NX3])]",
        "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]",
        "[$([#16X3](=[OX1])([#6])[#6]),$([#16X3+]([OX1-])([#6])[#6])]",
        "[$([#16X4](=[OX1])(=[OX1])([OX2H,OX1H0-])[OX2][#6]),$([#16X4+2]([OX1-])([OX1-])([OX2H,OX1H0-])[OX2][#6])]",
        "[$([SX4](=O)(=O)(O)O),$([SX4+2]([O-])([O-])(O)O)]",
        "[$([#16X4](=[OX1])(=[OX1])([OX2][#6])[OX2][#6]),$([#16X4](=[OX1])(=[OX1])([OX2][#6])[OX2][#6])]",
        "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2][#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2][#6])]",
        "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2H,OX1H0-]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2H,OX1H0-])]",
        "[#16X2][OX2H,OX1H0-]",
        "[#16X2][OX2H0]",
        # X
        "[#6][F,Cl,Br,I]",
        "[F,Cl,Br,I]",
        "[F,Cl,Br,I].[F,Cl,Br,I].[F,Cl,Br,I]",
    ]
    day_light_fg_mo_list = [Chem.MolFromSmarts(smarts) for smarts in day_light_fg_smarts_list]

    morgan_fp_N = 200
    morgan2048_fp_N = 2048
    maccs_fp_N = 167

    ### atom

    @staticmethod
    def get_atom_value(atom, name):
        """get atom values"""
        if name == 'atomic_num':
            return atom.GetAtomicNum()
        elif name == 'chiral_tag':
            return atom.GetChiralTag()
        elif name == 'degree':
            return atom.GetDegree()
        elif name == 'explicit_valence':
            return atom.GetExplicitValence()
        elif name == 'formal_charge':
            return atom.GetFormalCharge()
        elif name == 'hybridization':
            return atom.GetHybridization()
        elif name == 'implicit_valence':
            return atom.GetImplicitValence()
        elif name == 'is_aromatic':
            return int(atom.GetIsAromatic())
        elif name == 'mass':
            return int(atom.GetMass())
        elif name == 'total_numHs':
            return atom.GetTotalNumHs()
        else:
            raise ValueError(name)

    @staticmethod
    def get_atom_feature_id(atom, name):
        """get atom features id"""
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return CompoundKit.atom_vocab_dict[name].index(CompoundKit.get_atom_value(atom, name))

    @staticmethod
    def get_atom_feature_size(name):
        """get atom features size"""
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return len(CompoundKit.atom_vocab_dict[name])

    ### bond

    @staticmethod
    def get_bond_value(bond, name):
        """get bond values"""
        if name == 'bond_dir':
            return bond.GetBondDir()
        elif name == 'bond_type':
            return bond.GetBondType()
        elif name == 'is_in_ring':
            return int(bond.IsInRing())
        else:
            raise ValueError(name)

    @staticmethod
    def get_bond_feature_id(bond, name):
        """get bond features id"""
        assert name in CompoundKit.bond_vocab_dict, "%s not found in bond_vocab_dict" % name
        return CompoundKit.bond_vocab_dict[name].index(CompoundKit.get_bond_value(bond, name))

    @staticmethod
    def get_bond_feature_size(name):
        """get bond features size"""
        assert name in CompoundKit.bond_vocab_dict, "%s not found in bond_vocab_dict" % name
        return len(CompoundKit.bond_vocab_dict[name])

    ### fingerprint

    @staticmethod
    def get_morgan_fingerprint(mol, radius=2):
        """get morgan fingerprint"""
        nBits = CompoundKit.morgan_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return [int(b) for b in mfp.ToBitString()]
    
    @staticmethod
    def get_morgan2048_fingerprint(mol, radius=2):
        """get morgan2048 fingerprint"""
        nBits = CompoundKit.morgan2048_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_maccs_fingerprint(mol):
        """get maccs fingerprint"""
        fp = AllChem.GetMACCSKeysFingerprint(mol)
        return [int(b) for b in fp.ToBitString()]

    ### functional groups

    @staticmethod
    def get_daylight_functional_group_counts(mol):
        """get daylight functional group counts"""
        fg_counts = []
        for fg_mol in CompoundKit.day_light_fg_mo_list:
            sub_structs = Chem.Mol.GetSubstructMatches(mol, fg_mol, uniquify=True)
            fg_counts.append(len(sub_structs))
        return fg_counts


class Compound3DKit(object):
    """the 3Dkit of Compound"""
    @staticmethod
    def get_atom_poses(mol, numConfs=None):
        """the atoms of mol will be changed in some cases."""
        try:
            new_mol = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs)
            ### MMFF generates multiple conformations
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
            new_mol = Chem.RemoveHs(new_mol)
            index = np.argmin([x[1] for x in res])
            energy = res[index][1]
            conf = new_mol.GetConformer(id=int(index))
        except:
            new_mol = mol
            AllChem.Compute2DCoords(new_mol)
            energy = 0
            conf = new_mol.GetConformer()

        atom_poses = []
        for i, atom in enumerate(new_mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return [[0.0, 0.0, 0.0]] * len(new_mol.GetAtoms())
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])
        return new_mol, atom_poses, energy

    @staticmethod
    def get_2d_atom_poses(mol):
        """get 2d atom poses"""
        AllChem.Compute2DCoords(mol)
        energy = 0
        conf = mol.GetConformer()

        atom_poses = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])
        return mol, atom_poses, energy

    @staticmethod
    def get_bond_lengths(edges, atom_poses):
        """get bond lengths"""
        bond_lengths = []
        for src_node_i, tar_node_j in edges:
            bond_lengths.append(np.linalg.norm(atom_poses[tar_node_j] - atom_poses[src_node_i]))
        bond_lengths = np.array(bond_lengths, 'float32')
        return bond_lengths

    @staticmethod
    def get_superedge_angles(edges, atom_poses):
        """get superedge angles"""
        def _get_angle(vec1, vec2):
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0, False
            vec1 = vec1 / (norm1 + 1e-5)    # 1e-5: prevent numerical errors
            vec2 = vec2 / (norm2 + 1e-5)
            angle = np.arccos(np.dot(vec1, vec2))
            return angle, True

        E = len(edges)
        edge_indices = np.arange(E)
        super_edges = []
        super_edge_angles = []
        super_real_angle = []
        for tar_edge_i in range(E):
            tar_edge = edges[tar_edge_i]
            src_edge_indices = edge_indices[edges[:, 1] == tar_edge[0]]
            for src_edge_i in src_edge_indices:
                if src_edge_i == tar_edge_i:
                    continue
                src_edge = edges[src_edge_i]
                src_vec = atom_poses[src_edge[1]] - atom_poses[src_edge[0]]
                tar_vec = atom_poses[tar_edge[1]] - atom_poses[tar_edge[0]]
                super_edges.append([src_edge_i, tar_edge_i])
                angle, real_angle = _get_angle(src_vec, tar_vec)
                super_edge_angles.append(angle)
                super_real_angle.append(real_angle)
        if len(super_edges) == 0:
            super_edges = np.zeros([0, 2], 'int64')
            super_edge_angles = np.zeros([0,], 'float32')
            super_real_angle = np.zeros([0,], 'int64')
        else:
            super_edges = np.array(super_edges, 'int64')
            super_edge_angles = np.array(super_edge_angles, 'float32')
            super_real_angle = np.array(super_real_angle, 'int64')
        return super_edges, super_edge_angles, super_real_angle

    @staticmethod
    def get_polar_angles(edges, atom_poses):
        """get polar angles"""
        def _get_polar(src_atom_poses, tar_atom_pos):
            tar_atom_pos = tar_atom_pos.reshape([1, -1])
            vecs = tar_atom_pos - src_atom_poses
            polar = np.sum(vecs, 0)
            norm = np.linalg.norm(polar)
            if norm == 0:
                polar = np.ones(polar.shape)
                norm = np.linalg.norm(polar)
            polar /= norm
            return polar
        def _get_angle(vec1, vec2):
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            vec1 = vec1 / (norm1 + 1e-5)    # 1e-5: prevent numerical errors
            vec2 = vec2 / (norm2 + 1e-5)
            angle = np.arccos(np.dot(vec1, vec2))
            return angle

        atom_polars = []
        for atom_id in range(len(atom_poses)):
            src_atom_ids = edges[:, 0][edges[:, 1] == atom_id]
            src_atom_poses = atom_poses[src_atom_ids]
            tar_atom_pos = atom_poses[atom_id]
            atom_polar = _get_polar(src_atom_poses, tar_atom_pos)
            atom_polars.append(atom_polar)
        atom_polars = np.array(atom_polars)

        polar_edge_angles = []
        polar_polar_angles = []
        for src_atom_id, tar_atom_id in edges:
            src_polar = atom_polars[src_atom_id]
            tar_polar = atom_polars[tar_atom_id]
            edge_vec = atom_poses[tar_atom_id] - atom_poses[src_atom_id]
            polar_edge_angles.append(_get_angle(tar_polar, edge_vec))
            polar_polar_angles.append(_get_angle(tar_polar, src_polar))
        polar_edge_angles = np.array(polar_edge_angles)
        polar_polar_angles = np.array(polar_polar_angles)
        return atom_polars, polar_edge_angles, polar_polar_angles


def mol_to_graph_data(mol):
    """
    mol_to_graph_data

    Args:
        atom_features: Atom features.
        edge_features: Edge features.
        morgan_fingerprint: Morgan fingerprint.
        functional_groups: Functional groups.
    """
    if len(mol.GetAtoms()) == 0:
        return None

    atom_id_names = [
        "atomic_num", "chiral_tag", "degree", "explicit_valence", 
        "formal_charge", "hybridization", "implicit_valence", 
        "is_aromatic", "total_numHs",
    ]
    bond_id_names = [
        "bond_dir", "bond_type", "is_in_ring",
    ]
    
    data = {}
    for name in atom_id_names:
        data[name] = []
    data['mass'] = []
    for name in bond_id_names:
        data[name] = []
    data['edges'] = []

    ### atom features
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            return None
        for name in atom_id_names:
            data[name].append(CompoundKit.get_atom_feature_id(atom, name) + 1)  # 0: OOV
        data['mass'].append(CompoundKit.get_atom_value(atom, 'mass') * 0.01)

    ### bond features
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # i->j and j->i
        data['edges'] += [(i, j), (j, i)]
        for name in bond_id_names:
            bond_feature_id = CompoundKit.get_bond_feature_id(bond, name) + 1   # 0: OOV
            data[name] += [bond_feature_id] * 2

    ### self loop (+2)
    N = len(data[atom_id_names[0]])
    for i in range(N):
        data['edges'] += [(i, i)]
    for name in bond_id_names:
        bond_feature_id = CompoundKit.get_bond_feature_size(name) + 2   # N + 2: self loop
        data[name] += [bond_feature_id] * N

    ### check whether edge exists
    if len(data['edges']) == 0: # mol has no bonds
        for name in bond_id_names:
            data[name] = np.zeros((0,), dtype="int64")
        data['edges'] = np.zeros((0, 2), dtype="int64")

    ### make ndarray and check length
    N = len(data[atom_id_names[0]])
    E = len(data[bond_id_names[0]])
    for name in atom_id_names:
        data[name] = np.array(data[name], 'int64')
        assert len(data[name]) == N, '%s != %s' % (len(data[name]), N)
    data['mass'] = np.array(data['mass'], 'float32')
    assert len(data['mass']) == N, '%s != %s' % (len(data['mass']), N)
    for name in bond_id_names:
        data[name] = np.array(data[name], 'int64')
        assert len(data[name]) == E, '%s != %s' % (len(data[name]), E)
    data['edges'] = np.array(data['edges'], 'int64')
    assert len(data['edges']) == E, '%s != %s' % (len(data['edges']), E)

    ### morgan fingerprint
    data['morgan_fp'] = np.array(CompoundKit.get_morgan_fingerprint(mol), 'int64')
    data['morgan2048_fp'] = np.array(CompoundKit.get_morgan2048_fingerprint(mol), 'int64')
    data['maccs_fp'] = np.array(CompoundKit.get_maccs_fingerprint(mol), 'int64')
    data['daylight_fg_counts'] = np.array(CompoundKit.get_daylight_functional_group_counts(mol), 'int64')
    return data



def mol_to_md_graph_data(mol, add_3dpos=True, numConfs=10):
    """
    mol_to_md_graph_data

    Args:
        atom_pos: Atom position.
        energy: Energy.
    """
    if len(mol.GetAtoms()) == 0:
        return None
    
    data = {}
    if add_3dpos:
        mol, atom_poses, energy = Compound3DKit.get_atom_poses(mol, numConfs=numConfs)
        data['atom_pos'] = np.array(atom_poses, 'float32')
        data['energy'] = np.array([energy])
    data.update(mol_to_graph_data(mol))

    ### check length
    if add_3dpos:
        N = len(data['atomic_num'])
        assert len(data['atom_pos']) == N, '%s != %s' % (len(data['atom_pos']), N)
    return data


SUPEREDGE_BOND_ANGLE_NUM = 10
SUPEREDGE_BOND_LENGTH_NUM = 100
def mol_to_superedge_graph_data(mol, numConfs=10, cal3d_atom_num_thres=100):
    """
    mol_to_superedge_graph_data

    Args:
        atom_pos: Atom position.
        bond_length: Bond length.
        superedge_edges: Superedge edges.
        superedge_bond_angle_id: Superedge bond angle id.
    """
    if len(mol.GetAtoms()) == 0:
        return None

    data = {}
    if len(mol.GetAtoms()) <= cal3d_atom_num_thres:
        mol, atom_poses, energy = Compound3DKit.get_atom_poses(mol, numConfs=numConfs)
    else:
        mol, atom_poses, energy = Compound3DKit.get_2d_atom_poses(mol)
    data['atom_pos'] = np.array(atom_poses, 'float32')
    data['energy'] = np.array([energy])
    data.update(mol_to_graph_data(mol))

    bond_lengths = Compound3DKit.get_bond_lengths(data['edges'], data['atom_pos'])
    data['bond_length'] = bond_lengths
    data['bond_length_id'] = np.array(
            1.0 / (1.0 + bond_lengths) * SUPEREDGE_BOND_LENGTH_NUM, 'int64')
    super_edges, super_edge_angles, super_real_angle = \
            Compound3DKit.get_superedge_angles(data['edges'], data['atom_pos'])
    data['superedge_edges'] = super_edges
    data['superedge_bond_angle'] = np.array(super_edge_angles, 'float32')
    data['superedge_bond_angle_id'] = np.array(         # 0: fake angle, >=1: real angle
            super_edge_angles / np.pi * SUPEREDGE_BOND_ANGLE_NUM + 1, 'int64') * super_real_angle

    N = len(data['atomic_num'])
    E = len(data['edges'])
    SE = len(data['superedge_edges'])
    assert len(data['atom_pos']) == N, '%s != %s' % (len(data['atom_pos']), N)
    assert len(data['bond_length']) == E, '%s != %s' % (len(data['bond_length']), E)
    assert len(data['bond_length_id']) == E, '%s != %s' % (len(data['bond_length_id']), E)
    assert len(data['superedge_bond_angle']) == SE, '%s != %s' % (len(data['superedge_bond_angle']), SE)
    assert len(data['superedge_bond_angle_id']) == SE, '%s != %s' % (len(data['superedge_bond_angle_id']), SE)
    return data


POLAR_ANGLE_NUM = int(os.environ.get('POLAR_ANGLE_NUM', 10))
# print('[GLOBAL] POLAR_ANGLE_NUM:%s' % POLAR_ANGLE_NUM)
def mol_to_polar_graph_data(mol):
    """
    mol_to_polar_graph_data

    Args:
        atom_pos: Atom position.
        atom_polar: Atom polar.
        polar_edge_angle: Polar edge angle.
        polar_polar_angle: Polar polar angle.
    """
    if len(mol.GetAtoms()) == 0:
        return None
    
    data = {}
    mol, atom_poses, energy = Compound3DKit.get_2d_atom_poses(mol)
    data['atom_pos'] = np.array(atom_poses, 'float32')
    data['energy'] = np.array([energy])

    data.update(mol_to_graph_data(mol))

    atom_polars, polar_edge_angles, polar_polar_angles = \
            Compound3DKit.get_polar_angles(data['edges'], data['atom_pos'])
    data['atom_polar'] = np.array(atom_polars, 'float32')
    data['polar_edge_angle'] = np.array(polar_edge_angles, 'float32')
    data['polar_edge_angle_id'] = np.array(polar_edge_angles / np.pi * POLAR_ANGLE_NUM, 'int64')
    data['polar_polar_angle'] = np.array(polar_polar_angles, 'float32')
    data['polar_polar_angle_id'] = np.array(polar_polar_angles / np.pi * POLAR_ANGLE_NUM, 'int64')

    ### check length
    N = len(data['atomic_num'])
    E = len(data['edges'])
    assert len(data['atom_pos']) == N, '%s != %s' % (len(data['atom_pos']), N)
    assert len(data['atom_polar']) == N, '%s != %s' % (len(data['atom_polar']), N)
    assert len(data['polar_edge_angle']) == E, '%s != %s' % (len(data['polar_edge_angle']), E)
    assert len(data['polar_polar_angle']) == E, '%s != %s' % (len(data['polar_polar_angle']), E)
    return data


if __name__ == "__main__":
    smiles = "OCc1ccccc1CN"
    # smiles = r"[H]/[NH+]=C(\N)C1=CC(=O)/C(=C\C=c2ccc(=C(N)[NH3+])cc2)C=C1"
    mol = AllChem.MolFromSmiles(smiles)
    print(len(smiles))
    print(mol)
    # data = mol_to_md_graph_data(mol)
    data = mol_to_polar_graph_data(mol)
    print('atom_polar\n', data['atom_polar'])
    print('polar_edge_angle\n', data['polar_edge_angle'])
    print('polar_edge_angle_id\n', data['polar_edge_angle_id'])
    print('polar_polar_angle\n', data['polar_polar_angle'])
    print('polar_polar_angle_id\n', data['polar_polar_angle_id'])
    