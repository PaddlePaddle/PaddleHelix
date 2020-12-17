#!/usr/bin/python
#-*-coding:utf-8-*-
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
Processing of chembl filtered dataset.

The ChEMBL dataset containing 456K molecules with 1310 kinds of diverse and extensive biochemical assays. The database is unique because of its focus on all aspects of drug discovery and its size, containing information on more than 1.8 million compounds and over 15 million records of their effects on biological systems.

"""

import os
from os.path import join, exists, dirname
import pickle
import pandas as pd
import numpy as np
from itertools import repeat, product, chain

from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

from pahelix.datasets.inmemory_dataset import InMemoryDataset
from pahelix.datasets.bace_dataset import load_bace_dataset
from pahelix.datasets.bbbp_dataset import load_bbbp_dataset
from pahelix.datasets.clintox_dataset import load_clintox_dataset
from pahelix.datasets.esol_dataset import load_esol_dataset
from pahelix.datasets.freesolv_dataset import load_freesolv_dataset
from pahelix.datasets.hiv_dataset import load_hiv_dataset
from pahelix.datasets.lipophilicity_dataset import load_lipophilicity_dataset
from pahelix.datasets.muv_dataset import load_muv_dataset
from pahelix.datasets.sider_dataset import load_sider_dataset
from pahelix.datasets.tox21_dataset import load_tox21_dataset
from pahelix.datasets.toxcast_dataset import load_toxcast_dataset
from pahelix.utils.compound_tools import *
from pahelix.utils.splitters import ScaffoldSplitter


__all__ = ['get_chembl_filtered_task_num', 'load_chembl_filtered_dataset']


def get_chembl_filtered_task_num():
    """get that default bace task names and return class"""
    return 1310


def load_chembl_filtered_dataset(data_path, featurizer=None):
    """load chembl_filtered dataset ,process the classification labels and the input information.

    Introduction:
        Note that, in order to load this dataset, you should have other datasets (bace, bbbp, clintox,
        esol, freesolv, hiv, lipophilicity, muv, sider, tox21, toxcast) downloaded. Since the chembl
        dataset may overlap with the above listed dataset, the overlapped smiles for test will be filtered
        for a fair evaluation.

    Description:
        The data file contains a csv table, in which columns below are used:
            It contains the ID, SMILES/CTAB, InChI and InChIKey compound information
            smiles:SMILES representation of the molecular structure

    Args:
        data_path(str): the path to the cached npz path
        featurizer(pahelix.featurizers.Featurizer): the featurizer to use for 
            processing the data. If not none, The ``Featurizer.gen_features`` will be 
            applied to the raw data.
    
    Returns:
        an InMemoryDataset instance.
    
    Example:
        .. code-block:: python

            dataset = load_bbbp_dataset('./bace/raw')
            print(len(dataset))

    References:
    -- Gaulton, A; et al. (2011). “ChEMBL: a large-scale bioactivity database for drug discovery”. Nucleic Acids Research. 40 (Database issue): D1100-7.
    
    """
    downstream_datasets = [
        load_bace_dataset(join(dirname(dirname(data_path)), 'bace/raw')),
        load_bbbp_dataset(join(dirname(dirname(data_path)), 'bbbp/raw')),
        load_clintox_dataset(join(dirname(dirname(data_path)), 'clintox/raw')),
        load_esol_dataset(join(dirname(dirname(data_path)), 'esol/raw')),
        load_freesolv_dataset(join(dirname(dirname(data_path)), 'freesolv/raw')),
        load_hiv_dataset(join(dirname(dirname(data_path)), 'hiv/raw')),
        load_lipophilicity_dataset(join(dirname(dirname(data_path)), 'lipophilicity/raw')),
        load_muv_dataset(join(dirname(dirname(data_path)), 'muv/raw')),
        load_sider_dataset(join(dirname(dirname(data_path)), 'sider/raw')),
        load_tox21_dataset(join(dirname(dirname(data_path)), 'tox21/raw')),
        load_toxcast_dataset(join(dirname(dirname(data_path)), 'toxcast/raw')),
    ]
    downstream_inchi_set = set()
    splitter = ScaffoldSplitter()
    for c_dataset in downstream_datasets:
        train_dataset, valid_dataset, test_dataset = splitter.split(
                c_dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        ### remove both test and validation molecules
        # remove_smiles = test_smiles + valid_smiles
        remove_smiles = [d['smiles'] for d in valid_dataset] + [d['smiles'] for d in test_dataset]

        downstream_inchis = []
        for smiles in remove_smiles:
            species_list = smiles.split('.')
            for s in species_list:  # record inchi for all species, not just
             # largest (by default in create_standardized_mol_id if input has
             # multiple species)
                inchi = create_standardized_mol_id(s)
                downstream_inchis.append(inchi)
        downstream_inchi_set.update(downstream_inchis)

    smiles_list, rdkit_mol_objs, folds, labels = \
            _load_chembl_filtered_dataset(data_path)
    # print(smiles_list, rdkit_mol_objs, folds, labels)
    data_list = []
    for i in range(len(rdkit_mol_objs)):
        rdkit_mol = rdkit_mol_objs[i]
        if not rdkit_mol is None:
            mw = Descriptors.MolWt(rdkit_mol)
            if 50 <= mw <= 900:
                inchi = create_standardized_mol_id(smiles_list[i])
                if not inchi is None and inchi not in downstream_inchi_set:
                    raw_data = {
                        'smiles': smiles_list[i],
                        'label': labels[i].reshape([-1]),
                    }
                    
                    if not featurizer is None:
                        data = featurizer.gen_features(raw_data)
                    else:
                        data = raw_data

                    if not data is None:
                        data_list.append(data)
    
    dataset = InMemoryDataset(data_list)
    return dataset


def _load_chembl_filtered_dataset(root_path):
    """
<<<<<<< HEAD
    Description:
=======
    Description：
>>>>>>> f7fb468f8f5c7764763ff31376f2e9845c0b576c
        Data from 'Large-scale comparison of machine learning methods for drug target prediction on ChEMBL'
            param root_path: path to the folder containing the reduced chembl dataset
            return: list of smiles, preprocessed rdkit mol obj list, list of np.array
            containing indices for each of the 3 folds, np.array containing the labels
    """
    # adapted from https://github.com/ml-jku/lsc/blob/master/pythonCode/lstm/loadData.py
    # first need to download the files and unzip:
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced.zip
    # unzip and rename to chembl_with_labels
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20Smiles.pckl
    # into the dataPythonReduced directory
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20LSTM.pckl

    # 1. load folds and labels
    f = open(os.path.join(root_path, 'folds0.pckl'), 'rb')
    folds = pickle.load(f)
    f.close()

    f = open(os.path.join(root_path, 'labelsHard.pckl'), 'rb')
    targetMat = pickle.load(f)
    sampleAnnInd = pickle.load(f)
    targetAnnInd = pickle.load(f)
    f.close()

    targetMat = targetMat
    targetMat = targetMat.copy().tocsr()
    targetMat.sort_indices()
    targetAnnInd = targetAnnInd
    targetAnnInd = targetAnnInd - targetAnnInd.min()

    folds = [np.intersect1d(fold, sampleAnnInd.index.values).tolist() for fold in folds]
    targetMatTransposed = targetMat[sampleAnnInd[list(chain(*folds))]].T.tocsr()
    targetMatTransposed.sort_indices()
    # # num positive examples in each of the 1310 targets
    trainPosOverall = np.array([np.sum(targetMatTransposed[x].data > 0.5) 
            for x in range(targetMatTransposed.shape[0])])
    # # num negative examples in each of the 1310 targets
    trainNegOverall = np.array([np.sum(targetMatTransposed[x].data < -0.5) 
            for x in range(targetMatTransposed.shape[0])])
    # dense array containing the labels for the 456331 molecules and 1310 targets
    denseOutputData = targetMat.A # possible values are {-1, 0, 1}

    # 2. load structures
    f = open(os.path.join(root_path, 'chembl20LSTM.pckl'), 'rb')
    rdkitArr = pickle.load(f)
    f.close()

    assert len(rdkitArr) == denseOutputData.shape[0]
    assert len(rdkitArr) == len(folds[0]) + len(folds[1]) + len(folds[2])

    preprocessed_rdkitArr = []
    print('preprocessing')
    for i in range(len(rdkitArr)):
        m = rdkitArr[i]
        if m is None:
            preprocessed_rdkitArr.append(None)
        else:
            mol_species_list = split_rdkit_mol_obj(m)
            if len(mol_species_list) == 0:
                preprocessed_rdkitArr.append(None)
            else:
                largest_mol = get_largest_mol(mol_species_list)
                if len(largest_mol.GetAtoms()) <= 2:
                    preprocessed_rdkitArr.append(None)
                else:
                    preprocessed_rdkitArr.append(largest_mol)

    assert len(preprocessed_rdkitArr) == denseOutputData.shape[0]

    smiles_list = [AllChem.MolToSmiles(m) if not m is None else None for m in
                   preprocessed_rdkitArr]   # bc some empty mol in the
    # rdkitArr zzz...

    assert len(preprocessed_rdkitArr) == len(smiles_list)

    return [smiles_list, preprocessed_rdkitArr, \
            folds, denseOutputData]


