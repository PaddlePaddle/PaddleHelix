# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit.Chem import AllChem
from ogb.graphproppred import GraphPropPredDataset

def getmorganfingerprint(mol):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2))

def getmaccsfingerprint(mol):
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    return [int(b) for b in fp.ToBitString()]

def main(dataset_name):
    dataset = GraphPropPredDataset(name=dataset_name)

    df_smi = pd.read_csv(f"dataset/{dataset_name}/mapping/mol.csv.gz".replace("-", "_"))
    smiles = df_smi["smiles"]

    mgf_feat_list = []
    maccs_feat_list = []
    for ii in tqdm(range(len(smiles))):
        rdkit_mol = AllChem.MolFromSmiles(smiles.iloc[ii])

        mgf = getmorganfingerprint(rdkit_mol)
        mgf_feat_list.append(mgf)

        maccs = getmaccsfingerprint(rdkit_mol)
        maccs_feat_list.append(maccs)

    mgf_feat = np.array(mgf_feat_list, dtype="int64")
    maccs_feat = np.array(maccs_feat_list, dtype="int64")
    print("morgan feature shape: ", mgf_feat.shape)
    print("maccs feature shape: ", maccs_feat.shape)

    save_path = f"./dataset/{dataset_name}".replace("-", "_")
    print("saving feature in %s" % save_path)
    np.save(os.path.join(save_path, "mgf_feat.npy"), mgf_feat)
    np.save(os.path.join(save_path, "maccs_feat.npy"), maccs_feat)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='gnn')
    parser.add_argument("--dataset_name", type=str, default="ogbg-molhiv")
    args = parser.parse_args()

    main(args.dataset_name)

