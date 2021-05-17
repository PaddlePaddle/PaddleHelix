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
Preprocess downloaded MUTAG and PTC-MR datasets
"""

import os
import pandas as pd
from rdkit.Chem import AllChem

from pahelix.utils.compound_tools import mol_to_graph_data
from pahelix.utils.data_utils import save_data_list_to_npz

Datasets = {
    'mutag': ['mutag_188_data.can', 'mutag_188_target.txt'],
    'ptc_mr': ['ptc_MR_data.can', 'ptc_MR_target.txt']
}


def preprocess_dataset(name):
    """
    Preprocess raw datasets.

    Args:
        name (str): name of the dataset.
    """
    data_dir = os.path.join('data', name, 'raw')
    if not os.path.exists(data_dir):
        print('Ignore MUTAG dataset. Cannot find the corresponding folder: %s.' % data_dir)
        return

    can, txt = Datasets[name]
    smiles_path = os.path.join(data_dir, can)
    labels_path = os.path.join(data_dir, txt)
    smiles_list = pd.read_csv(smiles_path, sep=' ', header=None)[0]
    labels = pd.read_csv(labels_path, header=None)[0].replace(-1, 0).values

    data_list, data_smiles_list = [], []
    for i in range(len(smiles_list)):
        s = smiles_list[i]
        mol = AllChem.MolFromSmiles(s)
        if mol is not None:
            data = mol_to_graph_data(mol)
            data['label'] = labels[i].reshape([-1])
            data_list.append(data)
            data_smiles_list.append(smiles_list[i])

    processed_dir = os.path.join('data', name, 'processed')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    with open(os.path.join(processed_dir, 'smiles.txt'), 'w') as f:
        for smiles in smiles_list:
            f.write('%s\n' % smiles)

    save_data_list_to_npz(
        data_list, os.path.join(processed_dir, 'data.npz'))


if __name__ == '__main__':
    if not os.path.exists('data'):
        raise RuntimeError('Cannot find data folder, please check '
                           'README.md for more details.')

    for name in Datasets.keys():
        preprocess_dataset(name)
