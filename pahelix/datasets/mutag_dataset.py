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
Processsing of mutag dataset.

The MUTAG dataset is a dataset with 188 compounds with annotated mutagenic effect on bacterium.

You can download the dataset from
ftp://ftp.ics.uci.edu/pub/baldig/learning/mutag/
"""

# import os
# import pandas as pd
# from rdkit.Chem import AllChem

# from pahelix.utils.compound_tools import mol_to_graph_data


# def load_mutag_dataset(data_path):
#     """Load mutag dataset, process the raw dataset to graph data.
#     """
#     smiles_path = os.path.join(data_path, 'mutag_188_data.can')
#     labels_path = os.path.join(data_path, 'mutag_188_target.txt')

#     smiles_list = pd.read_csv(smiles_path, sep=' ', header=None)[0]
#     labels = pd.read_csv(labels_path, header=None)[0].replace(-1, 0).values

#     data_list, data_smiles_list = [], []
#     for i in range(len(smiles_list)):
#         s = smiles_list[i]
#         rdkit_mol = AllChem.MolFromSmiles(s)
#         if not rdkit_mol is None:  # ignore invalid mol objects
#             data = mol_to_graph_data(rdkit_mol)
#             data['label'] = labels[i].reshape([-1])
#             data_list.append(data)
#             data_smiles_list.append(smiles_list[i])

#     return data_list, data_smiles_list
