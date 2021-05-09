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
"""smiles preprocessing"""
from multiprocessing import Pool
from src.mol_tree import MolTree
import pickle
import rdkit
import os
import argparse

def tensorize(smiles, assm=True):
    """
    transform smiles into tree objects
    """
    try:
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        if assm:
            mol_tree.assemble()
            for node in mol_tree.nodes:
                if node.label not in node.cands:
                    node.cands.append(node.label)

        del mol_tree.mol
        for node in mol_tree.nodes:
            del node.mol
    except Exception as e:
        print(smiles, e)
        return None

    return mol_tree

if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", dest="train_path")
    parser.add_argument("--save_dir", dest="save_dir")
    parser.add_argument("--split", dest="nsplits", type=int, default=100)
    parser.add_argument("--num_workers", dest="num_workers", type=int, default=8)
    args = parser.parse_args()

    pool = Pool(args.num_workers)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    with open(args.train_path, 'r') as f:
        data = f.read().splitlines()
    num_splits = args.nsplits
    all_data = pool.map(tensorize, data)
    le = (len(all_data) + num_splits - 1) // num_splits

    for split_id in range(num_splits):
        st = split_id * le
        sub_data = all_data[st: st + le]
        if not sub_data:
            break
        with open(os.path.join(args.save_dir, 'tensors-%d.pkl' % split_id), 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

