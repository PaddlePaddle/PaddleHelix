#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

"""Inference"""

import os
from os.path import exists, join, dirname, basename
import argparse
import numpy as np
import time
import json
import ml_collections

import paddle
from paddle import distributed as dist

from utils.utils import get_model_parameter_size, tree_map
from utils.model_tape import RunTapeModel

from alphafold_paddle.model import features
from alphafold_paddle.model import config
from alphafold_paddle.model import utils
from alphafold_paddle.common import protein, residue_constants
from alphafold_paddle.data.data_utils import single_sequence_to_features


def read_fasta_file(fasta_file):
    """
    read fasta file
    """
    with open(fasta_file, 'r') as f:
        description = f.readline().strip()
        sequence = ''
        for line in f:
            if line.startswith('>'):
                break
            sequence += line.strip()
    return sequence, description


def sequence_to_batch(fasta_file, model_config):
    """
    make batch data with single sequence
    """
    sequence, description = read_fasta_file(fasta_file)
    raw_features = single_sequence_to_features(sequence, description)
    feat = features.np_example_to_features(np_example=raw_features, config=model_config)

    batch = {
        "name": [basename(fasta_file).replace('.fasta', '')],
        "feat": tree_map(lambda v: paddle.to_tensor(v[None, ...]), feat),
        "label": {},
    }
    return batch


def postprocess(batch, results, output_dir):
    """save unrelaxed pdb"""
    batch['feat'] = tree_map(lambda x: x[0].numpy(), batch['feat'])     # slice the 1st item
    results = tree_map(lambda x: x[0].numpy(), results)

    results.update(utils.get_confidence_metrics(results))
    plddt = results['plddt']
    plddt_b_factors = np.repeat(
            plddt[:, None], residue_constants.atom_type_num, axis=-1)
    prot = protein.from_prediction(batch['feat'], results, b_factors=plddt_b_factors)
    pdb_str = protein.to_pdb(prot)

    with open(join(output_dir, 'unrelaxed.pdb'), 'w') as f:
        f.write(pdb_str)


def main(args):
    """main function"""
    ### create model
    pwd = dirname(__file__)
    model_config = join(pwd, "./model_configs/tape-lnw4.json")
    tape_model_config = join(pwd, "./tape/configs/deberta_1B_bs_cp.json")
    af2_model_name = "seq512_pair64_l24_vio0"
    train_config = None
    model_config = ml_collections.ConfigDict(json.load(open(model_config, 'r')))
    tape_model_config = ml_collections.ConfigDict(json.load(open(tape_model_config, 'r')))
    af2_model_config = config.model_config(af2_model_name)
    model = RunTapeModel(train_config, model_config, tape_model_config, af2_model_config)

    print("model size:", get_model_parameter_size(model))
    model.load_params(args.init_model)

    ### make predictions
    af2_model_config.data.eval.delete_msa_block = False
    batch = sequence_to_batch(
            args.fasta_file, af2_model_config)
    model.eval()
    with paddle.no_grad():
        results = model(batch, compute_loss=False)
    
    ### postprocess
    os.makedirs(args.output_dir, exist_ok=True)
    postprocess(batch, results, args.output_dir)
    print(f'Done. output to {args.output_dir}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model", type=str, help='path to pretrained model')
    parser.add_argument("--fasta_file", type=str, help='path to fasta file to be predicted')
    parser.add_argument("--output_dir", type=str, help='path to prediction outputs')
    args = parser.parse_args()

    main(args)
