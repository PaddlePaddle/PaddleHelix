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

"""Full AlphaFold protein structure prediction script."""
import sys
import time
import json
import pickle
import random
import logging
import pathlib
import argparse
import numpy as np
from typing import Dict

from alphafold_paddle.model import config
from alphafold_paddle.model import model
from alphafold_paddle.relax import relax
from alphafold_paddle.data import pipeline, templates

logger = logging.getLogger(__file__)

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 20


def predict_structure(
        fasta_path: str,
        fasta_name: str,
        output_dir_base: str,
        data_pipeline: pipeline.DataPipeline,
        model_runners: Dict[str, model.RunModel],
        amber_relaxer: relax.AmberRelaxation,
        random_seed: int):
    timings = dict()
    output_dir = pathlib.Path(output_dir_base).joinpath(fasta_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    msa_output_dir = output_dir.joinpath('msas')
    msa_output_dir.mkdir(exist_ok=True)

    feature_dict = None
    features_pkl = output_dir.joinpath('features.pkl')
    if features_pkl.exists():
        logger.info('Use cached features.pkl')
        with open(features_pkl, 'rb') as f:
            feature_dict = pickle.load(f)
    else:
        t0 = time.time()
        feature_dict = data_pipeline.process(
            input_fasta_path=fasta_path,
            msa_output_dir=msa_output_dir)
        timings['features'] = time.time() - t0

        with open(features_pkl, 'wb') as f:
            pickle.dump(feature_dict, f, protocol=4)

    relaxed_pdbs, plddts = dict(), dict()
    for model_name, model_runner in model_runners.items():
        logger.info('Running model %s', model_name)

        input_features_pkl = output_dir.joinpath(f'{model_name}_input.pkl')
        has_cache = input_features_pkl.exists()

        t0 = time.time()
        processed_feature_dict = model_runner.preprocess(
            feature_dict, random_seed, input_features_pkl)
        if not has_cache:
            timings[f'process_features_{model_name}'] = time.time() - t0

        t0 = time.time()
        prediction = model_runner.predict(
            processed_feature_dict,
            ensemble_representations=True,
            return_representations=True)
        timings[f'predict_{model_name}'] = time.time() - t0

        aatype = processed_feature_dict['aatype'].numpy()[0, 0]
        residue_index = processed_feature_dict['residue_index'].numpy()[0, 0]
        relaxed_pdbs[model_name] = model_runner.postprocess(
            aatype, residue_index, amber_relaxer, prediction,
            output_dir, 0, timings)
        plddts[model_name] = np.mean(prediction['plddt'])

    # Rank by pLDDT and write out relaxed PDBs in rank order.
    ranked_order = []
    for idx, (model_name, _) in enumerate(
            sorted(plddts.items(), key=lambda x: x[1], reverse=True)):
        ranked_order.append(model_name)

        with open(output_dir.joinpath(f'ranked_{idx}.pdb'), 'w') as f:
            f.write(relaxed_pdbs[model_name])

    with open(output_dir.joinpath('ranking_debug.json'), 'w') as f:
        f.write(json.dumps({
            'plddts': plddts, 'order': ranked_order}, indent=4))

    logger.info('Final timings for %s: %s', fasta_name, timings)
    with open(output_dir.joinpath('timings.json'), 'w') as f:
        f.write(json.dumps(timings, indent=4))


def main():
    use_small_bfd = args.preset == 'reduced_dbs'
    if use_small_bfd:
        assert args.small_bfd_database_path is not None
    else:
        assert args.bfd_database_path is not None
        assert args.uniclust30_database_path is not None

    if args.preset in ['reduced_dbs', 'full_dbs']:
        num_ensemble = 1
    elif args.preset == 'casp14':
        num_ensemble = 8

    # Check for duplicate FASTA file names.
    fasta_names = [pathlib.Path(p).stem for p in args.fasta_paths.split(',')]
    if len(fasta_names) != len(set(fasta_names)):
        raise ValueError('All FASTA paths must have a unique basename.')

    template_featurizer = templates.TemplateHitFeaturizer(
        mmcif_dir=args.template_mmcif_dir,
        max_template_date=args.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=args.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=args.obsolete_pdbs_path)

    data_pipeline = pipeline.DataPipeline(
        jackhmmer_binary_path=args.jackhmmer_binary_path,
        hhblits_binary_path=args.hhblits_binary_path,
        hhsearch_binary_path=args.hhsearch_binary_path,
        uniref90_database_path=args.uniref90_database_path,
        mgnify_database_path=args.mgnify_database_path,
        bfd_database_path=args.bfd_database_path,
        uniclust30_database_path=args.uniclust30_database_path,
        small_bfd_database_path=args.small_bfd_database_path,
        pdb70_database_path=args.pdb70_database_path,
        template_featurizer=template_featurizer,
        use_small_bfd=use_small_bfd)

    model_runners = dict()
    for model_name in args.model_names.split(','):
        model_config = config.model_config(model_name)
        model_config.data.eval.num_ensemble = num_ensemble

        data_dir = pathlib.Path(args.data_dir)
        params = f'params_{model_name}'
        model_params = data_dir.joinpath('params', f'{params}.pd')
        if not model_params.exists():
            model_params = data_dir.joinpath('params', f'{params}.npz')

        model_runner = model.RunModel(model_name, model_config, model_params)
        model_runners[model_name] = model_runner

    logger.info('Have %d models: %s', len(model_runners),
                list(model_runners.keys()))

    amber_relaxer = relax.AmberRelaxation(
        max_iterations=RELAX_MAX_ITERATIONS,
        tolerance=RELAX_ENERGY_TOLERANCE,
        stiffness=RELAX_STIFFNESS,
        exclude_residues=RELAX_EXCLUDE_RESIDUES,
        max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS)

    random_seed = args.random_seed
    if random_seed is None:
        random_seed = random.randrange(sys.maxsize)

    logger.info('Using random seed %d for the data pipeline', random_seed)

    for fasta_path, fasta_name in zip(args.fasta_paths.split(','), fasta_names):
        predict_structure(
            fasta_path=fasta_path,
            fasta_name=fasta_name,
            output_dir_base=args.output_dir,
            data_pipeline=data_pipeline,
            model_runners=model_runners,
            amber_relaxer=amber_relaxer,
            random_seed=random_seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict protein structure')
    parser.add_argument('--fasta_paths', type=str,
                        default=None, required=True,
                        help='Paths to FASTA files, each containing '
                        'one sequence. Paths should be separated by commas. '
                        'All FASTA paths must have a unique basename as the '
                        'basename is used to name the output directories for '
                        'each prediction.')
    parser.add_argument('--output_dir', type=str,
                        default=None, required=True,
                        help='Path to a directory that will store results.')
    parser.add_argument('--model_names', type=str,
                        default=None, required=True,
                        help='Names of models to use.')
    parser.add_argument('--data_dir', type=str,
                        default=None, required=True,
                        help='Path to directory of supporting data.')

    parser.add_argument('--jackhmmer_binary_path', type=str,
                        default='/usr/bin/jackhmmer',
                        help='Path to the JackHMMER executable.')
    parser.add_argument('--hhblits_binary_path', type=str,
                        default='/usr/bin/hhblits',
                        help='Path to the HHblits executable.')
    parser.add_argument('--hhsearch_binary_path', type=str,
                        default='/usr/bin/hhsearch',
                        help='Path to the HHsearch executable.')
    parser.add_argument('--kalign_binary_path', type=str,
                        default='/usr/bin/kalign',
                        help='Path to the Kalign executable.')

    parser.add_argument('--uniref90_database_path', type=str,
                        default=None, required=True,
                        help='Path to the Uniref90 database for use '
                        'by JackHMMER.')
    parser.add_argument('--mgnify_database_path', type=str,
                        default=None, required=True,
                        help='Path to the MGnify database for use by '
                        'JackHMMER.')
    parser.add_argument('--bfd_database_path', type=str, default=None,
                        help='Path to the BFD database for use by HHblits.')
    parser.add_argument('--small_bfd_database_path', type=str, default=None,
                        help='Path to the small version of BFD used '
                        'with the "reduced_dbs" preset.')
    parser.add_argument('--uniclust30_database_path', type=str, default=None,
                        help='Path to the Uniclust30 database for use '
                        'by HHblits.')
    parser.add_argument('--pdb70_database_path', type=str,
                        default=None, required=True,
                        help='Path to the PDB70 database for use by '
                        'HHsearch.')
    parser.add_argument('--template_mmcif_dir', type=str,
                        default=None, required=True,
                        help='Path to a directory with template mmCIF '
                        'structures, each named <pdb_id>.cif')
    parser.add_argument('--max_template_date', type=str,
                        default=None, required=True,
                        help='Maximum template release date to consider. '
                        'Important if folding historical test sets.')
    parser.add_argument('--obsolete_pdbs_path', type=str,
                        default=None, required=True,
                        help='Path to file containing a mapping from '
                        'obsolete PDB IDs to the PDB IDs of their '
                        'replacements.')

    parser.add_argument('--preset',
                        default='full_dbs', required=True,
                        choices=['reduced_dbs', 'full_dbs', 'casp14'],
                        help='Choose preset model configuration - '
                        'no ensembling and smaller genetic database '
                        'config (reduced_dbs), no ensembling and full '
                        'genetic database config  (full_dbs) or full '
                        'genetic database config and 8 model ensemblings '
                        '(casp14).')
    parser.add_argument('--random_seed', type=int,
                        help='The random seed for the data pipeline. '
                        'By default, this is randomly generated.')

    args = parser.parse_args()
    main()
