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

"""Generate processed features from FASTA."""

import os
import pathlib
import pickle
import random
import sys
import time

from absl import app
from absl import flags
from absl import logging
from multiprocessing import Pool
from functools import partial

from alphafold_paddle.data import pipeline
from alphafold_paddle.data import templates
from alphafold_paddle.data.utils import generate_pkl_features_from_fasta

#### USER CONFIGURATION ####

# Default paths. Note that these arguments can be set directly by flags from command line.
default_database_dir = '.' 
default_fasta_dir = './example_data/fasta/'
default_output_dir = './out/features/'

# Set to database directory with Uniref90, MGnify, and BFD, Uniclust30 database.
# Path to the Uniref90 database for use by JackHMMER.
uniref90_database_path = os.path.join(
    default_database_dir, 'uniref90', 'uniref90.fasta')

# Path to the MGnify database for use by JackHMMER.
mgnify_database_path = os.path.join(
    default_database_dir, 'mgnify', 'mgy_clusters.fa')

# Path to the BFD database for use by HHblits.
bfd_database_path = os.path.join(
    default_database_dir, 'bfd',
    'bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt')

# Path to the Uniclust30 database for use by HHblits.
uniclust30_database_path = os.path.join(
    default_database_dir, 'uniclust30', 'UniRef30_2020_06/UniRef30_2020_06')

# Path to the PDB70 database for use by HHsearch.
pdb70_database_path = os.path.join(default_database_dir, 'pdb70', 'pdb70')

# Path to a directory with template mmCIF structures, each named <pdb_id>.cif')
template_mmcif_dir = os.path.join(default_database_dir, 'pdb_mmcif', 'mmcif_files')

# Path to a file mapping obsolete PDB IDs to their replacements.
obsolete_pdbs_path = os.path.join(default_database_dir, 'pdb_mmcif', 'obsolete.dat')

#### END OF USER CONFIGURATION ####

# Define flags.
flags.DEFINE_string('fasta_dir', default_fasta_dir, 
                    'Paths to FASTA files, each containing one sequence. '
                    'Paths should be separated by commas. All FASTA paths '
                    'must have a unique basename as the basename is used to '
                    'name the output directories for each prediction.')
flags.DEFINE_string('output_dir', default_output_dir,
                    'Path to a directory that will store the results.')
flags.DEFINE_string('data_dir', default_database_dir,
                    'Path to directory of supporting data.')
flags.DEFINE_string('jackhmmer_binary_path', 'jackhmmer',
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', 'hhblits',
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', 'hhsearch',
                    'Path to the HHsearch executable.')
flags.DEFINE_string('kalign_binary_path', 'kalign',
                    'Path to the Kalign executable.')
flags.DEFINE_string('uniref90_database_path', uniref90_database_path,
                    'Path to the Uniref90 database for use by JackHMMER.')
flags.DEFINE_string('mgnify_database_path', mgnify_database_path,
                    'Path to the MGnify database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', bfd_database_path, 
                    'Path to the BFD database for use by HHblits.')
flags.DEFINE_string('uniclust30_database_path', uniclust30_database_path,
                    'Path to the Uniclust30 database for use by HHblits.')
flags.DEFINE_string('pdb70_database_path', pdb70_database_path,
                    'Path to the PDB70 database for use by HHsearch.')
flags.DEFINE_string('template_mmcif_dir', template_mmcif_dir,
                    'Path to a directory with template mmCIF structures, '
                    'each named <pdb_id>.cif')
flags.DEFINE_string('max_template_date', '2020-4-30',
                    'Maximum template release date to consider. Important '
                    'if folding historical test sets.')
flags.DEFINE_string('obsolete_pdbs_path', obsolete_pdbs_path,
                    'Path to file containing a mapping from obsolete PDB IDs '
                    'to the PDB IDs of their replacements.')
flags.DEFINE_integer('random_seed', None,
                     'The random seed for the data pipeline. By default, this '
                     'is randomly generated. Note that even if this is set, '
                     'Uni-Fold may still not be deterministic, because '
                     'processes like GPU inference are nondeterministic.')
flags.DEFINE_integer('n_cpu', 8, 'Number of CPUs used in the data pipeline.')
flags.DEFINE_integer('num_workers', 1, 'Number of parallel workers.')

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20


def get_fasta_paths(fasta_dir: str):
  fasta_paths = []
  for root, dirs, files in os.walk(fasta_dir):
    for fname in files:
      # consider .fasta / .fas / .fa files
      if fname.endswith(".fasta") or fname.endswith(".fas") or fname.endswith(".fa"):
        fasta_paths.append(os.path.join(root, fname))
  return fasta_paths


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  
  # get fasta paths and check for duplicated names
  fasta_paths = get_fasta_paths(FLAGS.fasta_dir)
  fasta_names = [pathlib.Path(p).stem for p in fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  template_featurizer = templates.TemplateHitFeaturizer(
      mmcif_dir=FLAGS.template_mmcif_dir,
      max_template_date=FLAGS.max_template_date,
      max_hits=MAX_TEMPLATE_HITS,
      kalign_binary_path=FLAGS.kalign_binary_path,
      release_dates_path=None,
      obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)

  data_pipeline = pipeline.DataPipeline(
      jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
      hhblits_binary_path=FLAGS.hhblits_binary_path,
      hhsearch_binary_path=FLAGS.hhsearch_binary_path,
      uniref90_database_path=FLAGS.uniref90_database_path,
      mgnify_database_path=FLAGS.mgnify_database_path,
      bfd_database_path=FLAGS.bfd_database_path,
      uniclust30_database_path=FLAGS.uniclust30_database_path,
      pdb70_database_path=FLAGS.pdb70_database_path,
      template_featurizer=template_featurizer,
      n_cpu=FLAGS.n_cpu)

  # Predict structure by pool.
  par = partial(generate_pkl_features_from_fasta,
                output_dir_base=FLAGS.output_dir,
                data_pipeline=data_pipeline)
  pool = Pool(FLAGS.num_workers)
  pool.starmap(par, list(zip(FLAGS.fasta_paths, fasta_names)))
  pool.close()
  pool.join()


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'fasta_paths',
      'output_dir',
      'data_dir',
      'n_cpu'
  ])

  app.run(main)

