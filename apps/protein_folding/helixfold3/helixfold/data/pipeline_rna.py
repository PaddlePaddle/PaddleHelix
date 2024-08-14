# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# International License (the "License");  you may not use this file  except
# in compliance with the License. You may obtain a copy of the License at
#
#      http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for building the input RNA features for the HelixFold model."""

import os
from typing import Any, Mapping, Optional, Sequence
from absl import logging
from helixfold.common import residue_constants
from helixfold.data import msa_identifiers_rna
from helixfold.data import parsers
from helixfold.data.tools import hmmer
from helixfold.data.pipeline import FeatureDict

import numpy as np


def make_sequence_features(
    sequence: str, description: str, num_res: int) -> FeatureDict:
  """Constructs a feature dict of sequence features."""
  features = {}
  features['aatype'] = residue_constants.sequence_to_onehot(
      sequence=sequence,
      mapping=residue_constants.rna_nt_type_order,
      map_unknown_to_x=True, 
      x_token='N')
  features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
  features['domain_name'] = np.array([description.encode('utf-8')],
                                     dtype=np.object_)
  features['residue_index'] = np.array(range(num_res), dtype=np.int32)
  features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
  features['sequence'] = np.array([sequence.encode('utf-8')], dtype=np.object_)
  return features

def make_msa_features(msas: Sequence[parsers.Msa], species_identifer_df, max_align_depth: Optional[int]) -> FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  int_msa = []
  deletion_matrix = []
  species_ids = []
  msa_source_indices = []
  seen_sequences = set()
  for msa_index, msa in enumerate(msas):
    if max_align_depth and len(int_msa) >= max_align_depth:
      break

    if not msa:
      raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
    for sequence_index, sequence in enumerate(msa.sequences):
      if sequence in seen_sequences:
        continue
      seen_sequences.add(sequence)
      int_msa.append(
          [residue_constants.RNA_NT_TO_ID[res] for res in sequence])
      deletion_matrix.append(msa.deletion_matrix[sequence_index])
      identifiers = msa_identifiers_rna.get_identifiers(
          msa.descriptions[sequence_index], species_identifer_df)
      species_ids.append(identifiers.species_id.encode('utf-8'))
      msa_source_indices.append(msa_index)

  num_res = len(msas[0].sequences[0])
  num_alignments = len(int_msa)
  features = {}
  features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
  features['msa'] = np.array(int_msa, dtype=np.int32)
  features['num_alignments'] = np.array(
      [num_alignments] * num_res, dtype=np.int32)
  features['msa_species_identifiers'] = np.array(species_ids, dtype=np.object_)
  features['msa_source_indices'] = np.array(msa_source_indices, np.int32)
  return features


def run_msa_tool(msa_runner, input_fasta_path: str, msa_out_path: str,
                 msa_format: str, use_precomputed_msas: bool,
                 max_sto_sequences: Optional[int] = None
                 ) -> Mapping[str, Any]:
  """Runs an MSA tool, checking if output already exists first."""
  if not use_precomputed_msas or not os.path.exists(msa_out_path):
    if msa_format == 'sto' and max_sto_sequences is not None:
      result = msa_runner.query(input_fasta_path, max_sto_sequences)[0]  # pytype: disable=wrong-arg-count
    else:
      result = msa_runner.query(input_fasta_path)[0]
    with open(msa_out_path, 'w') as f:
      f.write(result[msa_format])
  else:
    logging.warning('Reading MSA from file %s', msa_out_path)
    if msa_format == 'sto' and max_sto_sequences is not None:
      precomputed_msa = parsers.truncate_stockholm_msa(
          msa_out_path, max_sto_sequences)
      result = {'sto': precomputed_msa}
    else:
      with open(msa_out_path, 'r') as f:
        result = {msa_format: f.read()}
  return result


class RNADataPipeline:

  """Runs the alignment tools and assembles the input features."""

  def __init__(self,
               hmmer_binary_path: str,
               rfam_database_path: str,
               rnacentral_database_path: Optional[str] = None,
               nt_database_path: Optional[str] = None,
               rfam_max_hits: int = 10000,
               rnacentral_max_hits: int = 10000,
               nt_max_hits: int = 10000,
               total_max_hits: int = 16384,
               species_identifer_map_path: str = None,
               use_precomputed_msas: bool = False):
    """Initializes the data pipeline."""

    self.nhmmer_rfam_runner = hmmer.Nhmmer(
        binary_path=hmmer_binary_path,
        database_path=rfam_database_path)

    if rnacentral_database_path:
      self.nhmmer_rnacentral_runner = hmmer.Nhmmer(
          binary_path=hmmer_binary_path,
          database_path=rnacentral_database_path)
    else:
      self.nhmmer_rnacentral_runner = None

    if nt_database_path:
      self.nhmmer_nt_runner = hmmer.Nhmmer(
          binary_path=hmmer_binary_path,
          database_path=nt_database_path)
    else:
      self.nhmmer_nt_runner = None

    self.rfam_max_hits = rfam_max_hits
    self.rnacentral_max_hits = rnacentral_max_hits
    self.nt_max_hits = nt_max_hits
    self.total_max_hits = total_max_hits
    self.use_precomputed_msas = use_precomputed_msas

    if species_identifer_map_path:
      self.species_identifer_df = pd.read_csv(species_identifer_map_path, sep='\t', compression='gzip') 
    else:
      self.species_identifer_df = None

  def process(self, input_fasta_path: str, msa_output_dir: str) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')

    input_sequence = input_seqs[0]
    input_description = input_descs[0]
    num_res = len(input_sequence)

    rfam_out_path = os.path.join(msa_output_dir, 'rfam_hits.sto')
    nhmmer_rfam_result = run_msa_tool(
        msa_runner=self.nhmmer_rfam_runner,
        input_fasta_path=input_fasta_path,
        msa_out_path=rfam_out_path,
        msa_format='sto',
        use_precomputed_msas=self.use_precomputed_msas,
        max_sto_sequences=self.rfam_max_hits)
    rfam_msa = parsers.parse_stockholm_RNA(nhmmer_rfam_result['sto'], input_sequence)
    logging.info('RFAM MSA size: %d sequences.', len(rfam_msa))
    
    if self.nhmmer_rnacentral_runner:
      rnacentral_out_path = os.path.join(msa_output_dir, 'rnacentral_hits.sto')
      nhmmer_rnacentral_result = run_msa_tool(
          msa_runner=self.nhmmer_rnacentral_runner,
          input_fasta_path=input_fasta_path,
          msa_out_path=rnacentral_out_path,
          msa_format='sto',
          use_precomputed_msas=self.use_precomputed_msas,
          max_sto_sequences=self.rnacentral_max_hits)
      rnacentral_msa = parsers.parse_stockholm_RNA(nhmmer_rnacentral_result['sto'], input_sequence)
      logging.info('RNAcentral MSA size: %d sequences.', len(rnacentral_msa))

      
    if self.nhmmer_nt_runner:
      nt_out_path = os.path.join(msa_output_dir, 'nt_hits.sto')
      nhmmer_nt_runner = run_msa_tool(
          msa_runner=self.nhmmer_nt_runner,
          input_fasta_path=input_fasta_path,
          msa_out_path=nt_out_path,
          msa_format='sto',
          use_precomputed_msas=self.use_precomputed_msas,
          max_sto_sequences=self.nt_max_hits)
      nt_msa = parsers.parse_stockholm_RNA(nhmmer_nt_runner['sto'], input_sequence)
      logging.info('NT MSA size: %d sequences.', len(nt_msa))
   
    sequence_features = make_sequence_features(
        sequence=input_sequence,
        description=input_description,
        num_res=num_res)

    msas = (rfam_msa,)
    if self.nhmmer_rnacentral_runner:
      msas += (rnacentral_msa,)
    if self.nhmmer_nt_runner:
      msas += (nt_msa,)

    msa_features = make_msa_features(msas, self.species_identifer_df, max_align_depth=self.total_max_hits)

    logging.info('Final (deduplicated) MSA size: %d sequences.',
                 msa_features['num_alignments'][0])

    return {**sequence_features, **msa_features}




