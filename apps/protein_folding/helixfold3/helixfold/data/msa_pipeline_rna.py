#   Copyright (c) 2024 PaddleHelix Authors. All Rights Reserved.
#
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# International License (the "License");  you may not use this file  except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for building the input RNA features for the HelixFold model."""

import os
import contextlib
import tempfile
from typing import Any, Mapping, Optional, Sequence, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from absl import logging

from helixfold.common import residue_constants
from helixfold.common import FeatureDict
from helixfold.data import msa_identifiers
from helixfold.data import parsers
from helixfold.data import msa_pairing
from helixfold.data.tools import hmmer


@contextlib.contextmanager
def temp_fasta_file(fasta_str: str):
  with tempfile.NamedTemporaryFile('w', suffix='.fasta') as fasta_file:
    fasta_file.write(fasta_str)
    fasta_file.seek(0)
    yield fasta_file.name


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


def make_msa_features(msas: Sequence[parsers.Msa], 
                      species_identifer_df: Optional[pd.DataFrame] = None, 
                      max_align_depth: Optional[int] = None) -> FeatureDict:
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
      if species_identifer_df is None:
        identifiers = msa_identifiers.get_identifiers(
          msa.descriptions[sequence_index])
      else:
        identifiers = msa_identifiers.get_identifiers_from_species_df(
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


def run_msa_tool_wrapper(args):
    """
    Helper function for wrapping run_msa_tool function.
    
    Args:
        args (tuple, list): 
          A tuple or list containing the parameters to be passed to the run_msa_tool function.
    
    Returns:
        value from the run_msa_tool function.
    """
    return run_msa_tool(*args)


def final_feature_correction(chain_features: FeatureDict) -> FeatureDict:
    """Final feature correction for RNA MSA.

      Args:
          chain_features (FeatureDict): The chain features to be corrected.
      Returns:
          FeatureDict: The processed chain features.
    """

    chain_features['aatype'] = transform_onehot_features(
      chain_features['aatype'],
      old_token_list=residue_constants.rna_nt_types_with_gap,
      new_token_id_map=residue_constants.HF3_restype_order,
      unmatched_token='N'
    )

    chain_features['msa'] = map_elements_vectorized(
      chain_features['msa'],
      old_token_list=residue_constants.rna_nt_types_with_gap,
      new_token_id_map=residue_constants.HF3_restype_order,
      unmatched_token='N'
    )

    chain_features['msa_all_seq'] = map_elements_vectorized(
      chain_features['msa_all_seq'],
      old_token_list=residue_constants.rna_nt_types_with_gap,
      new_token_id_map=residue_constants.HF3_restype_order,
      unmatched_token='N'
    )

    return chain_features


def transform_onehot_features(onehot_features: np.ndarray, 
                              old_token_list: List[str], 
                              new_token_id_map: Dict,
                              unmatched_token: str) -> np.ndarray:
    """Transform a set of one-hot encoded features based on a new token list.
    
    Args:
        onehot_features (np.ndarray[int]): A set of one-hot encoded features
        old_token_list (List[str]): The old token list, which is a list of strings.
        new_token_id_map (Dict[str]): A dictionary where the keys are old tokens,
                and the values are corresponding new token IDs.
        unmatched_token (str): A string representing the unmatched token. 
                It should have a corresponding entry in the new_token_list.
        
    Returns:
        np.ndarray[int]: The transformed feature set.
    """
    if unmatched_token not in new_token_id_map:
        raise ValueError(f"{unmatched_token} is not found in the new_token_id_map.")
   
    token_array = np.argmax(onehot_features, axis=1)
    transformed_tokens = map_elements_vectorized(
      token_array, 
      old_token_list=old_token_list, 
      new_token_id_map=new_token_id_map, 
      unmatched_token=unmatched_token
    )

    one_hot_matrix = np.eye(len(new_token_id_map))
    transformed_features = one_hot_matrix[transformed_tokens]
    
    return transformed_features


def map_elements_vectorized(array, old_token_list: List[str], 
                              new_token_id_map: Dict,
                              unmatched_token: str) -> np.ndarray:
    """Map each element in the array based on a predefined mapping.
    
    Args:
        array: The original possibly multidimensional np.ndarray[int] array
        old_token_list (List[str]): The old token list, which is a list of strings.
        new_token_id_map (Dict[str]): A dictionary where the keys are old tokens, 
            and the values are corresponding new token IDs.
        unmatched_token (str): A string representing the unmatched token. 
            It should have a corresponding entry in the new_token_list.
    
    Returns:
        The mapped array.
    """

    if unmatched_token not in new_token_id_map:
        raise ValueError(f"{unmatched_token} is not found in the new_token_id_map.")
        
    mapping_dict = {token_id: new_token_id_map[token] if token in new_token_id_map 
                        else new_token_id_map[unmatched_token] 
                 for token_id, token in enumerate(old_token_list)}
    
    unmatched_token_index = new_token_id_map[unmatched_token]
    mapper = np.vectorize(lambda x: mapping_dict.get(x, unmatched_token_index))
    
    mapped_array = mapper(array)
    
    return mapped_array


class DataPipeline:

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
               use_precomputed_msas: bool = False,
               max_workers: int = 3):
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
    self.max_workers = max_workers

    if species_identifer_map_path:
      self.species_identifer_df = pd.read_csv(
        species_identifer_map_path, 
        sep='\t', 
        compression='gzip'
      ) 
    else:
      self.species_identifer_df = None

  def _get_all_seq_msa_features_for_pairing(self,
                                rfam_msa: parsers.Msa) -> FeatureDict:
    """Get RNA MSA features for pairing."""
    all_seq_features = make_msa_features(
      [rfam_msa], 
      self.species_identifer_df, 
      self.total_max_hits
    )
    valid_feats = msa_pairing.MSA_FEATURES + (
        'msa_species_identifiers',
    )
    feats_for_pairing = {f'{k}_all_seq': v for k, v in all_seq_features.items()
             if k in valid_feats}
    return feats_for_pairing

  def monomer_process(self, input_fasta_path: str, msa_output_dir: str) -> FeatureDict:
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

    msa_tasks = []
    msa_tasks.append((
      self.nhmmer_rfam_runner,
      input_fasta_path,
      os.path.join(msa_output_dir, 'rfam_hits.sto'),
      'sto',
      self.use_precomputed_msas,
      self.rfam_max_hits))
    
    if self.nhmmer_rnacentral_runner:
      msa_tasks.append((
        self.nhmmer_rnacentral_runner,
        input_fasta_path,
        os.path.join(msa_output_dir, 'rnacentral_hits.sto'),
        'sto',
        self.use_precomputed_msas,
        self.rnacentral_max_hits))
      
    if self.nhmmer_nt_runner:
      msa_tasks.append((
        self.nhmmer_nt_runner,
        input_fasta_path,
        os.path.join(msa_output_dir, 'nt_hits.sto'),
        'sto',
        self.use_precomputed_msas,
        self.nt_max_hits))
 
    msas = tuple()
    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
      futures = {
        executor.submit(run_msa_tool_wrapper, msa_task): msa_task 
            for msa_task in msa_tasks
      }
      for future in as_completed(futures):
        task = futures[future]
        try:
          result = future.result()
          if 'rfam_hits.sto' in task[2]:
              rfam_msa = parsers.parse_stockholm_RNA(result['sto'], input_sequence)
              logging.info('RFAM MSA size: %d sequences.', len(rfam_msa))
              msas += (rfam_msa,)
          elif 'rnacentral_hits.sto' in task[2]:
              rnacentral_msa = parsers.parse_stockholm_RNA(result['sto'], input_sequence)
              logging.info('RNAcentral MSA size: %d sequences.', len(rnacentral_msa))
              msas += (rnacentral_msa,)
          elif 'nt_hits.sto' in task[2]:
              raise NotImplementedError("msa for nt database not supported yet.")
        except RuntimeError as exc:
          print(f'Task {task} generated an exception : {exc}')
   
    sequence_features = make_sequence_features(
        sequence=input_sequence,
        description=input_description,
        num_res=num_res)

    msa_features = make_msa_features(
      msas=msas,
      species_identifer_df=self.species_identifer_df,
      max_align_depth=self.total_max_hits)
    msa_pairing_features = self._get_all_seq_msa_features_for_pairing(rfam_msa)

    logging.info('Final (deduplicated) MSA size: %d sequences.',
                 msa_features['num_alignments'][0])

    return {**sequence_features, 
            **msa_features, 
            **msa_pairing_features}

  def process(self, chain_id: str, sequence: str,
                  description: str, msa_output_dir: str) -> FeatureDict:
    """Runs the monomer pipeline on a single chain."""
    chain_fasta_str = f'>chain_{chain_id}\n{sequence}\n'
    chain_msa_output_dir = os.path.join(msa_output_dir, chain_id)
    if not os.path.exists(chain_msa_output_dir):
      os.makedirs(chain_msa_output_dir)
    with temp_fasta_file(chain_fasta_str) as chain_fasta_path:
      logging.info('Running monomer pipeline on chain %s: %s',
                   chain_id, description)
      chain_features = self.monomer_process(
          input_fasta_path=chain_fasta_path,
          msa_output_dir=chain_msa_output_dir)
    
    ## final feature correction for RNA MSA
    chain_features = final_feature_correction(chain_features)
    return chain_features
