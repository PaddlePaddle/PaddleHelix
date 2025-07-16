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

"""Functions for building the input features for the HelixFold model."""

import os
import tempfile
import contextlib
import collections
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from absl import logging

from helixfold.common import FeatureDict
from helixfold.common import residue_constants
from helixfold.data import (
    msa_pairing,
    msa_identifiers,
    parsers,
    templates,
    feature_processing
)
from helixfold.data.tools import (
    hhblits,
    hhsearch,
    hmmsearch,
    jackhmmer
)

TemplateSearcher = Union[hhsearch.HHSearch, hmmsearch.Hmmsearch]


@contextlib.contextmanager
def temp_fasta_file(fasta_str: str):
  """Creates a temporary FASTA file containing the given FASTA string and deletes it after use.
    This function returns a context manager that can be used in a with statement.
  
  Args:
      fasta_str (str): FASTA format string including sequence name and sequence itself.
  
  Returns:
      str: Path to the temporary FASTA file.
  """
  with tempfile.NamedTemporaryFile('w', suffix='.fasta') as fasta_file:
    fasta_file.write(fasta_str)
    fasta_file.seek(0)
    yield fasta_file.name


def int_id_to_str_id(num: int) -> str:
  """Encodes a number as a string, using reverse spreadsheet style naming.

  Args:
    num: A positive integer.

  Returns:
    A string that encodes the positive integer using reverse spreadsheet style,
    naming e.g. 1 = A, 2 = B, ..., 27 = AA, 28 = BA, 29 = CA, ... This is the
    usual way to encode chain IDs in mmCIF files.
  """
  if num <= 0:
    raise ValueError(f'Only positive integers allowed, got {num}.')

  num = num - 1  # 1-based indexing.
  output = []
  while num >= 0:
    output.append(chr(num % 26 + ord('A')))
    num = num // 26 - 1
  return ''.join(output)


def make_sequence_features(
    sequence: str, description: str, num_res: int) -> FeatureDict:
  """Constructs a feature dict of sequence features."""
  features = {}
  features['aatype'] = residue_constants.sequence_to_onehot(
      sequence=sequence,
      mapping=residue_constants.restype_order_with_x,
      map_unknown_to_x=True)
  features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
  features['domain_name'] = np.array([description.encode('utf-8')],
                                     dtype=np.object_)
  features['residue_index'] = np.array(range(num_res), dtype=np.int32)
  features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
  features['sequence'] = np.array([sequence.encode('utf-8')], dtype=np.object_)
  return features


def make_msa_features(msas: Sequence[parsers.Msa], 
                      max_align_depth: Optional[int] = None) -> FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  int_msa = []
  deletion_matrix = []
  species_ids = []
  seen_sequences = set()
  for msa_index, msa in enumerate(msas):
    if not msa:
      raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
    for sequence_index, sequence in enumerate(msa.sequences):
      if sequence in seen_sequences:
        continue
      seen_sequences.add(sequence)
      int_msa.append(
          [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
      deletion_matrix.append(msa.deletion_matrix[sequence_index])
      identifiers = msa_identifiers.get_identifiers(
          msa.descriptions[sequence_index])
      species_ids.append(identifiers.species_id.encode('utf-8'))

  num_res = len(msas[0].sequences[0])
  num_alignments = len(int_msa)
  features = {}
  features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
  features['msa'] = np.array(int_msa, dtype=np.int32)
  features['num_alignments'] = np.array(
      [num_alignments] * num_res, dtype=np.int32)
  features['msa_species_identifiers'] = np.array(species_ids, dtype=np.object_)
  return features


def run_msa_tool(msa_runner,
                 input_fasta_path: str,
                 msa_out_path: str,
                 msa_format: str,
                 use_precomputed_msas: bool,
                 max_sto_sequences: Optional[int] = None
                 ) -> Mapping[str, Any]:
  """Runs an MSA tool, checking if output already exists first."""
  if not use_precomputed_msas or not os.path.exists(msa_out_path):
    if msa_format == 'sto' and max_sto_sequences is not None:
      print('pipeline:',input_fasta_path,max_sto_sequences)
      result = msa_runner.query(input_fasta_path, max_sto_sequences)[0]
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


def convert_monomer_features(
    monomer_features: FeatureDict,
    chain_id: str) -> FeatureDict:
  """Reshapes and modifies monomer features for multimer models."""
  converted = {}
  converted['auth_chain_id'] = np.asarray(chain_id, dtype=np.object_)
  unnecessary_leading_dim_feats = {
      'sequence', 'domain_name', 'num_alignments', 'seq_length'}
  for feature_name, feature in monomer_features.items():
    if feature_name in unnecessary_leading_dim_feats:
      # asarray ensures it's a np.ndarray.
      feature = np.asarray(feature[0], dtype=feature.dtype)
    elif feature_name == 'aatype':
      # The multimer model performs the one-hot operation itself.
      feature = np.argmax(feature, axis=-1).astype(np.int32)
    elif feature_name == 'template_aatype':
      feature = np.argmax(feature, axis=-1).astype(np.int32)
      new_order_list = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
      feature = np.take(new_order_list, feature.astype(np.int32), axis=0)
    # elif feature_name == 'template_all_atom_masks':
    #   feature_name = 'template_all_atom_mask'
    converted[feature_name] = feature
  return converted


def add_assembly_features(
    all_chain_features: MutableMapping[str, FeatureDict],
    ) -> MutableMapping[str, FeatureDict]:
  """Add features to distinguish between chains.
      Note: It may change the chain orders

  Args:
    all_chain_features: A dictionary which maps chain_id to a dictionary of
      features for each chain.

  Returns:
    new_all_chain_features: A dictionary which maps strings of the form
      `<seq_id>_<sym_id>` to the corresponding chain features. E.g. two
      chains from a homodimer would have keys A_1 and A_2. Two chains from a
      heterodimer would have keys A_1 and B_1.
  """
  # Group the chains by sequence
  seq_to_entity_id = {}
  grouped_chains = collections.defaultdict(list)
  grouped_chain_ids = collections.defaultdict(list)
  for chain_id, chain_features in all_chain_features.items():
    seq = str(chain_features['sequence'])
    if seq not in seq_to_entity_id:
      seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
    grouped_chains[seq_to_entity_id[seq]].append(chain_features)
    grouped_chain_ids[seq_to_entity_id[seq]].append(chain_id)

  new_all_chain_features = {}
  chain_id = 1
  new_ordered_chain_ids = []
  for entity_id, group_chain_features in grouped_chains.items():
    for sym_id, chain_features in enumerate(group_chain_features, start=1):
      new_all_chain_features[
          f'{int_id_to_str_id(entity_id)}_{sym_id}'] = chain_features
      seq_length = chain_features['seq_length']
      chain_features['asym_id'] = chain_id * np.ones(seq_length)
      chain_features['sym_id'] = sym_id * np.ones(seq_length)
      chain_features['entity_id'] = entity_id * np.ones(seq_length)
      chain_id += 1
    new_ordered_chain_ids += grouped_chain_ids[entity_id]

  return new_all_chain_features, new_ordered_chain_ids


def process_with_all_chain_features(
            all_chain_features: str,
            return_new_order: bool = False) -> FeatureDict:
  """convert all_chain_features to multimer features."""
  # using orderedcollection
  new_dict = collections.OrderedDict()

  # check if protein is homonmer with unique sequence
  input_seqs = set()
  for chain_id, chain_features in all_chain_features.items():
    input_seqs.add(str(chain_features["sequence"]))
  is_homomer_or_monomer = len(set(input_seqs)) == 1

  for chain_id, chain_features in all_chain_features.items():
    if is_homomer_or_monomer:
      # delete keys with _all_seq if is_homomer_or_monomer
      key_list = list(chain_features.keys())
      for key in key_list:
        if str(key).endswith("_all_seq"): chain_features.pop(key)

    new_dict[chain_id] = convert_monomer_features(chain_features,
                                              chain_id=chain_id)
  new_dict, new_ordered_chain_ids = add_assembly_features(new_dict)

  np_example = feature_processing.pair_and_merge(
      all_chain_features=new_dict)

  if return_new_order:
    return np_example, new_ordered_chain_ids
  return np_example


class DataPipeline:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self,
               jackhmmer_binary_path: str,
               hhblits_binary_path: str,
               uniref90_database_path: str,
               mgnify_database_path: str,
               uniprot_database_path: str,
               bfd_database_path: Optional[str],
               uniclust30_database_path: Optional[str],
               reduced_bfd_database_path: Optional[str],
               template_searcher: TemplateSearcher,
               template_featurizer: templates.TemplateHitFeaturizer,
               use_reduced_bfd: bool,
               bfd_max_hits: Optional[int] = None,
               reduced_bfd_max_hits: int = 5000,
               mgnify_max_hits: int = 5000,
               uniref_max_hits: int = 10000,
               uniprot_max_hits: int = 50000,
               use_precomputed_msas: bool = False,
               max_workers: int = 4):
    """Initializes the data pipeline. Constructs a feature dict for a given FASTA file."""
    self.use_reduced_bfd = use_reduced_bfd
    self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=uniref90_database_path)
    if use_reduced_bfd:
      self.jackhmmer_reduced_bfd_runner = jackhmmer.Jackhmmer(
          binary_path=jackhmmer_binary_path,
          database_path=reduced_bfd_database_path)
    else:
      self.hhblits_bfd_uniclust30_runner = hhblits.HHBlits(
          binary_path=hhblits_binary_path,
          databases=[bfd_database_path, uniclust30_database_path])
    self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=mgnify_database_path)
    self.jackhmmer_uniprot_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=uniprot_database_path)

    self.template_searcher = template_searcher
    self.template_featurizer = template_featurizer

    self.bfd_max_hits = bfd_max_hits
    self.reduced_bfd_max_hits = reduced_bfd_max_hits
    self.mgnify_max_hits = mgnify_max_hits
    self.uniref_max_hits = uniref_max_hits
    self.uniprot_max_hits = uniprot_max_hits

    self.use_precomputed_msas = use_precomputed_msas
    self.max_workers = max_workers

  def _get_all_seq_msa_features_for_pairing(self,
                                uniport_msa: parsers.Msa) -> FeatureDict:
    """Get MSA features for unclustered uniprot, for pairing."""
    msa = uniport_msa.truncate(max_seqs=self.uniprot_max_hits)
    all_seq_features = make_msa_features([msa])
    valid_feats = msa_pairing.MSA_FEATURES + (
        'msa_species_identifiers',
    )
    feats_for_pairing = {f'{k}_all_seq': v for k, v in all_seq_features.items()
             if k in valid_feats}

    return feats_for_pairing

  def _get_template_results(self,
                            input_sequence: str,
                            msa_results: dict,
                            msa_output_dir: str) -> FeatureDict:
    """Get template results from the template searcher."""
    msa_for_templates = msa_results['uniref90']['sto']
    msa_for_templates = parsers.deduplicate_stockholm_msa(msa_for_templates)
    msa_for_templates = parsers.remove_empty_columns_from_stockholm_msa(msa_for_templates)

    if self.template_searcher.input_format == 'sto':
      pdb_templates_result = self.template_searcher.query(msa_for_templates)
    elif self.template_searcher.input_format == 'a3m':
      uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(msa_for_templates)
      pdb_templates_result = self.template_searcher.query(uniref90_msa_as_a3m)
    else:
      raise ValueError('Unrecognized template input format: '
                       f'{self.template_searcher.input_format}')

    pdb_hits_out_path = os.path.join(
        msa_output_dir, f'pdb_hits.{self.template_searcher.output_format}')
    with open(pdb_hits_out_path, 'w') as f:
      f.write(pdb_templates_result)

    pdb_template_hits = self.template_searcher.get_template_hits(
        output_string=pdb_templates_result, input_sequence=input_sequence)

    templates_result = self.template_featurizer.get_templates(
        query_sequence=input_sequence,
        hits=pdb_template_hits,
        query_pdb_code=None,
        query_release_date=None)

    return templates_result

  def set_max_template_hits(self, max_hits: int):
    """Set the max template hits for the template searcher."""
    self.template_searcher.max_hits = max_hits

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
          self.jackhmmer_uniref90_runner,
          input_fasta_path,
          os.path.join(msa_output_dir, 'uniref90_hits.sto'),
          'sto',
          self.use_precomputed_msas,
          self.uniref_max_hits))
    msa_tasks.append((self.jackhmmer_mgnify_runner,                                                                                      
           input_fasta_path,                                                                                           
           os.path.join(msa_output_dir, 'mgnify_hits.sto'),                                                                
           'sto',                                                                                                                               
           self.use_precomputed_msas,
           self.mgnify_max_hits))
    msa_tasks.append((self.jackhmmer_uniprot_runner,
           input_fasta_path,
           os.path.join(msa_output_dir, 'uniprot_hits.sto'),
           'sto',
           self.use_precomputed_msas,
           self.uniprot_max_hits))

    if self.use_reduced_bfd:
      msa_tasks.append((
          self.jackhmmer_reduced_bfd_runner,
          input_fasta_path,
          os.path.join(msa_output_dir, 'reduced_bfd_hits.sto'),
          'sto',
          self.use_precomputed_msas,
          self.reduced_bfd_max_hits))
    else:
      ## TODO: support bfd + uniclust30
      raise NotImplementedError("bfd + uniclust30 not supported yet.")

    msa_results = {}
    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
      futures = {
        executor.submit(run_msa_tool_wrapper, msa_task): msa_task 
            for msa_task in msa_tasks
      }
      for future in as_completed(futures):
        task = futures[future]
        try:
          result = future.result()
          if 'uniref90_hits.sto' in task[2]:
              msa_results['uniref90'] = result
          elif 'mgnify_hits.sto' in task[2]:
              msa_results['mgnify'] = result
          elif 'reduced_bfd_hits.sto' in task[2]:
              msa_results['reduced_bfd'] = result
          elif 'bfd_uniclust30_hits.a3m' in task[2]:
              msa_results['bfd_uniclust30'] = result
          elif 'uniprot_hits.sto' in task[2]:
              msa_results['uniprot'] = result
        except RuntimeError as exc:
          logging.error(f'Task {task} generated an exception : {exc}')

    uniref90_msa = parsers.parse_stockholm(msa_results['uniref90']['sto'])
    mgnify_msa = parsers.parse_stockholm(msa_results['mgnify']['sto'])
    uniprot_msa = parsers.parse_stockholm(msa_results['uniprot']['sto'])
    if self.use_reduced_bfd:
        bfd_msa = parsers.parse_stockholm(msa_results['reduced_bfd']['sto'])
    else:
        raise NotImplementedError("bfd + uniclust30 not supported yet.")

    sequence_features = make_sequence_features(
        sequence=input_sequence,
        description=input_description,
        num_res=num_res)
    msa_features = make_msa_features((uniref90_msa, bfd_msa, mgnify_msa))
    msa_pairing_features = self._get_all_seq_msa_features_for_pairing(uniprot_msa)

    templates_result = self._get_template_results(
        input_sequence=input_sequence,
        msa_results=msa_results,
        msa_output_dir=msa_output_dir)

    logging.info('Uniprot MSA size: %d sequences.', len(uniprot_msa))
    logging.info('Uniref90 MSA size: %d sequences.', len(uniref90_msa))
    logging.info('BFD MSA size: %d sequences.', len(bfd_msa))
    logging.info('MGnify MSA size: %d sequences.', len(mgnify_msa))
    logging.info('Final (deduplicated) MSA size: %d sequences.',
                 msa_features['num_alignments'][0])
    logging.info('Total number of templates (NB: this can include bad '
                 'templates and is later filtered to top 4): %d.',
                 templates_result.features['template_domain_names'].shape[0])

    return {**sequence_features,
            **msa_features,
            **msa_pairing_features,
            **templates_result.features}

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
    return chain_features
