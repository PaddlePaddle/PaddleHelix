#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http: // www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for building the input features for the AlphaFold model."""

import os
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union
from absl import logging
from helixfold.common import residue_constants
from helixfold.data import msa_identifiers
from helixfold.data import parsers
from helixfold.data import templates
from helixfold.data.tools import hhblits
from helixfold.data.tools import hhsearch
from helixfold.data.tools import hmmsearch
from helixfold.data.tools import jackhmmer
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
# Internal import (7716).
from infer_scripts.tools.antibody_check import antibody_chain_check
FeatureDict = MutableMapping[str, np.ndarray]
TemplateSearcher = Union[hhsearch.HHSearch, hmmsearch.Hmmsearch]


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


def make_msa_features(msas: Sequence[parsers.Msa]) -> FeatureDict:
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


def run_msa_tool(msa_runner, input_fasta_path: str, msa_out_path: str,
                 msa_format: str, use_precomputed_msas: bool,
                 max_sto_sequences: Optional[int] = None
                 ) -> Mapping[str, Any]:
  """Runs an MSA tool, checking if output already exists first."""
  if not use_precomputed_msas or not os.path.exists(msa_out_path):
    if msa_format == 'sto' and max_sto_sequences is not None:
      print('pipeline:',input_fasta_path,max_sto_sequences)
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
    用于包装run_msa_tool函数的帮助程序，以便在使用argparse时可以更轻松地传递参数。
    
    Args:
        args (tuple, list): 一个元组或列表，其中包含要传递给run_msa_tool函数的参数。
    
    Returns:
        int: 返回run_msa_tool函数的返回值。
    """
    return run_msa_tool(*args)


class DataPipeline:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self,
               jackhmmer_binary_path: str,
               hhblits_binary_path: str,
               hhsearch_binary_path: str,
               uniref90_database_path: str,
               mgnify_database_path: str,
               bfd_database_path: Optional[str],
               uniclust30_database_path: Optional[str],
               small_bfd_database_path: Optional[str],
               template_searcher: TemplateSearcher,
               template_featurizer: templates.TemplateHitFeaturizer,
               use_small_bfd: bool,
               mgnify_max_hits: int=501,
               uniref_max_hits: int=10000,
               antiref_max_hits: int=10000,
               ab_mode: bool=False,
               antiref90_database_path: str=None,
               use_precomputed_msas: bool=False,
               use_msa_cache: bool=False,
               use_oas_msa_cache: bool=False,
               use_online_msa_cache: bool=False,
               msa_cache_dir: str=None,
               num_splits_uniref90=60,
               num_splits_antiref90=2,
               num_splits_small_bfd=50,
               num_splits_mgnify=60,
               split_dataset: bool=False,
               _max_works: int=8):
    """Initializes the data pipeline. Constructs a 
      feature dict for a given FASTA file."""
    self._max_works = _max_works
    self._use_small_bfd = use_small_bfd
    self._ab_mode = ab_mode
    self.antiref_max_hits = antiref_max_hits
    self.split_dataset = split_dataset
    if self.split_dataset:
      self.num_splits_uniref90 = num_splits_uniref90
      self.num_splits_antiref90 = num_splits_antiref90
      self.num_splits_small_bfd = num_splits_small_bfd
      self.num_splits_mgnify = num_splits_mgnify

      self.jackhmmer_uniref90_runners = [jackhmmer.Jackhmmer(
          binary_path=jackhmmer_binary_path,
          database_path=uniref90_database_path + '/fasta_part_{}.fasta'.format(i + 1)) \
          for i in range(num_splits_uniref90)]
      self.jackhmmer_uniref90_antibody_runners = [jackhmmer.Jackhmmer(
          binary_path=jackhmmer_binary_path,
          database_path=uniref90_database_path + '/uniref90_Ab.fasta')]


      if ab_mode:
        self.jackhmmer_antiref90_runners = [jackhmmer.Jackhmmer(
            binary_path=jackhmmer_binary_path,
            database_path=antiref90_database_path + '/fasta_part_{}.fasta'.format(i + 1)) \
            for i in range(num_splits_antiref90)]
      
      if use_small_bfd:
        self.jackhmmer_small_bfd_runners = [jackhmmer.Jackhmmer(
            binary_path=jackhmmer_binary_path,
            database_path=small_bfd_database_path + '/fasta_part_{}.fasta'.format(i + 1)) \
            for i in range(num_splits_small_bfd)]
      else:
        pass
        # self.hhblits_bfd_uniclust_runner = hhblits.HHBlits( 
        #     binary_path=hhblits_binary_path,
        #     databases=[bfd_database_path, uniclust30_database_path])

      self.jackhmmer_mgnify_runners = [jackhmmer.Jackhmmer(
          binary_path=jackhmmer_binary_path,
          database_path=mgnify_database_path + '/fasta_part_{}.fasta'.format(i + 1)) \
          for i in range(num_splits_mgnify)]
    else:
        self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
          binary_path=jackhmmer_binary_path,
          database_path=uniref90_database_path)
        if ab_mode:
          self.jackhmmer_antiref90_runner = jackhmmer.Jackhmmer(
            binary_path=jackhmmer_binary_path,
            database_path=antiref90_database_path)
    if use_small_bfd:
      self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
          binary_path=jackhmmer_binary_path,
          database_path=small_bfd_database_path)
    else:
      self.hhblits_bfd_uniclust_runner = hhblits.HHBlits(
          binary_path=hhblits_binary_path,
          databases=[bfd_database_path, uniclust30_database_path])
    self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=mgnify_database_path)
    
    self.template_searcher = template_searcher
    self.template_featurizer = template_featurizer
    self.mgnify_max_hits = mgnify_max_hits
    self.uniref_max_hits = uniref_max_hits
    self.use_precomputed_msas = use_precomputed_msas
    self.use_msa_cache = use_msa_cache
    self.use_oas_msa_cache = use_oas_msa_cache
    self.use_online_msa_cache = use_online_msa_cache
    self.msa_cache_dir = msa_cache_dir

  def _update_max_hits(
      self,
      sequence: str):
      """update max hits of uniprot based on sequence."""
      if len(sequence) > 2000:
          self.uniref_max_hits = self.uniref_max_hits // 6
          self.antiref_max_hits = self.antiref_max_hits // 6

  def update_parameters_by_protein_function(self, sequence: str):
      """update parameters based on sequence function."""
      if antibody_chain_check(sequence) in ['H', 'L']:   
          self.min_hits_for_task = 1 
          self._use_mgnify = False
          self._use_small_bfd = False 
          self._use_uniref90_antibody = True
          self._use_uniref90 = False
      else:
          self.min_hits_for_task = 5
          self._use_mgnify = True
          self._use_small_bfd = True
          self._use_uniref90_antibody = False
          self._use_uniref90 = True

  def process(self, input_fasta_path: str, msa_output_dir: str) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    if self.split_dataset:
      with open(input_fasta_path) as f:
        input_fasta_str = f.read()
      input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
      if len(input_seqs) != 1:
        raise ValueError(
            f'More than one input sequence found in {input_fasta_path}.')
      input_sequence = input_seqs[0]
      input_description = input_descs[0]
      num_res = len(input_sequence)
      self.update_parameters_by_protein_function(input_seqs[0])
      min_hits_for_task = self.min_hits_for_task

      
      msa_tasks = []
      if self._use_uniref90:
        msa_tasks.extend([(self.jackhmmer_uniref90_runners[i],                                                                                              
              input_fasta_path,                                                                                                                        
              os.path.join(msa_output_dir, 'uniref90_hits_{}.sto'.format(i + 1)),                                                                                       
              'sto',                                                                                                                                   
              self.use_precomputed_msas,                                                                                                               
              self.uniref_max_hits // self.num_splits_uniref90) 
              for i in range(self.num_splits_uniref90)])
      if self._use_uniref90_antibody:
        msa_tasks.extend([(self.jackhmmer_uniref90_antibody_runners[0],
            input_fasta_path,
            os.path.join(msa_output_dir, 'uniref90_antibody_hits.sto'),
            'sto',
            self.use_precomputed_msas,
            self.uniref_max_hits)])  


      if self._ab_mode:
        msa_tasks.extend([(                                                                                                                           
            self.jackhmmer_antiref90_runners[i],                                                                                                         
            input_fasta_path,                                                                                                                        
            os.path.join(msa_output_dir, 'antiref90_hits_{}.sto'.format(i + 1)),                                                                                      
            'sto',                                                                                                                                   
            self.use_precomputed_msas,                                                                                                               
            self.antiref_max_hits // self.num_splits_antiref90) 
            for i in range(self.num_splits_antiref90)])                                                                                                                    
      
      if self._use_mgnify:
        msa_tasks.extend([(self.jackhmmer_mgnify_runners[i],  
              input_fasta_path,                                                                                                                       
              os.path.join(msa_output_dir, 'mgnify_hits_{}.sto'.format(i + 1)),                                                                                        
              'sto',                                                                                                                                  
              self.use_precomputed_msas) for i in range(self.num_splits_mgnify)])

      if self._use_small_bfd:                                                                                                                     
        msa_tasks.extend([(                                                                                                                           
            self.jackhmmer_small_bfd_runners[i],                                                                                                         
            input_fasta_path,                                                                                                                        
            os.path.join(msa_output_dir, 'small_bfd_hits_{}.sto'.format(i + 1)),                                                                                      
            'sto',                                                                                                                                   
            self.use_precomputed_msas) 
            for i in range(self.num_splits_small_bfd)])                                                                                                              
      else:  
        pass                                                                                                                                        
        # msa_tasks.append((                                                                                                                           
        #     self.hhblits_bfd_uniclust_runner,                                                                                                        
        #     input_fasta_path,                                                                                                                        
        #     os.path.join(msa_output_dir, 'bfd_uniclust_hits.a3m'),                                                                                   
        #     'a3m',                                                                                                                                   
        #     self.use_precomputed_msas))
      #import pdb; pdb.set_trace()
      #run_msa_tool_wrapper(msa_tasks[0])
      msa_results = {'uniref90': [], 'antiref90': [], 'mgnify': [], 'small_bfd': [], 'bfd_uniclust': []}
      with ProcessPoolExecutor(max_workers=self._max_works) as executor:
        futures = {executor.submit(run_msa_tool_wrapper, msa_task): msa_task for msa_task in msa_tasks}
        for future in as_completed(futures):
          task = futures[future]
          try:
            result = future.result()
            if 'uniref90' in task[2]:                                                                                                       
              msa_results['uniref90'].append(result)# incloud uniref90_antibody_hits                                                                                                      
            elif 'antiref90_hits' in task[2]:                                                                                                    
              msa_results['antiref90'].append(result)                                                                                                    
            elif 'mgnify_hits' in task[2]:                                                                                                       
              msa_results['mgnify'].append(result)                                                                                                       
            elif 'small_bfd_hits' in task[2]:                                                                                                    
              msa_results['small_bfd'].append(result)                                                                                                    
            elif 'bfd_uniclust_hits' in task[2]:                                                                                                 
              msa_results['bfd_uniclust'].append(result)
          except Exception as exc:
            print(f'Task {task} generated an exception : {exc}')
      MAX_MSA_FOR_TEMPLATE = 2
      msa_for_templates = msa_results['uniref90'][0]['sto']
      #msa_for_templates = parsers.truncate_stockholm_msa(msa_for_templates, MAX_MSA_FOR_TEMPLATE)
      msa_for_templates = parsers.deduplicate_stockholm_msa(msa_for_templates, num_max_msa=MAX_MSA_FOR_TEMPLATE)
      msa_for_templates = parsers.remove_empty_columns_from_stockholm_msa(
          msa_for_templates)

      #MAX_MSA_FOR_TEMPLATE = 10
      #msa_for_templates = '\n'.join(msa_for_templates.splitlines()[:MAX_MSA_FOR_TEMPLATE]) + '\n'      

      logging.info('[Start search template hits.')
      import time; t0_template = time.time()

      if self.template_searcher.input_format == 'sto':
        pdb_templates_result = self.template_searcher.query(msa_for_templates)
      elif self.template_searcher.input_format == 'a3m':
        uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(msa_for_templates)
        pdb_templates_result = self.template_searcher.query(uniref90_msa_as_a3m)
      else:
        raise ValueError('Unrecognized template input format: '
                        f'{self.template_searcher.input_format}')

      logging.info('[Finished search template hits. %s s', str(time.time() - t0_template))

      pdb_hits_out_path = os.path.join(
          msa_output_dir, f'pdb_hits.{self.template_searcher.output_format}')
      with open(pdb_hits_out_path, 'w') as f:
        f.write(pdb_templates_result)

      uniref90_msas = [parsers.parse_stockholm(msa_results['uniref90'][i]['sto']) \
                      for i in range(len(msa_results['uniref90']))]
      if self._use_mgnify:
          mgnify_msas = [parsers.parse_stockholm(msa_results['mgnify'][i]['sto']) \
                    for i in range(len(msa_results['mgnify']))]
      else:
          mgnify_msas = []
      if self._ab_mode:
        antiref90_msas = [parsers.parse_stockholm(msa_results['antiref90'][i]['sto']) \
                         for i in range(len(msa_results['antiref90']))]
      else:
        antiref90_msas = []
      pdb_template_hits = self.template_searcher.get_template_hits(
          output_string=pdb_templates_result, input_sequence=input_sequence)

      logging.info('[Finished get_template_hits: %s s', str(time.time() - t0_template))
      logging.info('[Finished get_template_hits: %d hits.', len(msa_for_templates.splitlines()))
      if self._use_small_bfd:
        bfd_msas = [parsers.parse_stockholm(msa_results['small_bfd'][i]['sto']) \
                   for i in range(len(msa_results['small_bfd']))]                                                                       
      else:
        bfd_msas = []
        
      
      #templates_result = self.template_featurizer.get_templates(
      templates_result = self.template_featurizer.get_templates_block_parallel(
          query_sequence=input_sequence,
          hits=pdb_template_hits,
          query_pdb_code=None,
          query_release_date=None,
          min_hits_for_task=min_hits_for_task)

      logging.info('[Finished template_featurizer.get_templates. %s s', \
         str(time.time() - t0_template))

      sequence_features = make_sequence_features(
          sequence=input_sequence,
          description=input_description,
          num_res=num_res)
      if self._ab_mode:
        msa_features = make_msa_features(uniref90_msas + antiref90_msas + bfd_msas + mgnify_msas) 
        logging.info('Antiref90 MSA block size: %d sequences.', len(antiref90_msas))
      else:
        msa_features = make_msa_features(uniref90_msas + bfd_msas + mgnify_msas)
      logging.info('Uniref90 MSA block size: %d sequences.', len(uniref90_msas))
      logging.info('BFD MSA block size: %d sequences.', len(bfd_msas))
      logging.info('MGnify MSA block size: %d sequences.', len(mgnify_msas))
      logging.info('Final (deduplicated) MSA block size: %d sequences.',
                  msa_features['num_alignments'][0])
      logging.info('Total number of templates (NB: this can include bad '
                  'templates and is later filtered to top 4): %d.',
                  templates_result.features['template_domain_names'].shape[0])
    else:
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
      msa_tasks.append((self.jackhmmer_uniref90_runner,
            input_fasta_path,
            os.path.join(msa_output_dir, 'uniref90_hits.sto'),
            'sto',
            self.use_precomputed_msas,
            self.uniref_max_hits))
      if self._ab_mode:
          msa_tasks.append((                                                                                                                           
              self.jackhmmer_antiref90_runner,                                                                                                         
              input_fasta_path,                                                                                                                        
              os.path.join(msa_output_dir, 'antiref90_hits.sto'),                                                                                      
              'sto',                                                                                                                                   
              self.use_precomputed_msas,                                                                                                               
              self.antiref_max_hits                                                                                                                    
            ))

      msa_tasks.append((self.jackhmmer_mgnify_runner,                                                                                                     
            input_fasta_path,                                                                                                                    
            os.path.join(msa_output_dir, 'mgnify_hits.sto'),                                                                                  
            'sto',                                                                                                                               
            self.use_precomputed_msas))
      if self._use_small_bfd:
        msa_tasks.append((
            self.jackhmmer_small_bfd_runner,
            input_fasta_path,
            os.path.join(msa_output_dir, 'small_bfd_hits.sto'),
            'sto',
            self.use_precomputed_msas))
      else:
        msa_tasks.append((
            self.hhblits_bfd_uniclust_runner,
            input_fasta_path,
            os.path.join(msa_output_dir, 'bfd_uniclust_hits.a3m'),
            'a3m',
            self.use_precomputed_msas))

      msa_results = {}  
      with ProcessPoolExecutor(max_workers=self._max_works) as executor:
        futures = {executor.submit(run_msa_tool_wrapper, msa_task): msa_task for msa_task in msa_tasks}
        for future in as_completed(futures):
          task = futures[future]
          try:
            result = future.result()
            if 'uniref90_hits.sto' in task[2]:
                msa_results['uniref90'] = result
            elif 'antiref90_hits.sto' in task[2]:                                                                                                    
                msa_results['antiref90'] = result                                                                                                    
            elif 'mgnify_hits.sto' in task[2]:
                msa_results['mgnify'] = result
            elif 'small_bfd_hits.sto' in task[2]:
                msa_results['small_bfd'] = result
            elif 'bfd_uniclust_hits.a3m' in task[2]:
                msa_results['bfd_uniclust'] = result
          except Exception as exc:
            print(f'Task {task} generated an exception : {exc}')

      msa_for_templates = msa_results['uniref90']['sto']
      msa_for_templates = parsers.deduplicate_stockholm_msa(msa_for_templates)
      msa_for_templates = parsers.remove_empty_columns_from_stockholm_msa(
        msa_for_templates)

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

      uniref90_msa = parsers.parse_stockholm(msa_results['uniref90']['sto'])
      mgnify_msa = parsers.parse_stockholm(msa_results['mgnify']['sto'])
      if self._ab_mode:
        antiref90_msa = parsers.parse_stockholm(msa_results['antiref90']['sto'])
      pdb_template_hits = self.template_searcher.get_template_hits(
          output_string=pdb_templates_result, input_sequence=input_sequence)

      if self._use_small_bfd:
          bfd_msa = parsers.parse_stockholm(msa_results['small_bfd']['sto'])
      else:
          bfd_msa = parsers.parse_a3m(msa_results['bfd_uniclust']['a3m'])
          bfd_hhr_out_path = os.path.join(msa_output_dir, 'bfd_uniclust_hits.hhr')
          with open(bfd_hhr_out_path, 'w') as f:
            f.write(msa_results['bfd_uniclust']['hhr'])

      templates_result = self.template_featurizer.get_templates(
          query_sequence=input_sequence,
          hits=pdb_template_hits,
          query_pdb_code=None,
          query_release_date=None)

      sequence_features = make_sequence_features(
          sequence=input_sequence,
          description=input_description,
          num_res=num_res)
      if self._ab_mode:
          msa_features = make_msa_features((uniref90_msa, antiref90_msa, bfd_msa, mgnify_msa)) 
          logging.info('Antiref90 MSA size: %d sequences.', len(antiref90_msa))
      else:
          msa_features = make_msa_features((uniref90_msa, bfd_msa, mgnify_msa))

      logging.info('Uniref90 MSA size: %d sequences.', len(uniref90_msa))
      logging.info('BFD MSA size: %d sequences.', len(bfd_msa))
      logging.info('MGnify MSA size: %d sequences.', len(mgnify_msa))
      logging.info('Final (deduplicated) MSA size: %d sequences.',
                  msa_features['num_alignments'][0])
      logging.info('Total number of templates (NB: this can include bad '
                  'templates and is later filtered to top 4): %d.',
                  templates_result.features['template_domain_names'].shape[0])

    return {**sequence_features, **msa_features, **templates_result.features}
