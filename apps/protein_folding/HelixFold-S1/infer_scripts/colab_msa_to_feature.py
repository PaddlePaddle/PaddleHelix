"""Functions for building the input msa/template features from colabfold_search"""
import json
import pickle
import os
import copy
import pandas as pd
import numpy as np
import collections
import dataclasses
import pathlib
import gzip
import subprocess
import glob
from typing import MutableMapping, Optional, Sequence, Union, List
from absl import logging

from helixfold.common import residue_constants
from helixfold.data import msa_identifiers
from helixfold.data.msa_identifiers import Identifiers
from helixfold.data import feature_processing
from helixfold.data import parsers
from helixfold.data.tools import hhsearch, hmmsearch
from helixfold.data import pipeline, templates
from helixfold.data.feature_processing import _filter_features
from helixfold.data.pipeline_multimer import (
		temp_fasta_file,
		_make_chain_id_map, 
		convert_monomer_features,
		add_assembly_features,
	)

MAX_TEMPLATES = 4
TEMPLATE_CROP_SIZE = 20
MSA_CROP_SIZE = 16384

MSA_SPECIES_FEATURES = ('msa', 'msa_mask', 'deletion_matrix', 'deletion_matrix_int', 
						'msa_species_identifiers')
COLABFOLD_M8_FEAT_LIST = [
	'query_seq', 'target_seq', 'identity', 'align_lenth', 'mismatch_num',
	'gap_num', 'query_start', 'query_end', 'target_start', 'target_end',
	'E_value', 'bit_score'
]
FeatureDict = MutableMapping[str, np.ndarray]
TemplateSearcher = Union[hhsearch.HHSearch, hmmsearch.Hmmsearch]

# copied from parsers
def _get_indices(sequence: str, start: int) -> List[int]:
	"""Returns indices for non-gap/insert residues starting at the given index."""
	indices = []
	counter = start
	for symbol in sequence:
		# Skip gaps but add a placeholder so that the alignment is preserved.
		if symbol == '-':
			indices.append(-1)
		# Skip deleted residues, but increase the counter.
		elif symbol.islower():
			counter += 1
		# Normal aligned residue. Increase the counter and append to indices.
		else:
			indices.append(counter)
			counter += 1
	return indices


@dataclasses.dataclass(frozen=True)
class HitMetadata:
	pdb_id: str
	chain: str
	start: int
	end: int
	length: int
	text: str


@dataclasses.dataclass(frozen=True)
class TemplateHit:
	"""Class representing a template hit."""
	index: int
	name: str
	aligned_cols: int
	sum_probs: Optional[float]
	query: str
	hit_sequence: str
	indices_query: List[int]
	indices_hit: List[int]
	mode: Optional[str] = None


def colab_uniref30_get_identifiers(description: str, 
							species_identifer_df: pd.DataFrame) -> Identifiers:
	"""
		Computes extra MSA features from the description, only support UniRef30

		Example:
		>>> species_identifer_df:
			Taxon Id        Mnemonic        Scientific name
				7       AZOCA   Azorhizobium caulinodans
				9       9GAMM   Buchnera aphidicola
				11      9CELL   Cellulomonas gilvus
  	"""
	identifer = ''
	if species_identifer_df is None:
		return Identifiers(species_id=identifer)

	try:
		word_list = description.split('\t')[0]
		if 'UniRef100' in word_list:
			taxonID = int(word_list.split('-')[-1])
			matching_rows = species_identifer_df.loc[species_identifer_df['Taxon Id'] == taxonID, 'Mnemonic']  
			if not matching_rows.empty:  
				identifer = matching_rows.iloc[0]
	except:
		pass

	return Identifiers(
		species_id=identifer)


def make_sequence_features(sequence: str, description: str,
						   num_res: int) -> FeatureDict:
	"""Constructs a feature dict of sequence features."""
	features = {}
	features['aatype'] = residue_constants.sequence_to_onehot(
		sequence=sequence,
		mapping=residue_constants.restype_order_with_x,
		map_unknown_to_x=True)
	features['between_segment_residues'] = np.zeros((num_res, ),
													dtype=np.int32)
	features['domain_name'] = np.array([description.encode('utf-8')],
									   dtype=np.object_)
	features['residue_index'] = np.array(range(num_res), dtype=np.int32)
	features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
	features['sequence'] = np.array([sequence.encode('utf-8')],
									dtype=np.object_)
	return features


def make_msa_features(msas: Sequence[parsers.Msa], species_identifier_df: pd.DataFrame) -> FeatureDict:
	"""Constructs a feature dict of MSA features."""
	if not msas:
		raise ValueError('At least one MSA must be provided.')

	int_msa = []
	deletion_matrix = []
	species_ids = []
	seen_sequences = set()
	for msa_index, msa in enumerate(msas):
		if not msa:
			raise ValueError(
				f'MSA {msa_index} must contain at least one sequence.')
		for sequence_index, sequence in enumerate(msa.sequences):
			if sequence in seen_sequences:
				continue
			seen_sequences.add(sequence)
			int_msa.append(
				[residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
			deletion_matrix.append(msa.deletion_matrix[sequence_index])
			identifiers = colab_uniref30_get_identifiers(
					description=msa.descriptions[sequence_index],
					species_identifer_df=species_identifier_df)
			# identifiers = msa_identifiers.get_identifiers(
			# 	msa.descriptions[sequence_index])
			species_ids.append(identifiers.species_id.encode('utf-8'))

	num_res = len(msas[0].sequences[0])
	num_alignments = len(int_msa)
	features = {}
	features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
	features['msa'] = np.array(int_msa, dtype=np.int32)
	features['num_alignments'] = np.array([num_alignments] * num_res,
										  dtype=np.int32)
	features['msa_species_identifiers'] = np.array(species_ids,
												   dtype=np.object_)
	return features


def crop_and_further_msa_features(msa_features, max_msa_depth=16384) -> FeatureDict:
	msa_size = msa_features['msa'].shape[0]
	msa_crop_size = np.minimum(msa_size, max_msa_depth)

	msa_features['deletion_matrix_int'] = msa_features[
		'deletion_matrix_int'][:msa_crop_size, :]
	msa_features['msa'] = msa_features['msa'][:msa_crop_size, :]
	msa_features['msa_species_identifiers'] = msa_features[
		'msa_species_identifiers'][:msa_crop_size]
	msa_features['num_alignments'] = np.asarray([msa_crop_size], dtype=np.int32)

	msa_features['deletion_matrix'] = np.asarray(
					msa_features.pop('deletion_matrix_int'), dtype=np.float32)
	msa_features['deletion_mean'] = np.mean(msa_features['deletion_matrix'], axis=0)

	msa_features['cluster_bias_mask'] = np.zeros(msa_features['msa'].shape[0])
	msa_features['cluster_bias_mask'][0] = 1

	return msa_features


def postprocess_a3m_dimer(use_gzip: bool, output_dir: str, feat: FeatureDict, 
							fasta: str, a3m: str):
	"""
		Convert the a3m from colabfold when homomer is assigned.
		To keep the order with chain_id_map.json; that is input fasta order.
	"""
	fasta = pathlib.Path(fasta)
	a3m = pathlib.Path(a3m)
	output_dir = pathlib.Path(output_dir)
	chain_json = output_dir / 'chain_id_map.json'
	assert chain_json.exists() and a3m.exists()

	# import pdb;pdb.set_trace()
	with open(chain_json, 'r') as f:
		chains = json.load(f)

	msa_segs = dict()
	with open(a3m, 'r') as f:
		next(f)
		next(f)
		a3m_seq = f.readline().strip()

	if feat['msa'].shape[1] == feat['seq_mask'].shape[0]:
		return

	for ch in chains.values():
		seq = ch['sequence']
		i = a3m_seq.index(seq)
		j = i + len(seq)
		msa_segs[seq] = (i, j)

	for k in ['msa', 'deletion_matrix', 'msa_mask']:
		segs = []
		for c, v in chains.items():
			i, j = msa_segs[v['sequence']]
			segs.append(np.copy(feat[k][:, i:j]))

		new_feat_k = np.concatenate(segs, axis=1)
		feat[k] = new_feat_k

	for k in ['deletion_mean']:
		segs = []
		for c, v in chains.items():
			i, j = msa_segs[v['sequence']]
			segs.append(np.copy(feat[k][i:j]))

		new_feat_k = np.concatenate(segs, axis=0)
		feat[k] = new_feat_k
		#print(k, new_feat_k.shape)

	assert feat['msa'].shape[1] == feat['seq_mask'].shape[0]
	if use_gzip:
		with gzip.open(output_dir / 'features.pkl.gz', 'wb') as f:
			pickle.dump(feat, f)
	else:
		with open(output_dir / 'features.pkl', 'wb') as f:
			pickle.dump(feat, f)

	print(f'Update {output_dir}')


class DataPipelineColabSearchProtein:
	"""
		DataPipeline for colabfold search output.
		adapt to a3m, m8 results.
	"""
	def __init__(self,
				colabfold_database_path: str,
				colabfold_search_binary_path: str,
				mmseqs_binary_path: str,
				template_featurizer: templates.HmmsearchHitFeaturizer,
				pdb_seqres_database_path: str, 
				species_identifier_path: str = None):
		"""
			template_featurizer: templates.HmmsearchHitFeaturizer, 
			pdb_seqres_database_path: path to fasta library for template hit.
			species_identifier_path: path to species identifier file.
		"""
		self.colabfold_search_binary_path = colabfold_search_binary_path
		self.mmseqs_binary_path = mmseqs_binary_path
		self.colabfold_database_path = colabfold_database_path
		if species_identifier_path:
			self.species_identifer_df = pd.read_csv(
											species_identifier_path, 
											sep='\t', compression='gzip') 
			# print(f">>>> [DEBUG] {self.species_identifer_df.head()}")
		else:
			self.species_identifer_df = None

		self.template_featurizer = template_featurizer
		with open(pdb_seqres_database_path, 'r', encoding='utf-8') as f:
			# desc e.g., >1e0a_B mol:protein length:46  Serine/threonine-protein kinase PAK 1
			self.pdb_seqs, self.pdb_descs = parsers.parse_fasta(f.read())

	def _mock_msa_template(self, seq, result_dir):
		"""
			Mock a3m and m8 files.
		"""
		result_dir = pathlib.Path(result_dir)
		a3m_file = result_dir / 'mock_0.a3m'
		m8_file = result_dir / 'mock_0.m8'		
		with open(a3m_file, 'w') as fw:
			fw.write("# 101, 1\n")
			fw.write(">101\n")
			fw.write(f"{seq}\n")
		
		with open(m8_file, 'w') as f:
			f.write("")

		return a3m_file, m8_file

	def _dense_a3m_string(self, a3m_path, outpath):
		"""
			Convert a3m to dense format if multimer is inputed.
		"""
		with open(a3m_path, 'r') as f:
			a3m_lines = f.readlines()

		dense_a3m = ''
		N = 101
		paired_part =  ''
		unpaired_part = collections.defaultdict(list)

		## The first line is the annotation line, start with N = 101.
		unique_length = {str(N + idx):int(i) 
			for idx, i in enumerate(a3m_lines[0][1:].split('\t')[0].split(','))}
		
		updated_paired = False
		unpaired_N = None
		for line in a3m_lines[1:]:
			if line.startswith('>'):
				titles = line.strip()[1:].split('\t')
				if len(titles) > 1 and all([ t in unique_length for t in titles]):
					updated_paired = True
				elif len(titles) == 1 and titles[0] in unique_length:
					updated_paired = False
					unpaired_N = titles[0]

			if updated_paired:
				paired_part += line
			else:
				assert unpaired_N is not None, \
					("Error in find unpaired N when dense_a3m_string.")
				unpaired_part[unpaired_N].append(line)

		## Remove padding msa in unpaired part.
		def _get_unpaired_msa_points(ms, unpaired_a3m_line, unique_length):
			start_idx, end_idx = 0, 0
			ms_len = unique_length[ms]
			start_idx = 0 if ms == '101' \
				else sum([unique_length[str(t)] for t in range(int(ms) - 1, N - 1, -1)])
			
			end_idx = start_idx
			count = 0
			for i, c in enumerate(unpaired_a3m_line[start_idx:]):
				if not c.islower(): ## align and gap should be count.
					count += 1
				if count == ms_len:
					end_idx = start_idx + i + 1
					break

			return start_idx, end_idx

		_unpaired_part = collections.defaultdict(list)
		assert unpaired_part.keys() == unique_length.keys(), \
			(f"Error in find correct order when dense_a3m_string. Got {unpaired_part.keys()} != {unique_length.keys()}")
		for ms, lines in unpaired_part.items():
			for line in  lines:
				if line.startswith('>'):
					_unpaired_part[ms].append(line)
				else:
					start_idx, end_idx = _get_unpaired_msa_points(ms, line, unique_length)
					_unpaired_part[ms].append(line[start_idx: end_idx])
		unpaired_part = _unpaired_part 

		## Padding and Merge unpaired part.
		## padding 
		max_msa_len = max([len(v) for v in unpaired_part.values()])
		_unpaired_part = collections.defaultdict(list)
		for ms, lines in unpaired_part.items():
			pad_len = max_msa_len - len(lines)
			if pad_len > 0:
				_pad_content = [f">{ms}\tpadding\n", f"{'-' * unique_length[ms]}"] * (pad_len // 2)
				lines.extend(_pad_content)
				unpaired_part[ms] = lines
		## merge unpaired
		assert len(set([len(v) for v in unpaired_part.values()])) == 1
		unpaired_part_merge = ''
		for idx, pack_line in enumerate(zip(*unpaired_part.values())):
			_tmp_line = '\t'.join(pack_line)
			if idx == 0:
				_tmp_line += '\tunpaired'
			
			if not _tmp_line.startswith('>'):
				_tmp_line = _tmp_line.replace('\t', '')
			_tmp_line = _tmp_line.replace('\n', '') + '\n'
			unpaired_part_merge += _tmp_line

		## Merge paired and unpaired.
		dense_a3m = a3m_lines[0] + paired_part + unpaired_part_merge
		with open(outpath, 'w') as f:
			f.write(dense_a3m)

	def _process_single_chain(self, chain_id: str, sequence: str, 
									template_m8_path:str, 
									chain_name: str,
									description: str, 
									msa_output_dir: str,
									is_homomer_or_monomer: bool) -> FeatureDict:
		"""Runs the monomer pipeline on a single chain."""
		chain_fasta_str = f'>chain_{chain_id}\n{sequence}\n'
		with temp_fasta_file(chain_fasta_str) as chain_fasta_path:
			logging.info('Running monomer pipeline on chain %s: %s', chain_id, description)
			chain_features = self.process_chain(
						input_fasta_path=chain_fasta_path,
						a3m_msa_path=None,
						chain_name=chain_name,
						template_m8_path=template_m8_path)

		return chain_features

	def process_chain(self, input_fasta_path: str, a3m_msa_path: str, 
							template_m8_path: str, chain_name: str) -> FeatureDict:  
		""" 
			From colabfold_search's output:
			- Runs alignment tools on the input sequence and creates features for single chain;
				- a3m_msa_path: *.a3m, path from colabfold, msa results
				- template_m8_path: *.m8, path from colabfold, template-hit results.
				- chain_name: the description from query seqs; should be adapt to colabfold_search results.
					example: such as 101, 102, 103.
		"""
		if a3m_msa_path is not None:
			with open(a3m_msa_path, 'r') as f:
				next(f)  # First row is annotation, starting with '#'
				a3m_str = f.read()
			a3m_msa = parsers.parse_a3m(a3m_str)
			msa_features = make_msa_features([a3m_msa], self.species_identifer_df)
			all_seq_msa_features = {f'{k}_all_seq': v for k, v in msa_features.items()
										if k in MSA_SPECIES_FEATURES} # update species feature.
			msa_features.update(all_seq_msa_features)
		else:
			msa_features = {}

		template_df = pd.read_csv(template_m8_path, sep='\t',
								header=None, names=COLABFOLD_M8_FEAT_LIST, index_col=False)
		## FIXME, Use the 101, 102, raw name from colabfold_search results. 
		templates = template_df[template_df['query_seq'] == int(chain_name)]

		with open(input_fasta_path) as f:
			input_fasta_str = f.read()
		input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
		if len(input_seqs) != 1:
			raise ValueError(f'More than one input sequence found in {input_fasta_path}.')
		input_sequence = input_seqs[0]
		input_description = input_descs[0]
		num_res = len(input_sequence)

		hits = []
		for i in range(len(templates)):
			template = templates.iloc[i]
			_pdb_id, _chain_id = template['target_seq'].split('_')
			template_name = _pdb_id.lower() + '_' + _chain_id
			# print('>>>', template_name)
			# import pdb;pdb.set_trace()
			template_seq = None
			for j, desc in enumerate(self.pdb_descs):
				if desc.split()[0] == template_name:
					template_seq = self.pdb_seqs[j]
					# for oldder pdb_seqs.txt
					# template_lenth = int((desc.split()[2]).split(':')[1]) 

			if template_seq is None:
				continue

			# abstract templatehit attributions
			index = i + 1
			name = template['target_seq']
			aligned_cols = int(template['align_lenth'])
			query_sequence = input_sequence
			indices_query = _get_indices(query_sequence, start=0)
			hit_sequence_list = ['-'] * len(input_sequence)
			query_start, query_end = int(template['query_start']), int(
				template['query_end'])
			hit_start, hit_end = int(template['target_start']), int(
				template['target_end'])
			# ignore buggy hit in m8
			if hit_end <= hit_start:
				continue
			hit_lenth = hit_end - hit_start + 1
			hit_sequence_list[query_start:query_start + hit_lenth -
								1] = template_seq[hit_start:hit_end]
			hit_sequence = ''
			for aa in hit_sequence_list:
				hit_sequence += aa
			indices_hit = _get_indices(hit_sequence, start=hit_start - 1)
			hit = TemplateHit(index=index,
								name=name,
								aligned_cols=aligned_cols,
								sum_probs=None,
								query=query_sequence,
								hit_sequence=hit_sequence.upper(),
								indices_query=indices_query,
								indices_hit=indices_hit)
			hits.append(hit)

			if len(hits) >= TEMPLATE_CROP_SIZE:
				# *.m8 file order templates by e-score,
				break
		
		templates_result = self.template_featurizer.get_templates(
			query_sequence=input_sequence, hits=hits,
			query_pdb_code=None,
			query_release_date=None)

		sequence_features = make_sequence_features(
			sequence=input_sequence,
			description=input_description,
			num_res=num_res)
		
		## same seq features and template features.
		return {**sequence_features, **msa_features, **templates_result.features}

	def process_multimer(self, input_fasta_path: str, msa_output_dir: str,
								a3m_msa_path: str, template_m8_path: str):
		"""
			Process a3m and m8 to create features generated from colabfold_search.
		"""
		with open(input_fasta_path) as f:
			input_fasta_str = f.read()
			input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)

		with open(a3m_msa_path, 'r') as f:
			next(f)  # First row is annotation, starting with '#'
			a3m_str = f.read()
		
		## NOTE: Since colabfold_search is already pairing with species_identifiers, 
		## We make msa features from a3m string directly.
		a3m_msa = parsers.parse_a3m(a3m_str)
		msa_features = make_msa_features([a3m_msa])
		msa_features = crop_and_further_msa_features(msa_features, max_msa_depth=MSA_CROP_SIZE)

		## MSA HHBLITS AATYPE to standard residue type.
		new_order_list = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
		msa_features['msa'] = np.take(new_order_list, msa_features['msa'], axis=0)
		msa_features['msa'] = msa_features['msa'].astype(np.int32)

		# e.g., chin_id_map={'A':Fasta_chain,'B':Fasta_chian,...}
		chain_id_map = _make_chain_id_map(sequences=input_seqs,
										descriptions=input_descs)
		chain_id_map_path = os.path.join(msa_output_dir, 'chain_id_map.json')
		with open(chain_id_map_path, 'w') as f:
			chain_id_map_dict = {
				chain_id: dataclasses.asdict(fasta_chain)
				for chain_id, fasta_chain in chain_id_map.items()
			}
			json.dump(chain_id_map_dict, f, indent=4, sort_keys=True)

		all_chain_features = {}
		sequence_features = {}
		is_homomer_or_monomer = len(set(input_seqs)) == 1
		for chain_id, fasta_chain in chain_id_map.items():
			if fasta_chain.sequence in sequence_features:
				all_chain_features[chain_id] = copy.deepcopy(
					sequence_features[fasta_chain.sequence])
				continue
			# print(chain_id, fasta_chain)
			chain_features = self._process_single_chain(
				chain_id=chain_id,
				template_m8_path=template_m8_path,
				chain_name=fasta_chain.description,
				sequence=fasta_chain.sequence,
				description=fasta_chain.description,
				msa_output_dir=msa_output_dir,
				is_homomer_or_monomer=is_homomer_or_monomer)

			chain_features = convert_monomer_features(chain_features,
													chain_id=chain_id)
			all_chain_features[chain_id] = chain_features
			sequence_features[fasta_chain.sequence] = chain_features

		all_chain_features, new_ordered_chain_ids = add_assembly_features(all_chain_features)

		## final multimer features paring.
		np_example = feature_processing.pair_and_merge(
							all_chain_features=all_chain_features,
							use_colabfold_search_feats=True)
		np_example = {**msa_features, **np_example}

		np_example['msa_mask'] = np.ones_like(np_example['msa'], dtype=np.float32)
		np_example2 = _filter_features(np_example)

		return np_example2

	def colabfold_search(self, input_fasta_path: str, msa_output_dir: str):
		assert os.path.exists(self.colabfold_search_binary_path)
		assert os.path.exists(self.mmseqs_binary_path)
		assert os.path.exists(self.colabfold_database_path)

		os.makedirs(msa_output_dir, exist_ok=True)
		format_fasta = 'format_complex.fasta'
		format_fasta_path = os.path.join(msa_output_dir, format_fasta)
		with open(format_fasta_path, 'w') as fw:
			with open(input_fasta_path, 'r', encoding='utf-8') as f1:
				pdb_seqs, pdb_descs = parsers.parse_fasta(f1.read())
			
			pdb_descs_format = "::".join(pdb_descs)
			pdb_seqs_format = ":".join(pdb_seqs)
			fw.write(f">{pdb_descs_format}\n")
			fw.write(f"{pdb_seqs_format}\n")
		
		cmd = f"{self.colabfold_search_binary_path} " \
					"--use-env 1 " \
					"--use-templates 1 " \
					"--db-load-mode 2 " \
					f"--mmseqs {self.mmseqs_binary_path} " \
					"--db2 pdb100_230517 " \
					f"--max-accept {MSA_CROP_SIZE} " \
					"--threads 4 " \
					f"{format_fasta_path} " \
					f"{self.colabfold_database_path} " \
					f"{msa_output_dir}"
		logging.info(f"[Colabfold cmd]: {cmd}")
		ret = subprocess.run(cmd, shell=True, check=False, capture_output=True)
		if ret.returncode != 0:
			logging.error('Colabfold search failed, please check the error log below: {}'.format(msa_output_dir))
			with open(os.path.join(msa_output_dir, "ERROR.log"), "a") as f:
				print(ret.stdout.decode(), file=f, flush=True)
			
			if len(pdb_seqs) == 1:
				logging.warning(f"Mock msa and templates: {format_fasta_path}")
				mock_a3m_path, mock_m8_path = self._mock_msa_template(pdb_seqs_format, msa_output_dir)
				return mock_a3m_path, mock_m8_path
			else:
				raise RuntimeError("Colabfold search failed; But multi chains mode are not supported to mock msa and templates.")
		else:
			logging.info(f"Colabfold search successed. output saved at: {msa_output_dir}")
		
		## Now only support single *.a3m and *.m8.
		a3m_path = glob.glob(os.path.join(msa_output_dir, '*.a3m'))
		m8_path = glob.glob(os.path.join(msa_output_dir, '*.m8'))
		assert len(a3m_path) == 1 and len(m8_path) == 1
		a3m_path, m8_path = a3m_path[0], m8_path[0]

		## post process when only one chain is inputed.
		if len(pdb_seqs) == 1:
			## add annotation, starting with '#'
			with open(a3m_path, 'r+') as f:
				content = f.read()
				f.seek(0, 0)
				f.write('# 101, 1\n' + content)
		else:
			raise NotImplementedError("Multiple chains mode are not supported in colabfold_search any more.")
			## unpaired msa should be dense paired and saved in dense_a3m_path
			dense_a3m_path = os.path.join(msa_output_dir, 'dense_' + os.path.basename(a3m_path))
			self._dense_a3m_string(a3m_path, dense_a3m_path)
			a3m_path = dense_a3m_path

		return a3m_path, m8_path


if __name__ == '__main__':

	colabfold_database_path = './'
	colabfold_search_binary_path = './'
	mmseqs_binary_path = './'
	species_indenti_path = '/root/paddlejob/workspace/output/yexianbin/AF3_online/dev/af3_infer_dev/data/taxonomy_AND_taxonomies_with_1_uniprot_2024_07_14.tsv.gz'

	template_mmcif_dir = '/root/paddlejob/workspace/env_run/data_assembly/mmcif/'
	pdb_seqres_database_path = '/root/paddlejob/workspace/output/yexianbin/AF3_online/dev/af3_infer_dev/data/pdb_seqres.txt'

	MAX_TEMPLATE_HITS = 20
	max_template_date = '2024-09-30' # af3 infer cutoff date.
	kalign_binary_path = '/root/miniconda3/bin/kalign'
	template_featurizer = templates.HmmsearchHitFeaturizer(
		mmcif_dir=template_mmcif_dir,  #
		max_template_date=max_template_date,
		max_hits=MAX_TEMPLATE_HITS,
		kalign_binary_path=kalign_binary_path,
		release_dates_path=None,
		obsolete_pdbs_path=None)  #
	
	pipeline = DataPipelineColabSearchProtein(colabfold_database_path=colabfold_database_path,
											colabfold_search_binary_path=colabfold_search_binary_path,
											mmseqs_binary_path=mmseqs_binary_path,
											template_featurizer=template_featurizer, 
											pdb_seqres_database_path=pdb_seqres_database_path,
											species_identifier_path=species_indenti_path)
	

	file_root = '/root/paddlejob/workspace/output/yexianbin/AF3_online/dev/af3_infer_dev/data/colab_to_single_demo'
	## multi_chain; but no homonomer
	input_fasta_path = f'{file_root}/format_complex.fasta'
	a3m_msa_path = f'{file_root}/0_updated.a3m'
	template_m8_path = f'{file_root}/pdb100_230517.m8'
	msa_output_dir = f'{file_root}/output_debug/{os.path.basename(input_fasta_path)}/msas/'

	os.makedirs(msa_output_dir, exist_ok=True)
	feats = pipeline.process_chain(input_fasta_path=input_fasta_path, a3m_msa_path=a3m_msa_path, 
									template_m8_path=template_m8_path, chain_name='101')
	with open(os.path.join(msa_output_dir, 'update_features.pkl'), 'wb') as f:
		pickle.dump(feats, f)

	## test pair and merge.
	from helixfold.data import pipeline_multimer
	chain_group_feats = {}
	with open('./data/colab_to_single_demo/output_debug/format_complex.fasta/msas/update_features.pkl', 'rb') as f:
		chain_group_feats['A'] = pickle.load(f)
	with open('./data/colab_to_single_demo/old_process_features.pkl', 'rb') as f:
		chain_group_feats['B'] = pickle.load(f)
		old_letter = b'G'
		new_letter = b'X'
		chain_group_feats['B']['sequence'][0] = chain_group_feats['B']['sequence'][0].replace(old_letter, new_letter)
	# import pdb; pdb.set_trace()
	msa_feats = pipeline_multimer.process_with_all_chain_features(chain_group_feats)
	print(chain_group_feats['A']['msa'].shape, chain_group_feats['B']['msa'].shape,)
	print(msa_feats['msa'].shape)


	# import pdb; pdb.set_trace()

	# # postprocess_a3m_dimer(use_gzip=False, output_dir=msa_output_dir, feat=feats, 
	# # 				fasta=input_fasta_path, a3m=a3m_msa_path)


	# ## multi_chain; but no homonomer
	# input_fasta_path = './data/colab_demo/6e3k_2chain_shortone.fasta'
	# a3m_msa_path = './data/colab_demo/6e3k_2chain_shortone.a3m'
	# template_m8_path = './data/colab_demo/6e3k_2chain_shortone.m8'
	# msa_output_dir = f'./data/colab_demo/output/{os.path.basename(input_fasta_path)}/msas/'
	# os.makedirs(msa_output_dir, exist_ok=True)

	# pipeline = DataPipelineColabSearchProtein(
	# 	colabfold_database_path=None,
	# 	colabfold_search_binary_path=None,
	# 	mmseqs_binary_path=None,
	# 	template_featurizer=None,
	# 	pdb_seqres_database_path=None
	# )

	# dense_a3m_path = os.path.join(msa_output_dir, '6e3k_2chain_shortone_dense.a3m')
	# pipeline._dense_a3m_string(a3m_msa_path, dense_a3m_path)
	# a3m_msa_path = dense_a3m_path

	# feats = pipeline.process_multimer(input_fasta_path=input_fasta_path,
	# 							msa_output_dir=msa_output_dir,
	# 							a3m_msa_path=a3m_msa_path,
	# 							template_m8_path=template_m8_path)