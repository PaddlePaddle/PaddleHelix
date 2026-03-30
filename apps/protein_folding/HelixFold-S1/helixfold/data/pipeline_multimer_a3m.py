"""Functions for building the features for the HelixFold multimer model."""

import collections
import contextlib
import copy
import dataclasses
import json
import pickle
import os
import tempfile
from typing import Mapping, MutableMapping, Sequence

from absl import logging
from helixfold.common import protein
from helixfold.common import residue_constants
from helixfold.data import feature_processing_a3m
from helixfold.data import msa_pairing_a3m
from helixfold.data import parsers
from helixfold.data import pipeline_a3m
from helixfold.data.tools import jackhmmer
import numpy as np
import os
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union
from helixfold.data import msa_identifiers
from helixfold.data.tools import hhblits
from helixfold.data.tools import hhsearch
from helixfold.data.tools import hmmsearch
import numpy as np
# Internal import (7716).

# MAX_TEMPLATES = 4
# MSA_CROP_SIZE = 2048
MAX_TEMPLATES = 20
MSA_CROP_SIZE = 16384

FeatureDict = MutableMapping[str, np.ndarray]
TemplateSearcher = Union[hhsearch.HHSearch, hmmsearch.Hmmsearch]


#6--1>aatype 2>residue_index 3>seq_lenth 4,between_segment_residues5,domin_nane6,sequence
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
            raise ValueError(
                f'MSA {msa_index} must contain at least one sequence.')
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
    features['num_alignments'] = np.array([num_alignments] * num_res,
                                          dtype=np.int32)
    features['msa_species_identifiers'] = np.array(species_ids,
                                                   dtype=np.object_)
    return features


@dataclasses.dataclass(frozen=True)
class _FastaChain:
    sequence: str
    description: str


def _make_chain_id_map(
    *,
    sequences: Sequence[str],
    descriptions: Sequence[str],
) -> Mapping[str, _FastaChain]:
    """Makes a mapping from PDB-format chain ID to sequence and description."""
    if len(sequences) != len(descriptions):
        raise ValueError('sequences and descriptions must have equal length. '
                         f'Got {len(sequences)} != {len(descriptions)}.')
    if len(sequences) > protein.PDB_MAX_CHAINS:
        raise ValueError(
            'Cannot process more chains than the PDB format supports. '
            f'Got {len(sequences)} chains.')
    chain_id_map = {}
    for chain_id, sequence, description in zip(protein.PDB_CHAIN_IDS,
                                               sequences, descriptions):
        chain_id_map[chain_id] = _FastaChain(sequence=sequence,
                                             description=description)
    return chain_id_map


@contextlib.contextmanager
def temp_fasta_file(fasta_str: str):
    with tempfile.NamedTemporaryFile('w', suffix='.fasta') as fasta_file:
        fasta_file.write(fasta_str)
        fasta_file.seek(0)
        yield fasta_file.name


def convert_monomer_features(monomer_features: pipeline_a3m.FeatureDict,
                             chain_id: str) -> pipeline_a3m.FeatureDict:
    """Reshapes and modifies monomer features for multimer models."""
    converted = {}
    converted['auth_chain_id'] = np.asarray(chain_id, dtype=np.object_)
    #unnecessary_leading_dim_feats = {
    # 'sequence', 'domain_name', 'num_alignments', 'seq_length'}
    unnecessary_leading_dim_feats = {'sequence', 'domain_name', 'seq_length'}
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
        #elif feature_name == 'template_all_atom_masks':
        #feature_name = 'template_all_atom_mask'
        converted[feature_name] = feature
    return converted


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


def add_assembly_features(
    all_chain_features: MutableMapping[str, pipeline_a3m.FeatureDict],
) -> MutableMapping[str, pipeline_a3m.FeatureDict]:
    """Add features to distinguish between chains.

  Args:
    all_chain_features: A dictionary which maps chain_id to a dictionary of
      features for each chain.

  Returns:
    all_chain_features: A dictionary which maps strings of the form
      `<seq_id>_<sym_id>` to the corresponding chain features. E.g. two
      chains from a homodimer would have keys A_1 and A_2. Two chains from a
      heterodimer would have keys A_1 and B_1.
  """
    # Group the chains by sequence

    seq_to_entity_id = {}
    grouped_chains = collections.defaultdict(list)
    for chain_id, chain_features in all_chain_features.items():
        seq = str(chain_features['sequence'])
        if seq not in seq_to_entity_id:
            seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
        grouped_chains[seq_to_entity_id[seq]].append(chain_features)

    new_all_chain_features = {}
    chain_id = 1
    for entity_id, group_chain_features in grouped_chains.items():
        for sym_id, chain_features in enumerate(group_chain_features, start=1):
            new_all_chain_features[
                f'{int_id_to_str_id(entity_id)}_{sym_id}'] = chain_features
            seq_length = chain_features['seq_length']
            chain_features['asym_id'] = chain_id * np.ones(seq_length)
            chain_features['sym_id'] = sym_id * np.ones(seq_length)
            chain_features['entity_id'] = entity_id * np.ones(seq_length)
            chain_id += 1

    return new_all_chain_features


def pad_msa(np_example, min_num_seq):
    np_example = dict(np_example)
    num_seq = np_example['msa'].shape[0]
    if num_seq < min_num_seq:
        for feat in ('msa', 'deletion_matrix', 'bert_mask', 'msa_mask'):
            np_example[feat] = np.pad(np_example[feat],
                                      ((0, min_num_seq - num_seq), (0, 0)))
        np_example['cluster_bias_mask'] = np.pad(
            np_example['cluster_bias_mask'], ((0, min_num_seq - num_seq), ))
    return np_example


class DataPipeline:
    """Runs the alignment tools and assembles the input features."""

    def __init__(self,
                 monomer_data_pipeline: pipeline_a3m.DataPipeline):
        """Initializes the data pipeline.

    Args:
      monomer_data_pipeline: An instance of pipeline.DataPipeline - that runs
        the data pipeline for the monomer HelixFold system.
    """
        self._monomer_data_pipeline = monomer_data_pipeline

    def _process_single_chain(
            self, chain_id: str, sequence: str, template_df, chain_name: str,
            description: str, msa_output_dir: str,
            is_homomer_or_monomer: bool) -> pipeline_a3m.FeatureDict:
        """Runs the monomer pipeline on a single chain."""
        chain_fasta_str = f'>chain_{chain_id}\n{sequence}\n'
        with temp_fasta_file(chain_fasta_str) as chain_fasta_path:
            logging.info('Running monomer pipeline on chain %s: %s', chain_id,
                         description)
            chain_features = self._monomer_data_pipeline.process(
                input_fasta_path=chain_fasta_path,
                a3m_msa_path=None,
                chain_name=chain_name,
                template_df=template_df)

        return chain_features

    def process(self, input_fasta_path: str, msa_output_dir: str,
                a3m_msa_path: str, template_df) -> pipeline_a3m.FeatureDict:
        """Process a3m and m8 to create features."""
        with open(input_fasta_path) as f:
            input_fasta_str = f.read()
        input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)

        with open(a3m_msa_path, 'r') as f:
            next(f)  # First row is annotation, starting with '#'
            a3m_str = f.read()
        a3m_msa = parsers.parse_a3m(a3m_str)
        msa_features = make_msa_features([a3m_msa])

        msa_features['num_alignments'] = np.asarray(
            msa_features['msa'].shape[0], dtype=np.int32)
        msa_size = msa_features['num_alignments']
        msa_crop_size = np.minimum(msa_size, MSA_CROP_SIZE)

        msa_features['deletion_matrix_int'] = msa_features[
            'deletion_matrix_int'][:msa_crop_size, :]
        msa_features['msa'] = msa_features['msa'][:msa_crop_size, :]
        msa_features['msa_species_identifiers'] = msa_features[
            'msa_species_identifiers'][:msa_crop_size]
        msa_features['num_alignments'] = np.asarray(msa_crop_size,
                                                    dtype=np.int32)

        msa_features['deletion_matrix'] = np.asarray(
            msa_features.pop('deletion_matrix_int'), dtype=np.float32)
        msa_features['deletion_mean'] = np.mean(
            msa_features['deletion_matrix'], axis=0)

        msa_features['cluster_bias_mask'] = np.zeros(
            msa_features['msa'].shape[0])
        msa_features['cluster_bias_mask'][0] = 1

        # Initialize Bert mask with masked out off diagonals.
        # FIXME: shoule use multi-chains, not single merged chains
        msa_masks = [np.ones(msa_features['msa'].shape, dtype=np.float32)]
        msa_mask_block_diag = msa_pairing_a3m.block_diag(*msa_masks,
                                                         pad_value=0)
        msa_features['bert_mask'] = msa_mask_block_diag

        # process_final
        new_order_list = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
        msa_features['msa'] = np.take(new_order_list,
                                      msa_features['msa'],
                                      axis=0)
        msa_features['msa'] = msa_features['msa'].astype(np.int32)

        # We reduce the number of un-paired sequences, by the number of times a
        # sequence from this chain's MSA is included in the paired MSA.  This keeps
        # the MSA size for each chain roughly constant.

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
            print(chain_id, fasta_chain)
            chain_features = self._process_single_chain(
                chain_id=chain_id,
                template_df=template_df,
                chain_name=fasta_chain.description,
                sequence=fasta_chain.sequence,
                description=fasta_chain.description,
                msa_output_dir=msa_output_dir,
                is_homomer_or_monomer=is_homomer_or_monomer)

            chain_features = convert_monomer_features(chain_features,
                                                      chain_id=chain_id)
            all_chain_features[chain_id] = chain_features
            sequence_features[fasta_chain.sequence] = chain_features

            all_chain_features = add_assembly_features(all_chain_features)

        np_example = feature_processing_a3m.pair_and_merge(
            all_chain_features=all_chain_features)
        np_example = {**msa_features, **np_example}

        # NOTE: use all 1 here, in data transformation, assign
        # msa_mask = msa_mask & (msa != gap_idx)
        np_example['msa_mask'] = np.ones_like(np_example['msa'],
                                              dtype=np.float32)

        np_example2 = feature_processing_a3m._filter_features(np_example)
        # Pad MSA to avoid zero-sized extra_msa.
        np_example2 = pad_msa(np_example2, 512)

        return np_example2
