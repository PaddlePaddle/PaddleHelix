"""Functions for building the input features for the HelixFold model."""
import json
import pickle
import os
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union
from absl import logging
from helixfold.common import residue_constants
from helixfold.data import msa_identifiers
from helixfold.data import parsers
from helixfold.data import utils
from helixfold.data.tools import hhblits
from helixfold.data.tools import hhsearch
from helixfold.data.tools import hmmsearch
from helixfold.data.tools import jackhmmer
import numpy as np
import collections
import dataclasses
import itertools
import re
import string
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set


TEMPLATE_CROP_SIZE = 20


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


# Internal import (7716).

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


class DataPipeline:
    """Runs the alignment tools and assembles the input features."""

    def __init__(self,
                 template_featurizer,
                 pdb_seqres_database_path: str):
        """Initializes the data pipeline."""
        self.template_featurizer = template_featurizer

        with open(pdb_seqres_database_path, 'r') as f:
            # desc e.g., >1e0a_B mol:protein length:46  Serine/threonine-protein kinase PAK 1
            self.pdb_seqs, self.pdb_descs = parsers.parse_fasta(f.read())

    def process(
        self, input_fasta_path: str, a3m_msa_path: str, template_df,
        chain_name: str
    ) -> FeatureDict:  #***add 2 args: template_df,chain_name(e.g.,5ezf_A)
        """Runs alignment tools on the input sequence and creates features."""
        # TODO: support monomer mode, i.e. process msa features

        templates = template_df[template_df['query_seq'] == chain_name]

        with open(input_fasta_path) as f:
            input_fasta_str = f.read()
        input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
        if len(input_seqs) != 1:
            raise ValueError(
                f'More than one input sequence found in {input_fasta_path}.')
        input_sequence = input_seqs[0]
        input_description = input_descs[0]
        num_res = len(input_sequence)

        hits = []
        for i in range(len(templates)):
            template = templates.iloc[i]
            _pdb_id, _chain_id = template['target_seq'].split('_')
            template_name = _pdb_id.lower() + '_' + _chain_id
            template_seq = None
            for j, desc in enumerate(self.pdb_descs):
                if desc.split()[0] == template_name:
                    template_seq = self.pdb_seqs[j]
                    template_lenth = int((desc.split()[2]).split(':')[1])

            if template_seq is None:
                # pdb_seqres database is older than pdb70!!!
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
                # Since m8 file order templates by e-score,
                # we can crop templates in this way and keep the best ones
                break

        templates_result = self.template_featurizer.get_templates(
            query_sequence=input_sequence, hits=hits)

        sequence_features = make_sequence_features(
            sequence=input_sequence,
            description=input_description,
            num_res=num_res)

        pdb_id, chain_id = None, None
        id_match = re.match(r'[a-zA-Z\d]{4}_[a-zA-Z0-9.]+', chain_name)
        if id_match:
            pdb_id, chain_id = id_match.group(0).split('_')
            pdb_id = pdb_id.lower()

        if pdb_id is None:
            sequence_features['all_atom_positions'] = np.zeros(
                [num_res, 37, 3], dtype=np.float32)
            sequence_features['all_atom_mask'] = np.zeros(
                [num_res, 37], dtype=np.int32)
            sequence_features['resolution'] = \
                np.array([0.], dtype=np.float32)

        else:
            mmcif_dir = self.template_featurizer._mmcif_dir
            cif_path = os.path.join(mmcif_dir, pdb_id + '.cif')
            cif_gz_path = os.path.join(mmcif_dir, pdb_id[1:3].lower(),
                                       pdb_id.lower() + '.cif.gz')
            if not os.path.exists(cif_path) and os.path.exists(cif_gz_path):
                cif_path = cif_gz_path

            try:
                labels = utils.load_labels(cif_path, pdb_id, chain_id)
                sequence_features['all_atom_positions'] = \
                    labels['all_atom_positions'].astype(np.float32)
                sequence_features['all_atom_mask'] = \
                    labels['all_atom_mask'].astype(np.int32)
                sequence_features['resolution'] = \
                    labels['resolution'].astype(np.float32)

            except Exception as e:
                sequence_features['all_atom_positions'] = np.zeros(
                    [num_res, 37, 3], dtype=np.float32)
                sequence_features['all_atom_mask'] = np.zeros(
                    [num_res, 37], dtype=np.int32)
                sequence_features['resolution'] = \
                    np.array([0.], dtype=np.float32)

        return {**sequence_features, **templates_result.features}
