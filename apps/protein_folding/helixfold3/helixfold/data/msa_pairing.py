
"""Pairing logic for multimer data pipeline."""

import collections
from typing import Dict, Iterable, List, Sequence, Union

from helixfold.common import residue_constants
from helixfold.data import pipeline
from helixfold.data.pipeline_hybrid import HF2_PADDING_DIM, \
                                          feats_pad_and_concatenate
import numpy as np
import pandas as pd

MSA_GAP_IDX = residue_constants.restypes_with_x_and_gap.index('-')

MSA_PAD_VALUES = {'msa_all_seq': MSA_GAP_IDX,
                  'msa_mask_all_seq': 1,
                  'deletion_matrix_all_seq': 0,
                  'deletion_matrix_int_all_seq': 0,
                  'msa': MSA_GAP_IDX,
                  'msa_mask': 1,
                  'deletion_matrix': 0,
                  'deletion_matrix_int': 0}

MSA_FEATURES = ('msa', 'msa_mask', 'deletion_matrix', 'deletion_matrix_int')
SEQ_FEATURES = ('residue_index', 'aatype', 'all_atom_positions',
                'all_atom_mask', 'seq_mask', 'between_segment_residues',
                'has_alt_locations', 'has_hetatoms', 'asym_id', 'entity_id',
                'sym_id', 'num_sym', 'entity_mask', 'deletion_mean',
                'prediction_atom_mask',
                'literature_positions', 'atom_indices_to_group_indices',
                'rigid_group_default_frame')
TEMPLATE_FEATURES = ('template_aatype', 'template_all_atom_positions',
                     'template_all_atom_masks')
CHAIN_FEATURES = ('num_alignments', 'seq_length')

IS_FILTER_KEYS = set(['msa_source_indices'])

def create_paired_features(
    chains: Iterable[pipeline.FeatureDict]) ->  List[pipeline.FeatureDict]:
  """Returns the original chains with paired NUM_SEQ features.

  Args:
    chains:  A list of feature dictionaries for each chain.

  Returns:
    A list of feature dictionaries with sequence features including only
    rows to be paired.
  """
  chains = list(chains)
  chain_keys = chains[0].keys()

  if len(chains) < 2:
    return chains
  else:
    updated_chains = []
    paired_chains_to_paired_row_indices = pair_sequences(chains)
    paired_rows = reorder_paired_rows(
        paired_chains_to_paired_row_indices)

    for chain_num, chain in enumerate(chains):
      new_chain = {k: v for k, v in chain.items() if '_all_seq' not in k}
      for feature_name in chain_keys:
        if feature_name.endswith('_all_seq'):
          feats_padded = pad_features(chain[feature_name], feature_name)
          new_chain[feature_name] = feats_padded[paired_rows[:, chain_num]]
      new_chain['num_alignments_all_seq'] = np.asarray(
          len(paired_rows[:, chain_num]))
      updated_chains.append(new_chain)
    return updated_chains


def pad_features(feature: np.ndarray, feature_name: str) -> np.ndarray:
  """Add a 'padding' row at the end of the features list.

  The padding row will be selected as a 'paired' row in the case of partial
  alignment - for the chain that doesn't have paired alignment.

  Args:
    feature: The feature to be padded.
    feature_name: The name of the feature to be padded.

  Returns:
    The feature with an additional padding row.
  """
  assert feature.dtype != np.dtype(np.string_)
  if feature_name in ('msa_all_seq', 'msa_mask_all_seq',
                      'deletion_matrix_all_seq', 'deletion_matrix_int_all_seq'):
    num_res = feature.shape[1]
    padding = MSA_PAD_VALUES[feature_name] * np.ones([1, num_res],
                                                     feature.dtype)
  elif feature_name == 'msa_species_identifiers_all_seq':
    padding = [b'']
  else:
    return feature
  feats_padded = np.concatenate([feature, padding], axis=0)
  return feats_padded


def _make_msa_df(chain_features: pipeline.FeatureDict) -> pd.DataFrame:
  """Makes dataframe with msa features needed for msa pairing."""
  chain_msa = chain_features['msa_all_seq']
  query_seq = chain_msa[0]
  per_seq_similarity = np.sum(
      query_seq[None] == chain_msa, axis=-1) / float(len(query_seq))
  per_seq_gap = np.sum(chain_msa == 21, axis=-1) / float(len(query_seq))
  msa_df = pd.DataFrame({
      'msa_species_identifiers':
          chain_features['msa_species_identifiers_all_seq'],
      'msa_row':
          np.arange(len(
              chain_features['msa_species_identifiers_all_seq'])),
      'msa_similarity': per_seq_similarity,
      'gap': per_seq_gap
  })
  return msa_df


def _create_species_dict(msa_df: pd.DataFrame) -> Dict[bytes, pd.DataFrame]:
  """Creates mapping from species to msa dataframe of that species."""
  species_lookup = {}
  for species, species_df in msa_df.groupby('msa_species_identifiers'):
    species_lookup[species] = species_df
  return species_lookup


def _match_rows_by_sequence_similarity(this_species_msa_dfs: List[pd.DataFrame]
                                       ) -> List[List[int]]:
  """Finds MSA sequence pairings across chains based on sequence similarity.

  Each chain's MSA sequences are first sorted by their sequence similarity to
  their respective target sequence. The sequences are then paired, starting
  from the sequences most similar to their target sequence.

  Args:
    this_species_msa_dfs: a list of dataframes containing MSA features for
      sequences for a specific species.

  Returns:
   A list of lists, each containing M indices corresponding to paired MSA rows,
   where M is the number of chains.
  """
  all_paired_msa_rows = []

  num_seqs = [len(species_df) for species_df in this_species_msa_dfs
              if species_df is not None]
  take_num_seqs = np.min(num_seqs)

  sort_by_similarity = (
      lambda x: x.sort_values('msa_similarity', axis=0, ascending=False))

  for species_df in this_species_msa_dfs:
    if species_df is not None:
      species_df_sorted = sort_by_similarity(species_df)
      msa_rows = species_df_sorted.msa_row.iloc[:take_num_seqs].values
    else:
      msa_rows = [-1] * take_num_seqs  # take the last 'padding' row
    all_paired_msa_rows.append(msa_rows)
  all_paired_msa_rows = list(np.array(all_paired_msa_rows).transpose())
  return all_paired_msa_rows


def pair_sequences(examples: List[pipeline.FeatureDict]
                   ) -> Dict[int, np.ndarray]:
  """Returns indices for paired MSA sequences across chains."""

  num_examples = len(examples)

  all_chain_species_dict = []
  common_species = set()
  for chain_features in examples:
    msa_df = _make_msa_df(chain_features)
    species_dict = _create_species_dict(msa_df)
    all_chain_species_dict.append(species_dict)
    common_species.update(set(species_dict))

  common_species = sorted(common_species)
  common_species.remove(b'')  # Remove target sequence species.

  all_paired_msa_rows = [np.zeros(len(examples), int)]
  all_paired_msa_rows_dict = {k: [] for k in range(num_examples)}
  all_paired_msa_rows_dict[num_examples] = [np.zeros(len(examples), int)]

  for species in common_species:
    if not species:
      continue
    this_species_msa_dfs = []
    species_dfs_present = 0
    for species_dict in all_chain_species_dict:
      if species in species_dict:
        this_species_msa_dfs.append(species_dict[species])
        species_dfs_present += 1
      else:
        this_species_msa_dfs.append(None)

    # Skip species that are present in only one chain.
    if species_dfs_present <= 1:
      continue
    
    # Skip species that have too many sequences. Why?
    if np.any(
        np.array([len(species_df) for species_df in
                  this_species_msa_dfs if
                  isinstance(species_df, pd.DataFrame)]) > 600):
      continue

    paired_msa_rows = _match_rows_by_sequence_similarity(this_species_msa_dfs)
    all_paired_msa_rows.extend(paired_msa_rows)
    all_paired_msa_rows_dict[species_dfs_present].extend(paired_msa_rows)
  all_paired_msa_rows_dict = {
      num_examples: np.array(paired_msa_rows) for
      num_examples, paired_msa_rows in all_paired_msa_rows_dict.items()
  }
  return all_paired_msa_rows_dict


def reorder_paired_rows(all_paired_msa_rows_dict: Dict[int, np.ndarray]
                        ) -> np.ndarray:
  """Creates a list of indices of paired MSA rows across chains.

  Args:
    all_paired_msa_rows_dict: a mapping from the number of paired chains to the
      paired indices.

  Returns:
    a list of lists, each containing indices of paired MSA rows across chains.
    The paired-index lists are ordered by:
      1) the number of chains in the paired alignment, i.e, all-chain pairings
         will come first.
      2) e-values: the product of the sequence indexes of all chains in the pairing.
         The lower is the e-value, the more similar is the paring sequences to the target.
  """
  all_paired_msa_rows = []

  for num_pairings in sorted(all_paired_msa_rows_dict, reverse=True):
    paired_rows = all_paired_msa_rows_dict[num_pairings]
    paired_rows_product = abs(np.array([np.prod(rows) for rows in paired_rows]))
    paired_rows_sort_index = np.argsort(paired_rows_product)
    all_paired_msa_rows.extend(paired_rows[paired_rows_sort_index])

  return np.array(all_paired_msa_rows)


def dense_pad(*arrs: np.ndarray, padding_dim: Union[tuple, list], 
                        padding_value: float = 0.0) -> np.ndarray:
  """dense padding for unpaired MSA and deletions matrices."""
  merged_arrs = feats_pad_and_concatenate(arrs, padding_dim, value=padding_value)
  return merged_arrs


def _correct_post_merged_feats(
    np_example: pipeline.FeatureDict,
    np_chains_list: Sequence[pipeline.FeatureDict],
    pair_msa_sequences: bool) -> pipeline.FeatureDict:
  """Adds features that need to be computed/recomputed post merging."""

  np_example['seq_length'] = np.asarray(np_example['aatype'].shape[0],
                                        dtype=np.int32)
  np_example['num_alignments'] = np.asarray(np_example['msa'].shape[0],
                                            dtype=np.int32)

  if not pair_msa_sequences:
    # Generate a bias that is 1 for the first row of every block in the
    # block diagonal MSA - i.e. make sure the cluster stack always includes
    # the query sequences for each chain (since the first row is the query
    # sequence).
    cluster_bias_masks = []
    for chain in np_chains_list:
      mask = np.zeros(chain['msa'].shape[0])
      mask[0] = 1
      cluster_bias_masks.append(mask)
    np_example['cluster_bias_mask'] = np.concatenate(cluster_bias_masks)

  else:
    np_example['cluster_bias_mask'] = np.zeros(np_example['msa'].shape[0])
    np_example['cluster_bias_mask'][0] = 1

  return np_example


def _pad_templates(chains: Sequence[pipeline.FeatureDict],
                   max_templates: int) -> Sequence[pipeline.FeatureDict]:
  """For each chain pad the number of templates to a fixed size.

  Args:
    chains: A list of protein chains.
    max_templates: Each chain will be padded to have this many templates.

  Returns:
    The list of chains, updated to have template features padded to
    max_templates.
  """
  for chain in chains:
    for k, v in chain.items():
      if k in TEMPLATE_FEATURES:
        padding = np.zeros_like(v.shape)
        padding[0] = max_templates - v.shape[0]
        padding = [(0, p) for p in padding]
        chain[k] = np.pad(v, padding, mode='constant')
  return chains


def _merge_features_from_multiple_chains(
    chains: Sequence[pipeline.FeatureDict],
    pair_msa_sequences: bool) -> pipeline.FeatureDict:
  """Merge features from multiple chains.

  Args:
    chains: A list of feature dictionaries that we want to merge.
    pair_msa_sequences: Whether to concatenate MSA features along the
      num_res dimension (if True), or to block diagonalize them (if False).

  Returns:
    A feature dictionary for the merged example.
  """
  all_merge_keys = set()
  for chain in chains:
    all_merge_keys |= set(chain.keys())

  merged_example = {}
  for feature_name in all_merge_keys:
    if feature_name in IS_FILTER_KEYS:
      continue

    # NOTE: For RNA, the template related features are not present. For these features, we just cancatenate that for proteins.
    feats = [x[feature_name] for x in chains if feature_name in x]
    feature_name_split = feature_name.split('_all_seq')[0]
    if feature_name_split in MSA_FEATURES:
      if pair_msa_sequences or '_all_seq' in feature_name:
        merged_example[feature_name] = np.concatenate(feats, axis=1)
      else:
        # merged_example[feature_name] = block_diag(
        #     *feats, pad_value=MSA_PAD_VALUES[feature_name])
        ## NOTE: HF3 use the dense_pad function instead of block_diag for MSA_FEATURES.
        merged_example[feature_name] = dense_pad(
              *feats, padding_dim=HF2_PADDING_DIM[feature_name_split],
              padding_value=MSA_PAD_VALUES[feature_name])
    elif feature_name_split in SEQ_FEATURES:
      merged_example[feature_name] = np.concatenate(feats, axis=0)
    elif feature_name_split in TEMPLATE_FEATURES:
      merged_example[feature_name] = np.concatenate(feats, axis=1)
    elif feature_name_split in CHAIN_FEATURES:
      merged_example[feature_name] = np.array(sum(x for x in feats)).astype(np.int32)
    else:
      merged_example[feature_name] = feats[0]
  return merged_example


def _merge_homomers_dense_msa(
    chains: Iterable[pipeline.FeatureDict]) -> Sequence[pipeline.FeatureDict]:
  """Merge all identical chains, making the resulting MSA dense.

  Args:
    chains: An iterable of features for each chain.

  Returns:
    A list of feature dictionaries.  All features with the same entity_id
    will be merged - MSA features will be concatenated along the num_res
    dimension - making them dense.
  """
  entity_chains = collections.defaultdict(list)
  for chain in chains:
    entity_id = chain['entity_id'][0]
    entity_chains[entity_id].append(chain)

  grouped_chains = []
  for entity_id in sorted(entity_chains):
    chains = entity_chains[entity_id]
    grouped_chains.append(chains)
  chains = [
      _merge_features_from_multiple_chains(chains, pair_msa_sequences=True)
      for chains in grouped_chains]
  return chains


def _concatenate_paired_and_unpaired_features(
    example: pipeline.FeatureDict) -> pipeline.FeatureDict:
  """Merges paired and block-diagonalised features."""
  features = MSA_FEATURES
  for feature_name in features:
    if feature_name in example:
      feat = example[feature_name]
      feat_all_seq = example[feature_name + '_all_seq']
      merged_feat = np.concatenate([feat_all_seq, feat], axis=0)
      example[feature_name] = merged_feat
  example['num_alignments'] = np.array(example['msa'].shape[0],
                                       dtype=np.int32)
  return example


def merge_chain_features(np_chains_list: List[pipeline.FeatureDict],
                         pair_msa_sequences: bool,
                         max_templates: int) -> pipeline.FeatureDict:
  """Merges features for multiple chains to single FeatureDict.

  Args:
    np_chains_list: List of FeatureDicts for each chain.
    pair_msa_sequences: Whether to merge paired MSAs.
    max_templates: The maximum number of templates to include.

  Returns:
    Single FeatureDict for entire complex.
  """
  np_chains_list = _pad_templates(
      np_chains_list, max_templates=max_templates)
  np_chains_list = _merge_homomers_dense_msa(np_chains_list)
  # HF3 change: paired and unpaired MSA features will be always concatenated in dense manner.
  # It is different from HF2, where unpaired MSA features were always block-diagonalised; paired MSA
  # features were concatenated.
  np_example = _merge_features_from_multiple_chains(
      np_chains_list, pair_msa_sequences=False) # in HF2, pair_msa_sequences=False
  if pair_msa_sequences:
    np_example = _concatenate_paired_and_unpaired_features(np_example)
  np_example = _correct_post_merged_feats(
      np_example=np_example,
      np_chains_list=np_chains_list,
      pair_msa_sequences=pair_msa_sequences)

  return np_example


def deduplicate_unpaired_sequences(
    np_chains: List[pipeline.FeatureDict]) -> List[pipeline.FeatureDict]:
  """Removes unpaired sequences which duplicate a paired sequence."""

  feature_names = np_chains[0].keys()
  msa_features = MSA_FEATURES

  for chain in np_chains:
    # Convert the msa_all_seq numpy array to a tuple for hashing.
    sequence_set = set(tuple(s) for s in chain['msa_all_seq'])
    keep_rows = []
    # Go through unpaired MSA seqs and remove any rows that correspond to the
    # sequences that are already present in the paired MSA.
    for row_num, seq in enumerate(chain['msa']):
      if tuple(seq) not in sequence_set:
        keep_rows.append(row_num)
    for feature_name in feature_names:
      if feature_name in msa_features:
        chain[feature_name] = chain[feature_name][keep_rows]
    chain['num_alignments'] = np.array(chain['msa'].shape[0], dtype=np.int32)
  return np_chains
