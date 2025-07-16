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

"""Functions for building the features for the HelixFold-3 inference pipeline."""
import copy
import os
import time, gzip, pickle
from typing import Mapping, List, Union, Tuple
import numpy as np
import pathlib
import logging
import dataclasses
from concurrent.futures import ProcessPoolExecutor, as_completed

from helixfold.common import residue_constants
from helixfold.data import parsers
from helixfold.data import msa_pipeline_protein
from helixfold.data import pipeline_aa
from helixfold.data import utils
from helixfold.data.feature_processing import MAX_TEMPLATES as MAX_PROTEIN_TEMPLATE_HITS
from helixfold.data.feature_processing import MSA_CROP_SIZE as MAX_MSA_DEPTH
from infer_scripts.entity_bean import EntityBean


logger = logging.getLogger(__file__)

POLYMER_STANDARD_RESI_ATOMS = residue_constants.residue_atoms
STRING_FEATURES = ['all_chain_ids', 'all_ccd_ids','all_atom_ids', 'chain_ids',
                  'release_date', 'label_ccd_ids', 'label_atom_ids', 'atom_perm_str']
MAX_MSA_WORKERS = 1

@dataclasses.dataclass(frozen=True)
class MSATaskMeta:
  """Class representing a MSATaskMeta.

    data_pipeline (DataPipeline): DataPipeline object for processing individual chains.
    chain_id (str): Chain ID.
    seq (str): Chain sequence.
    desc (str): Chain description.
    msa_output_dir (PathLike): MSA output directory.
    features_pkl (PathLike): Feature file path.
    task_type (Literal): Task type, protein or rna
    processed_features (Mapping[str, any]): Processed features.
  """
  data_pipeline: msa_pipeline_protein.DataPipeline
  chain_id: str
  seq: str
  desc: str
  msa_output_dir: str
  features_pkl: str
  task_type: str
  processed_features: Mapping[str, any] = dataclasses.field(default_factory=dict)
  
  def __post_init__(self):
    assert self.task_type in ['protein', 'rna'],\
        (f"task_type: {self.task_type} not in ['protein', 'rna']")

  @property
  def features(self):
    return self.processed_features


def load_ccd_dict(ccd_preprocessed_path: str) -> Mapping[str, any]:
  """Load ccd preprocessed dict.

      Args:
          ccd_preprocessed_path: str

      Returns:
          Mapping[str, any]: CCD preprocessed dict.
  """
  assert os.path.exists(ccd_preprocessed_path),\
            (f'[CCD] ccd_preprocessed_path: {ccd_preprocessed_path} not exist.')
  st_1 = time.time()
  if 'pkl.gz' in ccd_preprocessed_path:
      with gzip.open(ccd_preprocessed_path, "rb") as fp:
          ccd_preprocessed_dict = pickle.load(fp)
  elif '.pkl' in ccd_preprocessed_path:
      with open(ccd_preprocessed_path, "rb") as fp:
          ccd_preprocessed_dict = pickle.load(fp)
  logger.info(f'[CCD] Load ccd dataset done. use: {time.time()-st_1}s; '\
                  f'Total length: {len(ccd_preprocessed_dict)}')
  
  return ccd_preprocessed_dict


def crop_msa(feat: Mapping[str, any], max_msa_depth: int = 16384) -> Mapping[str, any]:
  """ pad msa and generate msa_mask. """
  msa = feat['msa']
  deletion_mat = feat['deletion_matrix']
  has_deletion = feat['has_deletion']
  deletion_value_mat = feat['deletion_value']
  msa_mask = np.ones_like(feat['msa']).astype('float32') # [msa_depth, n_token]

  msa_depth, _ = msa_mask.shape
  if msa_depth > max_msa_depth:
      msa_mask = msa_mask[: max_msa_depth, :]
      msa = msa[: max_msa_depth, :]
      deletion_mat = deletion_mat[: max_msa_depth, :]
      has_deletion = has_deletion[: max_msa_depth, :]
      deletion_value_mat = deletion_value_mat[: max_msa_depth, :]
  
  feat['msa'] = msa.astype('int32')
  feat['msa_mask'] = msa_mask
  feat['deletion_matrix'] = deletion_mat
  feat['has_deletion'] = has_deletion
  feat['deletion_value'] = deletion_value_mat

  return feat


def get_padding_restype(ccd_id: str, 
                        ccd_preprocessed_dict: Mapping[str, any], 
                        chain_type: str, 
                        is_polymer_terminus: bool) -> Mapping[str, any]:
  """Get the standard atoms order and features for specific residue according to the CCD.

    Args:
        ccd_id: The CCD ID of the residue.
        ccd_preprocessed_dict: The chemical components dictionary.
        chain_type: The type of the chain, protein, rna, dna, ligand.
        is_polymer_terminus: Whether the residue is the terminus of the polymer.

    Returns:
        Mapping[str, any]: The padding features for the residue.
    """

  def _set_std_token_indices(pad_feats: Mapping[str, any], 
                             token_indice: np.ndarray, 
                             key_prefix: str):
      """
          Helper function to set token indices and masks.
      """
      token_indice = np.where(token_indice == 1)[0]
      if key_prefix == 'centra':
         suffix = 'token_indice'
      elif key_prefix == 'pseudo':
         suffix = 'beta'  
      assert len(token_indice) <= 1, f"residue should have only one {key_prefix}-token, Got {len(token_indice)}"
      pad_feats[f'{key_prefix}_{suffix}'] = token_indice if len(token_indice) == 1 else np.array([0], dtype=np.int32)
      pad_feats[f'{key_prefix}_{suffix}_mask'] = np.array([1] if len(token_indice) == 1 else [0], dtype=np.int32)

  def _drop_leaving_atoms(chain_type: str,
                          atom_ids_list: list, 
                          is_polymer_terminus: bool) -> List[str]:
    """Drop the leaving atoms when modified ccd gets linked in polymer."""
    _LEAVING_ATOMS_CONFIG = {
        'dna': ['OP3'],
        'rna': ['OP3'], 
        'protein': ['OXT', 'HXT']
    }
    _atom_ids_list = copy.deepcopy(atom_ids_list)
    if is_polymer_terminus or chain_type == 'ligand':
        return _atom_ids_list
    
    atoms_to_remove = _LEAVING_ATOMS_CONFIG.get(chain_type, [])
    return [atom for atom in _atom_ids_list if atom not in atoms_to_remove]

  def _get_ref_info():
    """Get residue information and determine if it's standard."""
    refs = ccd_preprocessed_dict.get(ccd_id, None)
    if refs is None: 
        raise ValueError(f'Not found ccd_id: {ccd_id} in ccd_preprocessed_dict')
    
    residue_is_standard = (ccd_id in residue_constants.STANDARD_LIST)
    if residue_is_standard:
        if is_polymer_terminus:
          standard_atom_ids_list = refs['atom_ids']
        else:
          standard_atom_ids_list = POLYMER_STANDARD_RESI_ATOMS[ccd_id]
    else:
        # for ligand/ion/modified_residues. ccd_id is not in STANDARD_LIST.
        standard_atom_ids_list = _drop_leaving_atoms(
          chain_type=chain_type, 
          atom_ids_list=refs['atom_ids'], 
          is_polymer_terminus=is_polymer_terminus
        )

    atom_positions_list = refs['position']
    return refs, residue_is_standard, standard_atom_ids_list, atom_positions_list

  def _filter_atom_positions(refs: Mapping[str, any], 
                             standard_atom_ids_list: List[str], 
                             atom_positions_list: List[np.ndarray]
                             ) -> Tuple[List[str], List[np.ndarray]]:
    """Filter atom positions for non-ligand types."""
    if chain_type == 'ligand':
        return standard_atom_ids_list, atom_positions_list
    
    ccd_ori_atom_ids_order = refs['atom_ids']
    new_atom_ids_list = []
    new_atom_positions_list = []
    for idx, key in enumerate(ccd_ori_atom_ids_order):
      if key in standard_atom_ids_list:
        new_atom_ids_list.append(key)
        new_atom_positions_list.append(atom_positions_list[idx])
    assert len(new_atom_ids_list) == len(standard_atom_ids_list) == len(new_atom_positions_list)
    return new_atom_ids_list, new_atom_positions_list

  def _process_atom_tokens(atom_ids_list: List[str], 
                           ref_atom_ids_index: Mapping[str, int], 
                           residue_is_standard: bool,
                           total_nums: int
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process atom tokens and set masks and indices."""
    padding_atom_pos = np.zeros([total_nums, 3], dtype=np.float32)  # Dummy Atom pos
    padding_atom_mask = np.zeros([total_nums], dtype=np.int32)
    centra_token_indice = np.zeros([total_nums], dtype=np.int32) 
    pseudo_token_indice = np.zeros([total_nums], dtype=np.int32)

    for at_id in atom_ids_list:
        if at_id in ref_atom_ids_index: 
            adjust_idx = ref_atom_ids_index[at_id]
            padding_atom_mask[adjust_idx] = 1
            if residue_is_standard:
                if at_id in residue_constants.CENTRA_TOKEN:
                    centra_token_indice[adjust_idx] = 1
                    if ccd_id.upper() == "GLY":
                        pseudo_token_indice[adjust_idx] = 1
                elif at_id in residue_constants.PSEUDO_TOKEN:
                    if at_id == 'CB': 
                        pseudo_token_indice[adjust_idx] = 1
                    elif at_id == 'P' and ccd_id.upper() in residue_constants.DNA_RNA_LIST: 
                        pseudo_token_indice[adjust_idx] = 1
            else:
                centra_token_indice[adjust_idx] = 1
                pseudo_token_indice[adjust_idx] = 1
    
    return padding_atom_pos, padding_atom_mask, centra_token_indice, pseudo_token_indice


  refs, residue_is_standard, atom_ids_layout, atom_positions_layout = _get_ref_info()
  
  atom_ids_layout, atom_positions_layout = _filter_atom_positions(
      refs, 
      atom_ids_layout, 
      atom_positions_layout
  )
  
  ref_atom_ids_index = {name: i for i, name in enumerate(refs['atom_ids'])}
  total_nums = len(ref_atom_ids_index)
  assert total_nums > 0, f'TODO filter - Got CCD <{ccd_id}>: 0 atom nums.'
  
  padding_atom_pos, padding_atom_mask, centra_token_indice, pseudo_token_indice = \
    _process_atom_tokens(
      atom_ids_layout, 
      ref_atom_ids_index, 
      residue_is_standard, 
      total_nums
  )

  frame_indice = utils.get_pae_frame_mask(
    atom_ids_list=atom_ids_layout, 
    atom_positions_list=atom_positions_layout,
    residue_name_3=ccd_id,
    residue_is_standard=residue_is_standard,
    residue_is_missing=False,
    ref_atom_ids_index=ref_atom_ids_index
  )

  pad_feats = {
    **frame_indice, 
    'ccd_ids': np.array([ccd_id] * total_nums, dtype=object),  # N_atom
    'atom_ids': refs['atom_ids'], # [N_atom]
    'atom_pos': padding_atom_pos,  # [N_atom]
    'atom_pos_mask': padding_atom_mask,  # [N_atom]
    'token_to_atom_nums': np.array([total_nums], dtype=np.int32) \
                      if residue_is_standard else np.ones([total_nums], dtype=np.int32)
  }

  if residue_is_standard:
    _set_std_token_indices(pad_feats, centra_token_indice, 'centra')
    _set_std_token_indices(pad_feats, pseudo_token_indice, 'pseudo')
  else:
    # if is non-standard, token_nums == atom_nums
    pad_feats['centra_token_indice'] = np.zeros_like(centra_token_indice, dtype=np.int32)
    pad_feats['centra_token_indice_mask'] = centra_token_indice
    pad_feats['pseudo_beta'] = np.zeros_like(pseudo_token_indice, dtype=np.int32)
    pad_feats['pseudo_beta_mask'] = pseudo_token_indice

  return pad_feats


def get_structure_layout(all_chain_features: Mapping[str, any], 
                          ccd_preprocessed_dict: Mapping[str, any]
                          ) -> Mapping[str, any]:
    """Generate structure layout from chain features.
    
    This function processes chain features to create comprehensive feature arrays
    for inference, including atom positions, token indices, and frame information.
    
    Args:
        all_chain_features: Dictionary mapping chain_id to chain_features.
            Each chain_features should contain 'ccd_seqs' and 'chain_type'.
        ccd_preprocessed_dict: Preprocessed CCD dictionary containing residue information.
        extra_feats: Optional extra features for non-standard residues.
        
    Returns:
        dict: Dictionary containing all processed features for inference.
    """
  
    def _is_polymer_terminus(idx: int, ccd_list: list, chain_type: str) -> bool:
        """Determine whether the residue is the terminus of the polymer.
            For protein, the terminus is the last residue. (c-terminal)
            For rna/dna, the terminus is the first residue. (5'-terminal)
        Args:
            idx: Current residue index in the chain.
            ccd_list: List of CCD IDs in the chain.
            chain_type: Chain type (protein, rna, dna).
            
        Returns:
            bool: True if the residue is a polymer terminus.
        """
        return (idx == len(ccd_list) - 1 and chain_type == 'protein') or (idx == 0 and chain_type in ['rna', 'dna'])

    def _update_frame_indices(pad_feats: dict, frame_indice_offset: int) -> int:
        """Update frame indices with the current offset and return new offset.
        
        Args:
            pad_feats: Padding features dictionary.
            frame_indice_offset: Current frame index offset.
            
        Returns:
            int: Updated frame index offset.
        """
        for key in ['ai_indice', 'bi_indice', 'ci_indice']:
            pad_feats[key] += frame_indice_offset
        return frame_indice_offset + pad_feats['frame_atom_offset']

    def _adjust_token_indices(feature_arrays: dict) -> dict:
        """Adjust token indices using cumulative sums.
        
        Args:
            feature_arrays: Dictionary containing feature arrays.
            
        Returns:
            dict: Updated feature arrays with adjusted token indices.
        """
        cumsum_array = np.cumsum(feature_arrays['all_token_to_atom_nums'])
        offset_array = np.insert(cumsum_array[:-1], 0, 0)
        
        feature_arrays['all_centra_token_indice'] += offset_array
        feature_arrays['pseudo_beta'] += offset_array

        ## pseudo_beta is the atom_pos of the pseudo_token_indice
        # so we need to adjust the pseudo_beta to the atom_pos of the pseudo_token_indice
        _all_atom_pos = feature_arrays['all_atom_pos']
        _all_pseudo_token_indice = feature_arrays['pseudo_beta']
        feature_arrays['pseudo_beta'] = _all_atom_pos[_all_pseudo_token_indice]
        
        return feature_arrays

    def _validate_feature_integrity(feature_arrays: dict) -> None:
        """Validate the integrity of feature arrays.
        
        Args:
            feature_arrays: Dictionary containing feature arrays.
            
        Raises:
            AssertionError: If feature integrity checks fail.
        """
        # Check atom-level feature consistency
        assert (feature_arrays['all_atom_pos'].shape[0] == 
                feature_arrays['all_atom_pos_mask'].shape[0] == 
                feature_arrays['label_atom_ids'].shape[0]), "Atom-level features have inconsistent shapes"
        
        # Check token-level feature consistency
        assert (feature_arrays['all_centra_token_indice'].shape[0] == 
                feature_arrays['all_centra_token_indice_mask'].shape[0]), "Centra token features have inconsistent shapes"
        
        assert (feature_arrays['pseudo_beta'].shape[0] == 
                feature_arrays['pseudo_beta_mask'].shape[0]), "Pseudo token features have inconsistent shapes"
        
        # Check frame-level feature consistency
        max_atom_idx = feature_arrays['all_atom_pos'].shape[0]
        frame_features = ['frame_ai_indice', 'frame_bi_indice', 'frame_ci_indice', 'frame_mask']
        assert all(feature_arrays[frame_features[0]].shape[0] == feature_arrays[key].shape[0] 
                  for key in frame_features), "Frame-level features have inconsistent shapes"
        assert all(np.max(feature_arrays[key]) < max_atom_idx for key in frame_features), \
                "Frame indices exceed atom array bounds"

    # Initialize feature arrays
    final_features = {
      "label_ccd_ids": [],
      "label_atom_ids": [],
      "all_atom_pos": [],
      "all_atom_pos_mask": [],
      "all_centra_token_indice": [],
      "all_centra_token_indice_mask": [],
      "all_token_to_atom_nums": [],
      "pseudo_beta": [],
      "pseudo_beta_mask": [],
      "frame_ai_indice": [],
      "frame_bi_indice": [],
      "frame_ci_indice": [],
      "frame_mask": []
    }

    _ignore_keys = ['frame_atom_offset']
    
    frame_indice_offset = 0
    for chain_id, chain_features in all_chain_features.items():
        chain_type = chain_features['chain_type']
        ccd_list = chain_features['ccd_seq']
        
        for idx, ccd_id in enumerate(ccd_list):
            is_polymer_terminus = _is_polymer_terminus(idx, ccd_list, chain_type)
            
            each_ccd_feats = get_padding_restype(
              ccd_id, 
              ccd_preprocessed_dict, 
              chain_type, 
              is_polymer_terminus=is_polymer_terminus
            )
            frame_indice_offset = _update_frame_indices(each_ccd_feats, frame_indice_offset)
            
            ## update final_features
            for key, value in each_ccd_feats.items():
                if key in _ignore_keys:
                  continue
                elif key in ['ai_indice', 'bi_indice', 'ci_indice']:
                  final_features[f'frame_{key}'].append(value)
                elif key in ['frame_indice_mask']:
                  final_features['frame_mask'].append(value)
                elif key in ['pseudo_beta', 'pseudo_beta_mask']:
                  final_features[key].append(value)
                elif key in ['ccd_ids', 'atom_ids']:
                  final_features[f'label_{key}'].append(value)
                else:
                  final_features[f'all_{key}'].append(value)
    
    for key in final_features.keys():
        final_features[key] = np.concatenate(final_features[key])
    
    final_features = _adjust_token_indices(final_features)  
    _validate_feature_integrity(final_features)
    
    return final_features


def get_complete_assembly_features(all_chain_features: Mapping[str, any], 
                          ccd_preprocessed_dict: Mapping[str, any], 
                          use_msa_templ_feats: bool = True) -> Mapping[str, any]:
  """Get complete assembly features for inference.
  
  Args:
    all_chain_features: Mapping[str, any], with keys: <chain_id>: <chain_features>
        all_chain_features: {
          <chain_id>: {
              'chain_type': str,
              'msa_seq': str,
              'ccd_seq': list of ccd,
              'extra_feats': list of extra_feats,
              'raw_info': raw_info,
              'msa_templ_feats': msa_templ_feats,
          }
    ccd_preprocessed_dict: The chemical components dictionary. 
      Mapping[str, any], with keys: <ccd_id>: <ccd_info>
    use_msa_templ_feats: bool, default: True
  Returns:
    Mapping[str, any]: Mapping of features for inference.
  """
  extra_ccd_infos = {}
  chain_msa_features = {}
  for chain_id, chain_features in all_chain_features.items():
    if (chain_features['chain_type'] not in ['protein', 'rna']) \
                  or (not use_msa_templ_feats):
      msa_features = None
    else:
      msa_features = chain_features.pop('msa_templ_feats')
    chain_msa_features[chain_id] = msa_features
    extra_ccd_infos.update(chain_features['extra_feats'])
  
  ## update ccd_preprocessed_dict with extra_ccd_infos
  if any(_ccd_id in ccd_preprocessed_dict for _ccd_id in extra_ccd_infos.keys()):
    raise ValueError(
        f'conflicting ligand ids {list(extra_ccd_infos.keys())} are in CCD '
        '- it is not supported to give '
        'ligands/modified residues created from SMILES the same name as CCD components.'
    )
  ccd_dict = {**ccd_preprocessed_dict, **extra_ccd_infos}

  ## 1. get msa features
  msa_features, chain_order = pipeline_aa.process_with_all_msa_chain_features(
    chain_msa_features, 
    all_chain_features, 
    ccd_dict
  )

  ## 2. get sequence features
  seq_features = pipeline_aa.get_assembly_sequence_features(
    chain_order, 
    all_chain_features, 
    coval_bonds_info=[],
    ccd_preprocessed_dict=ccd_dict
  )
        
  ## 3. combine sequence and msa features
  np_example = pipeline_aa.combine_assembly_seq_and_msa_features(
    seq_features, 
    msa_features
  )
  
  ## 4. get template further features
  np_example = pipeline_aa.add_further_assembly_template_feat(np_example)

  np_example["seq_mask"] = np.ones_like(np_example['restype']).astype('float32')
  np_example = crop_msa(np_example, max_msa_depth=MAX_MSA_DEPTH)
  
  ## 5. get inference pos mask
  all_chain_features_ordered = {
    chain_id: all_chain_features[chain_id] for chain_id in chain_order
  }
  label = get_structure_layout(
    all_chain_features_ordered, 
    ccd_dict
  )
  
  return {"feat": np_example, "label": label}


def process_chain_msa(msa_task: MSATaskMeta) -> MSATaskMeta:
    """Process a single chain MSA/Template search task.
    
    Args:
        msa_task: MSATaskMeta
    Returns:
        msa_task: MSATaskMeta with processed features.
    """
    if msa_task.task_type == 'protein':
        msa_task.data_pipeline.set_max_template_hits(max_hits=MAX_PROTEIN_TEMPLATE_HITS)
        logger.info(f'Set max template hits for protein to {MAX_PROTEIN_TEMPLATE_HITS}')
    
    if msa_task.features_pkl.exists():
        logger.info('MSA/Template features.pkl found, use offline features.pkl.')
        with open(msa_task.features_pkl, 'rb') as f:
            raw_features = pickle.load(f)
    else:
        t0 = time.time()
        raw_features = msa_task.data_pipeline.process(
            chain_id=msa_task.chain_id,
            sequence=msa_task.seq,
            description=msa_task.desc,
            msa_output_dir=msa_task.msa_output_dir
        )
        logger.info(f"[MSA/Template] {msa_task.desc};" \
                    f"seq length: {len(msa_task.seq)};" \
                    f"use: {time.time() - t0}")

        with open(msa_task.features_pkl, 'wb') as f:
            pickle.dump(raw_features, f, protocol=4)
                                                                                                                                                                                             
    return dataclasses.replace(msa_task, processed_features=raw_features)


def featurize_entities(all_entities: List[EntityBean], 
                       ccd_preprocessed_path: str, 
                       msa_templ_data_pipeline_dict: Mapping[str, any], 
                       msa_output_dir: Union[str, pathlib.Path],
                       use_msa_templ_feats: bool = True) -> Mapping[str, any]:

    """Featurize entities.

    Args:
        all_entities: List[EntityBean]
        ccd_preprocessed_path: str
        msa_templ_data_pipeline_dict: Mapping[str, any]
        msa_output_dir: str | pathlib.Path
        use_msa_templ_feats: bool, default: True

    Returns:
        Mapping[str, any]: Mapping of features for inference.
    """
    if isinstance(msa_output_dir, str):
        msa_output_dir = pathlib.Path(msa_output_dir)
    msa_output_dir.mkdir(parents=True, exist_ok=True)

    all_chain_features = {}
    for entity_items in all_entities:    
      chain_id = entity_items.raw_info['asym_chain_id']
      chain_features = {'msa_templ_feats': {},
                        'ccd_seq': parsers.parse_ccd_fasta(entity_items.seqs), 
                        'msa_seq': entity_items.msa_seqs,
                        'chain_type': entity_items.dtype,
                        'extra_feats': entity_items.extra_mol_infos,
                        'raw_info': entity_items.raw_info}
      all_chain_features[chain_id] = chain_features

    ## 1. get all_msa_seqs for protein/rna MSA/Template search.
    if use_msa_templ_feats:
      msa_tasks = []
      fasta_seq_to_type_chain_id = {}
      for chain_id, chain_features in all_chain_features.items():
        if chain_features['chain_type'] not in ['protein', 'rna']:
          continue
        
        chain_type = chain_features['chain_type']
        type_chain_id = chain_type + '_' + chain_id
        fasta_seq = chain_features['msa_seq']
        if fasta_seq not in fasta_seq_to_type_chain_id:
            fasta_seq_to_type_chain_id[fasta_seq] = [chain_id]
        else:
            fasta_seq_to_type_chain_id[fasta_seq].append(type_chain_id)
            continue
        
        features_pkl_dir = msa_output_dir.joinpath(f'{type_chain_id}')
        features_pkl_dir.mkdir(parents=True, exist_ok=True)
        features_pkl = features_pkl_dir.joinpath('features.pkl')

        meta_task = MSATaskMeta(
                    data_pipeline=msa_templ_data_pipeline_dict[chain_type], 
                    chain_id=chain_id, 
                    seq=fasta_seq, 
                    desc=type_chain_id, 
                    msa_output_dir=features_pkl_dir, 
                    features_pkl=features_pkl,
                    task_type=chain_type)
        msa_tasks.append(meta_task)

      print('MSA fastas:', list(fasta_seq_to_type_chain_id.items()))

      ## 2. multiprocessing for protein/rna MSA/Template search.
      seqs_to_msa_features = {}
      logger.info('[Multiprocess] starting MSA/Template search...')
      t0 = time.time()
      with ProcessPoolExecutor(max_workers=MAX_MSA_WORKERS) as executor:
          futures = [executor.submit(process_chain_msa, task) for task in msa_tasks]

          for future in as_completed(futures):
              try:
                  processed_msa = future.result()
                  seqs_to_msa_features[processed_msa.seq] = processed_msa.features
              except RuntimeError as exc:
                  import traceback; traceback.print_exc()
                  logger.error(f'Task generated an exception : {exc}')
      logger.info(f'[Multiprocess] All msa/template use: {time.time() - t0}')

      ## 3. add msa_templ_feats to all_chain_features.
      for chain_id in all_chain_features.keys():
        chain_features = all_chain_features[chain_id]
        if chain_features['chain_type'] not in ['protein', 'rna']:
          continue
        fasta_seq = chain_features['msa_seq']
        if fasta_seq in seqs_to_msa_features:
          chain_features['msa_templ_feats'] = copy.deepcopy(
             seqs_to_msa_features[fasta_seq]
          )

    ## 4. get complete features and check.
    assert len(all_entities) == len(all_chain_features.keys())
    ccd_preprocessed_dict = load_ccd_dict(ccd_preprocessed_path)
    all_feats = get_complete_assembly_features(all_chain_features, ccd_preprocessed_dict, 
                                        use_msa_templ_feats=use_msa_templ_feats)
    assert len(all_entities) == len(np.unique(all_feats['feat']['chain_ids']))

    sample = {
       **all_feats,
       'label_cropped': {}
    }

    for key in STRING_FEATURES:
      if key in sample['feat'].keys():
        sample['feat'][key] = ' '.join(sample['feat'][key])
      if key in sample['label'].keys():
        sample['label'][key] = ' '.join(sample['label'][key])

    return sample
