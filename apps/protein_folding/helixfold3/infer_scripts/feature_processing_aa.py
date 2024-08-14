"""Functions for building the features for the HelixFold-3 inference pipeline."""
import collections
import copy
import os
import time, gzip, pickle
import numpy as np
import logging
from helixfold.common import residue_constants
from helixfold.data import parsers
from helixfold.data import pipeline_multimer
from helixfold.data import pipeline_rna_multimer
from helixfold.data import pipeline_conf_bonds, pipeline_token_feature, pipeline_hybrid
from helixfold.data import label_utils
from concurrent.futures import ProcessPoolExecutor, as_completed
from .preprocess import digit2alphabet

logger = logging.getLogger(__file__)

POLYMER_STANDARD_RESI_ATOMS = residue_constants.residue_atoms
STRING_FEATURES = ['all_chain_ids', 'all_ccd_ids','all_atom_ids', 
                  'release_date','label_ccd_ids','label_atom_ids']

def load_ccd_dict(ccd_preprocessed_path):
    assert os.path.exists(ccd_preprocessed_path),\
              (f'[CCD] ccd_preprocessed_path: {ccd_preprocessed_path} not exist.')
    st_1 = time.time()
    if 'pkl.gz' in ccd_preprocessed_path:
        with gzip.open(ccd_preprocessed_path, "rb") as fp:
            ccd_preprocessed_dict = pickle.load(fp)
    elif '.pkl' in ccd_preprocessed_path:
        with open(ccd_preprocessed_path, "rb") as fp:
            ccd_preprocessed_dict = pickle.load(fp)
    print(f'[CCD] load ccd dataset done. use {time.time()-st_1}s;'\
                    f'Has length of {len(ccd_preprocessed_dict)}')
    
    return ccd_preprocessed_dict


def crop_msa(feat, max_msa_depth=16384):
    """ pad msa and generate msa_mask. """
    msa = feat['msa']
    msa_mask = np.ones_like(feat['msa']).astype('float32') # [msa_depth, n_token]
    delection_mat = feat['deletion_matrix']
    msa_depth, num_token = msa_mask.shape
    if msa_depth > max_msa_depth:
        msa_mask = msa_mask[: max_msa_depth, :]
        msa = msa[: max_msa_depth, :]
        delection_mat = delection_mat[: max_msa_depth, :]
    return msa.astype('int32'), msa_mask, delection_mat


def get_padding_restype(ccd_id, ccd_preprocessed_dict, extra_feats=None):
  if ccd_id in ccd_preprocessed_dict:
    refs = ccd_preprocessed_dict[ccd_id]  # O(1)
    if ccd_id in residue_constants.STANDARD_LIST:
      _residue_is_standard = True
      pdb_atom_ids_list = POLYMER_STANDARD_RESI_ATOMS[ccd_id] # NOTE: now is only support standard residue.
    else:
      # for ligand/ion. ccd_id.
      _residue_is_standard = False
      pdb_atom_ids_list = refs['atom_ids']
  else:
    # for ligand/ion. smiles.
    assert not extra_feats is None and ccd_id in extra_feats
    _residue_is_standard = False
    refs = extra_feats[ccd_id]
    pdb_atom_ids_list = refs['atom_ids']

  _atom_positions_list = refs['position']
  ## NOTE: map atom_ids to original atom_ids order from ccd; STANDARD_LIST
  if ccd_id in residue_constants.STANDARD_LIST:
    ccd_ori_atom_ids_order = refs['atom_ids']
    _new_atom_ids_list = []
    _new_atom_positions_list = []
    for idx, key in enumerate(ccd_ori_atom_ids_order):
      if key in pdb_atom_ids_list:
        _new_atom_ids_list.append(key)
        _new_atom_positions_list.append(_atom_positions_list[idx])
    assert len(_new_atom_ids_list) == len(pdb_atom_ids_list) == len(_new_atom_positions_list)
    pdb_atom_ids_list = _new_atom_ids_list
    _atom_positions_list = _new_atom_positions_list

  ref_atom_ids_index = { 
    name: i for i, name in enumerate(refs['atom_ids'])
  }
  total_nums = len(ref_atom_ids_index)
  assert total_nums > 0, f'TODO filter - Got CCD <{ccd_id}>: 0 atom nums.'
  padding_atom_pos = np.zeros([total_nums, 3], dtype=np.float32) ## Dummy Atom pos
  padding_atom_mask = np.zeros([total_nums], dtype=np.int32)
  centra_token_indice = np.zeros([total_nums], dtype=np.int32) 
  pseudo_token_indice = np.zeros([total_nums], dtype=np.int32)

  for at_id in pdb_atom_ids_list:
    if at_id in ref_atom_ids_index: 
      adjust_idx = ref_atom_ids_index[at_id]
      padding_atom_mask[adjust_idx] = 1
      if _residue_is_standard:
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

  frame_indice = label_utils.get_pae_frame_mask(atom_ids_list=pdb_atom_ids_list, 
                  atom_positions_list=_atom_positions_list,
                  residue_name_3=ccd_id,
                  residue_is_standard=_residue_is_standard,
                  residue_is_missing=False,
                  ref_atom_ids_index=ref_atom_ids_index)

  pad_feats = {
    **frame_indice, 
    'ccd_ids': np.array([ccd_id] * total_nums , dtype=object),  # N_atom
    'atom_ids': refs['atom_ids'], # [N_atom]
    'atom_pos': padding_atom_pos,  # [N_atom]
    'atom_mask': padding_atom_mask,  # [N_atom]
    'token_to_atom_nums': np.array([total_nums], dtype=np.int32) \
                     if _residue_is_standard else np.ones([total_nums], dtype=np.int32)
  }

  if _residue_is_standard:
    centra_token_indice = np.where(centra_token_indice == 1)[0]
    assert len(centra_token_indice) <= 1, f"residue should be has only one centra-token, Got {len(centra_token_indice)}"
    if len(centra_token_indice) == 1:
      pad_feats['centra_token_indice'] = centra_token_indice #[N_token]
      pad_feats['centra_token_indice_mask'] = np.array([1], dtype=np.int32)
    else:
      pad_feats['centra_token_indice'] = np.array([0], dtype=np.int32) #[N_token]
      pad_feats['centra_token_indice_mask'] = np.array([0], dtype=np.int32)
    
    pseudo_token_indice = np.where(pseudo_token_indice == 1)[0]
    assert len(pseudo_token_indice) <= 1, f"residue should be has only one pesudo-token. Got {len(pseudo_token_indice)}"
    if len(pseudo_token_indice) == 1:
      pad_feats['pseudo_token_indice'] = pseudo_token_indice
      pad_feats['pseudo_token_indice_mask'] = np.array([1], dtype=np.int32)
    else:
      pad_feats['pseudo_token_indice'] = np.array([0], dtype=np.int32)
      pad_feats['pseudo_token_indice_mask'] = np.array([0], dtype=np.int32)
  else:
    # if is non-standard, token_nums == atom_nums
    pad_feats['centra_token_indice'] = np.zeros_like(centra_token_indice, dtype=np.int32)
    pad_feats['centra_token_indice_mask'] = centra_token_indice

    pad_feats['pseudo_token_indice'] = np.zeros_like(pseudo_token_indice, dtype=np.int32)
    pad_feats['pseudo_token_indice_mask'] = pseudo_token_indice

  return pad_feats

def get_inference_restype_mask(all_chain_features, ccd_preprocessed_dict, extra_feats=None):
  """
    all_chain_features: <type>_<chain_id>: chain_features, chain_features should has the ccd_seqs
    ccd_preprocessed_dict: preprocessed CCD dict
  """

  all_ccd_ids = np.empty((0,), dtype=object)
  all_atom_ids = np.empty((0,), dtype=object)
  all_atom_pos = np.empty((0, 3), dtype=np.float32)
  all_atom_mask = np.empty((0,), dtype=np.int32)
  all_centra_token_indice = np.empty((0,), dtype=np.int32)
  all_centra_token_indice_mask = np.empty((0,), dtype=np.int32)
  all_token_to_atom_nums = np.empty((0,), dtype=np.int32)
  all_pseudo_token_indice = np.empty((0,), dtype=np.int32)
  all_pseudo_token_indice_mask = np.empty((0,), dtype=np.int32)
  frame_ai_indice = np.empty((0,), dtype=np.int32) # Ntoken
  frame_bi_indice = np.empty((0,), dtype=np.int32) # Ntoken
  frame_ci_indice = np.empty((0,), dtype=np.int32) # Ntoken
  frame_mask = np.empty((0,), dtype=np.int32) # Ntoken

  frame_indice_offset = 0
  for type_chain_id, ccd_list in all_chain_features.items():
    dtype, chain_id = type_chain_id.rsplit('_', 1) 
    for ccd_id in ccd_list:
      pad_feats = get_padding_restype(ccd_id, ccd_preprocessed_dict, extra_feats=extra_feats)
      pad_feats['ai_indice'] = pad_feats['ai_indice'] + frame_indice_offset
      pad_feats['bi_indice'] = pad_feats['bi_indice'] + frame_indice_offset
      pad_feats['ci_indice'] = pad_feats['ci_indice'] + frame_indice_offset
      frame_indice_offset += pad_feats['frame_atom_offset']	

      all_atom_ids = np.concatenate((all_atom_ids, pad_feats['atom_ids']))
      all_ccd_ids = np.concatenate((all_ccd_ids, pad_feats['ccd_ids']))
      all_atom_pos = np.concatenate((all_atom_pos, pad_feats['atom_pos']))
      all_atom_mask = np.concatenate((all_atom_mask, pad_feats['atom_mask']))
      all_centra_token_indice = np.concatenate((all_centra_token_indice, 
                            pad_feats['centra_token_indice']))
      all_centra_token_indice_mask = np.concatenate((all_centra_token_indice_mask, 
                            pad_feats['centra_token_indice_mask']))
      all_token_to_atom_nums = np.concatenate((all_token_to_atom_nums,
                            pad_feats['token_to_atom_nums']))
      all_pseudo_token_indice = np.concatenate((all_pseudo_token_indice, 
                            pad_feats['pseudo_token_indice']))
      all_pseudo_token_indice_mask = np.concatenate((all_pseudo_token_indice_mask, 
                            pad_feats['pseudo_token_indice_mask']))
      frame_ai_indice = np.concatenate((frame_ai_indice, pad_feats['ai_indice']))
      frame_bi_indice = np.concatenate((frame_bi_indice, pad_feats['bi_indice']))
      frame_ci_indice = np.concatenate((frame_ci_indice, pad_feats['ci_indice']))
      frame_mask = np.concatenate((frame_mask, pad_feats['frame_indice_mask']))

  cumsum_array = np.cumsum(all_token_to_atom_nums)
  assert all_atom_pos.shape[0] == all_atom_mask.shape[0] == all_atom_ids.shape[0]
  all_centra_token_indice = all_centra_token_indice + np.insert(cumsum_array[:-1], 0, 0)
  all_pseudo_token_indice = all_pseudo_token_indice + np.insert(cumsum_array[:-1], 0, 0)
  assert all_atom_pos.shape[0] == all_atom_mask.shape[0] == all_atom_ids.shape[0]
  assert all_centra_token_indice.shape[0] == all_centra_token_indice_mask.shape[0]
  assert all_pseudo_token_indice.shape[0] == all_pseudo_token_indice_mask.shape[0]
  assert frame_ai_indice.shape[0] == frame_bi_indice.shape[0] == frame_ci_indice.shape[0] == frame_mask.shape[0] # Ntoken
  assert np.max(frame_ai_indice) < all_atom_pos.shape[0] and \
      np.max(frame_bi_indice) < all_atom_pos.shape[0] and \
      np.max(frame_ci_indice) < all_atom_pos.shape[0]

  return {
    "label_ccd_ids": all_ccd_ids, # [N_atom, ]
    "label_atom_ids": all_atom_ids,	# [N_atom,]
    "all_atom_pos": all_atom_pos, # [N_atom, 3]
    "all_atom_pos_mask": all_atom_mask,  # [N_atom]
    "all_centra_token_indice": all_centra_token_indice, # [N_token, ]
    "all_centra_token_indice_mask": all_centra_token_indice_mask, # [N_token,]
    "all_token_to_atom_nums": all_token_to_atom_nums, # [N_token,]
    "pseudo_beta": all_atom_pos[all_pseudo_token_indice], # [N_token,]
    "pseudo_beta_mask": all_pseudo_token_indice_mask, # [N_token, ]

    ## for pae loss computation.
    "frame_ai_indice": frame_ai_indice, # [N_token, ]
    "frame_bi_indice": frame_bi_indice, # [N_token, ]
    "frame_ci_indice": frame_ci_indice, # [N_token, ]
    "frame_mask": frame_mask, # [N_token, ]
  }


def add_assembly_features(all_chain_features, ccd_preprocessed_dict, no_msa_templ_feats=True):
  '''
    ## NOTE: keep the type and chainID orders.
    all_chain_features: {
        <type>_<chain_id>: {
          'msa_templ_feats': [msa_templ_feats],
          'ccd_seqs': [ccd_list], 
          'extra_feats': [extra_mol_info],
          }
        }
    }
    1. include msa pair_and_merge, hf2 raw processing.
        pipeline_multimer.process_with_all_chain_features
        pipeline_rna_multimer.process_with_all_chain_features
    2. all type msa/template dense joint
        pipeline_hybrid
    3. include basic feature assembly, 
        such as token_index, entity_id, asym_id, sym_id, ref_space_uid, token_bonds, 
        perm_atom_index, ref_token2atom_idx, ref_atom_count, perm_entity_id, perm_asym_id
  '''
  ## first, Group the chains by ccd_seqs, and record all the chain type.
  dtype_grouped_chains = collections.defaultdict(dict)
  dtype_hf2_feats = collections.defaultdict(dict)
  for type_chain_id, chain_features in all_chain_features.items():
    dtype, chain_id = type_chain_id.rsplit('_', 1) 
    dtype_hf2_feats[dtype][chain_id] = chain_features.pop('msa_templ_feats')
    dtype_grouped_chains[dtype][chain_id] = chain_features # has keys: ccd_seqs, extra_feats
    # [dtype][chain_id]: chain_features.
  
  ## for ccd squence, token_seqs_features/ref_features/bond_features
  new_order_chain_infos = {}
  extra_feats_infos = {}
  for dtype, _ in dtype_hf2_feats.items():
    for _chaid, v in dtype_grouped_chains[dtype].items(): 
      new_order_chain_infos[dtype + '_' + _chaid] = v['ccd_seqs']
      extra_feats_infos.update(v['extra_feats'])

  total_feats = {}
  ## 1. msa pair_and_merge for protein/rna, use hf2 raw processing.
  for dtype, chain_group_feats in dtype_hf2_feats.items():
    hf2_msa_feats = {}

    if dtype == 'protein': 
      if not no_msa_templ_feats:
        hf2_msa_feats = pipeline_multimer.process_with_all_chain_features(chain_group_feats)
    elif dtype == 'rna':
      if not no_msa_templ_feats:
        # mapping RNA token ids to new ids
        for chain_id, features in chain_group_feats.items():
          chain_group_feats[chain_id] = pipeline_rna_multimer.process_feat_to_mapping_to_new_token_list(features)

        # pairing and merge RNA features
        hf2_msa_feats = pipeline_multimer.process_with_all_chain_features(chain_group_feats)
    else:
      pass
      
    total_feats[dtype] = hf2_msa_feats
    total_feats[dtype]["ccd_seqs"] = np.concatenate([np.array(v['ccd_seqs'], dtype=object) \
                                                          for k, v in dtype_grouped_chains[dtype].items()])
    
    total_feats[dtype]["extra_feats"] = {}
    for k, v in dtype_grouped_chains[dtype].items():
      total_feats[dtype]["extra_feats"].update(v['extra_feats'])


  ## 2. make token_seq_feats and conf_bond_feats.
  token_features = pipeline_token_feature.make_sequence_features(all_chain_info=new_order_chain_infos,
                                          ccd_preprocessed_dict=ccd_preprocessed_dict,
                                          extra_feats=extra_feats_infos)
  
  ## 3. Get reference features and bond features
  ref_features = pipeline_conf_bonds.make_ccd_conf_features(all_chain_info=new_order_chain_infos,
                                                      ccd_preprocessed_dict=ccd_preprocessed_dict,
                                                      extra_feats=extra_feats_infos)
  bond_features = pipeline_conf_bonds.make_bond_features(covalent_bond=[], 
                                                      all_chain_info=new_order_chain_infos, 
                                                      ccd_preprocessed_dict=ccd_preprocessed_dict,
                                                      extra_feats=extra_feats_infos)
  ## 4. post convert features
  total_feats['seq_token'] = token_features
  total_feats['conf_bond'] = {**ref_features, **bond_features}
  np_example = pipeline_hybrid._post_convert(ccd_preprocessed_dict=ccd_preprocessed_dict,
                                                  all_chain_feats_dict=total_feats)
  np_example = pipeline_hybrid.make_pseudo_beta(np_example, prefix='template_')
  np_example = pipeline_hybrid.make_template_further_feature(np_example)

  np_example["seq_mask"] = np.ones_like(
      np_example['restype']).astype('float32')
  np_example["msa"], np_example['msa_mask'], np_example['deletion_matrix'] = crop_msa(np_example)
  
  ## 5. get inference pos mask:
  label = get_inference_restype_mask(new_order_chain_infos, ccd_preprocessed_dict, extra_feats_infos)

  return {"feats": np_example,
          "label": label,}


def process_chain_msa(args):
    """
    处理链，如果缓存了特征文件，则直接使用缓存的特征文件，否则生成新的特征文件。
    
    Args:
        args (tuple): 包含以下元素：
            - data_pipeline (DataPipeline): DataPipeline对象，用于处理单个链。
            - chain_id (str): 链ID。
            - seq (Optional[str]): 链序列（可选）。
            - desc (Optional[str]): 链描述（可选）。## NOTE： 这里默认是type_chain_id.
            - msa_output_dir (PathLike): MSA输出目录。
            - features_pkl (PathLike): 特征文件路径。
    
    Returns:
        tuple: 返回一个元组，包含以下元素：
            - chain_id (str): 链ID。
            - raw_features (dict): 处理后的特征字典，包含预处理后的特征和其他相关信息。
            - desc (str): 链描述。
            - seq (str): 链序列。
    
    Raises:
        None.
    """
    data_pipeline, chain_id, seq, desc, \
    msa_output_dir, features_pkl = args
    if features_pkl.exists():
        logger.info('Use cached features.pkl')
        with open(features_pkl, 'rb') as f:
            raw_features = pickle.load(f)
    else:
        t0 = time.time()
        raw_features = data_pipeline._process_single_chain(
            chain_id, sequence=seq, description=desc,
            msa_output_dir=msa_output_dir,
            is_homomer_or_monomer=False)
        print(f'[MSA/Template] {desc}; seq length: {len(seq)}; use: {time.time() - t0}')

        with open(features_pkl, 'wb') as f:
            pickle.dump(raw_features, f, protocol=4)

    if 'template_all_atom_mask' in raw_features:                                                                                       
        raw_features['template_all_atom_masks'] = raw_features.pop('template_all_atom_mask')                                               
                                                                                                                                                
    return chain_id, raw_features, desc, seq


def process_input_json(all_entitys, ccd_preprocessed_path, 
                          msa_templ_data_pipeline_dict, msa_output_dir,
                          no_msa_templ_feats=True):

    ## load ccd dict.
    ccd_preprocessed_dict = load_ccd_dict(ccd_preprocessed_path)
    all_chain_features = {}
    sequence_features = {} 
    num_chains = 0
    for entity_items in all_entitys:
      # dtype(protein, dna, rna, ligand): no_chains,  msa_seqs, seqs
      dtype = list(entity_items.keys())[0]
      items = list(entity_items.values())[0]
      entity_count = items['count']
      ccd_seqs = items['seqs']
      msa_seqs = items['msa_seqs']
      extra_mol_infos = items.get('extra_mol_infos', {}) ## dict, 「extra-add, ccd_id」: ccd_features.

      for i in range(entity_count):
        chain_num_ids = num_chains + i
        chain_id = digit2alphabet(chain_num_ids) # increase ++
        type_chain_id = dtype + '_' + chain_id
        if ccd_seqs in sequence_features:
          all_chain_features[type_chain_id] = copy.deepcopy(sequence_features[ccd_seqs])
          continue
        
        ccd_list = parsers.parse_ccd_fasta(ccd_seqs)
        chain_features = {'msa_templ_feats': {},
                          'ccd_seqs': ccd_list, 
                          'msa_seqs': msa_seqs,
                          'extra_feats': extra_mol_infos}
        all_chain_features[type_chain_id] = chain_features
        sequence_features[ccd_seqs] = chain_features
      num_chains += entity_count

    if not no_msa_templ_feats:
      ## 1. get all msa_seqs for protein/rna MSA/Template search. Only for protein/rna.
      tasks = [] ## data_pipeline, chain_id, seq, desc, msa_output_dir, features_pkl
      fasta_seq_to_type_chain_id = {}
      type_chain_id_to_features_pkl = {}
      if isinstance(msa_output_dir, str):
        msa_output_dir = Path(msa_output_dir)

      for type_chain_id, chain_features in all_chain_features.items():
        dtype, chain_id = type_chain_id.rsplit('_', 1) 
        if dtype == 'protein':
          _data_pipeline = msa_templ_data_pipeline_dict['protein']
        elif dtype == 'rna':
          _data_pipeline = msa_templ_data_pipeline_dict['rna']
        else:
          ## others type is not used.
          continue
        
        fasta_seq = chain_features['msa_seqs']
        if fasta_seq not in fasta_seq_to_type_chain_id:
          fasta_seq_to_type_chain_id[fasta_seq] = []
          fasta_seq_to_type_chain_id[fasta_seq].append(type_chain_id)
        else:
          ## NOTE: same fasta_seq, but different chain_id. will only search once.
          fasta_seq_to_type_chain_id[fasta_seq].append(type_chain_id)
          continue
        
        features_pkl_dir = msa_output_dir.joinpath(f'{type_chain_id}')
        os.makedirs(features_pkl_dir,exist_ok=True)
        features_pkl = features_pkl_dir.joinpath('features.pkl')
        tasks.append((_data_pipeline, chain_id, fasta_seq, 
                        type_chain_id, features_pkl_dir, features_pkl))
        type_chain_id_to_features_pkl[type_chain_id] = features_pkl

      print('MSA fastas:', list(fasta_seq_to_type_chain_id.items()))
      print('features_pkl:', type_chain_id_to_features_pkl)

      ## 2. multiprocessing for protein/rna MSA/Template search.
      seqs_to_msa_features = {}
      logger.info('[Multiprocess] starting MSA/Template search...')
      t0 = time.time()
      with ProcessPoolExecutor() as executor:
          futures = [executor.submit(process_chain_msa, task) for task in tasks]

          for future in as_completed(futures):
              try:
                  _, raw_features, type_chain_id, seqs = future.result()
                  seqs_to_msa_features[seqs] = raw_features
              except Exception as exc:
                  import traceback; traceback.print_exc()
                  logger.error(f'Task generated an exception : {exc}')
      logger.info(f'[Multiprocess] All msa/template use: {time.time() - t0}')

      ## 3. add msa_templ_feats to all_chain_features.
      for type_chain_id in all_chain_features.keys():
        chain_features = all_chain_features[type_chain_id]
        fasta_seq = chain_features['msa_seqs']
        if fasta_seq in seqs_to_msa_features:
          for _type_chain_id in fasta_seq_to_type_chain_id[fasta_seq]:
            chain_features['msa_templ_feats'] = copy.deepcopy(seqs_to_msa_features[fasta_seq])

    assert num_chains == len(all_chain_features.keys())
    all_feats = add_assembly_features(all_chain_features, ccd_preprocessed_dict, no_msa_templ_feats)
    np_example, label = all_feats['feats'], all_feats['label']
    assert num_chains == len(np.unique(np_example['all_chain_ids']))

    sample = {
      "feat": np_example,
      "label": label, ## padding key
      'label_cropped': {}, ## padding key
    }

    for key in STRING_FEATURES:
      if key in sample['feat'].keys():
        sample['feat'][key] = ' '.join(sample['feat'][key])
      if key in sample['label'].keys():
        sample['label'][key] = ' '.join(sample['label'][key])

    return sample