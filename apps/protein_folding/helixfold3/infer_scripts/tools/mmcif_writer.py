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

"""mmcif writer for inference."""

import collections
from datetime import datetime

import numpy as np

from helixfold.data.utils import user_asymid_to_weight
from helixfold.common import residue_constants
from infer_scripts.rdkit_utils import RdkitConstants

REQUIRED_KEYS_FOR_SAVING = [
  'all_ccd_ids', 
  'all_atom_ids', 
  'all_chain_ids',
  'ref_token2atom_idx', 
  'restype', 
  'residue_index', 
  'asym_id',
  'all_atom_pos_mask',
  'is_ligand', 'is_protein', 'is_dna', 'is_rna',
  'token_bonds_type', 
  'ref_element', 
  'chain_ids',
]

META_REQUIRED_KEYS = {
    "_entry": ['id'],
    "_audit_author": ['name', 'pdbx_ordinal'],
    "_audit_conform": ['dict_location', 'dict_name', 'dict_version'],
    "_ma_data": ['content_type', 'id', 'name'],
    "_ma_model_list": ["data_id", "model_group_id", "model_group_name", "model_id", \
                        "model_name", "model_type", "ordinal_id"],
    "_ma_protocol_step": ["method_type", "ordinal_id", "protocol_id", "step_id"],
    "_ma_qa_metric": ["id", "mode", "name", "type"], 
    "_ma_qa_metric_global": ["metric_id", "metric_value", "model_id", "ordinal_id"],
    "_pdbx_data_usage": ["details", "id", "type", "url"],
}

MODEL_VERSION = "HelixFold3"

URL = "https://paddlehelix.baidu.com/app/ToS_helixfold3"

ANNOTATION = f"# \"See the Terms_of_Use.md in the model output for details. It can also be found at {URL}.\""

POLY_MAPPING_TYPE = {
    'protein': "polypeptide(L)",
    'dna': "polydeoxyribonucleotide",
    'rna': "polyribonucleotide",
}

META_REQUIRED_KEYS = {
    "_entry": ['id'],
    "_audit_author": ['name', 'pdbx_ordinal'],
    "_audit_conform": ['dict_location', 'dict_name', 'dict_version'],
    "_ma_data": ['content_type', 'id', 'name'],
    "_ma_model_list": ["data_id", "model_group_id", "model_group_name", "model_id", \
                        "model_name", "model_type", "ordinal_id"],
    "_ma_protocol_step": ["method_type", "ordinal_id", "protocol_id", "step_id"],
    "_ma_qa_metric": ["id", "mode", "name", "software_group_id", "type"], 
    "_ma_qa_metric_global": ["metric_id", "metric_value", "model_id", "ordinal_id"],
    "_pdbx_data_usage": ["details", "id", "type", "url"],
}

STRUCTURE_REQUIRED_KEYS = {
    "_entity": ['id', 'pdbx_description', 'type'],
    "_entity_poly": ['entity_id', 'pdbx_strand_id', 'type'],
    "_entity_poly_seq": ['entity_id', 'hetero', 'mon_id', 'num'],
    "_ma_target_entity": ['data_id', 'entity_id', 'origin'],
    "_ma_target_entity_instance": ["asym_id", "details", "entity_id"],
    "_pdbx_nonpoly_scheme": ["asym_id", "auth_seq_num", "entity_id", "mon_id", "pdb_ins_code", \
                                "pdb_seq_num", "pdb_strand_id"],
    "_pdbx_poly_seq_scheme": ['asym_id', 'auth_seq_num', 'entity_id', 'hetero', 
                            'mon_id', 'pdb_ins_code', 'pdb_seq_num', 'pdb_strand_id', 'seq_id'],
    "_struct_asym": ["entity_id", "id"],
    "_ma_qa_metric_local": ["label_asym_id", "label_comp_id", "label_seq_id", "metric_id", \
                                "metric_value", "model_id", "ordinal_id"],
    "_atom_site": ["group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id", \
                    "label_asym_id", "label_entity_id", "label_seq_id", "pdbx_PDB_ins_code", \
                    "Cartn_x", "Cartn_y", "Cartn_z", "occupancy", "B_iso_or_equiv", "auth_seq_id", "auth_asym_id", \
                    "pdbx_PDB_model_num"],
}

## BEELOW IS THE CONSTANTS.
MMCIF_CONSTANTS = {
    "_entry.id": "<PAD>", 
    "_audit_author.name": "\"Baidu PaddleHelix Team\"",
    "_audit_author.pdbx_ordinal": "1",
    "_audit_conform.dict_location": "https://raw.githubusercontent.com/ihmwg/ModelCIF/master/dist/mmcif_ma.dic",
    "_audit_conform.dict_name": "mmcif_ma.dic",
    "_audit_conform.dict_version": "1.4.6",
    "_ma_data.content_type": "\"model coordinates\"",
    "_ma_data.id": "1",
    "_ma_data.name": "Model",
    "_ma_model_list.data_id": "1",
    "_ma_model_list.model_group_id": "1",
    "_ma_model_list.model_group_name": "<PAD>",
    "_ma_model_list.model_id": "1",
    "_ma_model_list.model_name": "\"Top ranked model\"",
    "_ma_model_list.model_type": "\"Ab initio model\"",
    "_ma_model_list.ordinal_id": "1",
    "_ma_qa_metric_global.metric_id": "1",
    "_ma_qa_metric_global.metric_value": "<PAD>",
    "_ma_qa_metric_global.model_id": "1",
    "_ma_qa_metric_global.ordinal_id": "1",
}

LOOP_MMCIF_CONSTANTS = {
    "_ma_qa_metric.id": ["1", "2"],
    "_ma_qa_metric.mode": ["global", "local"],
    "_ma_qa_metric.name": ["pLDDT", "pLDDT"],
    "_ma_qa_metric.software_group_id": ["1", "1"],
    "_ma_qa_metric.type": ["pLDDT", "pLDDT"],
    "_ma_protocol_step.method_type": ["\"coevolution MSA\"", "\"template search\"", "modeling"],
    "_ma_protocol_step.ordinal_id": ["1", "2", "3"],
    "_ma_protocol_step.protocol_id": ["1", "1", "1"],
    "_ma_protocol_step.step_id": ["1", "2", "3"],
    "_pdbx_data_usage.details": ["\"See the Terms_of_Use.md in the model output for details.\"", \
                                    "\"See the Terms_of_Use.md in the model output for details.\""],
    "_pdbx_data_usage.id": ["1", "2"],
    "_pdbx_data_usage.type": ["license", "disclaimer"],
    "_pdbx_data_usage.url": [URL, URL]
}


def format_pad(string: str, max_width: int, pad_value: str = ' ') -> str:
    return string.ljust(max_width, pad_value)


def _format_mmcif_loop_data(contexts: dict, value_formatter=None) -> list:
    """Format mmCIF loop data.
        
    Args:
        contexts: dict, data to format
        value_formatter: callable, optional value formatter function
    Returns:
        list: formatted mmCIF lines
    """
    def _check_shape(context_dict):
        _length = []
        for k in contexts:
            _length.append(len(contexts[k]))
            if not k.startswith('_'):
                context_dict[f'_{k}'] = contexts.pop(k)
        
        if len(set(_length)) != 1:
            raise ValueError("All values must have same length.")
    
    cif = dict(contexts)
    _check_shape(cif)
    _key = list(cif.keys())[0]

    context_list = []
    if len(cif[_key]) > 1:
        if value_formatter:
            for k, vs in cif.items():
                cif[k] = [value_formatter(v) for v in vs]
        else:
            for k, vs in cif.items():
                max_val_length = max([len(str(v)) for v in vs])
                cif[k] = [format_pad(str(v), max_val_length) for v in vs]
            
        context_list = ['loop_'] + list(cif.keys())
        for tuple_val in zip(*cif.values()):
            tuple_val = list(map(str, tuple_val))
            string_line = ' '.join(tuple_val)
            context_list.append(string_line)
        context_list.append('#')
    elif len(cif[_key]) == 1:
        max_key_length = max([len(k) for k in cif])
        for key, val in cif.items():
            val = list(map(str, val))
            pad_key = format_pad(key, max_key_length)
            assert len(val) == 1, \
                (f"{key} has diferent lengths, Please check your input")
            val = val[0]
            context_list.append(f'{pad_key} {val}')
        context_list.append('#')
    else:
        ## some keys have no values, skip them
        ## such as only *_nonpoly_* and *_poly_*
        pass

    return context_list


def format_mmcif_dict(contexts:dict) -> list:
    """Format mmCIF dict data."""
    return _format_mmcif_loop_data(contexts)


def make_mmcif_dict(key: str, subkeys: list) -> dict:
    """Make mmCIF dict data."""
    assert len(subkeys) > 0 and isinstance(subkeys, list)
    mmcif_dict = {}
    for subkey in subkeys:
        mmcif_dict[f'{key}.{subkey}'] = []
    return mmcif_dict


def mmcif_meta_append(mmcif_path: str, extra_infos: dict) -> str:
    """Append meta information to the start of mmcif file.
        ref: ANNOTATION, MMCIF_CONSTANTS, LOOP_MMCIF_CONSTANTS
    """
    with open(mmcif_path) as fh:
        lines = fh.readlines()
    head_lines = lines[:2] ## data_xxxx + #
    head_lines = [f"{ANNOTATION}\n"] + head_lines
    rest_lines = lines[2:]

    contexts = []
    for k, subkey in META_REQUIRED_KEYS.items():
        cif = {f"{k}.{v}": [] for v in subkey}
        for v in subkey:
            meta_keys = f"{k}.{v}"
            if meta_keys in MMCIF_CONSTANTS:
                cif[meta_keys].append(MMCIF_CONSTANTS[meta_keys])
            else:
                cif[meta_keys].extend(LOOP_MMCIF_CONSTANTS[meta_keys])

        #### special keys, need to be updated. ####
        if k == '_entry':
            cif['_entry.id'] = [extra_infos['entry_id']]
        elif k == '_ma_qa_metric_global':
            cif['_ma_qa_metric_global.metric_value'] = [extra_infos['global_plddt']]
        elif k == '_ma_model_list':
            _time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            _mod_group_name = f"\"{MODEL_VERSION} @ {_time_stamp}\""
            cif['_ma_model_list.model_group_name'] = [_mod_group_name]
        #### special keys, need to be updated. ####

        assert len(cif[f"{k}.{subkey[0]}"]) > 0, \
                (f"{k}.{subkey[0]} is not assigned values. Please check your input")

        context_list = _format_mmcif_loop_data(cif)
        contexts.extend(context_list)

    contexts.append("\n")
    with open(mmcif_path, 'w') as fh:
        fh.write(''.join(head_lines))
        fh.write('\n'.join(contexts))
        fh.write(''.join(rest_lines))
        
    return mmcif_path


def mmcif_bonds_append(mmcif_path:str, contexts:dict, rm_duplicates=False) -> str:
    """Append a context to an mmCIF file.

        Ref: https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Index/
            For example:
                _chem_comp_atom.comp_id: ['UNK', 'UNK'] 
                _chem_comp_atom.atom_id: ['C', 'CA']
                _chem_comp_atom.alt_atom_id : ['C', 'CA']
        
        Args:
            mmcif_path: str, path to the mmCIF file
            contexts: dict of contexts to append, followed by a the mmcif format.
            rm_duplicates: bool, whether to remove duplicate lines
        Returns:
            str: path to the mmCIF file
    """
    context_list = _format_mmcif_loop_data(
        contexts, 
        value_formatter=lambda x: str(x).ljust(4, ' ')
    )

    if rm_duplicates:
        loop_start = context_list.index('loop_')
        data_start = loop_start + 1
        data_end = context_list.index('#')
        
        seen_lines = set()
        unique_lines = []
        for line in context_list[data_start:data_end]:
            if line not in seen_lines:
                seen_lines.add(line)
                unique_lines.append(line)
        context_list = context_list[:data_start] + unique_lines + context_list[data_end:]
    context_list[-1] = context_list[-1] + '\n'

    with open(mmcif_path) as fh:
        lines = fh.readlines()
    with open(mmcif_path, 'w') as fh:
        fh.write(''.join(lines))
        fh.write('\n'.join(context_list))

    return mmcif_path


def prediction_to_mmcif(entry_name: str,
                      atom_positions: np.ndarray, 
                      feats_dict: dict, 
                      mmcif_path: str,
                      extra_infos: dict, 
                      ligand_intra_bonds_info=None) -> str:
    """Convert prediction position, to mmcif file.
    
    Args:
        entry_name: str, name of the mmcif file
        prediction_atom_pos: np.ndarray, prediction atom positions, shape: (N_atom, 3)
        feats_dict: dict, features dict, features dict for saving.
        mmcif_path: path to save *.cif
        extra_infos: dict, extra infos to be written in mmcif file.
    
    Returns:
        str: path to the mmcif file
    """
    def _validate_input(feats_dict, atom_positions, mmcif_path):
        assert mmcif_path.endswith('.cif'), \
            f'mmcif_path should endswith .cif; got {mmcif_path}'
        assert isinstance(atom_positions, np.ndarray), \
            f"atom_positions should be a numpy array; got {type(atom_positions)}"
        assert atom_positions.shape[0] == len(feats_dict["all_atom_ids"]), \
            f"The number of atoms in the prediction does not match the number of atoms in the features. "
        assert np.all(feats_dict["restype"] < residue_constants.HF3_restype_nums), \
            f'Invalid aatypes. Got: {np.unique(feats_dict["restype"])}'    
    _validate_input(feats_dict, atom_positions, mmcif_path)

    def _reorder_feats_by_ori_chain_ids(feats_dict, ori_chain_ids):
        ent_weights, sym_weights = user_asymid_to_weight(ori_chain_ids)
        ## np.lexsort is stable sort: https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html#numpy.lexsort
        sorted_indices = np.lexsort((sym_weights, ent_weights)) 
        reordered_feats_dict = {key: np.take(feat, sorted_indices, axis=0) 
                                        for key, feat in feats_dict.items()}

        return reordered_feats_dict

    ligand_intra_bonds_info = _prepare_bonds(feats_dict) if ligand_intra_bonds_info is None else ligand_intra_bonds_info
    ref_token2atom_idx = feats_dict["ref_token2atom_idx"] 
    
    aatype = feats_dict["restype"][ref_token2atom_idx] 
    residue_index = feats_dict["residue_index"][ref_token2atom_idx].astype(np.int32) 
    residue_index += 1 ## asym_id start with 1
    chain_index = feats_dict["asym_id"][ref_token2atom_idx].astype(np.int32)
    chain_index -= 1 ## resid start with 1
    is_ligand = feats_dict['is_ligand'][ref_token2atom_idx]  
    is_protein = feats_dict['is_protein'][ref_token2atom_idx] 
    is_dna = feats_dict['is_dna'][ref_token2atom_idx]  
    is_rna = feats_dict['is_rna'][ref_token2atom_idx]
    
    ori_feat_dict = {
        'original_chain_ids': feats_dict["all_chain_ids"],
        'ccd_ids': feats_dict["all_ccd_ids"],
        'atom_ids': feats_dict["all_atom_ids"],
        'atom_mask': feats_dict["all_atom_pos_mask"],
        'b_factors': feats_dict['atom_plddts'],
        'aatype': aatype,
        'residue_index': residue_index,
        'chain_index': chain_index,
        'is_ligand': is_ligand,
        'is_protein': is_protein,
        'is_dna': is_dna,
        'is_rna': is_rna,
        'atom_positions': atom_positions,
        'atom_plddts': feats_dict['atom_plddts']
    }
    reordered_feats_dict = _reorder_feats_by_ori_chain_ids(ori_feat_dict, feats_dict["all_chain_ids"])
    del ori_feat_dict

    cif_path = _write_cif_body(reordered_feats_dict, entry_name, mmcif_path)
    _write_cif_context(cif_path, entry_name, reordered_feats_dict, extra_infos, ligand_intra_bonds_info)

    return cif_path


def _write_cif_body(reordered_feats_dict, entry_name, mmcif_path):
    original_chain_ids = reordered_feats_dict['original_chain_ids']
    ccd_ids = reordered_feats_dict['ccd_ids']
    atom_ids = reordered_feats_dict['atom_ids']
    atom_mask = reordered_feats_dict['atom_mask']
    b_factors = reordered_feats_dict['b_factors']
    aatype = reordered_feats_dict['aatype']
    residue_index = reordered_feats_dict['residue_index']
    chain_index = reordered_feats_dict['chain_index']
    is_ligand = reordered_feats_dict['is_ligand']
    is_protein = reordered_feats_dict['is_protein']
    is_dna = reordered_feats_dict['is_dna']
    is_rna = reordered_feats_dict['is_rna']
    atom_positions = reordered_feats_dict['atom_positions']


    ######### NOTE: start writing cif file ##########
    #### key for basic structure info #### 
    basic_infos_dict = {}
    for k, subkeys in STRUCTURE_REQUIRED_KEYS.items():
        basic_infos_dict[k] = make_mmcif_dict(k, subkeys)
    #### key for basic structure info #### 

    atom_index = 1
    entity_id = 0
    seen_chain_ids = {}
    seen_entity_poly_seq = set()
    seen_pdbx_poly_seq_scheme = set()
    residue_level_plddt = collections.OrderedDict() ## (label_asym_id, label_comp_id, label_seq_id): list of atom score.

    # Add all atom sites.
    for i in range(aatype.shape[0]):
        res_name_3 = ccd_ids[i]
        atom_name, pos, mask, b_factor = atom_ids[i], atom_positions[i], \
                                atom_mask[i], b_factors[i]

        if mask < 0.5:
            continue

        record_type = 'ATOM' if res_name_3 in residue_constants.STANDARD_LIST else 'HETATM'
        name = atom_name if len(atom_name) == 4 else f' {atom_name}'
        occupancy = 1.00
        element = atom_name[0]
        if record_type == "HETATM" and res_name_3.lower() in \
                            map(str.lower, residue_constants.ATOM_ELEMENT.keys()):
            element = atom_name
        
        ## record data_type, entity_id, chain_asym_id, ccd_name ####
        # chain_asym_id = chain_ids[chain_index[i]]
        chain_asym_id = original_chain_ids[i]
        if chain_asym_id not in seen_chain_ids:
            entity_id += 1

            if is_ligand[i] == 1:
                _dtype = 'ligand'
            elif is_protein[i] == 1:
                _dtype = 'protein'
            elif is_dna[i] == 1:
                _dtype = 'dna'
            elif is_rna[i] == 1:
                _dtype = 'rna'
            else:
                raise ValueError('Unknown residue type.')            
            seen_chain_ids[chain_asym_id] = {'entity_id': entity_id, 
                                            "dtype": _dtype, 'ccd_name': res_name_3}

        ## _atom_site ####
        basic_infos_dict['_atom_site']['_atom_site.group_PDB'].append(record_type)
        basic_infos_dict['_atom_site']['_atom_site.id'].append(atom_index)
        basic_infos_dict['_atom_site']['_atom_site.type_symbol'].append(element)
        basic_infos_dict['_atom_site']['_atom_site.label_atom_id'].append(name)
        basic_infos_dict['_atom_site']['_atom_site.label_alt_id'].append('.')
        basic_infos_dict['_atom_site']['_atom_site.label_comp_id'].append(res_name_3)
        basic_infos_dict['_atom_site']['_atom_site.label_asym_id'].append(chain_asym_id)
        basic_infos_dict['_atom_site']['_atom_site.label_entity_id'].append(entity_id)
        basic_infos_dict['_atom_site']['_atom_site.label_seq_id'].append(residue_index[i])
        basic_infos_dict['_atom_site']['_atom_site.pdbx_PDB_ins_code'].append('?')
        basic_infos_dict['_atom_site']['_atom_site.Cartn_x'].append(f"{pos[0]:.3f}")
        basic_infos_dict['_atom_site']['_atom_site.Cartn_y'].append(f"{pos[1]:.3f}")
        basic_infos_dict['_atom_site']['_atom_site.Cartn_z'].append(f"{pos[2]:.3f}")
        basic_infos_dict['_atom_site']['_atom_site.occupancy'].append(f"{occupancy:.2f}")
        basic_infos_dict['_atom_site']['_atom_site.B_iso_or_equiv'].append(f"{b_factor:.2f}")
        basic_infos_dict['_atom_site']['_atom_site.auth_seq_id'].append(residue_index[i])
        basic_infos_dict['_atom_site']['_atom_site.auth_asym_id'].append(chain_asym_id)
        basic_infos_dict['_atom_site']['_atom_site.pdbx_PDB_model_num'].append(1)

        if (chain_asym_id, res_name_3, residue_index[i]) not in residue_level_plddt:
            residue_level_plddt[(chain_asym_id, res_name_3, residue_index[i])] = []
        residue_level_plddt[(chain_asym_id, res_name_3, residue_index[i])].append(b_factor)

        ## _entity_poly_seq, 
        ## _pdbx_poly_seq_scheme
        if is_ligand[i] != 1 and (entity_id, 'n', res_name_3, residue_index[i]) not in seen_entity_poly_seq:
            basic_infos_dict['_entity_poly_seq']['_entity_poly_seq.entity_id'].append(entity_id)
            basic_infos_dict['_entity_poly_seq']['_entity_poly_seq.hetero'].append('n')
            basic_infos_dict['_entity_poly_seq']['_entity_poly_seq.mon_id'].append(res_name_3)
            basic_infos_dict['_entity_poly_seq']['_entity_poly_seq.num'].append(residue_index[i])
            seen_entity_poly_seq.add((entity_id, 'n', res_name_3, residue_index[i]))

        if is_ligand[i] != 1 and (entity_id, chain_asym_id, 'n', res_name_3, residue_index[i]) not in seen_pdbx_poly_seq_scheme:
            basic_infos_dict['_pdbx_poly_seq_scheme']['_pdbx_poly_seq_scheme.asym_id'].append(chain_asym_id)
            basic_infos_dict['_pdbx_poly_seq_scheme']['_pdbx_poly_seq_scheme.auth_seq_num'].append(residue_index[i])
            basic_infos_dict['_pdbx_poly_seq_scheme']['_pdbx_poly_seq_scheme.entity_id'].append(entity_id)
            basic_infos_dict['_pdbx_poly_seq_scheme']['_pdbx_poly_seq_scheme.hetero'].append('n')
            basic_infos_dict['_pdbx_poly_seq_scheme']['_pdbx_poly_seq_scheme.mon_id'].append(res_name_3)
            basic_infos_dict['_pdbx_poly_seq_scheme']['_pdbx_poly_seq_scheme.pdb_ins_code'].append('.')
            basic_infos_dict['_pdbx_poly_seq_scheme']['_pdbx_poly_seq_scheme.pdb_seq_num'].append(residue_index[i])
            basic_infos_dict['_pdbx_poly_seq_scheme']['_pdbx_poly_seq_scheme.pdb_strand_id'].append(chain_asym_id)
            basic_infos_dict['_pdbx_poly_seq_scheme']['_pdbx_poly_seq_scheme.seq_id'].append(residue_index[i])
            seen_pdbx_poly_seq_scheme.add((entity_id, chain_asym_id, 'n', res_name_3, residue_index[i]))

        atom_index += 1

    ## followed by af3_server ####
    ## NOTE：entity_id == asym_id
    assert entity_id == len(seen_chain_ids)

    for chain_id, chain_info in seen_chain_ids.items():
        
        enti_id = chain_info['entity_id']
        dtype = chain_info['dtype']
        ccd_name = chain_info['ccd_name']

        ## _entity
        basic_infos_dict['_entity']['_entity.id'].append(enti_id)
        basic_infos_dict['_entity']['_entity.pdbx_description'].append('.')
        if dtype != 'ligand':
            basic_infos_dict['_entity']['_entity.type'].append('polymer')
        else:
            basic_infos_dict['_entity']['_entity.type'].append('non-polymer')
        
        ## _struct_asym
        basic_infos_dict['_struct_asym']['_struct_asym.entity_id'].append(enti_id)
        basic_infos_dict['_struct_asym']['_struct_asym.id'].append(chain_id)

        ## _ma_target_entity
        basic_infos_dict['_ma_target_entity']['_ma_target_entity.data_id'].append(1)
        basic_infos_dict['_ma_target_entity']['_ma_target_entity.entity_id'].append(enti_id)
        basic_infos_dict['_ma_target_entity']['_ma_target_entity.origin'].append('.')

        ## _ma_target_entity_instance
        basic_infos_dict['_ma_target_entity_instance']['_ma_target_entity_instance.asym_id'].append(chain_id)
        basic_infos_dict['_ma_target_entity_instance']['_ma_target_entity_instance.details'].append('.')
        basic_infos_dict['_ma_target_entity_instance']['_ma_target_entity_instance.entity_id'].append(enti_id)

        if dtype != 'ligand':
            ## _entity_poly
            basic_infos_dict['_entity_poly']['_entity_poly.entity_id'].append(enti_id)
            basic_infos_dict['_entity_poly']['_entity_poly.pdbx_strand_id'].append(chain_id)
            basic_infos_dict['_entity_poly']['_entity_poly.type'].append(POLY_MAPPING_TYPE[dtype])
        else:
            ## _pdbx_nonpoly_scheme
            basic_infos_dict['_pdbx_nonpoly_scheme']['_pdbx_nonpoly_scheme.asym_id'].append(chain_id)
            basic_infos_dict['_pdbx_nonpoly_scheme']['_pdbx_nonpoly_scheme.auth_seq_num'].append(1)
            basic_infos_dict['_pdbx_nonpoly_scheme']['_pdbx_nonpoly_scheme.entity_id'].append(enti_id)
            basic_infos_dict['_pdbx_nonpoly_scheme']['_pdbx_nonpoly_scheme.mon_id'].append(ccd_name)
            basic_infos_dict['_pdbx_nonpoly_scheme']['_pdbx_nonpoly_scheme.pdb_ins_code'].append('.')
            basic_infos_dict['_pdbx_nonpoly_scheme']['_pdbx_nonpoly_scheme.pdb_seq_num'].append(1)
            basic_infos_dict['_pdbx_nonpoly_scheme']['_pdbx_nonpoly_scheme.pdb_strand_id'].append(chain_id)
    
    # residue_level_plddt[(chain_asym_id, res_name_3, residue_index[i])]: b_factor_list
    for (asym_id, res_name_3, residue_index), b_factor_list in residue_level_plddt.items():
        basic_infos_dict['_ma_qa_metric_local']['_ma_qa_metric_local.label_asym_id'].append(asym_id)
        basic_infos_dict['_ma_qa_metric_local']['_ma_qa_metric_local.label_comp_id'].append(res_name_3)
        basic_infos_dict['_ma_qa_metric_local']['_ma_qa_metric_local.label_seq_id'].append(residue_index)
        basic_infos_dict['_ma_qa_metric_local']['_ma_qa_metric_local.metric_id'].append(2)
        basic_infos_dict['_ma_qa_metric_local']['_ma_qa_metric_local.metric_value'].append(np.round(np.mean(b_factor_list), 2))
        basic_infos_dict['_ma_qa_metric_local']['_ma_qa_metric_local.model_id'].append(1)
        basic_infos_dict['_ma_qa_metric_local']['_ma_qa_metric_local.ordinal_id'].append(residue_index)


    cif_lines = []
    cif_lines.append(f"data_{entry_name}\n#")
    for key in STRUCTURE_REQUIRED_KEYS:
        context_dict = basic_infos_dict[key]
        _cif_line = format_mmcif_dict(context_dict)
        cif_lines.extend(_cif_line)

    cif_lines_string = '\n'.join(cif_lines)
    cif_lines_string += '\n'
    with open(mmcif_path, 'w') as f:
        f.write(cif_lines_string)

    return mmcif_path


def _prepare_bonds(common_feat_masked):
    """Prepare ligand bond type info."""
    ccd_ids = common_feat_masked['all_ccd_ids'] # N_atom
    atom_ids = common_feat_masked['all_atom_ids'] # N_atom

    token_bond_type = common_feat_masked['token_bonds_type'] # N_token 
    ref_token2atom_idx = common_feat_masked["ref_token2atom_idx"]
    is_ligand_aa = common_feat_masked['is_ligand'][ref_token2atom_idx]
    bond_mat = token_bond_type[ref_token2atom_idx][:, ref_token2atom_idx]

    ret = []
    # (ccd_id, atom_id_1, atom_id_2, bond_type)
    for i, j in zip(*np.where(bond_mat>0)):
        if i < j and is_ligand_aa[i] and is_ligand_aa[j]:
            assert ccd_ids[i] == ccd_ids[j]
            ret.append((ccd_ids[i], atom_ids[i], atom_ids[j],
                    RdkitConstants.INVERSE_ALLOWED_LIGAND_BONDS_TYPE_MAP[bond_mat[i][j]]))
    return ret


def _write_cif_context(pred_cif_path, entry_name, common_feat_masked, extra_infos, ligand_intra_bonds_info):
    """Write extra context(like license or bondtype) to cif.
    """
    # 1. license
    mmcif_extra_infos = {'entry_id': entry_name, "global_plddt": f"{float(common_feat_masked['atom_plddts'].mean()):.2f}"}
    mmcif_extra_infos.update(extra_infos)
    mmcif_meta_append(pred_cif_path, mmcif_extra_infos)
    
    # 2. add ligand bond type;
    if len(ligand_intra_bonds_info) > 0:
        columns = zip(*ligand_intra_bonds_info)
        contexts = {}
        contexts['_chem_comp_bond.comp_id'], contexts['_chem_comp_bond.atom_id_1'], \
        contexts['_chem_comp_bond.atom_id_2'], contexts['_chem_comp_bond.value_order'] = (list(col) for col in columns)
        mmcif_bonds_append(pred_cif_path, contexts, rm_duplicates=True)
