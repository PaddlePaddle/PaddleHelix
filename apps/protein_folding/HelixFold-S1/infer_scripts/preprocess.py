"""
    Convert online server json to hf3 input/json
"""
import logging
import dataclasses
from typing import Optional, List
import copy
import itertools

from helixfold.common import residue_constants
from infer_scripts.rdkit_utils import SMILESParseError, ConformerGenerationError
from infer_scripts.rdkit_utils import (
    RdkitConstants, smiles_to_rdMol,
    sidechain_smiles_to_rdMol, make_basic_feature_fromMol)
from infer_scripts.entity_bean import EntityBean
from infer_scripts.tools.utils import read_json
from infer_scripts.validation.error_handling import handle_entity_errors

logger = logging.getLogger(__file__)
PROTEIN_1to3_with_x = residue_constants.PROTEIN_1to3_with_x
DNA_1to2_with_x = residue_constants.DNA_RNA_1to2_with_x_and_gap['dna']
RNA_1to2_with_x = residue_constants.DNA_RNA_1to2_with_x_and_gap['rna']
EXCLUDE_CCD_LIST = set(residue_constants.crystallization_aids) | set(residue_constants.ligand_exclusion_list)
USER_LIG_IDS = 'abcdefghijklmnopqrstuvwxyz0123456789'
USER_LIG_IDS_3 = [''.join(pair) for pair in itertools.product(USER_LIG_IDS, repeat=3)]


def polymer_convert(items) -> EntityBean:
    """
        "type": "protein",                          
        "sequence": "GPDSMEEVVVPEEPPKLVSALATYVQQERLCTMFLSIANKLLPLKP",  
        "count": 1
            OR :
        "type": "protein",                             
        "sequence": "AGPDSMEEVVVPEEPPKLVSALATYVQQERLCTMFLSIANKLLPLKP",    
        "count": 1,                                    
        "modification": [                              
            {
                "type": "residue_replace",             
                "index": 1,                            
                "ccd": "CIR"                           
            },
            {
                "type": "residue_replace",                      
                "index": 5,                            
                "ccd": "SEP"               
            }
        ]
    """
    def _modified_residue_convert(modifications: list, ccd_seqs: list, dtype: str):
        modi_index_to_ccd = {}
        smiles_replace = {}
        backbone_smiles = RdkitConstants.MODIFIED_BACKBONE_NAME_MAPPING[dtype]['backbone']
        backbone_infos = RdkitConstants.MODIFIED_BACKBONE_NAME_MAPPING[dtype]['name']
        modified_indices = []
        
        for modify in modifications:
            index = modify['index']
            modified_indices.append(index)
            
            if modify['type'] == 'residue_replace':
                ccd = modify['ccd']
                modi_index_to_ccd[index] = ccd
            elif modify['type'] == 'sidechain_replace':
                user_ccd = f"UNK-{index}"
                modi_index_to_ccd[index] = user_ccd
                R_replace_smiles = modify['R_smiles']
                R_connect_idx = modify['R_connect_idx']

                gen_mol2 = sidechain_smiles_to_rdMol(backbone_smiles, R_replace_smiles, 
                                connect_idx=R_connect_idx, backbone_infos=backbone_infos)
                smiles_replace.update({
                    user_ccd: make_basic_feature_fromMol(gen_mol2, reset_atom_ids=False)})

        for index, ccd in modi_index_to_ccd.items():
            ccd_seqs[index - 1] = f"({ccd})"

        return ccd_seqs, smiles_replace

    dtype = items['type']
    one_letter_seqs = items['sequence']
    count = items['count']
    modifications = items.get('modification', [])
    smiles_replace = {}

    msa_seqs = one_letter_seqs
    ccd_seqs = []
    for resi_name_1 in one_letter_seqs:
        if dtype == 'protein':
            ccd_seqs.append(f"({PROTEIN_1to3_with_x[resi_name_1]})")
        elif dtype == 'dna':
            ccd_seqs.append(f"({DNA_1to2_with_x[resi_name_1]})")
        elif dtype == 'rna':
            ccd_seqs.append(f"({RNA_1to2_with_x[resi_name_1]})")
    
    if len(modifications) > 0:
        try:
            ccd_seqs, smiles_replace = _modified_residue_convert(modifications, ccd_seqs, dtype)
        except Exception as e:
            logger.error(f'[modified_polymer_convert] {e}')
            return EntityBean.create_with_kwargs(error_code=16, message=e.message)

    ccd_seqs = ''.join(ccd_seqs) ## (GLY)(ALA).....
    raw_info = {
        'sequence': one_letter_seqs
    }
    entity = EntityBean(
        dtype=dtype,
        seqs=ccd_seqs,
        msa_seqs=msa_seqs,
        count=count,
        extra_mol_infos=smiles_replace,
        raw_info=raw_info,
        error_code=0
    )
    return entity


def ligand_convert(items) -> EntityBean:
    """
        "type": "ligand" or "ion",
        "ccd": "ATP", or "smiles": "CCccc(O)ccc",
        "count": 1
    """
    dtype = items['type']
    count = items['count']
    msa_seqs = ""
    _ccd_seqs = []
    ccd_to_extra_mol_infos = {}
    if 'ccd' in items and len(items['ccd']) > 0:
        _ccd_seqs.append(f"({items['ccd']})")
    elif 'smiles' in items and len(items['smiles']) > 0:
        _ccd_seqs.append(f"(UNK-)")
        try:
            mol_wo_h = smiles_to_rdMol(items['smiles'])
            ccd_to_extra_mol_infos = {
                "UNK-": make_basic_feature_fromMol(mol_wo_h)
            }
        except SMILESParseError as e:
            logger.error(f"[ligand_convert] {e}")
            return EntityBean.create_with_kwargs(error_code=1, smiles=items['smiles'])
        except ConformerGenerationError as e:
            logger.error(f"[ligand_convert] {e}")
            return EntityBean.create_with_kwargs(error_code=1, smiles=items['smiles'])
        except Exception as e:
            logger.error(f"[ligand_convert] Failed when converting SMILES to MOL: {e}")
            return EntityBean.create_with_kwargs(error_code=1, smiles=items['smiles'])
    else:
        logger.error(f"[ligand_convert] Neither CCD nor SMILES provided for {items}")
        return EntityBean(error_code=2)

    ccd_seqs = ''.join(_ccd_seqs) ## (GLY)(ALA).....
    raw_info = {
        'ccd': items['ccd'] if 'ccd' in items else '',
        'smiles': items['smiles'] if 'smiles' in items else '',
    }

    entity = EntityBean(
        dtype="ligand",
        seqs=ccd_seqs,
        msa_seqs=msa_seqs,
        count=count,
        extra_mol_infos=ccd_to_extra_mol_infos,
        raw_info=raw_info,
        error_code=0
    )
    return entity


def extra_ccd_rename(entities: List[EntityBean]) -> List[EntityBean]:
    """
        rename the extra ccd to user-defined ccd.
        such as: (UNK-1) -> (aaa), (UNK-2) -> (aab), ...
    """
    cur_idx = 0
    for entity_items in entities:
        ## dict, 「extra-add, ccd_id」: ccd_features.
        extra_mol_infos = entity_items.extra_mol_infos 
        extra_ccd_ids = list(extra_mol_infos.keys())
        ## rename UNK-* to aaa, aab, aac, ...
        for k in extra_ccd_ids:
            user_name_3 = USER_LIG_IDS_3[cur_idx]
            entity_items.seqs = entity_items.seqs.replace(f"({k})", f"({user_name_3})")
            entity_items.extra_mol_infos[user_name_3] = extra_mol_infos.pop(k)
            cur_idx += 1

    return entities


def postprocess_entity(entities: List[EntityBean], raw_json: dict) -> List[EntityBean]:
    """
        Postprocess the entity list.
        1. Rename the extra ccd to user-defined ccd.
        2. Expand the entity by the count and add asym_chain_id.
            asym_chain_id follows the format: first number is the `entity_id`, the second number is the `sym_chain_id`.
            For example, 1-1, 1-2, 1-3, 2-1, 2-2...
    """
    ## NOTE: entities is the list of EntityBean, follows the order of the input JSON file.
    entities = extra_ccd_rename(entities)
    
    final_entities = []
    for entity_id, entity in enumerate(entities, start=1):
        sym_chain_id = 1
        for _ in range(entity.count):
            entity_copy = copy.deepcopy(entity)
            entity_copy.raw_info['asym_chain_id'] = f"{entity_id}-{sym_chain_id}"
            entity_copy.raw_info['model_type'] = raw_json.get('model_type', "HelixFold3")
            entity_copy.count = 1
            final_entities.append(entity_copy)
            sym_chain_id += 1
    
    return final_entities


def online_json_parser(json_path, out_dir) -> List[EntityBean]:
    """
        Convert online server json to HF3 input entity list.
    """
    # 直接读取 JSON 文件，不再使用 validate_input_file
    logger.info(f'Start to convert entities from JSON file: {json_path}')
    
    data = read_json(json_path)
    if 'random_seed' in data:
        del data['random_seed']
    entity_dicts = data['entities']
    
    # 转换实体
    converted_entities = []
    for entity in entity_dicts:
        if entity['type'] in ['ligand', 'ion']:
            entity_bean = ligand_convert(entity)
        elif entity['type'] in ['protein', 'dna', 'rna']:
            entity_bean = polymer_convert(entity)
        converted_entities.append(entity_bean)
    
    if any(entity.error_code != 0 for entity in converted_entities):
            valid_entities = handle_entity_errors(
                input_json_dict=data,
                entity_beans=converted_entities,
                out_dir=out_dir
            )

    logger.info(f'Successfully converted {len(converted_entities)} entities')
    return postprocess_entity(converted_entities, data)

