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

"""Convert json input to list of EntityBean."""

import logging
import copy
import itertools
from typing import Dict, List

from helixfold.common import residue_constants
from helixfold.data.utils import int_id_to_str_id
from infer_scripts.rdkit_utils import SMILESParseError, ConformerGenerationError
from infer_scripts.rdkit_utils import smiles_to_rdMol, make_basic_feature_fromMol
from infer_scripts.entity_bean import EntityBean
from infer_scripts.tools.utils import read_json

logger = logging.getLogger(__file__)
PROTEIN_1to3_with_x = residue_constants.PROTEIN_1to3_with_x
DNA_1to2_with_x = residue_constants.DNA_RNA_1to2_with_x_and_gap['dna']
RNA_1to2_with_x = residue_constants.DNA_RNA_1to2_with_x_and_gap['rna']
EXCLUDE_CCD_LIST = set(residue_constants.crystallization_aids) | set(residue_constants.ligand_exclusion_list)
USER_LIG_IDS = 'abcdefghijklmnopqrstuvwxyz0123456789'
USER_LIG_IDS_3 = [''.join(pair) for pair in itertools.product(USER_LIG_IDS, repeat=3)]


def polymer_convert(items: Dict) -> EntityBean:
    """Convert polymer from raw json_dict to EntityBean.

    Args:
        items (dict): The polymer entity dict.

    Returns:
        entity (EntityBean): The entity bean of polymer.
    """
    def _modified_residue_convert(modifications: List[Dict], 
                                    ccd_seqs: List[str]) -> List[str]:
        """Handling the modification of residue.
        
        Args:
            modifications (list): The modification list 
            ccd_seqs (list): The ccd sequence list

        Returns:
            ccd_seqs (list): The updated ccd sequence list with modification.
        """
        modi_index_to_ccd = {}
        modified_indices = []
        
        for modify in modifications:
            index = modify['index']
            modified_indices.append(index)
            
            if modify['type'] == 'residue_replace':
                ccd = modify['ccd']
                modi_index_to_ccd[index] = ccd
            else:
                raise ValueError(f'Unknown modification type: {modify["type"]}')

        for index, ccd in modi_index_to_ccd.items():
            ccd_seqs[index - 1] = f"({ccd})"

        return ccd_seqs

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
            ccd_seqs = _modified_residue_convert(modifications, ccd_seqs)
        except Exception as e:
            logger.error(f'[modified_polymer_convert] {e}')
            return EntityBean.create_with_kwargs(error_code=16, message=str(e))

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


def ligand_convert(items: Dict) -> EntityBean:
    """Convert ligand from raw json_dict to EntityBean.
        
    Args:
        items (dict): The polymer entity dict. For example:
            {
                "type": "ligand" or "ion",
                "ccd": "ATP", or "smiles": "CCccc(O)ccc",
                "count": 1
            }
    Returns:
        entity (EntityBean): The entity bean of polymer.
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
    """Rename the extra ccd to user-defined ccd.
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


def postprocess_entity(entities: List[EntityBean]) -> List[EntityBean]:
    """Postprocess the entity list. Include the following steps:
    - Rename the extra ccd to user-defined ccd.
    - Expand the entity by the count and add asym_chain_id.
        asym_chain_id follows the format: first number is the `entity_id`, the second number is the `sym_chain_id`.
        For example, A-1, A-2, A-3, B-1, B-2...
    """
    ## NOTE: entities is the list of EntityBean, follows the order of the input JSON file.
    entities = extra_ccd_rename(entities)
    
    final_entities = []
    for entity_id, entity in enumerate(entities, start=1):
        sym_chain_id = 1
        entity_id_letters = int_id_to_str_id(entity_id)
        for _ in range(entity.count):
            entity_copy = copy.deepcopy(entity)
            entity_copy.raw_info['asym_chain_id'] = f"{entity_id_letters}-{sym_chain_id}"
            entity_copy.count = 1
            final_entities.append(entity_copy)
            sym_chain_id += 1
    
    return final_entities


def online_json_parser(json_path: str, out_dir: str = None) -> List[EntityBean]:
    """Convert raw input json to simple HF3 input entity list."""

    logger.info(f'Start to convert entities from JSON file: {json_path}')
    
    data = read_json(json_path)
    entity_dicts = data['entities']
    
    converted_entities = []
    for entity in entity_dicts:
        if entity['type'] in ['ligand', 'ion']:
            entity_bean = ligand_convert(entity)
        elif entity['type'] in ['protein', 'dna', 'rna']:
            entity_bean = polymer_convert(entity)
        converted_entities.append(entity_bean)
    
    logger.info(f'Successfully converted {len(converted_entities)} entities')
    return postprocess_entity(converted_entities)

