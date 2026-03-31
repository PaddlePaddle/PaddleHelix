"""
Error handling utilities for input validation and preprocessing.

This module contains functions for handling errors found during
entity validation and preprocessing.
"""

import logging
import os
import sys
import copy
from typing import List, Dict, Any, Tuple

from infer_scripts.tools.utils import write_json
from infer_scripts.entity_bean import EntityBean

logger = logging.getLogger(__file__)


def process_entity_errors(
    entities: List[Dict[str, Any]], 
    entity_beans: List[EntityBean]
) -> Tuple[List[EntityBean], List[str], List[Dict[str, Any]]]:
    """
    Process entities with errors by marking them as UNSUPPORTED
    and collecting error messages.
    
    Args:
        entities: Original input entities
        entity_beans: List of EntityBean objects (may contain errors)
        
    Returns:
        Tuple containing:
        - List of valid entities
        - List of error messages
        - Modified entities list with UNSUPPORTED markers
    """
    success_entity = []
    error_message_list = []
    
    # Create a deep copy of original entities to avoid modifying the input
    modified_entities = copy.deepcopy(entities)
    
    for idx, entity_bean in enumerate(entity_beans):
        if entity_bean.error_code == 0:
            success_entity.append(entity_bean)
        else:
            error_message_list.append(entity_bean.error_message)
            _type = modified_entities[idx]['type']
            if 'ion' in _type or 'ligand' in _type:
                if 'ccd' in modified_entities[idx] and modified_entities[idx]['ccd']:
                    modified_entities[idx]['ccd'] = 'UNSUPPORTED'
                elif 'smiles' in modified_entities[idx] and modified_entities[idx]['smiles']:
                    modified_entities[idx]['smiles'] = 'UNSUPPORTED'
            else:
                modified_entities[idx]['sequence'] = 'UNSUPPORTED'
    
    logger.warning(f'Error occurs on {len(error_message_list)}/{len(entities)} entities.')
    return success_entity, error_message_list, modified_entities


def write_output_files(
    input_json_dict: Dict[str, Any],
    error_message_list: List[str],
    out_dir: str
) -> None:
    """
    Write output files for error cases.
    
    Args:
        entities: List of entities (may contain UNSUPPORTED markers)
        error_message_list: List of error messages
        out_dir: Directory to write output files
        
    Raises:
        SystemExit: If there are any errors in the entities
    """
    user_input_path = os.path.join(out_dir, 'user_input.json')
    job_status_path = os.path.join(out_dir, 'job_status.json')
    
    error_infos = {"status": "failed"}
    error_infos['job_fail_reason'] = ';'.join(set(error_message_list))
    write_json(input_json_dict, user_input_path)
    write_json(error_infos, job_status_path, cn=True)
    logger.error(f'[Failed] Error occurred when processing entities: [{user_input_path}, {job_status_path}]')


def handle_entity_errors(
    input_json_dict: Dict[str, Any], 
    entity_beans: List[EntityBean],
    out_dir: str
) -> List[EntityBean]:
    """
    Handle entities with errors by marking them as UNSUPPORTED
    and generating appropriate error messages.
    
    Args:
        entities: Original input entities
        entity_beans: List of EntityBean objects (may contain errors)
        out_dir: Directory to write output files
        
    Returns:
        List of valid entities
        
    Raises:
        SystemExit: If there are any errors in the entities
    """
    entities = input_json_dict['entities']
    success_entity, error_message_list, modified_entities = process_entity_errors(entities, entity_beans)
    input_json_dict['entities'] = modified_entities
    write_output_files(input_json_dict, error_message_list, out_dir)
    sys.exit(1)
