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

"""Validation functions for input preprocessing.

This module contains functions to validate different aspects of the input data
before it's processed by the preprocessing pipeline.
"""

import logging
import json
import re
from typing import List, Dict, Any, Tuple, Optional

from rdkit import Chem
from jsonschema import validate, ValidationError

from helixfold.common import residue_constants
from ..entity_bean import EntityBean
from . import ValidationException

logger = logging.getLogger(__file__)

# Constants for validation
EXCLUDE_CCD_LIST = set(residue_constants.crystallization_aids) | set(residue_constants.ligand_exclusion_list)

# Sequence patterns for validation
SEQUENCE_PATTERNS = {
    "protein": r"^[ACDEFGHIKLMNPQRSTVWYX]+$",
    "dna": r"^[ACGTX]+$",
    "rna": r"^[ACGUX]+$"
}

def validate_polymer_sequence(entity: Dict[str, Any]) -> Tuple[bool, Optional[EntityBean]]:
    """Validate a polymer sequence (protein, DNA, RNA).
    
    Args:
        entity: The entity dictionary
        
    Returns:
        Tuple of (is_valid, error_entity)
    """
    if entity['type'] not in ['protein', 'dna', 'rna']:
        return True, None  # Not a polymer type, skip validation
        
    entity_type = entity['type']
    sequence = entity['sequence']
    
    # Check sequence length
    if len(sequence) < 4:
        return False, EntityBean.create_with_kwargs(error_code=9, length=len(sequence))
            
    # Check sequence characters
    if entity_type in SEQUENCE_PATTERNS:
        if not re.match(SEQUENCE_PATTERNS[entity_type], sequence):
            return False, EntityBean.create_with_kwargs(error_code=11, type=entity_type, sequence=sequence)
    
    return True, None


def validate_ccd(entity: Dict[str, Any]) -> Tuple[bool, Optional[EntityBean]]:
    """Validate CCD codes.
    
    Args:
        entity: The entity dictionary
        
    Returns:
        Tuple of (is_valid, error_entity)
    """
    if 'ccd' not in entity or not entity['ccd']:
        return True, None  # No CCD to validate
        
    ccd = entity['ccd'].upper()
    
    # Check against excluded CCD list
    if ccd in EXCLUDE_CCD_LIST:
        logger.error(f"[validate_ccd] CCD not supported: {ccd}")
        return False, EntityBean.create_with_kwargs(error_code=4, CCD=ccd)
        
    return True, None


def validate_smiles(entity: Dict[str, Any]) -> Tuple[bool, Optional[EntityBean]]:
    """Validate SMILES strings.
    
    Args:
        entity: The entity dictionary
        
    Returns:
        Tuple of (is_valid, error_entity)
    """
    if 'ccd' in entity and entity['ccd']:
        return True, None
    if 'smiles' not in entity or not entity['smiles']:
        return True, None  # No SMILES to validate
        
    smiles = entity['smiles']
    
    # Check for disconnected components
    if '.' in smiles:
        error_msg = "SMILES containing a period (\".\") are not supported. Please submit each disconnected component separately."
        logger.error(f"[validate_smiles] {error_msg}")
        return False, EntityBean.create_with_kwargs(error_code=1, smiles=smiles)
    
    # Use RDKit for validation
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            logger.error(f"[validate_smiles] Cannot read invalid SMILES string: {smiles}")
            return False, EntityBean.create_with_kwargs(error_code=1, smiles=smiles)
            
    except Exception as e:
        logger.error(f"[validate_smiles] Error validating SMILES: {smiles}, {str(e)}")
        return False, EntityBean.create_with_kwargs(error_code=14, smiles=smiles, error=str(e))
    
    return True, None


def validate_modification(entity: Dict[str, Any]) -> Tuple[bool, Optional[EntityBean]]:
    """Validate modifications in a polymer entity.
    
    Args:
        entity: The entity dictionary
        
    Returns:
        Tuple of (is_valid, error_entity)
    """
    if 'modification' not in entity or not entity['modification']:
        return True, None  # No modifications to validate
        
    if 'sequence' not in entity:
        # Can't validate modifications without a sequence
        return True, None
        
    sequence_length = len(entity['sequence'])
    modified_indices = []
    
    for modification in entity['modification']:
        index = modification['index']
        
        # Validate modification index is within range
        if index <= 0 or index > sequence_length:
            return False, EntityBean.create_with_kwargs(error_code=6, index=index, max_index=sequence_length)
        
        # Validate no duplicate modifications
        if index in modified_indices:
            return False, EntityBean.create_with_kwargs(error_code=7, index=index)
                                
        # Check for unsupported modification types
        if modification['type'] not in ['residue_replace']:
            return False, EntityBean.create_with_kwargs(error_code=8, mod_type=modification['type'])
                
        modified_indices.append(index)
    
    return True, None


def validate_json_schema(json_path: str, schema_path: str) -> Dict[str, Any]:
    """Validate input data from JSON file against schema.
    
    Args:
        json_path: Path to the input JSON file
        schema_path: Path to JSON schema file
        
    Returns:
        The validated data
        
    Raises:
        ValidationException: If the JSON data doesn't conform to the schema
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON format: {str(e)}"
        logger.error(f"[validate_json_schema] {error_msg}")
        raise ValidationException(error_msg)
    except FileNotFoundError:
        error_msg = f"File not found: {json_path}"
        logger.error(f"[validate_json_schema] {error_msg}")
        raise ValidationException(error_msg)
    
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid schema JSON format: {str(e)}"
        logger.error(f"[validate_json_schema] {error_msg}")
        raise ValidationException(error_msg)
    except FileNotFoundError:
        error_msg = f"Schema file not found: {schema_path}"
        logger.error(f"[validate_json_schema] {error_msg}")
        raise ValidationException(error_msg)
    
    try:
        validate(instance=data, schema=schema)
    except ValidationError as e:
        error_msg = f"JSON schema validation failed: {str(e)}"
        logger.error(f"[validate_json_schema] {error_msg}")
        raise ValidationException(error_msg)
    
    return data


def validate_input_file(json_path: str, out_dir: str = None, schema_path: str = None) -> List[Dict[str, Any]]:
    """Validate all entities in an input file.
    
    Args:
        json_path: Path to input JSON file
        out_dir: Directory to write output files (optional)
        schema_path: Path to JSON schema file (optional)
        
    Returns:
        List of validated entities if all are valid
        
    Raises:
        ValidationException: If any validation fails
    """
    # First validate JSON schema if schema_path is provided
    data = validate_json_schema(json_path, schema_path)
    
    # Validate all entities using the independent validation functions
    entities = data['entities']
    converted_entities = []
    
    for idx, entity in enumerate(entities):
        # Run each validation function independently
        validation_results = [
            validate_polymer_sequence(entity),
            validate_ccd(entity),
            validate_smiles(entity),
            validate_modification(entity)
        ]
        
        # Check if any validation failed
        failed_validations = [(is_valid, error_entity) for is_valid, error_entity in validation_results if not is_valid]
        
        if failed_validations:
            # Take the first validation failure
            _, error_entity = failed_validations[0]
            error_entity.set_error_message(f'Entity validation failed for entity {idx+1}. {error_entity.error_message}')
            converted_entities.append(error_entity)
        else:
            # All validations passed
            valid_entity = EntityBean(error_code=0)
            converted_entities.append(valid_entity)
    
    # Handle any validation errors
    if any(entity.error_code != 0 for entity in converted_entities):
        total_error = 0
        for idx, entity in enumerate(converted_entities, start=1):
            if entity.error_code != 0:
                logger.error(f'Entity {idx} validation failed. {entity.error_message}')
                total_error += 1
        
        raise ValidationException(f'Error occurs on {total_error}/{len(entities)} entities.')
    
    return
