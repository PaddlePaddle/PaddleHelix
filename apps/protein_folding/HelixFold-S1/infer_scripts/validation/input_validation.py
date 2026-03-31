"""
Validation functions for input preprocessing.

This module contains functions to validate different aspects of the input data
before it's processed by the preprocessing pipeline.
"""

from typing import List, Dict, Any, Tuple, Optional, Set
import logging
import os
import copy
import json
import re
import tempfile
from jsonschema import validate, ValidationError

from helixfold.common import residue_constants
from ..entity_bean import EntityBean
from ..rdkit_utils import validate_R_smiles
from .error_handling import handle_entity_errors
from ..tools.utils import read_json
from ..tools.token_calc import compute_token
from rdkit import Chem
from . import ValidationException

logger = logging.getLogger(__file__)

# Constants for validation
EXCLUDE_CCD_LIST = set(residue_constants.crystallization_aids) | set(residue_constants.ligand_exclusion_list)
COMMON_ION_CCDS = {"MG", "ZN", "CL", "CA", "NA", "MN", "MN3", "K", "FE", "FE2", "CU", "CU1", "CU3", "CO"}
MAX_TOKENS = 1000
MAX_TOKENS_RDNA = 1000
MAX_SEQUENCE_LENGTH = 1000

# Sequence patterns for validation
SEQUENCE_PATTERNS = {
    "protein": r"^[ACDEFGHIKLMNPQRSTVWY]+$",
    "dna": r"^[ACGT]+$",
    "rna": r"^[ACGU]+$"
}


def validate_model_type(data: Dict[str, Any]) -> None:
    """
    Validate model_type.
    """    
    model_type = data['model_type']

    if model_type == 'HelixFold-S1' and 'constraint' in data and len(data['constraint']) > 0:
        error_msg = f"constraint is not supported by HelixFold-S1 model"
        logger.error(f"[validate_model_type] {error_msg}")
        raise ValidationException(error_msg)
    
    if model_type == 'HelixFold3' and 's1_sample_constraint' in data and len(data['s1_sample_constraint']) > 0:
        error_msg = f"s1_sample_constraint is not supported by HelixFold3 model"
        logger.error(f"[validate_model_type] {error_msg}")
        raise ValidationException(error_msg)


def validate_s1_chain_count(data: Dict[str, Any]) -> None:
    """
    Validate chain count for S1 model.
    """
    if data['model_type'] != "HelixFold-S1":
        return
    
    total_chain_count = sum(entity['count'] for entity in data["entities"])
    if total_chain_count < 2:
        error_msg = f"Number of entities ({total_chain_count}) less than minimum limit of 2"
        logger.error(f"[validate_s1_chain_count] {error_msg}")
        raise ValidationException(error_msg)


def validate_s1_sample_constraint(data: Dict[str, Any]) -> None:
    """
    Validate s1_sample_constraint.
    """
    if 's1_sample_constraint' not in data:
        return
    
    if data['model_type'] != "HelixFold-S1" and len(data['s1_sample_constraint']) > 0:
        error_msg = f"s1_sample_constraint is only supported for model type 'HelixFold-S1'."
        logger.error(f"[validate_s1_sample_constraint] {error_msg}")
        raise ValidationException(error_msg)

    entity_count = len(data["entities"])
    constraints = data["s1_sample_constraint"]
    
    # Check constraint count
    if len(constraints) > 10:
        error_msg = f"Number of s1_sample_constraints ({len(constraints)}) exceeds maximum limit of 10"
        logger.error(f"[validate_s1_sample_constraint] {error_msg}")
        raise ValidationException(error_msg)

    for constraint in constraints:
        if constraint["left_entity"] == constraint["right_entity"]:
            error_msg = f"left_entity and right_entity cannot be the same; But got {constraint['left_entity']} and {constraint['right_entity']}"
            logger.error(f"[validate_s1_sample_constraint] {error_msg}")
            raise ValidationException(error_msg)
        for entity_ref in ["left_entity", "right_entity"]:
            # Validate entity index
            try:
                entity_idx = int(constraint[entity_ref].split("-")[0]) - 1
                if entity_idx < 0:
                    error_msg = f"Entity index in {entity_ref} must be positive"
                    logger.error(f"[validate_s1_sample_constraint] {error_msg}")
                    raise ValidationException(error_msg)

                if entity_idx >= entity_count:
                    error_msg = f"Entity index {entity_idx + 1} in constraint exceeds entity count {entity_count}"
                    logger.error(f"[validate_s1_sample_constraint] {error_msg}")
                    raise ValidationException(error_msg)

            except (ValueError, IndexError):
                error_msg = f"Invalid {entity_ref} format in constraint: {constraint[entity_ref]}"
                logger.error(f"[validate_s1_sample_constraint] {error_msg}")
                raise ValidationException(error_msg)
                        
            # Validate chain index
            try:
                chain_idx = int(constraint[entity_ref].split("-")[1]) - 1
                if chain_idx < 0:
                    error_msg = f"Chain index in {entity_ref} must be positive"
                    logger.error(f"[validate_s1_sample_constraint] {error_msg}")
                    raise ValidationException(error_msg)
                
                entity = data["entities"][entity_idx]
                if chain_idx >= entity["count"]:
                    error_msg = f"Chain index {chain_idx + 1} in constraint exceeds entity count {entity['count']}"
                    logger.error(f"[validate_s1_sample_constraint] {error_msg}")
                    raise ValidationException(error_msg)
            except (ValueError, IndexError) as e:
                if isinstance(e, ValidationException):
                    raise
                error_msg = f"Invalid {entity_ref} format in constraint: {constraint[entity_ref]}"
                logger.error(f"[validate_s1_sample_constraint] {error_msg}")
                raise ValidationException(error_msg)


def validate_polymer_sequence(entity: Dict[str, Any]) -> Tuple[bool, Optional[EntityBean]]:
    """
    Validate a polymer sequence (protein, DNA, RNA).
    
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
        
    if len(sequence) > MAX_SEQUENCE_LENGTH:
        return False, EntityBean.create_with_kwargs(error_code=10, length=len(sequence), max_length=MAX_SEQUENCE_LENGTH)
    
    # Check sequence characters
    if entity_type in SEQUENCE_PATTERNS:
        if not re.match(SEQUENCE_PATTERNS[entity_type], sequence):
            return False, EntityBean.create_with_kwargs(error_code=11, type=entity_type, sequence=sequence)
    
    return True, None


def validate_ccd(entity: Dict[str, Any]) -> Tuple[bool, Optional[EntityBean]]:
    """
    Validate CCD codes.
    
    Args:
        entity: The entity dictionary
        
    Returns:
        Tuple of (is_valid, error_entity)
    """
    if 'ccd' not in entity or not entity['ccd']:
        return True, None  # No CCD to validate
        
    ccd = entity['ccd'].upper()
    entity_type = entity.get('type', '')
    
    # Special validation for ion type entities
    if entity_type == "ion" and ccd not in COMMON_ION_CCDS:
        logger.error(f"[validate_ccd] Invalid ion CCD code: {ccd}")
        return False, EntityBean.create_with_kwargs(error_code=12, ccd=ccd)
    
    # Check against excluded CCD list
    if ccd in EXCLUDE_CCD_LIST:
        logger.error(f"[validate_ccd] CCD not supported: {ccd}")
        return False, EntityBean.create_with_kwargs(error_code=4, CCD=ccd)
        
    return True, None


def validate_smiles(entity: Dict[str, Any]) -> Tuple[bool, Optional[EntityBean]]:
    """
    Validate SMILES strings.
    
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
            
        if mol.GetNumHeavyAtoms() > 100:
            logger.error(f"[validate_smiles] SMILES contains more than 100 heavy atoms: {smiles}")
            return False, EntityBean.create_with_kwargs(error_code=13, smiles=smiles)
    except Exception as e:
        logger.error(f"[validate_smiles] Error validating SMILES: {smiles}, {str(e)}")
        return False, EntityBean.create_with_kwargs(error_code=14, smiles=smiles, error=str(e))
    
    return True, None


def validate_modification(entity: Dict[str, Any]) -> Tuple[bool, Optional[EntityBean]]:
    """
    Validate modifications in a polymer entity.
    
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
        
        # Validate R-SMILES for sidechain_replace
        if modification['type'] == 'sidechain_replace':
            try:
                validate_R_smiles(modification['R_smiles'], modification['R_connect_idx'])
            except Exception as e:
                return False, EntityBean.create_with_kwargs(error_code=15, error=str(e))
                
        # Validate ccd for residue_replace
        if modification['type'] == 'residue_replace' and 'ccd' in modification:
            ccd = modification['ccd']
            # TODO 这里不应该做这个判断，修饰的 ccd 有专门的列表，不应该走 ligand 的逻辑
            # if ccd in EXCLUDE_CCD_LIST:
            #     return False, EntityBean.create_with_kwargs(error_code=4, CCD=ccd)
        
        # Check for unsupported modification types
        if modification['type'] not in ['sidechain_replace', 'residue_replace']:
            return False, EntityBean.create_with_kwargs(error_code=8, mod_type=modification['type'])
                
        modified_indices.append(index)
    
    return True, None


def validate_token_count(data: Dict[str, Any]) -> None:
    """
    Validate token count of the input data.
    
    Args:
        data: The complete input data dictionary
        
    Raises:
        ValidationException: If token count exceeds maximum allowed
    """
    try:
        # Create a temporary file for token calculation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(data, tmp)
            tmp_path = tmp.name
        
        try:
            n_token = compute_token(tmp_path)
            has_rdna = any(entity["type"] in ["rna", "dna"] for entity in data["entities"])
            
            if not has_rdna and n_token > MAX_TOKENS:
                error_msg = f"Token count ({n_token}) exceeds maximum allowed ({MAX_TOKENS})"
                logger.error(f"[validate_token_count] {error_msg}")
                raise ValidationException(error_msg)
            elif has_rdna and n_token > MAX_TOKENS_RDNA:
                error_msg = f"Token count ({n_token}) exceeds maximum allowed for RNA/DNA structures ({MAX_TOKENS_RDNA})"
                logger.error(f"[validate_token_count] {error_msg}")
                raise ValidationException(error_msg)
        except Exception as e:
            logger.error(f"[validate_token_count] Error calculating token count: {str(e)}")
            raise ValidationException(f"{str(e)}")
        finally:
            # Clean up temporary file
            os.remove(tmp_path)
                
    except Exception as e:
        if isinstance(e, ValidationException):
            raise
        error_msg = f"Error calculating token count: {str(e)}"
        logger.error(f"[validate_token_count] {error_msg}")
        raise ValidationException(error_msg)


def validate_constraints(data: Dict[str, Any]) -> None:
    """
    Validate constraints if present.
    
    Args:
        data: The complete input data dictionary
        
    Raises:
        ValidationException: If constraints are invalid
    """
    if "constraint" not in data:
        return

    entity_count = len(data["entities"])
    constraints = data["constraint"]
    
    # Check constraint count
    if len(constraints) > 10:
        error_msg = f"Number of constraints ({len(constraints)}) exceeds maximum limit of 10"
        logger.error(f"[validate_constraints] {error_msg}")
        raise ValidationException(error_msg)
    
    for constraint in constraints:
        for entity_ref in ["left_entity", "right_entity"]:
            # Validate entity index
            try:
                entity_idx = int(constraint[entity_ref].split("-")[0]) - 1
                if entity_idx < 0:
                    error_msg = f"Entity index in {entity_ref} must be positive"
                    logger.error(f"[validate_constraints] {error_msg}")
                    raise ValidationException(error_msg)
            except (ValueError, IndexError):
                error_msg = f"Invalid {entity_ref} format in constraint: {constraint[entity_ref]}"
                logger.error(f"[validate_constraints] {error_msg}")
                raise ValidationException(error_msg)
            
            if entity_idx >= entity_count:
                error_msg = f"Entity index {entity_idx + 1} in constraint exceeds entity count {entity_count}"
                logger.error(f"[validate_constraints] {error_msg}")
                raise ValidationException(error_msg)
            
            # Validate chain index
            try:
                chain_idx = int(constraint[entity_ref].split("-")[1]) - 1
                if chain_idx < 0:
                    error_msg = f"Chain index in {entity_ref} must be positive"
                    logger.error(f"[validate_constraints] {error_msg}")
                    raise ValidationException(error_msg)
                
                entity = data["entities"][entity_idx]
                if chain_idx >= entity["count"]:
                    error_msg = f"Chain index {chain_idx + 1} in constraint exceeds entity count {entity['count']}"
                    logger.error(f"[validate_constraints] {error_msg}")
                    raise ValidationException(error_msg)
            except (ValueError, IndexError) as e:
                if isinstance(e, ValidationException):
                    raise
                error_msg = f"Invalid {entity_ref} format in constraint: {constraint[entity_ref]}"
                logger.error(f"[validate_constraints] {error_msg}")
                raise ValidationException(error_msg)
            
            # Validate residue index for residue-level constraints
            if constraint["level"] == "residue":
                entity = data["entities"][entity_idx]
                if "sequence" in entity:
                    try:
                        residue_idx = int(constraint[entity_ref].split("-")[2]) - 1
                        if residue_idx < 0:
                            error_msg = f"Residue index in {entity_ref} must be positive"
                            logger.error(f"[validate_constraints] {error_msg}")
                            raise ValidationException(error_msg)
                        
                        if residue_idx >= len(entity["sequence"]):
                            error_msg = f"Residue index {residue_idx + 1} in constraint exceeds sequence length {len(entity['sequence'])}"
                            logger.error(f"[validate_constraints] {error_msg}")
                            raise ValidationException(error_msg)
                    except (ValueError, IndexError) as e:
                        if isinstance(e, ValidationException):
                            raise
                        error_msg = f"Invalid {entity_ref} format in constraint: {constraint[entity_ref]}"
                        logger.error(f"[validate_constraints] {error_msg}")
                        raise ValidationException(error_msg)


def validate_ref_structures(data: Dict[str, Any]) -> None:
    """
    Validate reference structures if present.
    
    Args:
        data: Full input data dictionary
        
    Raises:
        ValidationException: If reference structures are invalid
    """
    if "ref_structures" not in data:
        return
        
    entity_count = len(data["entities"])
    
    for ref_structure in data["ref_structures"]:
        for pair in ref_structure["refer_target_pairs"]:
            target = pair["target"]
            try:
                # Format should be "entity-chain"
                parts = target.split("-")
                if len(parts) != 2:
                    error_msg = f"Invalid target format: {target}. Expected format: 'entity-chain'"
                    logger.error(f"[validate_ref_structures] {error_msg}")
                    raise ValidationException(error_msg)
                
                entity_idx = int(parts[0]) - 1
                chain_idx = int(parts[1]) - 1
                
                # Validate entity index
                if entity_idx < 0:
                    error_msg = f"Entity index in target must be positive"
                    logger.error(f"[validate_ref_structures] {error_msg}")
                    raise ValidationException(error_msg)
                
                if entity_idx >= entity_count:
                    error_msg = f"Entity index {entity_idx + 1} in target exceeds entity count {entity_count}"
                    logger.error(f"[validate_ref_structures] {error_msg}")
                    raise ValidationException(error_msg)
                
                # Validate chain index
                if chain_idx < 0:
                    error_msg = f"Chain index in target must be positive"
                    logger.error(f"[validate_ref_structures] {error_msg}")
                    raise ValidationException(error_msg)
                
                entity = data["entities"][entity_idx]
                if chain_idx >= entity["count"]:
                    error_msg = f"Chain index {chain_idx + 1} in target exceeds entity count {entity['count']}"
                    logger.error(f"[validate_ref_structures] {error_msg}")
                    raise ValidationException(error_msg)
                    
            except (ValueError, IndexError) as e:
                if isinstance(e, ValidationException):
                    raise
                error_msg = f"Invalid target format in reference structure: {target}"
                logger.error(f"[validate_ref_structures] {error_msg}")
                raise ValidationException(error_msg)


def validate_json_schema(json_path: str, schema_path: str) -> Dict[str, Any]:
    """
    Validate input data from JSON file against schema.
    
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


def validate_input_file(json_path: str, out_dir: str, schema_path: str = None) -> List[Dict[str, Any]]:
    """
    Validate all entities in an input file.
    
    Args:
        json_path: Path to input JSON file
        out_dir: Directory to write output files
        schema_path: Path to JSON schema file (optional)
        
    Returns:
        List of validated entities if all are valid
        
    Raises:
        ValidationException: If any validation fails
    """
    # First validate JSON schema if schema_path is provided
    data = validate_json_schema(json_path, schema_path)
    
    if 'random_seed' in data:
        del data['random_seed']
    
    # Validate data aspects
    validate_model_type(data)
    validate_token_count(data)
    validate_constraints(data)
    validate_ref_structures(data)
    validate_s1_chain_count(data)
    validate_s1_sample_constraint(data)
    
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
        valid_entities = handle_entity_errors(
            input_json_dict=data,
            entity_beans=converted_entities,
            out_dir=out_dir
        )
    
    return
