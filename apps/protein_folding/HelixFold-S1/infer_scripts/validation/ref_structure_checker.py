"""
Reference structure validation module.
author: gaojie01
"""
import json

from helixfold.data.templates_input import get_refer_structure_features
from helixfold.data.templates import Error as TemplateError
from infer_scripts.validation import ValidationException

def ref_structure_check(all_entities: list, kalign_binary_path: str, json_path: str):
    """Validate reference structure sequences against target sequences.
    
    Args:
        all_entities: List of processed entities from the input JSON
        kalign_binary_path: Path to kalign binary
        json_path: Path to the input JSON file
            
    Returns:
        bool: True if all sequences match, raises ValidationException otherwise
        
    Raises:
        ValidationException: If sequence mismatches are found
    """
    # Create entity_id2sequence mapping from all_entities
    entity_id2sequence = {}
    for entity in all_entities:
        entity_id = entity.raw_info['asym_chain_id']
        entity_id2sequence[entity_id] = entity.msa_seqs

    # Validate reference structures
    mismatch_pairs = []
    with open(json_path, 'r') as f:
        config_json = json.load(f)

    if 'ref_structures' in config_json:
        for ref_structures in config_json['ref_structures']:
            cif_path = ref_structures['ref_file']
            refer_target_pairs = ref_structures['refer_target_pairs']
            hit_pdb_code = cif_path.split('/')[-1].split('.')[0]

            for refer_target in refer_target_pairs:
                try:
                    refer_chain_id = refer_target['refer']
                    target_entity_id = refer_target['target']
                    sequence = entity_id2sequence[target_entity_id]
                
                    get_refer_structure_features(
                        cif_path,
                        refer_chain_id=refer_chain_id,
                        target_sequence=sequence,
                        hit_pdb_code=hit_pdb_code,
                        kalign_binary_path=kalign_binary_path
                    )
                except (TemplateError, Exception) as e:
                    if isinstance(e, FileNotFoundError):
                        raise
                    mismatch_pairs.append(refer_target)

    if mismatch_pairs:
        msg = f'Sequence mismatch in reference structure: {", ".join(map(str, mismatch_pairs))}. Ensure reference and target sequences are at least 90% similar and at least 6 residues long.'
        raise ValidationException(msg, details={"mismatch_pairs": mismatch_pairs})
    
    return True

if __name__ == '__main__':
    cif_path = '/data/helixfold/20250210/7quh_fail_test_531937/7quh_e9d2.cif'
    refer_chain_id = 'H'
    sequence = 'EVQLVESGGGLVQPGGSLRLSCAASGFSLTIYGAHWVRQAPGKGLEWVSVIWAGGSTNYNSALMSRFTISKDNSKNTVYLQMNSLRAEDTAVYYCARDGSSPYYYSMEYWGQGTTVTVSS'
    hit_pdb_code = '7quh'
    kalign_binary_path = '/data/helixfold/envs/helixfold_share/bin/kalign'
    get_refer_structure_features(cif_path,
        refer_chain_id=refer_chain_id, target_sequence = sequence,
        hit_pdb_code=hit_pdb_code, kalign_binary_path=kalign_binary_path) 