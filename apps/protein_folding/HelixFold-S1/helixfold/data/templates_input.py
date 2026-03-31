#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for getting templates and calculating template features."""
import os
import logging
from helixfold.data import mmcif_parsing, parsers
from helixfold.data.templates import (_extract_template_features, 
                                     _build_query_to_hit_index_mapping,
                                     _read_file)

def get_refer_structure_features(
        cif_path: str,
        refer_chain_id: str,
        target_sequence: str,
        hit_pdb_code: str,
        kalign_binary_path: str):
    """Get reference structure features for single chain.
    
    Args:
        cif_path: Path to the mmCIF file
        refer_chain_id: Chain ID in the reference structure
        target_sequence: The target protein sequence
        hit_pdb_code: PDB code of the hit structure
        kalign_binary_path: Path to kalign binary
        
    Returns:
        Dictionary containing the extracted template features
    """
    query = query_sequence = hit_sequence = target_sequence
    
    indices_hit = parsers._get_indices(hit_sequence, start=0)
    indices_query = parsers._get_indices(query_sequence, start=0)

    mapping = _build_query_to_hit_index_mapping(
        query, hit_sequence, indices_hit, indices_query, query_sequence)
    
    template_sequence = hit_sequence.replace('-', '')

    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"CIF file not found: {cif_path}")
        
    logging.debug('Reading PDB entry from %s. Query: %s, template: %s', 
                 cif_path, query_sequence, template_sequence)
    
    cif_string = _read_file(cif_path)
    parsing_result = mmcif_parsing.parse(
        file_id=hit_pdb_code, 
        mmcif_string=cif_string)

    features, _ = _extract_template_features(
        mmcif_object=parsing_result.mmcif_object,
        pdb_id=hit_pdb_code,
        mapping=mapping,
        template_sequence=template_sequence,
        query_sequence=query_sequence,
        template_chain_id=refer_chain_id,
        kalign_binary_path=kalign_binary_path,
        ref_structure=True)
        
    return features

if __name__ == '__main__':
    
    cif = '/home/gaojie01/7e9p.cif'
    fasta = '/home/gaojie01/7e9p.fasta'
    hit_pdb_code = '7e9p'
    mmcif_dir = '/home/gaojie01'
    hit_chain_id = 'H'
    kalign_binary_path = '/data/helixfold/conda_envs/helixfold_share/bin/kalign'

    sequences, disc = parsers.parse_fasta(fasta)
    hit = parsers.TemplateHit()

    seq = sequences[0]
    hit.query = seq
    hit.hit_sequence = seq
    query_sequence = seq

    hit.indices_hit = parsers._get_indices(seq)
    hit.indices_query = parsers._get_indices(seq)

    mapping = _build_query_to_hit_index_mapping(
        hit.query, hit.hit_sequence, hit.indices_hit, hit.indices_query,
        query_sequence)

    # The mapping is from the query to the actual hit sequence, so we need to
    # remove gaps (which regardless have a missing confidence score).
    template_sequence = hit.hit_sequence.replace('-', '')
    cif_path = os.path.join(mmcif_dir, hit_pdb_code + '.cif')

    if os.path.exists(cif_path):
        logging.debug('Reading PDB entry from %s. Query: %s, template: %s', cif_path,
                    query_sequence, template_sequence)
        # Fail if we can't find the mmCIF file.
        cif_string = _read_file(cif_path)

    parsing_result = mmcif_parsing.parse(
        file_id=hit_pdb_code, mmcif_string=cif_string)

    features, realign_warning = _extract_template_features(
        mmcif_object=parsing_result.mmcif_object,
        pdb_id=hit_pdb_code,
        mapping=mapping,
        template_sequence=template_sequence,
        query_sequence=query_sequence,
        template_chain_id=hit_chain_id,
        kalign_binary_path=kalign_binary_path)

    if hit.sum_probs is None:
        features['template_sum_probs'] = [0]
    else:
        features['template_sum_probs'] = [hit.sum_probs]