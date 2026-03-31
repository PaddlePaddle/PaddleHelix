import json
import sys
import os
import traceback
import anarci

"""
This script checks whether the input JSON sequences contain antibodies.
"""

FAIL_MESSAGE = "MSA diversion check failed! The input is not supported currently."

def read_json(path):
    """
    Read a JSON file and return its contents as a Python object.
    """
    with open(path, 'r') as f:
        return json.load(f)


def is_antibody(path):
    """
    Check whether the input JSON sequences contain antibodies.
    Args:
        path (str): Path to the JSON file.
    Returns:
        bool: True if antibodies are found, False otherwise.
    """
    obj = read_json(path)
    entities = obj['entities']
    protein_sequences = [( index, items['sequence']) for index, items in enumerate(entities) if items['type'] == 'protein']
    # If no protein sequences, return False early.
    if not protein_sequences:
        return False
    _, _, alignment_details, _ = anarci.run_anarci(protein_sequences)
    alignment_details
    for details in alignment_details:
        if not details:
            continue
        chain_type = details[0]['chain_type']
        if chain_type in ['H', 'L', 'K']:
            return True
    return False

def antibody_chain_check(sequence):
    """ 
    Check whether the input sequence function type.
    Args:
        sequence (str): sequence.
    Returns:
        chain type: str. three type: ['H', 'L', 'A']
    """
    
    protein_sequences = [( 0, sequence)]
    # If no protein sequences, return False early.
    if not protein_sequences:
        return False
    _, _, alignment_details, _ = anarci.run_anarci(protein_sequences)
    alignment_details
    for details in alignment_details:
        if not details:
            continue
        chain_type = details[0]['chain_type']
        if chain_type in ['L', 'K']:
            return 'L'
        else:
            return chain_type
    return 'A'
    
if __name__ == "__main__":
    path = sys.argv[1]
    outpath = sys.argv[2]
    try:
        result = is_antibody(path)
        print(result)
    except Exception as e:
        traceback.print_exc()

        ## revised the task status
        os.makedirs(outpath, exist_ok=True)
        json_status = os.path.join(outpath, 'job_status.json')

        status_dict = {}
        status_dict["status"] = 'failed'
        status_dict["job_fail_reason"] = FAIL_MESSAGE
        with open(json_status, 'w') as f:
            json.dump(status_dict, f, indent=4)
        sys.exit(1)
