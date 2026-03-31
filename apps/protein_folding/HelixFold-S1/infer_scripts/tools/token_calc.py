import os
import argparse
import json
import gzip
from rdkit import Chem

class ModificationConstants:
	DNA_SMILES_BACKBONE = "O=P(O)(O)OCC1OC*CC1(O)"
	DNA_NAME_MAPPING = {
		0: "OP1",
		1: "P",
		2: "OP2",
		3: "OP3",
		4: "O5'",
		5: "C5'",
		6: "C4'",
		7: "O4'",
		8: "C1'",
		-1: "O3'",
		-2: "C3'",
		-3: "C2'",
	}
	RNA_SMILES_BACKBONE = "O=P(O)(O)OCC1OC*C(O)C1(O)"
	RNA_NAME_MAPPING = {
		0: "OP1",
		1: "P",
		2: "OP2",
		3: "OP3",
		4: "O5'",
		5: "C5'",
		6: "C4'",
		7: "O4'",
		8: "C1'",
		-1: "O3'",
		-2: "C3'",
		-3: "O2'",
		-4: "C2'",
	}
	PROT_SMILES_BACKBONE = "NC*C(=O)O"
	PROT_NAME_MAPPING = {
		0: "N",
		1: "CA",
		-1: "OXT",
		-2: "O",
		-3: "C",
	}


MAX_TOKEN = 3000
MAX_TOKEN_RDNA = 3000
CCD_JSON_DICT = json.load(gzip.open(f"{os.path.dirname(__file__)}/test_data/ccd_database.json.gz", 'rb'))

def get_modified_token(modifications, poly_type): 
    ## 对polymer的修饰，返回修饰的 token 数量
    index_count = 0
    token_count = 0 
    for modi_entity in modifications:
        index_count += 1

        if modi_entity['type'] == 'residue_replace':
            ccd = modi_entity['ccd']
            n_token = CCD_JSON_DICT[ccd]['n_tokens']
            token_count += n_token
        elif modi_entity['type'] == 'sidechain_replace':
            R_smiles = modi_entity['R_smiles']
            mol = Chem.MolFromSmiles(R_smiles, sanitize=False)          
            token_count += mol.GetNumHeavyAtoms()
            if poly_type == 'protein':
                token_count += len(ModificationConstants.PROT_NAME_MAPPING)
            elif poly_type == 'rna':
                token_count += len(ModificationConstants.RNA_NAME_MAPPING)
            elif poly_type == 'dna':
                token_count += len(ModificationConstants.DNA_NAME_MAPPING)
        else:
            raise ValueError(f"Unknown modification type: {modi_entity['type']}")
        
    return token_count - index_count


def compute_token(json_path):
    input_json = json.load(open(json_path, 'r'))
    n_token = 0
    n_token_rdna = 0
    has_rdna = False
    for e in input_json['entities']:
        if e['type'] == "protein":
            n_token += len(e["sequence"]) * e['count']  # 一个氨基酸是一个 token
            if 'modification' in e:
                n_token += (get_modified_token(e['modification'], poly_type="protein") * e['count'])
        elif e['type'] == "rna" or e['type'] == "dna":
            n_token += len(e["sequence"]) * e['count']  # 一个核苷酸是一个 token                                
            n_token_rdna += len(e["sequence"]) * e['count']
            if 'modification' in e:
                _tmp_token = get_modified_token(e['modification'], poly_type=e['type']) * e['count']
                n_token += _tmp_token
                n_token_rdna += _tmp_token
            has_rdna = True
        elif e['type'] == "ion":
            n_token += 1 * e['count']  # 一个离子是一个 token，我们只支持单原子离子                                                 
        elif e['type'] == "ligand":
            if e.get("ccd", "") != "":              # 如果有 CCD，那么就用 CCD 数据库来找到 SMILES
                e['smiles'] = CCD_JSON_DICT[e["ccd"]]['SMILES']            # 注意这里会覆盖已有的 SMILES

            mol = Chem.MolFromSmiles(e["smiles"], sanitize=False)           # 如果这里失败，给用户报 SMILES 不合法
            if mol is None:
                raise ValueError(f"Invalid SMILES: {e['smiles']}")
            n_token += mol.GetNumHeavyAtoms() * e['count']                  # 一个重核是一个 token
        else:
            raise ValueError("Unknown entity type")

    print("Total number of tokens:", n_token)
    return n_token


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate tokens for input JSON')
    parser.add_argument('--input_json', type=str, required=True, help='Path to input JSON file')
    args = parser.parse_args()

    compute_token(args.input_json)
