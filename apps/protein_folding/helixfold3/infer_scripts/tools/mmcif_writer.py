"""
    tools for online inference.
"""
import os

_DISCLAIMER = """ xxxxxx need to supplement. """

# Authors of the Nature methods paper we reference in the mmCIF.
_MMCIF_PAPER_AUTHORS = (
    "PaddleHelix",
    "Baidu",
)

# Authors of the mmCIF - we set them to be equal to the authors of the paper.
_MMCIF_AUTHORS = _MMCIF_PAPER_AUTHORS

def mmcif_meta_append(mmcif_path:str):
    cif = {}
    # License and disclaimer.
    cif["_pdbx_data_usage.id"] = ["1", "2"]
    cif["_pdbx_data_usage.type"] = ["license", "disclaimer"]
    cif["_pdbx_data_usage.details"] = [
        "NON-COMMERCIAL USE ONLY, BY USING THIS FILE YOU AGREE TO THE TERMS OF USE FOUND AT helixfoldserver.com/output-terms.",
        _DISCLAIMER,
    ]
    cif["_pdbx_data_usage.url"] = [
        "?",
        "?",
    ]

    # Structure author details.
    cif["_audit_author.name"] = []
    cif["_audit_author.pdbx_ordinal"] = []
    for author_index, author_name in enumerate(_MMCIF_AUTHORS, start=1):
        cif["_audit_author.name"].append(author_name)
        cif["_audit_author.pdbx_ordinal"].append(str(author_index))

    # Paper author details.
    cif["_citation_author.citation_id"] = []
    cif["_citation_author.name"] = []
    cif["_citation_author.ordinal"] = []
    for author_index, author_name in enumerate(_MMCIF_PAPER_AUTHORS, start=1):
        cif["_citation_author.citation_id"].append("primary")
        cif["_citation_author.name"].append(author_name)
        cif["_citation_author.ordinal"].append(str(author_index))

    context_list = ['loop_'] + list(cif.keys())
    for tuple_val in zip(*cif.values()):
        tuple_val = list(map(str, tuple_val))
        string_line = ' '.join(tuple_val)
        context_list.append(string_line)
    context_list.append('#' + '\n')

    with open(mmcif_path) as fh:
        lines = fh.readlines()
    head_lines = lines[:2] ## data_xxxx + #
    with open(mmcif_path, 'w') as fh:
        fh.write(''.join(head_lines))
        fh.write('\n'.join(context_list))
        fh.write(''.join(lines[2:]))
        
    return mmcif_path


def mmcif_append(mmcif_path:str, contexts:dict, rm_duplicates=False):
    """
        Append a context to an mmCIF file.
        contexts: dict of contexts to append, followed by a the mmcif format.
        ## https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Index/
            For example:
                _chem_comp_atom.comp_id: ['UNK', 'UNK'] 
                _chem_comp_atom.atom_id: ['C', 'CA']
                _chem_comp_atom.alt_atom_id : ['C', 'CA']
    """
    def _check_shape(context_dict):
        _length = []
        _key_nums = 0
        for k in contexts:
            _key_nums += 1
            _length.append(len(contexts[k]))
            if not k.startswith('_'):
                context_dict[f'_{k}'] = contexts.pop(k)
        
        if len(set(_length)) != 1:
            raise ValueError("All values must have same length.")
            
    _check_shape(contexts)

    context_list = ['loop_'] + list(contexts.keys())
    _seen_lines = set()
    for tuple_val in zip(*contexts.values()):
        tuple_val = list(map(str, tuple_val))
        tuple_val = list(map(lambda x: x.ljust(4, ' '), tuple_val))
        string_line = ' '.join(tuple_val)
        if rm_duplicates and (string_line in _seen_lines): 
            continue
        _seen_lines.add(string_line)
        context_list.append(string_line)
    context_list.append('#' + '\n')
    
    # os.system(f"cp {mmcif_path} {mmcif_path.replace('.cif', '.tmp.cif')}")
    with open(mmcif_path) as fh:
        lines = fh.readlines()
    with open(mmcif_path, 'w') as fh:
        fh.write(''.join(lines))
        fh.write('\n'.join(context_list))

    return mmcif_path


if __name__ == "__main__": 
    mmcif_path = './test_data/B7L_wprotein.cif'
    contexts = {'_chem_comp_bond.comp_id': ['B7L', 'B7L'], 
                '_chem_comp_bond.atom_id_1': ['C5', 'C5'], 
                '_chem_comp_bond.atom_id_2 ': ['N3', 'C6'],
                '_chem_comp_bond.value_order': ['SING', 'SING'],
                '_chem_comp_bond.pdbx_ordinal': ['1', '2']}
    mmcif_append(mmcif_path, contexts)
