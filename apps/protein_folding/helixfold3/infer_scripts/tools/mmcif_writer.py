"""
    tools for online inference.
"""

META_REQUIRED_KEYS = {
    "_entry": ['id'],
    "_audit_author": ['name', 'pdbx_ordinal'],
    "_audit_conform": ['dict_location', 'dict_name', 'dict_version'],
    "_ma_data": ['content_type', 'id', 'name'],
    "_ma_model_list": ["data_id", "model_group_id", "model_group_name", "model_id", \
                        "model_name", "model_type", "ordinal_id"],
    "_ma_protocol_step": ["method_type", "ordinal_id", "protocol_id", "step_id"],
    "_ma_qa_metric": ["id", "mode", "name", "type"], 
    "_ma_qa_metric_global": ["metric_id", "metric_value", "model_id", "ordinal_id"],
    "_pdbx_data_usage": ["details", "id", "type", "url"],
}

ANNOTATION = "# \"See the terms_of_use.md in the model output for details. It can also be found at URL.\""

## BEELOW IS THE CONSTANTS.
MMCIF_CONSTANTS = {
    "_entry.id": "<PAD>",  ## TODO: need to be updated, global plddt
    "_audit_author.name": "\"Baidu PaddleHelix Team\"",
    "_audit_author.pdbx_ordinal": "1",
    "_audit_conform.dict_location": "https://raw.githubusercontent.com/ihmwg/ModelCIF/master/dist/mmcif_ma.dic",
    "_audit_conform.dict_name": "mmcif_ma.dic",
    "_audit_conform.dict_version": "1.4.6",
    "_ma_data.content_type": "\"model coordinates\"",
    "_ma_data.id": "1",
    "_ma_data.name": "Model",
    "_ma_model_list.data_id": "1",
    "_ma_model_list.model_group_id": "1",
    "_ma_model_list.model_group_name": "\"HelixFoldAA-20240731 (729bbc29-8a2e-4b22-8e2a-50ca55a8b71c @ 2024-05-13 04:52:55)\"",
    "_ma_model_list.model_id": "1",
    "_ma_model_list.model_name": "\"Top ranked model\"",
    "_ma_model_list.model_type": "\"Ab initio model\"",
    "_ma_model_list.ordinal_id": "1",
    "_ma_qa_metric.id": "1",
    "_ma_qa_metric.mode": "global",
    "_ma_qa_metric.name": "pLDDT",
    "_ma_qa_metric.type": "pLDDT",
    "_ma_qa_metric_global.metric_id": "1",
    "_ma_qa_metric_global.metric_value": "<PAD>", ## TODO: need to be updated, global plddt
    "_ma_qa_metric_global.model_id": "1",
    "_ma_qa_metric_global.ordinal_id": "1",
}

LOOP_MMCIF_CONSTANTS = {
    "_ma_protocol_step.method_type": ["\"coevolution MSA\"", "\"template search\"", "modeling"],
    "_ma_protocol_step.ordinal_id": ["1", "2", "3"],
    "_ma_protocol_step.protocol_id": ["1", "1", "1"],
    "_ma_protocol_step.step_id": ["1", "2", "3"],
    "_pdbx_data_usage.details": ["\"See the terms_of_use.md in the model output for details.\"", \
                                    "\"See the terms_of_use.md in the model output for details.\""],
    "_pdbx_data_usage.id": ["1", "2"],
    "_pdbx_data_usage.type": ["license", "disclaimer"],
    "_pdbx_data_usage.url": ["URL", "URL"] ## TODO: need to updata.
}

def format_pad(string, max_width, pad_value=' '):
    return string.ljust(max_width, pad_value)

def mmcif_meta_append(mmcif_path: str, extra_infos: dict):
    """
        TODO: we only support appending one context at a time.
    """
    with open(mmcif_path) as fh:
        lines = fh.readlines()
    head_lines = lines[:2] ## data_xxxx + #
    head_lines = [f"{ANNOTATION}\n"] + head_lines
    rest_lines = lines[2:]

    contexts = []
    for k, subkey in META_REQUIRED_KEYS.items():
        context_list = []
        cif = {f"{k}.{v}": [] for v in subkey}
        for v in subkey:
            meta_keys = f"{k}.{v}"
            if meta_keys in MMCIF_CONSTANTS:
                cif[meta_keys].append(MMCIF_CONSTANTS[meta_keys])
            else:
                cif[meta_keys].extend(LOOP_MMCIF_CONSTANTS[meta_keys])

        ## special keys, need to be updated.
        if k == '_entry':
            cif['_entry.id'] = [extra_infos['entry_id']]
        elif k == '_ma_qa_metric_global':
            cif['_ma_qa_metric_global.metric_value'] = [extra_infos['global_plddt']]

        assert len(cif[f"{k}.{subkey[0]}"]) > 0, \
                (f"{k}.{subkey[0]} is not assigned values. Please check your input")

        if len(cif[f"{k}.{subkey[0]}"]) > 1:
            for k, vs in cif.items():
                max_val_length = max([len(v) for v in vs])
                cif[k] = [format_pad(str(v), max_val_length) for v in vs]
                
            context_list = ['loop_'] + list(cif.keys())
            for tuple_val in zip(*cif.values()):
                tuple_val = list(map(str, tuple_val))
                string_line = ' '.join(tuple_val)
                context_list.append(string_line)
            context_list.append('#')
        else:
            max_key_length = max([len(k) for k in cif])
            for key, val in cif.items():
                pad_key = format_pad(key, max_key_length)
                val = list(map(str, val))
                assert len(val) == 1, \
                    (f"{k} has diferent lengths, Please check your input")
                val = val[0]
                context_list.append(f'{pad_key} {val}')
            context_list.append('#')
        contexts.extend(context_list)

    contexts.append("\n")
    with open(mmcif_path, 'w') as fh:
        fh.write(''.join(head_lines))
        fh.write('\n'.join(contexts))
        fh.write(''.join(rest_lines))
        
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

