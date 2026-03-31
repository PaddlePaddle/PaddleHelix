import os
import json
import gzip
import re
import numpy as np
from datetime import datetime


def get_extra_infos_for_mmcif(args, length=5) -> dict:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    commit_file_path = os.path.join(current_file_dir, '../commit_id')
    md5_file_path = os.path.join(current_file_dir, '../ckpt_md5')
    
    extra_cif_infos = {}
    extra_cif_infos['time_stamp'] = get_timestamp()
    extra_cif_infos['ckpt_md5'] = get_md5(md5_file_path, length=length)
    commit, tag = get_commit_tag_hash(commit_file_path, length=length)
    extra_cif_infos['commit_id'] = commit
    extra_cif_infos['tag'] = tag
    return extra_cif_infos


def get_commit_tag_hash(file_path, length=None):
    try:
        with open(file_path, 'r') as f:
            context = f.readlines()
            if len(context) > 1:
                commit_id = context[0].strip()
                tag = context[1].strip()
            else:
                commit_id = context[0].strip()
                tag = 'HelixFold3'
        if length:
            commit_id = commit_id[:length]
        return commit_id, tag
    except FileNotFoundError:
        return "unknown_commit", "HelixFold3"


def get_timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def get_md5(file_path, length=None):
    """
        Calculates the MD5 hash of the given file.
        Args:
            file_path (str): 
                The path of the file to calculate the MD5 hash.
            length (int, optional): 
                The length (prefix) of the expected MD5 hash. 
                    If not specified, the full MD5 hash value is returned.
        Returns:
            str: The prefix or full MD5 hash of the given file.
    """
    try:
        with open(file_path, 'r') as f:
            md5_hash = f.readline().strip()
        if length:
            return md5_hash[:length]
        return md5_hash
    except FileNotFoundError:
        return "unknown_ckpt"


def read_json(path):
    if path.endswith('.json.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    else:
        with open(path, 'r') as f:
            return json.load(f)


def filter_job_name(name: str, max_length=200) -> str:
    """
    从输入的json文件中过滤并重命名作业名称。
    
    Args:
        name (str): 作业名称字符串。
        max_length (int, optional): 过滤后的作业名称的最大长度，默认为200。
    
    Returns:
        str: 过滤后的作业名称字符串，长度不超过max_length。
    
    """
    pattern = r'[^a-zA-Z0-9\-_\.]'
    filtered_string = re.sub(pattern, '', name)
    return filtered_string[:max_length]


def write_json(context, path, cn=False):
    if not cn:
        with open(path, 'w') as f:
            json.dump(context, f, indent=4)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(context, f, ensure_ascii=False, indent=4)


def alphabet2digit(alphabet):
    return sum((ord(a) - 65) * (26 ** e) for e, a in enumerate(reversed(alphabet)))


def digit2alphabet(digit):
    mod, remainder = divmod(digit, 26)
    alphabet = chr(65 + remainder)
    while mod:
        mod, remainder = divmod(mod, 26)
        alphabet = chr(65 + remainder) + alphabet
    return alphabet


def format_floats(obj, precision=2, nan_to_none=False):
    """
        Format floats in an object or list of objects to have a specified number of decimal places.
        这里进行了性能优化，避免使用递归。主要假设是 obj 一定是一个多级 list 或者 float，而一旦发现 list 的第一个元素是 float，
        则认为整个 list 都是 float，并进行一次性转换。
        
        Parameters:
        - obj: An object or list of objects that may contain floating point numbers.
        - precision: The desired number of decimal places to include in the formatted output. Defaults to 2.
        
        Returns:
        - The input object or list with all floating point values rounded to the specified precision.
    """
    if isinstance(obj, list):
        if not obj:
            return obj
        current = obj
        while isinstance(current, list) and current:
            current = current[0]
        if isinstance(current, (float, np.floating)):
            arr = np.array(obj)
            arr = np.around(arr, precision)
            if nan_to_none:
                arr = np.where(np.isnan(arr), None, arr)
            obj = arr.tolist()
    elif isinstance(obj, float):
        if nan_to_none and np.isnan(obj):
            return None
        obj = round(obj, precision)
    return obj


def write_format_json(data: dict, file_path: str, format_float=True, nan_to_none=False):
    """
        Write the data to a JSON file with formatting applied.
        
        Parameters:
        - data: A dictionary containing the data to be written to the file.
        - file_path: The path where the JSON file should be saved.
    """
    with open(file_path, 'w') as f:
        f.write('{\n')
        for i, (k, v) in enumerate(data.items()):
            if format_float:
                if 'pae' in k or 'ptm' in k or 'ranking_confidence' in k:
                    v = format_floats(v, precision=4, nan_to_none=nan_to_none)
                else:
                    v = format_floats(v, nan_to_none=nan_to_none)
            json_key = json.dumps(k, ensure_ascii=False)
            json_value = json.dumps(v, ensure_ascii=False)
            if i < len(data) - 1:
                f.write(f'    {json_key}: {json_value},\n')
            else:
                f.write(f'    {json_key}: {json_value}\n')
        f.write('}\n')


def convert_to_json_compatible(obj):
    """
        Convert a Python object to a JSON-compatible format.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj): 
            return None 
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_compatible(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return np.array(obj).tolist()
    else:
        return obj
