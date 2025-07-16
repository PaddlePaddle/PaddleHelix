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

"""Utils for infer_scripts."""

import json
import gzip
import re
import numpy as np

def filter_job_name(name: str, max_length=200) -> str:
    """Filter and rename the job name from the input JSON file.

    Args:
        name (str): The job name string.
        max_length (int, optional): The maximum length of the filtered job name, defaults to 200.

    Returns:
        str: The filtered job name string with a length not exceeding max_length.
    """
    pattern = r'[^a-zA-Z0-9\-_\.]'
    filtered_string = re.sub(pattern, '', name)
    return filtered_string[:max_length]


def read_json(path) -> dict:
    """read json
    
    Args:
        path (str): The path to the json file.

    Returns:
        dict: The json file content.
    """
    if path.endswith('.json.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    else:
        with open(path, 'r') as f:
            return json.load(f)


def write_json(context, path, cn=False):
    """write json
    
    Args:
        context: The context to write to the file.
        path: The path to the file to write to.
        cn: Whether to write in Chinese. Defaults to False.
    """
    if not cn:
        with open(path, 'w') as f:
            json.dump(context, f, indent=4)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(context, f, ensure_ascii=False, indent=4)


def format_floats(obj, precision=2, nan_to_none=False):
    """Format floats in an object or list of objects to have a specified number of decimal places.
        
    Args:
        obj: An object or list of objects that may contain floating point numbers.
        precision: The desired number of decimal places to include in the formatted output. Defaults to 2.
        nan_to_none: Whether to convert NaN values to None. Defaults to False.
    
    Returns:
        The input object or list with all floating point values rounded to the specified precision.
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
    """Write the data to a JSON file with formatting applied.
        
        Args:
            data: A dictionary containing the data to be written to the file.
            file_path: The path where the JSON file should be saved.
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
    """Convert a Python object to a JSON-compatible format."""
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
