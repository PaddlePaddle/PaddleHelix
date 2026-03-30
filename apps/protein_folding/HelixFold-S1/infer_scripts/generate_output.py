'''
    Tools for online output to display/download.
'''
import re
import os
import argparse
import glob
import logging
import json
import time
import zipfile
import tempfile
import os.path as osp
from typing import Union

import numpy as np
import pandas as pd

from tools.draw_results import draw_pae, draw_interface_heatmap
from tools.utils import *

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)

CC_INFO_PATH = os.path.abspath(osp.join(os.path.dirname(__file__), 'tools', 'cc_infos'))

def cmd_retry(cmd, retry_times=3):
    """
    :param cmd:
    :param retry_times:
    :return:
    """
    for i in range(retry_times):
        ret = os.system(cmd)
        if ret == 0:
            break
    if ret:
        assert 0, "cmd:{} retry_times:{} error ret:{}".format(cmd, retry_times, ret)


def get_chain_indices(chain_ids: Union[np.ndarray, list]) -> dict:
    """Returns a list of tuples indicating the start and end indices for each chain.

        Args:
            chain_ids: np.ndarray (N_token), the chain ids for the token. such as: [1-1, 1-1, 1-2, 2-1, 2-2, ...]
        Returns:
            chain_starts_ends: dict, the start and end indices for each chain.
                such as: {1-1: (0, 1), 1-2: (2, 3), 2-1: (4, 5), 2-2: (6, 7), ...}
    """
    if isinstance(chain_ids, list):
        chain_ids = np.array(chain_ids)
    
    chain_starts_ends = {}
    unique_chains = np.unique(chain_ids) # chains are numbered 1-1, 1-2, 2-1, 2-2, ...
    for chain in unique_chains:
        positions = np.where(chain_ids == chain)[0]
        chain_starts_ends[chain] = (positions[0], positions[-1])

    return chain_starts_ends


def get_chain_id_mapping(input_dir):
    chain_id_mapping_path = glob.glob(os.path.join(input_dir, "job*rank1/chain_id_mapping.csv"))
    _df = pd.read_csv(chain_id_mapping_path[0], na_values='', keep_default_na=False)
    mapping = {r['display_chain_id']: r['download_cif_chain_id'] for i, r in _df.iterrows()}
    return mapping


def get_summary_results(input_dir, cif_dir, output_file='summary.csv', chain_id_mapping=None):
    required_keys_json = ['has_clash','global_ptm',
                         'global_iptm', 'global_plddt', 'ranking_confidence']

    interface_infos_path = os.path.join(input_dir, "interface_infos/sample_infos.csv")
    interface_infos_df = pd.read_csv(interface_infos_path, na_values='', keep_default_na=False)
    all_pred_results_json = glob.glob(os.path.join(cif_dir, "job*rank1/all_results.json"))
    ## 1. add required keys to interface_infos_df
    for json_f in all_pred_results_json:
        model_id = re.search(r'(\d+)-rank1/', json_f).group(1)
        _json_dict = read_json(json_f)
        for key in required_keys_json:
            interface_infos_df.loc[interface_infos_df['model_id'] == int(model_id), key] = _json_dict[key]

    ## 2. convert chain_id to alpha chain_id by chain_id_mapping.csv
    if chain_id_mapping is not None:
        for i, row in interface_infos_df.iterrows():
            left_entity, right_entity = row['sampling_left_entity'], row['sampling_right_entity']
            _left_enti_id = '-'.join(left_entity.split('-')[0:2])
            _right_enti_id = '-'.join(right_entity.split('-')[0:2])
            interface_infos_df.at[i, 'sampling_left_entity'] = chain_id_mapping[_left_enti_id] + '-' + '-'.join(left_entity.split('-')[2:])
            interface_infos_df.at[i, 'sampling_right_entity'] = chain_id_mapping[_right_enti_id] + '-' + '-'.join(right_entity.split('-')[2:])

    interface_infos_df.to_csv(output_file, index=False)


def get_task_status(args, output_dir):
    """
        {
            "status": "failed",
            "job_fail_reason": "Invalid entity convert."
        }
    """
    assert osp.exists(osp.join(args.input_dir, "job_status.json"))
    with open(osp.join(args.input_dir, "job_status.json"), 'r') as fp:
        job_status = json.load(fp)
    job_status['status'] = "success"
    job_status['job_fail_reason'] = ""
    with open(osp.join(output_dir, "job_status.json"), 'w') as fp:
        json.dump(job_status, fp, indent=4)
    

def get_download(args, output_dir, file_suffix='', model_type="HelixFold3"):

    def _mapping_json_with_chain_id(res, chain_id_mapping):
        """Mapping the key with `chain_id` in json file
            
            numerical -> alphabetical; such as '1-1' -> A, '2-1' -> B
        """
        for key in res.keys():
            if 'chain_ids' in key:
                chain_ids = res[key] if isinstance(res[key], list) else [res[key]]
                chain_ids_set = set(chain_ids)
                if (len(chain_ids_set & set(chain_id_mapping.values())) 
                                    == len(set(chain_id_mapping.values()))):
                    logging.warning(f'numerical to alphabetical mapping is complete, skip.')
                    continue
                res[key] = [chain_id_mapping[cid] for cid in chain_ids]
        return res

    to_download_files = [
        'user_input.json',
    ]   
    if model_type == "HelixFold3":
        to_download_files.append('*-rank*/')
        download_zip_file_name = 'helixfold3_result_to_download_%s' % file_suffix
    elif model_type == "HelixFold-S1":
        to_download_files.extend([
            'module2/*-rank1/',
            'interface_infos/'
        ])
        download_zip_file_name = 'helixfold_s1_result_to_download_%s' % file_suffix
    else:
        raise ValueError(f'Invalid model type: {model_type}')

    keep_files = [
        "使用条款.md",
        "Terms_of_Use.md",
        "user_input.json",
        "all_results.json",
        "predicted_structure.cif",
        "chain_id_mapping.csv",
        "summary.csv",
        "predicted_interface.json",
        "predicted_interface.png",
    ]

   
    output_zip_path = os.path.join(output_dir, download_zip_file_name + '.zip')

    ## TODO: There is a certain probability that an error will occur when writing to disk in the temp directory.
    with tempfile.TemporaryDirectory(dir='/dev/shm/') as tmp_dir:
        temp_zip_path = os.path.join(tmp_dir, download_zip_file_name)
        os.makedirs(temp_zip_path, exist_ok=True)
        for pattern in to_download_files:
            base_dir = glob.glob(os.path.join(args.input_dir, pattern))
            assert len(base_dir) > 0
            cmd = f"cp -r {' '.join(base_dir)} {temp_zip_path}/"
            cmd_retry(cmd)
        
        ## copy cc_infos
        cmd_cc_infos = f"cp -r {CC_INFO_PATH}/*.md {temp_zip_path}/"
        cmd_retry(cmd_cc_infos)

        chain_id_mapping = get_chain_id_mapping(temp_zip_path)
        ## get summary.csv
        summary_csv_path = os.path.join(temp_zip_path, "summary.csv")
        get_summary_results(temp_zip_path, temp_zip_path,
                            output_file=summary_csv_path,
                            chain_id_mapping=chain_id_mapping)
        
        with zipfile.ZipFile(output_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(tmp_dir):
                for file in files:
                    if file not in keep_files:
                        continue
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, tmp_dir)

                    if file == "predicted_structure.cif" and osp.exists(file_path+'ABC'):
                        # 判断是否生成了 ABC 链名的 cif 结果，如果有，加入压缩包并重命名
                        file_path = file_path + 'ABC'
                    elif file in ["all_results.json", "predicted_interface.json"] and chain_id_mapping is not None:
                        # 映射all_results中的链名，从前端链名映射到 ABC 链名
                        res = read_json(file_path)
                        res = _mapping_json_with_chain_id(res, chain_id_mapping)
                        write_format_json(data=res, file_path=file_path, format_float=False)
                    elif file == "predicted_interface.png":
                        _json_res = read_json(file_path.replace('.png', '.json'))
                        _json_res = _mapping_json_with_chain_id(_json_res, chain_id_mapping)
                        draw_interface_heatmap(_json_res['token_pair_interface_probs'], 
                                                chain_idx_map=get_chain_indices(_json_res['token_chain_ids']), 
                                                path=file_path, title='Predicted Interface Probability',  
                                                verbose=False)
                    
                    zipf.write(file_path, arcname=arcname)
                    logging.info(f"Added {file_path} as {arcname}")

    logging.info(f"[Output zip file]: {output_zip_path}")
        

def get_display(args, output_dir, file_suffix='', model_type="HelixFold3"):
    
    to_display_files = ['user_input.json', 'summary.csv']

    if model_type == "HelixFold3":
        to_display_dir = ["*-rank1/"]
        display_json_file_name = 'helixfold3_result_to_display_%s' % file_suffix
    elif model_type == "HelixFold-S1":
        to_display_dir = ["module2/*-rank1/", "interface_infos/"]
        display_json_file_name = 'helixfold_s1_result_to_display_%s' % file_suffix
    else:
        raise ValueError(f'Invalid model type: {model_type}')

    keep_files = [
        "all_results.json",
        "summary.csv",
        "predicted_structure.cif",
        "predicted_interface.json",
        "predicted_interface.png",
    ]
    for pattern in to_display_dir:
        base_dir_list = glob.glob(os.path.join(args.input_dir, pattern))
        assert len(base_dir_list) > 0

        for base_dir in base_dir_list:
            base_dir = base_dir.rstrip('/') ## select the first one
            target_dir = os.path.join(output_dir, os.path.basename(base_dir))
            os.makedirs(target_dir, exist_ok=True)
            
            for file in os.listdir(base_dir):
                if file in keep_files:
                    cmd = f"cp -r {base_dir}/{file} {target_dir}/"
                    logging.info(cmd)
                    cmd_retry(cmd)

    ## 3. get summary.csv
    get_summary_results(args.input_dir, args.input_dir + '/module2/',
                        output_file=os.path.join(args.input_dir, "summary.csv"))

    for pattern in to_display_files:
        base_file = glob.glob(os.path.join(args.input_dir, pattern))
        assert len(base_file) == 1, f'Got {len(base_file)} {pattern} in model_type: {model_type}'
        base_file = base_file[0]
        cmd = f"cp -r {base_file} {output_dir}/"
        logging.info(cmd)
        cmd_retry(cmd)

    TEMPLATE_JSON = {
        "user_input": "",
        "summary": "",
        "interface": {
            "predicted_interface_json": "",
            "predicted_interface_png": "",
        },
        "prediction": {
            "all_results": "",
            "predicted_structure": "",
            "pae_figure_svg": "",
        },
    }

    user_input_path = glob.glob(osp.join(output_dir, 'user_input.json'))[0]
    TEMPLATE_JSON['user_input'] = user_input_path
    if model_type == "HelixFold3":
        all_results_path = glob.glob(osp.join(output_dir, '*-rank1/all_results.json'))
        assert len(all_results_path) == 1, \
            f'Got {len(all_results_path)} all_results.json in model_type: {model_type}'
    elif model_type == "HelixFold-S1":
        all_results_path = glob.glob(osp.join(output_dir, '*-rank1/all_results.json'))
        
        summary_path = glob.glob(osp.join(output_dir, 'summary.csv'))[0]
        interface_json_path = glob.glob(osp.join(output_dir, 'interface_infos/predicted_interface.json'))[0]
        interface_png_path = glob.glob(osp.join(output_dir, 'interface_infos/predicted_interface.png'))[0]
        TEMPLATE_JSON['interface']['predicted_interface_json'] = interface_json_path
        TEMPLATE_JSON['interface']['predicted_interface_png'] = interface_png_path
        TEMPLATE_JSON['summary'] = summary_path

    chain_idx_map = None
    for res_path in all_results_path:
        predicted_structure_path = os.path.join(os.path.dirname(res_path), "predicted_structure.cif")
        _res_dict = read_json(res_path)
        if chain_idx_map is None:
            chain_idx_map = get_chain_indices(_res_dict['token_chain_ids'])
        
        ## draw the pae figure
        pae_svg_path = os.path.join(os.path.dirname(res_path), "pae.svg")
        pae_matrix = _res_dict.pop('pae')
        draw_pae(pae_matrix, chain_idx_map, pae_svg_path, figure_size=(6, 6), verbose=False)

        ## update the all_results.json without pae
        write_format_json(data=_res_dict, file_path=res_path, format_float=False)

        ## update the to_display json
        if model_type == "HelixFold3":
            TEMPLATE_JSON['prediction']['pae_figure_svg'] = pae_svg_path
            TEMPLATE_JSON['prediction']['all_results'] = res_path
            TEMPLATE_JSON['prediction']['predicted_structure'] = predicted_structure_path
        elif model_type == "HelixFold-S1":
            model_id = re.search(r'(\d+)-rank1/', res_path).group(1)
            if model_id == '1':
                TEMPLATE_JSON['prediction']['pae_figure_svg'] = pae_svg_path
                TEMPLATE_JSON['prediction']['all_results'] = res_path
                TEMPLATE_JSON['prediction']['predicted_structure'] = predicted_structure_path
            TEMPLATE_JSON['prediction_{}'.format(model_id)] = {}
            TEMPLATE_JSON['prediction_{}'.format(model_id)]['pae_figure_svg'] = pae_svg_path
            TEMPLATE_JSON['prediction_{}'.format(model_id)]['predicted_structure'] = predicted_structure_path
            TEMPLATE_JSON['prediction_{}'.format(model_id)]['all_results'] = res_path


    display_json_file_name = display_json_file_name + '.json'
    with open(osp.join(output_dir, display_json_file_name), 'w') as fp:
        json.dump(TEMPLATE_JSON, fp, indent=4)


def main(args):
    start_time = time.time()

    assert osp.exists(args.input_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = os.path.abspath(args.output_dir)
    logging.info(f'[OUTPUT_DIR] {output_dir}')

    user_json = read_json(osp.join(args.input_dir, "user_input.json"))
    model_type = user_json.get('model_type', "HelixFold3")
    logging.info(f'[MODEL_TYPE] {model_type}')

    suffix = time.strftime('%Y%m%d%H%M%S')
    job_name = user_json['job_name']
    suffix = filter_job_name(job_name) + '_' + suffix

    ## check the json_status info;
    assert osp.exists(osp.join(args.input_dir, "job_status.json"))
    job_status = read_json(osp.join(args.input_dir, "job_status.json"))

    def failed_cleanup():
        cmd = f"cp -r {args.input_dir}/user_input.json {args.input_dir}/job_status.json {output_dir}/"
        cmd_retry(cmd)

        user_input_path = glob.glob(osp.join(output_dir, 'user_input.json'))[0]
        job_status_path = glob.glob(osp.join(output_dir, 'job_status.json'))[0]
        template_json = {}
        template_json['user_input'] = user_input_path
        template_json['job_status'] = job_status_path
        display_json_file_name = 'helixfold3_result_to_display_%s' % suffix
        display_json_file_name = display_json_file_name + '.json'
        with open(osp.join(output_dir, display_json_file_name), 'w') as fp:
            json.dump(template_json, fp, indent=4)

    if job_status["status"] == "failed":
        logging.error('failed, please check the original input file and job status.')
        failed_cleanup()
        return
    elif job_status["status"] == "running":
        # 一种特殊的失败情况，当MSA模块被OOM杀死时，状态是running，但无结果文件
        if len(glob.glob(os.path.join(args.input_dir, "module2", "*-rank1/"))) == 0:
            logging.error('MSA module gets killed.')
            job_status["status"] = "failed"
            job_status["job_fail_reason"] = "OOM. Sorry, the input protein/RNA is not supported currently."
            write_format_json(data=job_status, file_path=osp.join(args.input_dir, "job_status.json"), format_float=False)

            for ent in user_json['entities']:
                if ent['type'] == 'protein' or ent['type'] == 'rna':
                    ent['sequence'] = "UNSUPPORTED"
            write_format_json(data=user_json, file_path=osp.join(args.input_dir, "user_input.json"), format_float=False)

            failed_cleanup()
            return 

    ## 1. file to download
    # helixfold3_result_to_download_{job_name}_{datetime}.zip
    get_download(args, output_dir, file_suffix=suffix, model_type=model_type)

    ## 2. file to display
    # helixfold3_result_to_display_{datetime}.json
    get_display(args, output_dir, file_suffix=suffix, model_type=model_type)

    ## final revised the task status to success.
    get_task_status(args, output_dir)

    ## clean final_features.pkl
    if not os.environ.get('DEBUG'):
        if os.path.exists(osp.join(args.input_dir, "final_features.pkl")):
            os.remove(osp.join(args.input_dir, "final_features.pkl"))
        if os.path.exists(osp.join(args.input_dir, "final_features.pdparams")):
            os.remove(osp.join(args.input_dir, "final_features.pdparams"))
    
    logging.info(f'Generate output finished, time cost: {time.time() - start_time}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    logging.info(f'[ARG] {args}')

    main(args)
