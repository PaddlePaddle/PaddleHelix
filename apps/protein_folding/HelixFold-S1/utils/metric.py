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

"""Metrics."""

import sys
import os
from os.path import join, basename, dirname, exists
import numpy as np
import subprocess

import paddle
import paddle.distributed as dist
import json
import re
from collections import defaultdict
from scipy import stats
from helixfold.common import protein
from helixfold.model.utils import get_confidence_metrics
from utils.utils import tree_map, tree_flatten, tree_filter
from ppfleetx.distributed.protein_folding import dp

def dist_all_reduce(x, return_num=False, distributed=False):
    x_num = len(x)
    x_sum = 0 if x_num == 0 else np.sum(x)
    if distributed:
        if os.environ.get('DCU_MODE_SOFTMAX_CLIP', '0') == '1':
            import paddle.distributed as dist
            x_num = int(dist.all_reduce(paddle.to_tensor(x_num, dtype='int64')))
            x_sum = int(dist.all_reduce(paddle.to_tensor(x_sum, dtype='float32')))
        else:
            x_num = int(dp.all_reduce(paddle.to_tensor(x_num, dtype='int64')))
            x_sum = float(dp.all_reduce(paddle.to_tensor(x_sum, dtype='float32')))
    x_mean = 0 if x_num == 0 else x_sum / x_num
    if return_num:
        return x_mean, x_num
    else:
        return x_mean


def get_tm_scores(tm_score_bin, pred_pdb_file, exp_pdb_file):
    assert exists(pred_pdb_file), pred_pdb_file
    assert exists(exp_pdb_file), exp_pdb_file
    cmd = f'{tm_score_bin} {pred_pdb_file} {exp_pdb_file} > {pred_pdb_file}.tmscore'
    # print(f"cmd: {cmd}")
    # s = os.popen(cmd).readlines()
    res = {}
    # for line in s:
    os.system(cmd)
    for line in open(f'{pred_pdb_file}.tmscore','r'):
        line = line.strip()
        if line[:8] == "TM-score":
            res['TM-score'] = float(line.split()[2])
        elif line[:6] == "GDT-TS":
            res['GDT-TS'] = float(line.split()[1])
        elif line[:6] == "GDT-HA":
            res['GDT-HA'] = float(line.split()[1])
        elif line[:4] == 'RMSD':
            res['RMSD'] = float(line.split()[5])
    return res


def get_lddt_scores(lddt_score_bin, pred_pdb_file, exp_pdb_file):
    def _get_lddt(cmd):
        s = os.popen(cmd).readlines()
        for line in s:
            segs = line.strip().split(':')
            if len(segs) == 2 and segs[0] == "Global LDDT score":
                return float(segs[1].strip())
        return np.nan

    assert exists(pred_pdb_file), pred_pdb_file
    assert exists(exp_pdb_file), exp_pdb_file
    res = {}
    # get lddt score
    cmd = f'{lddt_score_bin} {pred_pdb_file} {exp_pdb_file}'
    res['LDDT'] = _get_lddt(cmd)
    # get lddta score
    cmd = f'{lddt_score_bin} -c {pred_pdb_file} {exp_pdb_file}'
    res['LDDTa'] = _get_lddt(cmd)
    return res

def get_dockq_scores(dockq_score_dir, pred_pdb_file, exp_pdb_file, chain_info_list):
    """ tbd."""
    export_cmd = f"export PATH=./EMBOSS-6.6.0/emboss:$PATH"
    run_dockq_cmd = f'{sys.executable} DockQ.py {pred_pdb_file} {exp_pdb_file}'
    dockq_group = []
    for chain_info in chain_info_list:
        if chain_info.get("is_ligand", False):
            dockq_group.append(chain_info["id"])
    if len(dockq_group) > 0:
        run_dockq_cmd += f" -native_chain1 {' '.join(dockq_group)}"
    run_cmd = f"cd {dockq_score_dir} && {export_cmd} && {run_dockq_cmd}"
    try: run_output = subprocess.check_output(run_cmd, shell=True)
    except Exception as e:
        print(e)
        return {}

    res = {}
    dockq_score = 0
    for line in run_output.decode('utf8').split("\n"):
        for metric in ['DockQ', 'Fnat', 'Fnonnat', 'iRMS', 'LRMS']:
            if line.startswith(metric):
                res[metric] = float(line.split(' ')[1])
    if "DockQ" in res:
        res['DockQ80'] = 1. if res['DockQ'] >= 0.8 else 0.
        res['DockQ49'] = 1. if res['DockQ'] >= 0.49 else 0.
    return res

def get_subchain_pdb_file(pdb_file, subchain_id):
    try: subchain_id = protein.PDB_CHAIN_IDS[int(subchain_id)]
    except: pass
    subchain_name = f"-subchain_{subchain_id}"
    subchain_pdb_file = pdb_file + subchain_name
    pdb_file_prot_name = re.findall("[(?:exp)(?:pred)]-(.*?)(?:\.masked)?\.pdb$", pdb_file)
    if len(pdb_file_prot_name) > 0:
        pdb_file_prot_name = pdb_file_prot_name[0]
        subchain_pdb_file = pdb_file.replace(pdb_file_prot_name, pdb_file_prot_name + subchain_name)
    return subchain_pdb_file

def per_sample_pearsonr(log_dir, protein_info):
    # collect dockq and confidence for pearsonr
    run_cmd = f"cat {os.path.join(log_dir,f'score-*{protein_info}.txt')}"
    run_output = subprocess.check_output(run_cmd, shell=True)
    dockq_ls = []
    for line in run_output.decode('utf8').split("\n"):
        try:
            dockq_ls.append(float(re.findall("DockQ:(\d+\.\d+)\s", line)[0]))
        except:
            print(f"failed at line: {line}")
    if len(set(dockq_ls)) == 1: dockq_ls = [dockq_ls[0] + 0.00001] + dockq_ls[1:] # change value 0 to avoid nan pearsonr
        
    conf_ls = []
    run_cmd = f"cat {os.path.join(log_dir,f'score_confidence-*{protein_info}.txt')}"
    run_output = subprocess.check_output(run_cmd, shell=True)
    for line in run_output.decode('utf8').split("\n"):
        try:
            conf_ls.append(float(re.findall("ranking_confidence:(\d+\.\d+)\s", line)[0]))
        except:
            print(f"failed at line: {line}")
    if len(set(conf_ls)) == 1: conf_ls = [conf_ls[0] + 0.00001] + conf_ls[1:] # change value 0 to avoid nan pearsonr
      
    pearsonr, _ = stats.pearsonr(conf_ls, dockq_ls)
    with open(os.path.join(log_dir,f'pearsonr-{protein_info}.txt'), 'w', encoding='utf-8') as pf:
        pf.write("\t".join([protein_info, str(float(pearsonr)),
                            f"Confidence:{','.join([str(c) for c in conf_ls])}",
                            f"DockQ:{','.join([str(c) for c in dockq_ls])}"])+"\n")
    return pearsonr

def per_sample_dockq_ensembled(log_dir, protein_info):
    # collect dockq and confidence for ensembled dockq
    run_cmd = f"cat {os.path.join(log_dir,f'score-*{protein_info}.txt')}"
    run_output = subprocess.check_output(run_cmd, shell=True)
    dockq_ls = []
    for line in run_output.decode('utf8').split("\n"):
        try:
            dockq_ls.append(float(re.findall("DockQ:(\d+\.\d+)\s", line)[0]))
        except:
            print(f"failed at line: {line}")
    conf_ls = []
    run_cmd = f"cat {os.path.join(log_dir,f'score_confidence-*{protein_info}.txt')}"
    run_output = subprocess.check_output(run_cmd, shell=True)
    for line in run_output.decode('utf8').split("\n"):
        try:
            conf_ls.append(float(re.findall("ranking_confidence:(\d+\.\d+)\s", line)[0]))
        except:
            print(f"failed at line: {line}")
            
    ensembled_score = dockq_ls[np.argmax(conf_ls)]
    with open(os.path.join(log_dir,f'ensembled_score-{protein_info}.txt'), 'w', encoding='utf-8') as pf:
        pf.write("\t".join([protein_info, str(float(ensembled_score)),
                            f"Confidence:{','.join([str(c) for c in conf_ls])}",
                            f"DockQ:{','.join([str(c) for c in dockq_ls])}"])+"\n")
    return ensembled_score

def multimer_prediction_to_pdb_file(res_dict, pdb_file, b_factors=None, save_subchain=True):
    """ convert prediction to pdb file. """
    feat = {
        'chain_ids': res_dict.get('chain_ids', "_"),
        'aatype': res_dict['aatype'],
        'asym_id': res_dict['asym_id'],
        'residue_index': res_dict['residue_index'],
    }
    result = {
        "structure_module": {
            "final_atom_mask": res_dict["atom_mask"],
            "final_atom_positions": res_dict["atom_positions"],
        }
    }
    pdb_str = protein.to_pdb(protein.from_prediction(
            feat, result, b_factors=b_factors, remove_leading_feature_dimension=False))
    open(pdb_file, 'w').write(pdb_str)

    if save_subchain:
      # save subchain from results
      for unique_id in np.unique(res_dict['asym_id']):
        indicies = (res_dict['asym_id'] == unique_id).nonzero()[0]
        subchain_id = protein.PDB_CHAIN_IDS[int(unique_id)]
        subchain_feat = {
            'chain_ids': subchain_id,
            'aatype': np.array(res_dict['aatype'])[indicies],
            'asym_id': np.array(res_dict['asym_id'])[indicies],
            'residue_index': np.array(res_dict['residue_index'])[indicies],
        }
        subchain_result = {
            "structure_module": {
                "final_atom_mask": np.array(res_dict["atom_mask"])[indicies],
                "final_atom_positions": np.array(res_dict["atom_positions"])[indicies],
            }
        }

        subchain_pdb_str = protein.to_pdb(protein.from_prediction(
                subchain_feat, subchain_result, remove_leading_feature_dimension=False))
        open(get_subchain_pdb_file(pdb_file, subchain_id=unique_id), 'w').write(subchain_pdb_str)

def asym_id_to_chain(asym_id, chain_info_list):
    """
    chain_info_list: [{"id": "ID"}, {"id": "ID", "is_ligand": TRUE}, … ]
    replace asym_id with chain_id of each chain
    """
    asym_id_converted = np.zeros_like(asym_id)
    for idx, chain_info in zip(range(int(asym_id.max()) + 1), chain_info_list):
        chain_id = chain_info["id"][0] # FIXME: chain_id with two or more characters are not supported
        index = np.where(asym_id == idx + 1)[0]
        chain_id = protein.PDB_CHAIN_IDS.index(chain_id)
        asym_id_converted[index] = chain_id
    return asym_id_converted

def save_train_multimer_results(batch, results, batch_i, pred_pdb_file, exp_pdb_file,
                                score_file, exp_diff_file, label_key='label'):
    """save multimer results from training"""
    ## support at most 62 chains
    if len(np.unique(batch[label_key]['asym_id'][batch_i].numpy())) > protein.PDB_MAX_CHAINS:
        return
    ## save predictions
    ## note that common.protein assume chain_index starts from 0
    ## such that we need asym_id - 1
    res_dict = {
        'aatype': batch['feat']['aatype'][batch_i, 0],
        'asym_id': batch['feat']['asym_id'][batch_i, 0] - 1,
        'residue_index': batch['feat']['residue_index'][batch_i, 0],
        'atom_mask': results['structure_module']["final_atom_mask"][batch_i],
        'atom_positions': results['structure_module']["final_atom_positions"][batch_i],
    }
    seq_mask = batch['feat']['seq_mask'][batch_i, 0].numpy() == 1
    res_dict = tree_map(lambda x: x.numpy()[seq_mask], res_dict)
    multimer_prediction_to_pdb_file(res_dict, pred_pdb_file, save_subchain=False)

    ## save labels
    res_dict = {
        'aatype': batch[label_key]['aatype_index'][batch_i],
        'asym_id': batch[label_key]['asym_id'][batch_i] - 1,
        'residue_index': batch[label_key]['residue_index'][batch_i],
        'atom_mask': batch[label_key]["all_atom_mask"][batch_i],
        'atom_positions': batch[label_key]["all_atom_positions"][batch_i],
    }
    seq_mask = batch[label_key]['seq_mask'][batch_i].numpy() == 1
    res_dict = tree_map(lambda x: x.numpy()[seq_mask], res_dict)
    multimer_prediction_to_pdb_file(res_dict, exp_pdb_file, save_subchain=False)

    if "pos_diffused" in batch["feat"]:
        ## save diffused exp
        res_dict = {
            'aatype': batch[label_key]['aatype_index'][batch_i],
            'asym_id': batch[label_key]['asym_id'][batch_i] - 1,
            'residue_index': batch[label_key]['residue_index'][batch_i],
            'atom_mask': batch[label_key]["all_atom_mask"][batch_i],
            'atom_positions': batch["feat"]["pos_diffused"][batch_i],
        }
        seq_mask = batch[label_key]['seq_mask'][batch_i].numpy() == 1
        res_dict = tree_map(lambda x: x.numpy()[seq_mask], res_dict)
        multimer_prediction_to_pdb_file(res_dict, exp_diff_file, save_subchain=False)

    ## save scores/losses
    loss_res = tree_filter(lambda k: 'loss' in k or 'fape' in k, 
            lambda v: len(v.shape) == 1, tree_flatten(results))
    loss_values = tree_map(lambda v: v.numpy()[batch_i], loss_res)
    with open(score_file, 'w') as f:
        f.write('\t'.join([f'{k}:{v}' for k, v in loss_values.items()]) + '\n')

def save_multimer_results(
        batch, results, batch_i, 
        pred_pdb_file=None, pred_pdb_masked_file=None, 
        exp_pdb_file=None, exp_pdb_masked_file=None, b_factors=None, save_subchain=True):
    def _remove_G_linker(res_dict, G_eval_info):
        mask = G_eval_info['mask'][batch_i] == 1
        res_dict = tree_map(lambda x: x[mask], res_dict)
        ## replace asym_id and residue_index
        res_dict['asym_id'] = G_eval_info['asym_id'][batch_i]
        res_dict['residue_index'] = G_eval_info['residue_index'][batch_i]
        return res_dict
    def _save_pdb(res_dict, chain_info_list, seq_mask, pdb_file, b_factors=None, save_subchain=True):
        if pdb_file is None:
            return
        if not seq_mask is None:
            res_dict = tree_map(lambda x: x[seq_mask], res_dict)
        chain_ids = "_".join([chain_info["id"] for chain_info in chain_info_list])
        res_dict = {'chain_ids': chain_ids, **res_dict}
        if 'asym_id' in res_dict:  res_dict.update({'asym_id': 
                asym_id_to_chain(res_dict['asym_id'], chain_info_list=chain_info_list)})
        multimer_prediction_to_pdb_file(res_dict, pdb_file, b_factors=b_factors, save_subchain=save_subchain)

    chain_info_list = json.loads(batch['chain_info_list'][batch_i])
    pred_res_dict = {
        'aatype': batch['feat']['aatype'][batch_i, 0],
        'asym_id': batch['feat']['asym_id'][batch_i, 0],
        'residue_index': batch['feat']['residue_index'][batch_i, 0],
        'atom_mask': results['structure_module']["final_atom_mask"][batch_i],
        'atom_positions': results['structure_module']["final_atom_positions"][batch_i],
    }
    exp_res_dict = {}
    if 'label' in batch and 'all_atom_mask' in batch['label']:
        exp_res_dict = {
            'aatype': batch['feat']['aatype'][batch_i, 0],
            'asym_id': batch['feat']['asym_id'][batch_i, 0],
            'residue_index': batch['feat']['residue_index'][batch_i, 0],
            'atom_mask': batch['label']["all_atom_mask"][batch_i],
            'atom_positions': batch['label']["all_atom_positions"][batch_i],
        }

    if 'G_eval' in batch:
        pred_res_dict = _remove_G_linker(pred_res_dict, batch['G_eval'])
        exp_res_dict = _remove_G_linker(exp_res_dict, batch['G_eval'])

    pred_res_dict = tree_map(lambda x: x.numpy(), pred_res_dict)
    exp_res_dict = tree_map(lambda x: x.numpy(), exp_res_dict)

    ## save pdbs
    _save_pdb(pred_res_dict, chain_info_list, None, pred_pdb_file, b_factors=b_factors, save_subchain=save_subchain)
    if len(exp_res_dict) > 0:
        _save_pdb(exp_res_dict, chain_info_list, None, exp_pdb_file, b_factors=b_factors, save_subchain=save_subchain)
    
    ## remove masked residues, such that the eval of DockQ
    ## can be consistent
    if len(exp_res_dict) > 0:
        seq_mask = np.logical_and(pred_res_dict['atom_mask'].sum(-1) > 0, 
                exp_res_dict['atom_mask'].sum(-1) > 0)
        ## save masked pdbs
        _save_pdb(pred_res_dict, chain_info_list, seq_mask, pred_pdb_masked_file, 
                  b_factors=b_factors, save_subchain=save_subchain)
        _save_pdb(exp_res_dict, chain_info_list, seq_mask, exp_pdb_masked_file, 
                  b_factors=b_factors, save_subchain=save_subchain)

    ## visualize pair pocket on exp pdb for dimers
    if 'pair_pocket_mask' in batch["feat"]:
        asym_id = exp_res_dict['asym_id']
        if len(np.unique(asym_id)) == 2:
            pair_pocket_mask = batch['feat']['pair_pocket_mask'][batch_i, 0].numpy()
            ## keep residues that has any pocket mask in interface
            is_interface = asym_id[:, None] != asym_id[None]
            pair_mask = pair_pocket_mask * is_interface
            seq_mask = np.sum(pair_mask, 0) > 0
            if np.sum(seq_mask) > 0:
                filename = exp_pdb_masked_file.replace('.masked.pdb', '') + '.pocket_mask.pdb'
                _save_pdb(exp_res_dict, chain_info_list, seq_mask, filename, save_subchain=False)

class TMScore(object):
    """
    Utilize exe `cal_score` to get TMScore, GDT-TS and GDT-HA
    """
    def __init__(self, tm_score_bin, lddt_score_bin, dockq_score_dir, output_dir, cal_dockq_score=True):
        self.tm_score_bin = tm_score_bin
        self.lddt_score_bin = lddt_score_bin
        self.output_dir = output_dir
        self.dockq_score_dir = dockq_score_dir
        self.cal_dockq_score = cal_dockq_score

        assert exists(self.tm_score_bin),f'({self.tm_score_bin} not exists)'
        assert exists(self.lddt_score_bin),f'({self.lddt_score_bin} not exists)'
        assert exists(self.dockq_score_dir), f'({self.dockq_score_dir} not exists)'
        os.makedirs(self.output_dir, exist_ok=True)

        self.proteins = []
        self.seq_lens = []
        self.msa_depths = []
        ## key: [] must be init here otherwise some cards may 
        ## hand due to no test data
        if self.cal_dockq_score:
            self.score_dict = {
                'DockQ': [],
                'Fnat': [],
                'Fnonnat': [],
                'iRMS': [],
                'LRMS': [],
                'DockQ80': [],
                'DockQ49': [],
                'Receptor-TMScore': [],
                'Ligand-TMScore': [],
                'Receptor-RMSD': [],
                'Ligand-RMSD': [],             
            }
        else:
            self.score_dict = {
                'TM-score': [],
                'GDT-TS': [],
                'GDT-HA': [],
                'LDDT': [],
                'LDDTa': [],
            }

    def _update_results(self, name, pred_pdb_file, 
        # exp_mmcif_file, exp_fasta_file, 
        exp_pdb_file, msa_depth, seq_len, chain_info_list):
        """calculate tm_score, gdt-ha, gdt-ts and lddt"""
        subchain_key="subchain"
        if self.cal_dockq_score:
            pred_pdb_file_abs = os.path.abspath(pred_pdb_file)
            exp_pdb_file_abs = os.path.abspath(exp_pdb_file)
            cur_scores = get_dockq_scores(self.dockq_score_dir, pred_pdb_file_abs, exp_pdb_file_abs, chain_info_list)
            # Compute metrics for subchain selected
            # Put chain with rigid to the front if appliable
            chain_ids_grouped = defaultdict(list)
            for chain_info in chain_info_list:
                chain_id = chain_info["id"]
                if chain_info.get("is_ligand", False): chain_ids_grouped["Ligand"].append(chain_id)
                else: chain_ids_grouped["Receptor"].append(chain_id)

            for chain_group_name, chain_id_ls in chain_ids_grouped.items():
                # chain_group_name: Receptor/ Ligand
                subchain_tmscore, subchain_rmsd = [],[]
                for chain_id in chain_id_ls:
                    print(f"Computing metrics for subchain {chain_id}")
                    subchain_pred_pdb_file = os.path.abspath(get_subchain_pdb_file(pred_pdb_file, subchain_id=chain_id))
                    subchain_exp_pdb_file = os.path.abspath(get_subchain_pdb_file(exp_pdb_file, subchain_id=chain_id))
                    subchain_metrics = {}
                    try:
                        subchain_metrics = get_tm_scores(self.tm_score_bin, subchain_pred_pdb_file, subchain_exp_pdb_file)
                        subchain_tmscore.append(subchain_metrics["TM-score"])
                        subchain_rmsd.append(subchain_metrics["RMSD"])
                    except Exception as e: 
                        print(f"subchain metrics failed at {pred_pdb_file}, skipped. {e}")
                    cur_scores[f"{subchain_key}_{chain_id}-TMScore"]=subchain_metrics.get("TM-score",0)
                    cur_scores[f"{subchain_key}_{chain_id}-RMSD"]=subchain_metrics.get("RMSD",0)
                cur_scores[f"{chain_group_name}-TMScore"] = np.mean(subchain_tmscore) if len(subchain_metrics) >0 else 0.
                cur_scores[f"{chain_group_name}-RMSD"] = np.mean(subchain_rmsd) if len(subchain_rmsd) >0 else 0.
        else:
            cur_scores = get_tm_scores(self.tm_score_bin, pred_pdb_file, exp_pdb_file)
            cur_scores.update(get_lddt_scores(self.lddt_score_bin, pred_pdb_file, exp_pdb_file))

        if not re.findall("pred\d+", name): # skip predictions for ensembling
            for k, v in cur_scores.items():
                if k.startswith(subchain_key): continue
                self.score_dict[k].append(v)

        self.proteins.append(name)
        self.msa_depths.append(msa_depth)
        self.seq_lens.append(seq_len)
        chain_ids = "_".join([chain_info["id"] for chain_info in chain_info_list])
        print('[TMScore]', f'{name}_{chain_ids}', msa_depth, seq_len, cur_scores)
        with open(f'{self.output_dir}/score-{name}_{chain_ids}.txt', 'w') as f:
            out_str = f'{name}_{chain_ids}\t{msa_depth}\t{seq_len}\t'
            out_str += '\t'.join([f'{k}:{v}' for k, v in cur_scores.items()])
            f.write(out_str + '\n')

    def _save_confidence_scores(self, batch, results, batch_i):
        ## get multimer confidence
        single_pred = {
            "predicted_lddt": {
                "logits": results['predicted_lddt']['logits'][batch_i].numpy(),
            }
        }
        if "predicted_aligned_error" in results:
            single_pred.update({"predicted_aligned_error": {
                "logits": results['predicted_aligned_error']['logits'][batch_i].numpy(),
                "breaks": results['predicted_aligned_error']['breaks'].numpy(),
                "asym_id": batch['feat']['asym_id'][batch_i, 0].numpy(),
            }})
            metrics = get_confidence_metrics(single_pred, 
                    multimer_mode=self.cal_dockq_score)
            metrics['mean_plddt'] = np.mean(metrics['plddt'])
            keys = ['mean_plddt', 'ptm', 'iptm', 'ranking_confidence']
            metrics = tree_filter(lambda k: k in keys, None, metrics)

            name = batch['name'][batch_i]
            chain_info_list = json.loads(batch['chain_info_list'][batch_i])
            chain_ids = "_".join([chain_info["id"] for chain_info in chain_info_list])
            with open(f'{self.output_dir}/score_confidence-{name}_{chain_ids}.txt', 'w') as f:
                out_str = f'{name}_{chain_ids}\t'
                out_str += '\t'.join([f'{k}:{v}' for k, v in metrics.items()])
                f.write(out_str + '\n')

    def add(self, batch, results):
        """tbd"""
        protein_names = batch['name']
        features = batch['feat']
        batch_size = len(protein_names)
        for i in range(batch_size):
            name = batch['name'][i]
            msa_depth, seq_len = features['msa_feat'].shape[2:4]
            chain_info_list = json.loads(batch['chain_info_list'][i])
            chain_ids = "_".join([chain_info["id"] for chain_info in chain_info_list])

            if False: # 'structure_file' in batch:
                pred_pdb_file = f'{self.output_dir}/pred-{name}_{chain_ids}.pdb'
                exp_mmcif_file = batch['structure_file'][i]
                exp_fasta_file = batch['fasta_file'][i]
                exp_pdb_file =  f'{self.output_dir}/exp-{name}_{chain_ids}.pdb'
                save_multimer_results(batch, results, i, pred_pdb_file=pred_pdb_file)
                self._update_results(name, pred_pdb_file, exp_mmcif_file, exp_fasta_file, exp_pdb_file, msa_depth, seq_len, chain_ids)

            else:
                pred_pdb_file = f'{self.output_dir}/pred-{name}_{chain_ids}.pdb'
                pred_pdb_masked_file = f'{self.output_dir}/pred-{name}_{chain_ids}.masked.pdb'
                exp_pdb_file = f'{self.output_dir}/exp-{name}_{chain_ids}.pdb'
                exp_pdb_masked_file = f'{self.output_dir}/exp-{name}_{chain_ids}.masked.pdb'
                save_multimer_results(
                        batch, results, i, 
                        pred_pdb_file=pred_pdb_file,
                        pred_pdb_masked_file=pred_pdb_masked_file,
                        exp_pdb_file=exp_pdb_file,
                        exp_pdb_masked_file=exp_pdb_masked_file)
                if "pdb_struct_file" in batch:  
                    exp_pdb_masked_file = batch["pdb_struct_file"][i] # use assigned pdb file as ground truth
                    pred_pdb_masked_file = pred_pdb_file # pred_pdb_masked_file not exists when exp_pdb is not provided
                self._update_results(name, pred_pdb_masked_file, exp_pdb_masked_file,
                                      msa_depth, seq_len, chain_info_list)
            self._save_confidence_scores(batch, results, i)
        
    def get_proteins(self):
        return self.proteins

    def get_result(self, seq_range=None, msa_depth_range=None):
        """
        filter the results by seq_len range and msa_depth range
        seq_range: [left, right)
        msa_depth_range: [left, right)
        """
        def _get_flag(values, v_range):
            left, right = v_range
            return np.logical_and(values >= left, values < right)

        seq_lens = np.array(self.seq_lens)
        msa_depths = np.array(self.msa_depths)
        res = {}
        for k, v in self.score_dict.items():
            v = np.array(v)
            flag = np.ones([len(v)], dtype='bool')
            if not seq_range is None:
                flag = np.logical_and(flag, _get_flag(seq_lens, seq_range))
            if not msa_depth_range is None:
                flag = np.logical_and(flag, _get_flag(msa_depths, msa_depth_range))

            res[k] = v[flag]
        return res



class ResultsCollect(object):
    def __init__(self, 
            eval_tm_score=False, 
            tm_score_bin=None,
            lddt_score_bin=None,
            dockq_score_dir=None,
            cache_dir=None, 
            distributed=False,
            cal_dockq_score=True, log_dir=None):
        self.eval_tm_score = eval_tm_score
        self.distributed = distributed
        self.log_dir = log_dir

        self.res_dict_list = []
        if self.eval_tm_score:
            self.tm_score = TMScore(tm_score_bin, lddt_score_bin, dockq_score_dir, cache_dir, cal_dockq_score)

    def add(self, batch, results, extra_dict):
        """
        batch, results: 
        extra_dict: {key: float, ...}
        """
        res_dict = self._extract_loss_dict(results)     # {key: float, ...}
        res_dict.update(extra_dict)
        self.res_dict_list.append(res_dict)
        if self.eval_tm_score:
            self.tm_score.add(batch, results)

    def _get_max_key_len(self, res_dict_list):
        if len(res_dict_list) > 0:
            key_len = len(list(res_dict_list[0]))
        else:
            key_len = 0
        if self.distributed:
            max_key_len = dp.all_reduce(
                    paddle.to_tensor(key_len, dtype='int64'),
                    dist.ReduceOp.MAX)
        else:
            max_key_len = key_len
        return max_key_len
    
    def get_result(self):
        res = {}
        # get results in res_dict_list
        max_key_len = self._get_max_key_len(self.res_dict_list)
        if len(self.res_dict_list) > 0:
            keys = sorted(list(self.res_dict_list[0].keys()))
            for k in keys:
                if os.environ.get('DCU_MODE_SOFTMAX_CLIP', '0') == '1': 
                    res[k] = np.mean([d[k] for d in self.res_dict_list])
                else:
                    res[k] = dist_all_reduce(
                            [d[k] for d in self.res_dict_list], distributed=self.distributed)
        else:
            # This is tricky here for not blocking dist_all_reduce,
            # since during test stage, some cards may have no data to evaluate.
            for _ in range(max_key_len):
                dist_all_reduce([], distributed=self.distributed)

        # get tm_score results
        if self.eval_tm_score:
            if os.environ.get('DCU_MODE_SOFTMAX_CLIP', '0') == '1': 
                for k, l in self.tm_score.get_result().items():
                    avg_score, num = np.mean(l), len(l)
                    res.update({k: avg_score, 'sample_num': num})
            else:
                tmscore_results = self.tm_score.get_result()
                keys = sorted(list(tmscore_results.keys()))
                max_key_len = self._get_max_key_len([tmscore_results])
                if len(tmscore_results) > 0:
                    for k in keys:
                        l = tmscore_results[k]
                        avg_score, num = dist_all_reduce(
                                l, return_num=True, distributed=self.distributed)
                        res.update({k: avg_score, 'sample_num': num})
                else:
                    # Pseudo reduce for cards that have no data to evaluate.
                    for _ in range(max_key_len):
                        dist_all_reduce([], distributed=self.distributed)


            # # get tm_score by seq_len range
            # seq_range_list = [(0, 100), (100, 400), (400, 1400)]
            # for v_range in seq_range_list:
            #     prefix = f'seq{v_range[0]}_{v_range[1]}'
            #     for k, l in self.tm_score.get_result(seq_range=v_range).items():
            #         avg_score, num = dist_all_reduce(
            #                 l, return_num=True, distributed=self.distributed)
            #         res.update({
            #             f'{prefix}-{k}': avg_score, 
            #             f'{prefix}-sample_num': num
            #         })
            
            # # get tm_score by msa_depth range
            # # TODO: such way of getting msa_depth is wrong
            # msa_depth_range_list = [(0, 100), (100, 500), (500, np.inf)]
            # for v_range in msa_depth_range_list:
            #     prefix = f'depth{v_range[0]}_{v_range[1]}'
            #     for k, l in self.tm_score.get_result(msa_depth_range=v_range).items():
            #         avg_score, num = dist_all_reduce(
            #                 l, return_num=True, distributed=self.distributed)
            #         res.update({
            #             f'{prefix}-{k}': avg_score, 
            #             f'{prefix}-sample_num': num
            #         })
        return res

    def _extract_loss_dict(self, results):
        """extract value with 'loss' or 'fape' in key"""
        def _calc_tensor_mean(x):
            if x.dtype == paddle.bfloat16:
                x = x.cast("float32")
            if len(x.shape) == 0:
                return x.item()
            else:
                return x.numpy().mean()

        res = tree_flatten(results)
        res = tree_filter(lambda k: 'loss' in k or 'fape' in k, None, res)
        res = tree_map(lambda x: _calc_tensor_mean(x), res)
        return res
