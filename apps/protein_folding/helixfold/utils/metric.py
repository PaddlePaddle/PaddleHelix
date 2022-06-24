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

import os
from os.path import join, basename, dirname, exists
import numpy as np

import paddle
import paddle.distributed as dist

from alphafold_paddle.common import protein
from utils.utils import tree_map, tree_flatten, tree_filter


def dist_all_reduce(x, return_num=False, distributed=False):
    x_num = len(x)
    x_sum = 0 if x_num == 0 else np.sum(x)
    if distributed:
        x_num = dist.all_reduce(paddle.to_tensor(x_num, dtype='int64')).numpy()[0]
        x_sum = dist.all_reduce(paddle.to_tensor(x_sum, dtype='float32')).numpy()[0]
    x_mean = 0 if x_num == 0 else x_sum / x_num
    if return_num:
        return x_mean, x_num
    else:
        return x_mean


def get_tm_scores(tm_score_bin, pred_pdb_file, exp_pdb_file):
    assert exists(pred_pdb_file), pred_pdb_file
    assert exists(exp_pdb_file), exp_pdb_file
    cmd = f'{tm_score_bin} {pred_pdb_file} {exp_pdb_file}'
    # print(f"cmd: {cmd}")
    s = os.popen(cmd).readlines()
    res = {}
    for line in s:
        line = line.strip()
        if line[:8] == "TM-score":
            res['TM-score'] = float(line.split()[2])
        elif line[:6] == "GDT-TS":
            res['GDT-TS'] = float(line.split()[1])
        elif line[:6] == "GDT-HA":
            res['GDT-HA'] = float(line.split()[1])
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


class TMScore(object):
    """
    Utilize exe `cal_score` to get TMScore, GDT-TS and GDT-HA
    """
    def __init__(self, tm_score_bin, lddt_score_bin, output_dir):
        self.tm_score_bin = tm_score_bin
        self.lddt_score_bin = lddt_score_bin
        self.output_dir = output_dir

        assert exists(self.tm_score_bin),f'({self.tm_score_bin} not exists)'
        assert exists(self.lddt_score_bin),f'({self.lddt_score_bin} not exists)'
        os.makedirs(self.output_dir, exist_ok=True)

        self.proteins = []
        self.seq_lens = []
        self.msa_depths = []
        self.score_dict = {
            'TM-score': [],
            'GDT-TS': [],
            'GDT-HA': [],
            'LDDT': [],
            'LDDTa': [],
        }

    def _result_to_pdb_file(self, res_dict, pdb_file):
        feat = {
            'aatype': res_dict['aatype'],
            'residue_index': res_dict['residue_index'],
        }
        result = {
            "structure_module": {
                "final_atom_mask": res_dict["final_atom_mask"],
                "final_atom_positions": res_dict["final_atom_positions"],
            }
        }
        pdb_str = protein.to_pdb(protein.from_prediction(feat, result))
        open(pdb_file, 'w').write(pdb_str)

    def _update_results(self, name, pred_pdb_file, exp_pdb_file, msa_depth, seq_len):
        """calculate tm_score, gdt-ha, gdt-ts and lddt"""
        cur_scores = get_tm_scores(self.tm_score_bin, pred_pdb_file, exp_pdb_file)
        cur_scores.update(get_lddt_scores(self.lddt_score_bin, pred_pdb_file, exp_pdb_file))
        for k, v in cur_scores.items():
            self.score_dict[k].append(v)

        self.proteins.append(name)
        self.msa_depths.append(msa_depth)
        self.seq_lens.append(seq_len)
        print('[TMScore]', name, msa_depth, seq_len, cur_scores)
    
    def add(self, batch, results):
        """tbd"""
        def _update_dict(cur_d, d):
            for k in cur_d:
                d[k].append(cur_d[k])

        protein_names = batch['name']
        features = batch['feat']
        labels = batch['label']

        batch_size = len(protein_names)
        for i in range(batch_size):
            name = batch['name'][i]
            msa_depth, seq_len = features['msa_feat'].shape[2:4]
            # generate pred pdb
            res_dict = {
                'aatype': features['aatype'],
                'residue_index': features['residue_index'],
                "final_atom_mask": results['structure_module']["final_atom_mask"],
                "final_atom_positions": results['structure_module']["final_atom_positions"],
            }
            res_dict = tree_map(lambda x: x[i].numpy(), res_dict)
            pred_pdb_file = f'{self.output_dir}/pred-{name}.pdb'
            self._result_to_pdb_file(res_dict, pred_pdb_file)

            # generate exp pdb
            if 'struct_file' in batch:
                exp_pdb_file = batch['struct_file'][i]
            else:
                res_dict = {
                    'aatype': features['aatype'],
                    'residue_index': features['residue_index'],
                    "final_atom_mask": labels["all_atom_mask"],
                    "final_atom_positions": labels["all_atom_positions"],
                }
                res_dict = tree_map(lambda x: x[i].numpy(), res_dict)
                exp_pdb_file = f'{self.output_dir}/exp-{name}.pdb'
                self._result_to_pdb_file(res_dict, exp_pdb_file)

            # update 
            self._update_results(name, pred_pdb_file, exp_pdb_file, msa_depth, seq_len)
        
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
            cache_dir=None, 
            distributed=False):
        self.eval_tm_score = eval_tm_score
        self.distributed = distributed

        self.res_dict_list = []
        if self.eval_tm_score:
            self.tm_score = TMScore(tm_score_bin, lddt_score_bin, cache_dir)

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
    
    def get_result(self):
        res = {}
        # get results in res_dict_list
        if len(self.res_dict_list) > 0:
            keys = list(self.res_dict_list[0].keys())
            for k in keys:
                res[k] = dist_all_reduce(
                        [d[k] for d in self.res_dict_list], distributed=self.distributed)

        # get tm_score results
        if self.eval_tm_score:
            for k, l in self.tm_score.get_result().items():
                avg_score, num = dist_all_reduce(
                        l, return_num=True, distributed=self.distributed)
                res.update({k: avg_score, 'sample_num': num})

            # get tm_score by seq_len range
            seq_range_list = [(0, 100), (100, 400), (400, 1400)]
            for v_range in seq_range_list:
                prefix = f'seq{v_range[0]}_{v_range[1]}'
                for k, l in self.tm_score.get_result(seq_range=v_range).items():
                    avg_score, num = dist_all_reduce(
                            l, return_num=True, distributed=self.distributed)
                    res.update({
                        f'{prefix}-{k}': avg_score, 
                        f'{prefix}-sample_num': num
                    })
            
            # get tm_score by msa_depth range
            msa_depth_range_list = [(0, 100), (100, 500), (500, np.inf)]
            for v_range in msa_depth_range_list:
                prefix = f'depth{v_range[0]}_{v_range[1]}'
                for k, l in self.tm_score.get_result(msa_depth_range=v_range).items():
                    avg_score, num = dist_all_reduce(
                            l, return_num=True, distributed=self.distributed)
                    res.update({
                        f'{prefix}-{k}': avg_score, 
                        f'{prefix}-sample_num': num
                    })
        return res

    def _extract_loss_dict(self, results):
        """extract value with 'loss' or 'fape' in key"""
        res = tree_flatten(results)
        res = tree_filter(lambda k: 'loss' in k or 'fape' in k, None, res)
        res = tree_map(lambda x: x.numpy().mean(), res)
        return res

