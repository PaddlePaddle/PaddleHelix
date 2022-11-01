#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""
Convert CCLE&GDSC datasets into npz file which can be trained directly.
"""
import os
import sys
import json
import random
import pickle
import csv
import argparse
import numpy as np
import pandas as pd
import hickle as hkl
import deepchem as dc
from rdkit import Chem

import pgl

from pahelix.utils.data_utils import save_data_list_to_npz
import pdb

def raw_drug_feature(Drug_smiles_file, save_dir):
    """
    Generate drug features using Deepchem library

    :param Drug_smiles_file: drugs' SMILES
    :param save_dir: save path of raw drug features
    """
    pubchemid2smile = {item.split('\t')[0]: item.split('\t')[1].strip() for item in open(Drug_smiles_file).readlines()}
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for each in pubchemid2smile.keys():
        molecules = []
        molecules.append(Chem.MolFromSmiles(pubchemid2smile[each]))
        featurizer = dc.feat.graph_features.ConvMolFeaturizer()
        mol_object = featurizer.featurize(molecules)
        features = mol_object[0].atom_features
        degree_list = mol_object[0].deg_list
        adj_list = mol_object[0].canon_adj_list
        hkl.dump([features, adj_list, degree_list], '%s/%s.hkl' % (save_dir, each))

def metadata_generate(Drug_info_file, Cell_line_info_file, Genomic_mutation_file, Drug_feature_file,
                      Gene_expression_file, Methylation_file):
    """
    Generate sample(or its index) including omics features, drug index and labels, which correspond to cell lines

    :param Drug_info_file: drugs' list
    :param Cell_line_info_file: cell lines list
    :param Genomic_mutation_file: genomic mutation raw data
    :param Drug_feature_file:  drug feature generated from Deepchem
    :param Gene_expression_file: gene expression raw data
    :param Methylation_file: methylation raw data
    :return: a dict with all omics features, drug index and labels
    """
    # drug_id --> pubchem_id
    reader = csv.reader(open(Drug_info_file, 'r'))
    rows = [item for item in reader]
    drugid2pubchemid = {item[0]: item[5] for item in rows if item[5].isdigit()}
    cellline2cancertype = {}
    for line in open(Cell_line_info_file).readlines()[1:]:
        cellline_id = line.split('\t')[1]
        TCGA_label = line.strip().split('\t')[-1]
        cellline2cancertype[cellline_id] = TCGA_label
    mutation_feature = pd.read_csv(Genomic_mutation_file, sep=',', header=0, index_col=[0])
    drug_pubchem_id_set = []
    drug_feature = {}
    for each in os.listdir(Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        feat_mat, adj_list, degree_list = hkl.load('%s/%s' % (Drug_feature_file, each))
        drug_feature[each.split('.')[0]] = [feat_mat, adj_list, degree_list]
    assert len(drug_pubchem_id_set) == len(drug_feature.values())
    gexpr_feature = pd.read_csv(Gene_expression_file, sep=',', header=0, index_col=[0])
    mutation_feature = mutation_feature.loc[list(gexpr_feature.index)]
    methylation_feature = pd.read_csv(Methylation_file, sep=',', header=0, index_col=[0])
    methylation_feature = methylation_feature.loc[list(gexpr_feature.index)]
    assert methylation_feature.shape[0] == gexpr_feature.shape[0] == mutation_feature.shape[0]
    experiment_data = pd.read_csv(Cancer_response_exp_file, sep=',', header=0, index_col=[0])
    drug_match_list = [item for item in experiment_data.index if item.split(':')[1] in drugid2pubchemid.keys()]
    experiment_data_filtered = experiment_data.loc[drug_match_list]
    data_idx = []
    for each_drug in experiment_data_filtered.index:
        for each_cellline in experiment_data_filtered.columns:
            pubchem_id = drugid2pubchemid[each_drug.split(':')[-1]]
            if str(pubchem_id) in drug_pubchem_id_set and each_cellline in mutation_feature.index:
                if not np.isnan(experiment_data_filtered.loc[
                                    each_drug, each_cellline]) and each_cellline in cellline2cancertype.keys():
                    
                    ln_IC50 = float(experiment_data_filtered.loc[each_drug, each_cellline])
                    data_idx.append((each_cellline, pubchem_id, ln_IC50, cellline2cancertype[each_cellline]))
    nb_celllines = len(set([item[0] for item in data_idx]))
    nb_drugs = len(set([item[1] for item in data_idx]))
    print('36314 in data_idx:', '36314' in list(set([item[1] for item in data_idx])))
    print(
        '{} instances across {} cell lines and {} drugs were generated.'.format(len(data_idx), nb_celllines, nb_drugs))
    return {'metadata': (drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_idx)}


def random_adjacency_matrix(n):
    """
    Generate random sub-matrix for adjacent matrix
    :param n: dims of sub-matrix
    """
    matrix = [[random.randint(0, 1) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 0
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]
    return matrix


def data_split(data_idx, ratio=0.95):
    """
    Dataset splitting
    :param data_idx: Dataset index for all data
    :param ratio: splitting ratio
    :return: partitioned data index for training and evaluating
    """
    data_train_idx, data_test_idx = [], []
    for each_type in TCGA_label_set:
        data_subtype_idx = [item for item in data_idx if item[-1] == each_type]
        train_list = random.sample(data_subtype_idx, int(ratio * len(data_subtype_idx)))
        test_list = [item for item in data_subtype_idx if item not in train_list]
        data_train_idx += train_list
        data_test_idx += test_list
    return data_train_idx, data_test_idx


def gen_drug_feature(drug_feature, israndom=False):
    """
    generate drug feature ready for graph construction from raw drug feature and use random matrix padding
    :param drug_feature: raw drug feature from
    :param israndom: whether using random matrix padding
    :return: a dict of all drug features ready for graph construction
    """
    for drug in drug_feature:
        drug_feature[drug].append([])
        for idx, positions in enumerate(drug_feature[drug][1]):
            for position in positions:
                if (idx, position) not in drug_feature[drug][-1] and (position, idx) not in drug_feature[drug][-1]:
                    drug_feature[drug][-1].append((idx, position))
    for drug in drug_feature:
        feat_mat = drug_feature[drug][0].astype('float32')
        feat = np.zeros((Max_atoms, 75), dtype='float32')
        adj_mat = np.zeros((Max_atoms, Max_atoms), dtype='float32')
        drug_feature[drug][-1] += [(i, i) for i in range(Max_atoms)]

        if israndom:
            feat = np.random.rand(Max_atoms, 75)
            adj_mat[feat_mat.shape[0]:, feat_mat.shape[0]:] = random_adjacency_matrix(Max_atoms - feat_mat.shape[0])
            adj_mat_1 = np.where(adj_mat[feat_mat.shape[0]:, feat_mat.shape[0]:] == 1)
            for i, j in zip(adj_mat_1[0], adj_mat_1[1]):
                if (i + feat_mat.shape[0], j + feat_mat.shape[0]) not in drug_feature[drug][-1]:
                    drug_feature[drug][-1].append((i + feat_mat.shape[0], j + feat_mat.shape[0]))

        feat[:feat_mat.shape[0], :] = feat_mat
        drug_feature[drug][0] = feat

    return drug_feature


def gen_drug_graph(drug_feature, data_id):
    """
    Construct graphs from current drug features
    :param drug_feature: current drug features generated from preceding steps
    :param data_id: sample index
    :return: a list of pgl.graph
    """
    graph_list = []
    for i in data_id:
        g = pgl.Graph(edges=drug_feature[i[1]][-1],
                      num_nodes=Max_atoms,
                      node_feat={'nfeat': drug_feature[i[1]][0].astype('float32')},
                      )
        graph_list.append(g)

    return graph_list


def gen_omics_feature(data_idx, mutation_feature, gexpr_feature, methylation_feature):
    """
    Generate sequential omics feature matrix corresponding to sample index
    :param data_idx: sample index
    :param mutation_feature: raw mutation features generated from preceding steps
    :param gexpr_feature: raw gene expression features generated from preceding steps
    :param methylation_feature: raw methylation features generated from preceding steps
    :return: sequential omics feature matrix
    """
    cancer_type_list = []
    nb_instance = len(data_idx)
    nb_mutation_feature = mutation_feature.shape[1]  # 34673
    nb_gexpr_features = gexpr_feature.shape[1]  # 697
    nb_methylation_features = methylation_feature.shape[1]  # 808
    mutation_data = np.zeros((nb_instance, 1, nb_mutation_feature, 1), dtype='float32')
    gexpr_data = np.zeros((nb_instance, nb_gexpr_features), dtype='float32')
    methylation_data = np.zeros((nb_instance, nb_methylation_features), dtype='float32')
    target = np.zeros(nb_instance, dtype='float32')
    for idx in range(nb_instance):
        cell_line_id, pubchem_id, ln_IC50, cancer_type = data_idx[idx]
        mutation_data[idx, 0, :, 0] = mutation_feature.loc[cell_line_id].values
        gexpr_data[idx, :] = gexpr_feature.loc[cell_line_id].values
        methylation_data[idx, :] = methylation_feature.loc[cell_line_id].values
        target[idx] = ln_IC50
        cancer_type_list.append([cancer_type, cell_line_id, pubchem_id])
    return mutation_data, gexpr_data, methylation_data, target, cancer_type_list

def main(args):
    """Entry for data preprocessing."""

    processed_name = args.save_dir 
    os.makedirs(os.path.join(data_dir, processed_name), exist_ok=True)
    processed_dir = os.path.join(data_dir, processed_name)

    save_dir = '%s/GDSC/drug_graph_feat' % args.data_dir
    raw_drug_feature(Drug_smiles_file, save_dir)
    Drug_feature_file = '%s/GDSC/drug_graph_feat' % data_dir

    metadata = metadata_generate(Drug_info_file,
                                 Cell_line_info_file,
                                 Genomic_mutation_file,
                                 Drug_feature_file,
                                 Gene_expression_file,
                                 Methylation_file)
    drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_idx = metadata['metadata']
    drug_feature = gen_drug_feature(drug_feature, israndom=args.israndom)
    train_idx, test_idx = data_split(data_idx, args.split_ratio)
    _, _, _, target_all, cancer_type_list_all = gen_omics_feature(data_idx, mutation_feature, gexpr_feature, methylation_feature)
    expdata_name = Gene_expression_file.split('/')[-1][:-4]
    print('==============================')
    print("train_set : test_set == %.2f" % (len(train_idx) / len(test_idx)))
    for split in ['train', 'test']:
        index = train_idx if split == 'train' else test_idx
        mutation_data, gexpr_data, methylation_data, target, cancer_type_list = gen_omics_feature(index,
                                                                                                  mutation_feature,
                                                                                                  gexpr_feature,
                                                                                                  methylation_feature)
        drug_list = gen_drug_graph(drug_feature, index)
        data_lst = [{'drug_list': drug_list, 'mutation_data': mutation_data, 'gexpr_data': gexpr_data,
                     'methylation_data': methylation_data, 'target': target, 'cancer_type_list': cancer_type_list}]
        npz = os.path.join(processed_dir, '{}_{}.npz'.format(split, args.split_ratio))
        save_data_list_to_npz(data_lst, npz)
    print('==============================')
    print('{} training samples and {} testing samples have been generated and saved in {} '.format(len(train_idx),
                                                                                                   len(test_idx),
                                                                                                   processed_dir))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--split_ratio', type=float, default=0.95)
    parser.add_argument('--israndom', type=bool, default=False)
    parser.add_argument('--save_dir', type=str, default='processed_data')
    args = parser.parse_args()
    ####################################Constants Settings###########################
    TCGA_label_set = ["ALL", "BLCA", "BRCA", "CESC", "DLBC", "LIHC", "LUAD",
                      "ESCA", "GBM", "HNSC", "KIRC", "LAML", "LCML", "LGG",
                      "LUSC", "MESO", "MM", "NB", "OV", "PAAD", "SCLC", "SKCM",
                      "STAD", "THCA", 'COAD/READ']
    data_dir = args.data_dir
    Drug_info_file = '%s/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv' % data_dir
    Cell_line_info_file = '%s/CCLE/Cell_lines_annotations_20181226.txt' % data_dir
    Drug_smiles_file = '%s/GDSC/223drugs_pubchem_smiles_old.txt' % data_dir
    Genomic_mutation_file = '%s/CCLE/genomic_mutation_34673_demap_features.csv' % data_dir
    Cancer_response_exp_file = '%s/GDSC/GDSC_IC50.csv' % data_dir
    Gene_expression_file = '%s/CCLE/genomic_expression_560celllines_697genes_demap_features.csv' % data_dir
    Methylation_file = '%s/CCLE/genomic_methylation_561celllines_808genes_demap_features.csv' % data_dir
    Max_atoms = 100
    #####################################Main#########################################
    main(args)
