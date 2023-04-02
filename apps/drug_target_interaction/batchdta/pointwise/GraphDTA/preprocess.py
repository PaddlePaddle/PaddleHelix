"""Preprocessing scripts for GraphDTA."""

import pandas as pd
import numpy as np
import os
import rdkit
import sklearn
import torch
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *

# Global setting
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

def one_of_k_encoding(x, allowable_set):
    """tbd."""
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    """Atom feat."""
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def smile_to_graph(smile):
    """SMILES to graph."""
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index

def seq_cat(prot):
    """tbd."""
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

def process_data(df):
    """Process data."""
    pairs=[]
    i = 0
    for _,row in df.iterrows():
        try:
            pair = []
            lg = Chem.MolToSmiles(Chem.MolFromSmiles(row[1]), isomericSmiles=True) # smiles
            pair.append(lg)
            pair.append(seq_cat(row[0]))
            pair.append(row[4]) # label
            pair.append(row[2]) # target name
            pairs.append(pair)
        except:
            i += 1
    
    print('discard {} SMILES'.format(i))
    pairs=pd.DataFrame(pairs)
    #Drug
    compound_iso_smiles = pairs.iloc[:,0]
    compound_iso_smiles = set(compound_iso_smiles)
    smile_graph = {}
    outlier_smiles = []
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
        _, _, edge_index = g
        edge_index=torch.LongTensor(edge_index)
        if len(edge_index.shape) == 1:
            outlier_smiles.append(smile)
    print('we discard smiles sequence : {}'.format(outlier_smiles))
        
    train_drugs, train_prots, train_Y, target_name= list(pairs.iloc[:,0]),list(pairs.iloc[:,1]),list(pairs.iloc[:,2]), list(pairs.iloc[:,3])
    target_name, train_drugs, train_prots, train_Y = np.asarray(target_name), np.asarray(train_drugs), np.asarray(train_prots), np.asarray(train_Y)
    mask = np.full(len(train_drugs),True)
    for i in outlier_smiles:
        temp = train_drugs != i
        mask = mask & temp

    target_name = target_name[mask]
    train_drugs = train_drugs[mask]
    train_prots = train_prots[mask]
    train_Y = train_Y[mask]
    return (target_name, train_drugs, train_prots, train_Y, smile_graph)