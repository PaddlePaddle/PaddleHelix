#-*- coding: utf-8 -*-
from ogb.graphproppred import GraphPropPredDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
import pickle
import ogbcomp_helper as helper
import numpy as np
import pandas as pd
import json
import os
import argparse
import sys
from tqdm import tqdm

def main(args):
    ds = args.dataset_name
    all_probs = {}
    all_ap = {}
    all_rocs = {}
    train_label_props = {}

    n_estimators = 1000
    max_tasks = None
    nreps = 10

    eval_scores = []
    test_scores = []

    # Read in dataset
    print("Reading dataset")
    dataset = GraphPropPredDataset(name=ds)
    
    # By default, the line above will save data to the path below
    df_smi = pd.read_csv(f"dataset/{ds}/mapping/mol.csv.gz".replace("-", "_"))
    smiles = df_smi["smiles"]
    outcomes = df_smi.set_index("smiles").drop(["mol_id"], axis=1)
    
    # Generate features
    print("Extracting features...")
    X = helper.parallel_apply(smiles, func=helper.getmorganfingerprint2)
    
    soft_mgf_file = "./dataset/%s/soft_mgf_feat.npy" % (args.dataset_name.replace("-", "_"))
    maccs_file = "./dataset/%s/maccs_feat.npy" % (args.dataset_name.replace("-", "_"))
    soft_mgf_feat = np.load(soft_mgf_file)
    maccs_feat = np.load(maccs_file)
    maccs_feat_dim = maccs_feat.shape[1]

    print("concat mgf and maccs feature")
    gnn_feat = pd.DataFrame(soft_mgf_feat, columns=[2048+i for i in range(2048)])
    index = X.index
    new_X = pd.concat([X.reset_index(drop=True), gnn_feat.reset_index(drop=True)], axis=1)
    X = pd.DataFrame(new_X.values, index=index)

    maccs_feat = pd.DataFrame(maccs_feat, columns=[X.values.shape[1]+i for i in range(maccs_feat_dim)])
    index = X.index
    new_X = pd.concat([X.reset_index(drop=True), maccs_feat.reset_index(drop=True)], axis=1)
    X = pd.DataFrame(new_X.values, index=index)


    for rep in range(nreps):
        # Split into train/val/test
        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]

        for oo in tqdm(outcomes.columns[:max_tasks]):
            # Get probabilities
            val_key = ds, oo, rep, "val"
            test_key = ds, oo, rep, "test"
            
            # If re-running, skip finished runs
            if val_key in all_probs:
                print("Skipping", val_key[:-1])
                continue
    
            # Split outcome in to train/val/test
            Y = outcomes[oo]
            y_train, y_val, y_test = Y.loc[X_train.index], Y.loc[X_val.index], Y.loc[X_test.index]
            
            # Skip outcomes with no positive training examples
            if y_train.sum() == 0:
                continue
            
            # Remove missing labels in validation
            y_val, y_test = y_val.dropna(), y_test.dropna()
            X_v, X_t = X_val.loc[y_val.index], X_test.loc[y_test.index]
            
            # Remove missing values in the training labels, and downsample imbalance to cut runtime
            y_tr = helper.process_y(y_train)
            train_label_props[ds, oo, rep] = y_tr.mean()
            print(f"Sampled label balance:\n{y_tr.value_counts()}")
            
            # Fit model
            print("Fitting model...")
            #  rf = RandomForestClassifier(min_samples_leaf=2, n_estimators=n_estimators, n_jobs=-1)
            rf = RandomForestClassifier(min_samples_leaf=2,
                    n_estimators=n_estimators,
                    n_jobs=-1,
                    criterion='entropy',
                    class_weight={0:1, 1:10}
                    )
            rf.fit(X_train.loc[y_tr.index], y_tr)

            # Calculate probabilities
            all_probs[val_key] = pd.Series(rf.predict_proba(X_v)[:, 1], index=X_v.index)
            all_probs[test_key] = pd.Series(rf.predict_proba(X_t)[:, 1], index=X_t.index)

            if y_val.sum() > 0:
                all_ap[val_key] = average_precision_score(y_val, all_probs[val_key])
                all_rocs[val_key] = roc_auc_score(y_val, all_probs[val_key])
            
            if y_test.sum() > 0:
                all_ap[test_key] = average_precision_score(y_test, all_probs[test_key])
                all_rocs[test_key] = roc_auc_score(y_test, all_probs[test_key])

            print(f'{oo}, rep {rep}, AP (val, test): {all_ap.get(val_key, np.nan):.3f}, {all_ap.get(test_key, np.nan):.3f}')
            print(f'\tROC (val, test): {all_rocs.get(val_key, np.nan):.3f}, {all_rocs.get(test_key, np.nan):.3f}')
            eval_scores.append(all_rocs.get(val_key, np.nan))
            test_scores.append(all_rocs.get(test_key, np.nan))

    eval_avg = np.mean(eval_scores)
    eval_std = np.std(eval_scores, ddof=1)
    test_avg = np.mean(test_scores)
    test_std = np.std(test_scores, ddof=1)
    print("eval: ", eval_scores)
    print("test: ", test_scores)
    print("%s | eval and test: %.4f (%.4f),%.4f (%.4f)" % (args.dataset_name, eval_avg, eval_std, test_avg, test_std))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='gnn')
    parser.add_argument("--dataset_name", type=str, default="ogbg-molhiv")
    args = parser.parse_args()

    main(args)


