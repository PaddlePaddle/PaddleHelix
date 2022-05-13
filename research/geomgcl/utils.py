import paddle
import paddle.nn as nn
import numpy as np
from random import Random
from dataset import Subset

def split_dataset(dataset, idx, num_folds=10):
    random = Random(0)
    indices = np.repeat(np.arange(num_folds), 1 + len(dataset) // num_folds)[:len(dataset)]
    random.shuffle(indices)
    test_index = idx % num_folds
    val_index = (idx + 1) % num_folds

    train_ids, val_ids, test_ids = [], [], []
    for d, index in zip(range(len(dataset)), indices):
        if index == test_index:
            test_ids.append(d)
        elif index == val_index:
            val_ids.append(d)
        else:
            train_ids.append(d)
    
    return Subset(dataset, train_ids), Subset(dataset, val_ids), Subset(dataset, test_ids)

def split_dataset_gcl(dataset, ratio=0.1):
    np.random.seed(123)
    idx = np.random.permutation(len(dataset))
    data_shuffle = np.arange(len(dataset))[idx]
    spl = int(len(data_shuffle) * ratio)
    train_ids, val_ids = data_shuffle[spl:], data_shuffle[:spl]
    return Subset(dataset, train_ids), Subset(dataset, val_ids)

def rmse(y,f):
    rmse = np.sqrt(((y - f)**2).mean(axis=0))
    return rmse