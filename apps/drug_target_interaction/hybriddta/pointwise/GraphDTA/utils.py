"""Utils scripts for GraphDTA."""

import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch


class TestbedDataset(InMemoryDataset):
    """TestbedDataset."""
    def __init__(self, root='/tmp', dataset='DAVIS',
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None):
        # Root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y,smile_graph)
    
    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass
    
    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, xt, y,smile_graph):
        """Customize the process method to fit the task of drug-target affinity prediction.

        Args:
            xd: List of SMILES.
            xt: List of encoded target (categorical or one-hot).
            y: List of labels.

        Returns:
            PyTorch-Geometric format processed data.
        """
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # Convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # Make the graph ready for PyTorch Geometrics GCN algorithms
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
            GCNData.target = torch.LongTensor([target])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # Append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        self.data, self.slices = self.collate(data_list)

def rmse(y,f):
    """RMSE."""
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

def mse(y,f):
    """MSE."""
    mse = ((y - f)**2).mean(axis=0)
    return mse

def pearson(y,f):
    """Pearson."""
    rp = np.corrcoef(y, f)[0,1]
    return rp

def spearman(y,f):
    """Spearman."""
    rs = stats.spearmanr(y, f)[0]
    return rs

def ci(y,f):
    """CI."""
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci