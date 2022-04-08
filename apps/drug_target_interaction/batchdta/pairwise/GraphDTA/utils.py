import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric.data import Dataset
from torch_geometric import data as DATA
import torch
import pdb

class TrainDataset(Dataset):
    def __init__(self, root='./', train_x1_index=None, train_x2_index=None, train_d=None, train_t=None, y=None, onehot_train_mixed=None,smile_graph=None,transform=None,
                 pre_transform=None):
        super(TrainDataset, self).__init__(root,transform, pre_transform)
        #root is required for save preprocessed data, default is '/tmp'
        self.train_x1_index = train_x1_index
        self.train_x2_index = train_x2_index
        self.train_d = train_d
        self.train_t = train_t
        self.y = y
        self.onehot_train_mixed = onehot_train_mixed
        self.smile_graph = smile_graph
    
    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, xd1, xd2, xt1, xt2, y, train_mixed, smile_graph):
        smiles1 = xd1
        target1 = xt1

        smiles2 = xd2
        target2 = xt2

        labels = y

        # convert SMILES to molecular representation using rdkit
        c_size1, features1, edge_index1 = smile_graph[smiles1]
        c_size2, features2, edge_index2 = smile_graph[smiles2]
        # make the graph ready for PyTorch Geometrics GCN algorithms:
        GCNData1 = DATA.Data(x=torch.Tensor(features1),
                            edge_index=torch.LongTensor(edge_index1).transpose(1, 0),
                            y=torch.FloatTensor([labels]))
        GCNData1.target = torch.LongTensor([target1])
        GCNData1.train_mixed = torch.LongTensor([train_mixed])
        GCNData1.__setitem__('c_size', torch.LongTensor([c_size1]))

        GCNData2 = DATA.Data(x=torch.Tensor(features2),
                            edge_index=torch.LongTensor(edge_index2).transpose(1, 0),
                            y=torch.FloatTensor([labels]))
        GCNData2.target = torch.LongTensor([target2])
        GCNData2.train_mixed = torch.LongTensor([train_mixed])
        GCNData2.__setitem__('c_size', torch.LongTensor([c_size2]))

        return GCNData1, GCNData2

    def len(self):
        return len(self.train_x1_index)

    def get(self, idx):
        x1_index = self.train_x1_index[idx]
        x2_index = self.train_x2_index[idx]

        xd1 = self.train_d[x1_index]
        xd2 = self.train_d[x2_index]
        xt1 = self.train_t[x1_index]
        xt2 = self.train_t[x2_index]
        Y = self.y[idx]
        train_mixed = self.onehot_train_mixed[idx]

        data1, data2 = self.process(xd1, xd2, xt1, xt2, Y, train_mixed, self.smile_graph)

        return data1, data2

class TrainDataset1(Dataset):
    def __init__(self, root='./', train_x1_index=None, train_x2_index=None, train_d=None, train_t=None, y=None, smile_graph=None,transform=None,
                 pre_transform=None):
        super(TrainDataset1, self).__init__(root,transform, pre_transform)
        #root is required for save preprocessed data, default is '/tmp'
        self.train_x1_index = train_x1_index
        self.train_x2_index = train_x2_index
        self.train_d = train_d
        self.train_t = train_t
        self.y = y
        self.smile_graph = smile_graph
    
    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, xd1, xd2, xt1, xt2, y, smile_graph):
        smiles1 = xd1
        target1 = xt1

        smiles2 = xd2
        target2 = xt2

        labels = y

        # convert SMILES to molecular representation using rdkit
        c_size1, features1, edge_index1 = smile_graph[smiles1]
        c_size2, features2, edge_index2 = smile_graph[smiles2]
        # make the graph ready for PyTorch Geometrics GCN algorithms:
        GCNData1 = DATA.Data(x=torch.Tensor(features1),
                            edge_index=torch.LongTensor(edge_index1).transpose(1, 0),
                            y=torch.FloatTensor([labels]))
        GCNData1.target = torch.LongTensor([target1])
        GCNData1.__setitem__('c_size', torch.LongTensor([c_size1]))

        GCNData2 = DATA.Data(x=torch.Tensor(features2),
                            edge_index=torch.LongTensor(edge_index2).transpose(1, 0),
                            y=torch.FloatTensor([labels]))
        GCNData2.target = torch.LongTensor([target2])
        GCNData2.__setitem__('c_size', torch.LongTensor([c_size2]))

        return GCNData1, GCNData2

    def len(self):
        return len(self.train_x1_index)

    def get(self, idx):
        x1_index = self.train_x1_index[idx]
        x2_index = self.train_x2_index[idx]

        xd1 = self.train_d[x1_index]
        xd2 = self.train_d[x2_index]
        xt1 = self.train_t[x1_index]
        xt2 = self.train_t[x2_index]
        Y = self.y[idx]

        data1, data2 = self.process(xd1, xd2, xt1, xt2, Y, self.smile_graph)

        return data1, data2



class TestDataset(Dataset):
    def __init__(self, root='./', xd=None, xt=None, y=None, smile_graph=None,test_index=None,transform=None,
                 pre_transform=None):
        #root is required for save preprocessed data, default is '/tmp'
        super(TestDataset, self).__init__(root,transform, pre_transform)
        self.test_index = test_index
        self.max_len = max([len(i) for i in self.test_index])
        self.data_list = self.process(xd, xt, y,smile_graph)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, xd, xt, y, smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
            GCNData.target = torch.LongTensor([target])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))

        
            data_list.append(GCNData)
        return data_list

    def len(self):
        return len(self.test_index)

    def get(self, idx):
        return_test_index = self.test_index[idx]
        return_data = [ self.data_list[index] for index in return_test_index]
        return return_data


class TestDataset1(Dataset):
    def __init__(self, root='./', xd=None, xt=None, y=None, smile_graph=None,groupID=None,transform=None,
                 pre_transform=None):
        #root is required for save preprocessed data, default is '/tmp'
        super(TestDataset1, self).__init__(root,transform, pre_transform)
        self.xd = xd
        self.xt = xt
        self.y = y
        self.groupID = groupID

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, xd, xt, y, smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
            GCNData.target = torch.LongTensor([target])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))

        
            data_list.append(GCNData)
        return data_list

    def len(self):
        return len(self.test_index)

    def get(self, idx):
        return_data = process( xd[idx], xt[idx], y[idx], smile_graph)
        return_group = self.groupID[idx]
        return (return_data, return_group)


class Data_Encoder(Dataset):
    def __init__(self, root='./', data=None, transform=None, pre_transform=None):
        #root is required for save preprocessed data, default is '/tmp'
        super(Data_Encoder, self).__init__(root,transform, pre_transform)
        self.data = data

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    
    def len(self):
        return len(self.data)

    def get(self, idx):
        return_data = self.data[idx]
        return return_data