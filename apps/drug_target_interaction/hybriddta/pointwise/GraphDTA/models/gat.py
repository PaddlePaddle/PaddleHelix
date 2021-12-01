"""GraphDTA_GAT backbone model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp


# GAT backbone model
class GATNet(torch.nn.Module):
    """GAT model.

    Args:
        data: Input data.
    
    Returns:
        out: Prediction results.
    """
    def __init__(self, num_features_xd=78, n_output=1, num_features_xt=25,
                     n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(GATNet, self).__init__()
        # Basic config
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # SMILES graph branch
        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)
        # Protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc_xt1 = nn.Linear(32*121, output_dim)
        # Combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

    def forward(self, data):
        """tbd."""
        # Get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Get protein input
        target = data.target

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)          # global max pooling

        x = self.fc_g1(x)
        x = self.relu(x)
        # 1d conv layers
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt1(embedded_xt)
        conv_xt = self.relu(conv_xt)
        # Flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc_xt1(xt)
        # Concat
        xc = torch.cat((x, xt), 1)
        # Add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
