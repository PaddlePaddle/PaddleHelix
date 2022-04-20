import paddle

paddle.disable_static()
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
from layers import (
    Node2EdgeLayer,
    Edge2EdgeLayer,
    Edge2NodeLayer,
    FuncPredictor,
    Readout,
)


class ProteinSIGN(nn.Layer):
    def __init__(self, args):
        super(ProteinSIGN, self).__init__()
        num_convs = args.num_convs
        dense_dims = args.dense_dims
        n_channels = args.n_channels
        hidden_dim = args.hidden_dim
        self.num_convs = num_convs

        cmap_thresh = args.cmap_thresh
        num_angle = args.num_angle
        merge_e2e = args.merge_e2e
        merge_e2n = args.merge_e2n

        activation = args.activation
        num_heads = args.num_heads
        feat_drop = args.feat_drop
        self.n_labels = args.n_labels
        self.drop = args.feat_drop
        self.n_channels = args.n_channels
        self.activation = args.activation

        self.dist_emb = nn.Embedding(int(cmap_thresh) + 1, hidden_dim)
        self.n_emb = nn.Embedding(self.n_channels, hidden_dim)

        self.node2edge_layers = nn.LayerList()
        self.edge2edge_layers = nn.LayerList()
        self.edge2node_layers = nn.LayerList()
        for i in range(num_convs):
            if i == 0:
                node_dim = hidden_dim
            else:
                node_dim = hidden_dim * num_heads if "cat" in merge_e2n else hidden_dim
            edge_dim = hidden_dim * num_angle if "cat" in merge_e2e else hidden_dim

            self.node2edge_layers.append(
                Node2EdgeLayer(node_dim, edge_dim=hidden_dim, activation=activation)
            )
            self.edge2edge_layers.append(
                Edge2EdgeLayer(
                    hidden_dim,
                    hidden_dim,
                    num_angle,
                    feat_drop,
                    merge=merge_e2e,
                    activation=None,
                )
            )
            self.edge2node_layers.append(
                Edge2NodeLayer(
                    hidden_dim,
                    node_dim,
                    hidden_dim,
                    num_heads,
                    num_angle,
                    feat_drop,
                    merge=merge_e2n,
                    activation=activation,
                )
            )

        self.readout = Readout(hidden_dim)
        self.predictor = FuncPredictor(hidden_dim, dense_dims, self.n_labels, self.drop)

    def forward(self, n2n_g, e2e_g):
        n_feats = n2n_g.node_feat["seq"]
        dist_feat = paddle.cast(n2n_g.edge_feat["dist"], "int64")
        n_feats = self.n_emb(n_feats)
        dist_feat = self.dist_emb(dist_feat)

        dist_h = dist_feat
        for i in range(self.num_convs):
            edge_h = self.node2edge_layers[i](n2n_g, n_feats, dist_h)
            edge_h = self.activation(edge_h)
            edge_h = self.edge2edge_layers[i](e2e_g, edge_h)
            n_feats = self.edge2node_layers[i](n2n_g, n_feats, edge_h, dist_h)

        prot_chains_emb = self.readout(n2n_g, n_feats)
        prot_chains_emb = self.activation(prot_chains_emb)

        return self.predictor(prot_chains_emb)
