from itertools import chain
import paddle
from paddle import nn
import paddle.nn.functional as F
from pgl.nn import GCNConv, GATConv, GraphSageConv
from pgl.nn.pool import GraphPool


class DeepFRI(nn.Layer):
    def __init__(self, args):
        super().__init__()
        self.n_channels = args.n_channels
        self.gc_dims = args.gc_dims
        self.fc_dims = args.fc_dims
        self.drop = args.drop
        self.n_labels = args.n_labels

        # Load language model
        self.lm_model = None
        if args.lm_model_name:
            pass  # TODO: Load a pre-trained lm_model for protein sequence learning

        self.lm_dim = args.lm_dim
        self.aa_emb = nn.Embedding(self.n_channels, self.lm_dim)
        self.gcnn_list = nn.LayerList()

        self.gc_layer = args.gc_layer
        if self.gc_layer == "GAT":
            GConvLayer = GATConv
            params = dict(feat_drop=0, attn_drop=0, num_heads=2, concat=False)
            in_feats = [self.lm_dim] + [
                (d * params["num_heads"] if params["concat"] else d)
                for d in self.gc_dims[:-1]
            ]
        elif self.gc_layer == "GraphConv":
            GConvLayer = GCNConv
            params = {}
            in_feats = [self.lm_dim] + self.gc_dims[:-1]
        elif self.gc_layer == "SAGEConv":
            GConvLayer = GraphSageConv
            params = {}
            in_feats = [self.lm_dim] + self.gc_dims[:-1]
        # More graph convolution networks can be added here
        else:
            GConvLayer = None
            raise ValueError("gc_layer not specified.")

        for in_f, out_f in zip(in_feats, self.gc_dims):
            self.gcnn_list.append(GConvLayer(in_f, out_f, **params))

        self.fc_list = nn.LayerList()
        in_feats = sum(self.gc_dims)
        for out_feats in self.fc_dims:
            self.fc_list.append(nn.Linear(in_feats, out_feats))
            in_feats = out_feats

        self.global_pool = GraphPool(pool_type="sum")
        self.func_predictor = nn.Linear(in_feats, self.n_labels)

    def forward(self, graphs, padded_feats):

        out = self.aa_emb(graphs.node_feat["seq"])

        if self.lm_model is not None:
            pass  # TODO: Sum output from lm_model using 'padded_feats' as input and variable 'out' above.

        gcnn_concat = []
        for gcnn in self.gcnn_list:
            out = gcnn(graphs, out)
            out = F.elu(out)
            gcnn_concat.append(out)

        out = paddle.concat(gcnn_concat, axis=1)
        out = self.global_pool(graphs, out)

        for fc in self.fc_list:
            out = fc(out)
            out = F.relu(out)
            out = F.dropout(out, p=self.drop, training=self.training)

        out = self.func_predictor(out)

        return out
