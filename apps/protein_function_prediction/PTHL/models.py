import paddle 
from paddle import nn 
from paddle.nn import functional as F 
import pgl 
import pgl.math as math
from layers import GVP, RR_VPConv, Linear

class PTHL(nn.Layer):
    def __init__(self, args):
        super().__init__()
        self.n_h_dim = args.n_h_dim
        self.v_n_feats = args.v_n_feats
        self.s_h_dim = args.s_h_dim
        self.v_h_dim = args.v_h_dim
        # self.e_v_n_feats = args.e_v_n_feats
        # self.e_s_h_dim = args.e_s_h_dim
        # self.e_v_h_dim = args.e_v_h_dim
        self.num_convs = args.num_convs
        self.n_channels = args.n_channels
        self.n_labels = args.n_labels
        self.t_out_dim = args.t_out_dim 
        self.t_n_layers = args.t_n_layers 
        self.t_n_heads = args.t_n_heads
        self.n_emb = nn.Embedding(self.n_channels, self.s_h_dim)

        self.conv_layers = nn.LayerList()
        
        out_dims = (self.s_h_dim, self.v_h_dim)
        for i in range(self.num_convs):
            in_dims = [self.s_h_dim, self.v_h_dim]
            if i == 0:
                in_dims[1] = self.v_n_feats 
            self.conv_layers.append(PG_GNN_Layer(in_dims, out_dims))

        self.output_gvp = GVP((self.s_h_dim, self.v_h_dim), (self.n_h_dim, 0), activations=(None, None))
        self.seq_encoder = PT_Encoder(self.n_h_dim, self.t_out_dim, self.t_n_layers, self.t_n_heads, self.n_labels)

    def forward(self, data):
        n_graph, rr_graphs = data
        seq_feats = self.n_emb(n_graph.node_feat['seq'])
        v_feats = n_graph.node_feat['v_feats']
        local_sys = n_graph.node_feat['local_sys']
        node_indx = n_graph.node_feat['node_indx']
        node_batch_id = n_graph.graph_node_id
        batch_size = paddle.max(node_batch_id) + 1
        x = (seq_feats, v_feats)
        for i, conv in enumerate(self.conv_layers):
            x = conv(n_graph, rr_graphs, x, local_sys)
            # Aggregation from the domains
        
        x = self.output_gvp(x)
        prot_emb = self.seq_encoder(batch_size, node_batch_id, x, node_indx)

        return prot_emb


class PG_GNN_Layer(nn.Layer):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.self_vp_layer = GVP(in_dims, out_dims, h_dim=out_dims[1]) 
        self.rr_layer = nn.LayerList(
            RR_VPConv(in_dims, out_dims) for _ in range(2)
        )
        self.agg_layer = AggregationLayer(out_dims, out_dims)
    
    def forward(self, n_graph, rr_graphs, node_feats, local_sys):
        assert len(rr_graphs) == len(self.rr_layer)

        hid_feats = []
        for rr_g, rr_conv in zip(rr_graphs, self.rr_layer):
            out  = rr_conv(rr_g, node_feats, local_sys)
            hid_feats.append(out)
        
        s_feats, v_feats = zip(*hid_feats)
        s_feats = paddle.stack(s_feats, axis=1)
        v_feats = paddle.stack(v_feats, axis=1)
        node_feats = self.self_vp_layer(node_feats)

        out = self.agg_layer(n_graph, node_feats, (s_feats, v_feats))
        out = node_feats[0] + out[0], node_feats[1] + out[1]
        return out


class AggregationLayer(nn.Layer):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.s_i, self.v_i = in_dims 
        self.s_o, self.v_o = out_dims

        self.s_ln = nn.LayerList(
            Linear(self.s_i, self.s_o, bias_attr=False) for _ in range(2)
        )
        self.v_ln = nn.LayerList(
            Linear(self.v_i, self.v_o, bias_attr=False) for _ in range(2)
        )
    
    def forward(self, n_graph, node_feats, rr_x):
        assert  2 == paddle.shape(rr_x[0])[1] == paddle.shape(rr_x[1])[1]
        s_feats, v_feats = rr_x
        v_feats = paddle.transpose(v_feats, [0, 1, 3, 2])
        for i in range(2):
            ang_s = self.s_ln[i](s_feats[:, i])
            ang_v = self.v_ln[i](v_feats[:, i])
            s_h = ang_s if i == 0 else s_h + ang_s
            v_h = ang_v if i == 0 else v_h + ang_v 
        v_h = paddle.transpose(v_h, [0, 2, 1])

        return (s_h, v_h)

class PT_Encoder(nn.Layer):
    def __init__(self, in_dim, out_dim, n_layers, n_heads, n_labels):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_labels = n_labels
        self.ln = Linear(self.in_dim, self.out_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=out_dim, nhead=self.n_heads, dim_feedforward=out_dim)
        self.transformer_encoder_layer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.pred = nn.Linear(self.out_dim, self.n_labels)
    
    def forward(self, batch_size, node_batch_id, feats, node_indx):

        feats = self.ln(feats)
        seqs = paddle.zeros(shape=(batch_size, 1000, feats.shape[1]), dtype='float32')
        attn_mask = paddle.full(shape=(batch_size, 1000),fill_value=False, dtype='bool')

        indices = (node_batch_id, node_indx)
        seqs[indices] = feats
        attn_mask[indices] = True
        attn_mask_expand = paddle.unsqueeze(attn_mask, 1)
        attn_mask_expand = paddle.unsqueeze(attn_mask_expand, 1)
        
        prot_emb = self.transformer_encoder_layer(seqs, src_mask=attn_mask_expand)
        # prot_emb = seqs 
        prot_emb = prot_emb * paddle.unsqueeze(paddle.cast(attn_mask, 'float32'), -1)
        prot_emb = paddle.sum(prot_emb, 1)

        prot_emb = F.relu(prot_emb)

        return self.pred(prot_emb)
