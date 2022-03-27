import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
from pgl.utils import op
import pgl.math as math
from pgl.nn.pool import GraphPool


class Node2EdgeLayer(nn.Layer):
    """Implementation of Node->Edge Aggregation Layer."""

    def __init__(self, node_dim, edge_dim, activation=F.relu):
        super(Node2EdgeLayer, self).__init__()
        self.src_ln = nn.Linear(node_dim, edge_dim, bias_attr=False)
        self.dst_ln = nn.Linear(node_dim, edge_dim, bias_attr=False)
        self.d_ln = nn.Linear(node_dim, edge_dim)

    def agg_func(self, src_feat, dst_feat, edge_feat):
        h_agg = src_feat["h"] + dst_feat["h"] + edge_feat["h"]
        return {"h": h_agg}

    def forward(self, g, node_feat, edge_feat_dist):
        src_feat = self.src_ln(node_feat)
        dst_feat = self.dst_ln(node_feat)
        edge_feat_dist = self.d_ln(edge_feat_dist)
        msg = g.send(
            self.agg_func,
            src_feat={"h": src_feat},
            dst_feat={"h": dst_feat},
            edge_feat={"h": edge_feat_dist},
        )
        edge_feat = msg["h"]
        return edge_feat


class Edge2NodeAttentionLayer(nn.Layer):
    def __init__(self, hidden_dim, edge_dim, num_angle, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.e_in_dim = edge_dim
        self.out_dim = hidden_dim
        self.num_angle = num_angle
        self.drop = dropout

        self.edg_fc = nn.Linear(hidden_dim, hidden_dim, bias_attr=False)

        self.dst_fc = nn.Linear(hidden_dim, hidden_dim, bias_attr=False)
        self.src_fcs = nn.LayerList()

        for i in range(self.num_angle):
            self.src_fcs.append(nn.Linear(edge_dim, hidden_dim, bias_attr=False))

        self.weight_src = nn.Linear(hidden_dim, 1, bias_attr=False)
        self.weight_dst = nn.Linear(hidden_dim, 1, bias_attr=False)
        self.weight_edg = nn.Linear(hidden_dim, 1, bias_attr=False)
        self.drop = nn.Dropout(p=self.drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def attn_send_func(self, src_feat, dst_feat, edge_feat):
        alpha = dst_feat["attn"] + edge_feat["attn_dist"] + edge_feat["attn_edge"]
        alpha = self.leaky_relu(alpha)
        return {"alpha": alpha, "edge_h": edge_feat["edge_h"]}

    def attn_recv_func(self, msg):
        alpha = msg.reduce_softmax(msg["alpha"])
        alpha = self.drop(alpha)

        feature = msg["edge_h"]
        assert feature.ndim == alpha.ndim
        feature = feature * alpha
        feature = msg.reduce(feature, pool_type="sum")
        return feature

    def forward(self, n2n_g, n_feats, edge_feats, edge_dist_feat):

        assert len(edge_feats) == self.num_angle
        for i in range(self.num_angle):
            if i == 0:
                temp_bond_feats = self.src_fcs[i](self.drop(edge_feats[i]))
            else:
                temp_bond_feats += self.src_fcs[i](self.drop(edge_feats[i]))

        edge_feats = temp_bond_feats
        edge_dist_feat = self.edg_fc(self.drop(edge_dist_feat))
        n_feats = self.dst_fc(self.drop(n_feats))

        attn_src = self.weight_src(edge_feats)
        attn_dst = self.weight_dst(n_feats)
        attn_edg_dist = self.weight_edg(edge_dist_feat)

        msg = n2n_g.send(
            self.attn_send_func,
            dst_feat={"attn": attn_dst},
            edge_feat={
                "attn_dist": attn_edg_dist,
                "attn_edge": attn_src,
                "edge_h": edge_feats,
            },
        )
        rst = n2n_g.recv(reduce_func=self.attn_recv_func, msg=msg)

        return rst


class Edge2NodeLayer(nn.Layer):
    """Implementation of Distance-aware Edge->Node Aggregation Layer."""

    def __init__(
        self,
        edge_dim,
        node_dim,
        hidden_dim,
        num_heads,
        num_angle,
        dropout,
        merge="mean",
        activation=F.relu,
    ):
        super(Edge2NodeLayer, self).__init__()
        self.merge = merge
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.activation = activation

        self.att_layers = nn.LayerList(
            Edge2NodeAttentionLayer(hidden_dim, edge_dim, num_angle, dropout)
            for _ in range(num_heads)
        )

    def forward(self, n2n_g, n_feats, edge_feats, edge_feat_dist):

        feature = []
        for att_l in self.att_layers:
            feature.append(att_l(n2n_g, n_feats, edge_feats, edge_feat_dist))

        feature = paddle.stack(feature, axis=1)
        if self.merge == "cat":
            feature = paddle.reshape(feature, [-1, self.num_heads * self.hidden_dim])
        if self.merge == "mean":
            feature = paddle.mean(feature, axis=1)

        return feature


class DomainAttentionLayer(nn.Layer):
    """Implementation of Angle Domain-speicific Attention Layer."""

    def __init__(self, edge_dim, hidden_dim, dropout, activation=F.relu):
        super(DomainAttentionLayer, self).__init__()
        self.attn_fc_scr = nn.Linear(edge_dim, hidden_dim)
        self.attn_fc_dst = nn.Linear(edge_dim, hidden_dim)
        self.attn_out = nn.Linear(hidden_dim, 1, bias_attr=False)

        self.feat_drop = nn.Dropout(p=dropout)
        self.attn_drop = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.activation = activation

    def attn_send_func(self, src_feat, dst_feat, edge_feat):
        h_c = src_feat["h"] + dst_feat["h"]
        h_c = self.tanh(h_c)
        h_s = self.attn_out(h_c)
        return {"alpha": h_s, "h": src_feat["neig_h"]}

    def attn_recv_func(self, msg):
        alpha = msg.reduce_softmax(msg["alpha"])
        alpha = self.attn_drop(alpha)  # [-1, 1]
        feature = msg["h"]  # [-1, hidden_dim]
        feature = feature * alpha
        feature = msg.reduce(feature, pool_type="sum")
        return feature

    def forward(self, g, edge_feats):
        edge_feats = self.feat_drop(edge_feats)
        msg = g.send(
            self.attn_send_func,
            src_feat={"h": self.attn_fc_scr(edge_feats), "neig_h": edge_feats},
            dst_feat={"h": self.attn_fc_dst(edge_feats)},
        )
        rst = g.recv(reduce_func=self.attn_recv_func, msg=msg)
        if self.activation:
            rst = self.activation(rst)
        return rst


class Edge2EdgeLayer(nn.Layer):
    """Implementation of Angle-oriented Edge->Edge Aggregation Layer."""

    def __init__(
        self, edge_dim, hidden_dim, num_angle, dropout, merge="cat", activation=None
    ):
        super(Edge2EdgeLayer, self).__init__()
        self.num_angle = num_angle
        self.hidden_dim = hidden_dim
        self.merge = merge
        self.conv_layer = nn.LayerList()
        for _ in range(num_angle):
            conv = DomainAttentionLayer(edge_dim, hidden_dim, dropout, activation=None)
            self.conv_layer.append(conv)
        self.activation = activation

    def forward(self, g_list, edge_feats):
        assert len(g_list) == self.num_angle
        h_list = []
        for k in range(self.num_angle):
            h = self.conv_layer[k](g_list[k], edge_feats)
            if self.activation:
                h = self.activation(h)
            h_list.append(h)

        return h_list


class FuncPredictor(nn.Layer):
    def __init__(self, in_feats, dense_dims, n_labels, drop):
        super().__init__()
        self.n_labels = n_labels
        self.in_feats = in_feats
        self.dense_dims = dense_dims
        self.drop = drop
        self.mlp = nn.LayerList()
        self.n_layers = len(dense_dims)
        for i in range(self.n_layers):
            self.mlp.append(nn.Linear(in_feats, dense_dims[i]))
            in_feats = dense_dims[i]
        self.out_layer = nn.Linear(in_feats, n_labels)

    def forward(self, feats):
        for i in range(self.n_layers):
            feats = self.mlp[i](feats)
            feats = F.dropout(feats, p=self.drop, training=self.training)
            feats = F.relu(feats)
        out = self.out_layer(feats)

        return out


class Readout(nn.Layer):
    def __init__(self, in_feats):
        super().__init__()
        self.in_feats = in_feats
        self.ln = nn.Linear(in_feats, 1, bias_attr=False)

    def forward(self, n2n_g, n_feats):
        scores = self.ln(n_feats)
        scores = math.segment_softmax(scores, n2n_g.graph_node_id)
        n_feats = scores * n_feats
        graph_reprs = math.segment_sum(n_feats, n2n_g.graph_node_id)
        return graph_reprs
