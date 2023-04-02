import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from pgl.sampling.custom import subgraph
import pgl
import math

class DenseLayer(nn.Layer):
    def __init__(self, in_dim, out_dim, activation=F.relu, bias=True):
        super(DenseLayer, self).__init__()
        self.activation = activation
        if not bias:
            self.fc = nn.Linear(in_dim, out_dim, bias_attr=False)
        else:
            self.fc = nn.Linear(in_dim, out_dim)
    
    def forward(self, input_feat):
        return self.activation(self.fc(input_feat))


class GeometryEmbedding(nn.Layer):
    def __init__(self, rbf_dim, hidden_dim, max_dist, activation):
        super(GeometryEmbedding, self).__init__()
        self.dist_rbf = DistRBF(rbf_dim, max_dist)
        self.angle_rbf = AngleRBF(rbf_dim)
        self.dist_fc = DenseLayer(rbf_dim, hidden_dim, activation, bias=True)
        self.angle_fc = DenseLayer(rbf_dim, hidden_dim, activation, bias=True)

    def forward(self, dist_feat_list, angle_feat_list):
        dist_h_list = []
        for dist_feat in dist_feat_list:
            dist_h = self.dist_rbf(dist_feat)
            dist_h = self.dist_fc(dist_h)
            dist_h_list.append(dist_h)
        angle_h_list = []
        for angle_feat in angle_feat_list:
            angle_feat = angle_feat * (np.pi/180)
            angle_h = self.angle_rbf(angle_feat)
            angle_h = self.angle_fc(angle_h)
            angle_h_list.append(angle_h)
        return dist_h_list, angle_h_list


class EdgeAggConv(nn.Layer):
    def __init__(self, node_in_dim, link_in_dim, hidden_dim, activation=F.relu):
        super(EdgeAggConv, self).__init__()
        in_dim = node_in_dim * 2 + link_in_dim
        self.fc_agg = DenseLayer(in_dim, hidden_dim, activation=activation, bias=True)

    def agg_func(self, src_feat, dst_feat, edge_feat):
        h_agg = paddle.concat([src_feat['h'], dst_feat['h'], edge_feat['h']], axis=-1)
        return {'h': h_agg}

    def forward(self, graph, node_feat, lnk_feat):
        msg = graph.send(self.agg_func, src_feat={'h': node_feat}, dst_feat={'h': node_feat}, edge_feat={'h': lnk_feat})
        edge_feat = msg['h']
        edge_feat = self.fc_agg(edge_feat)
        return edge_feat


class Angle2DConv(nn.Layer):
    def __init__(self, in_dim, hidden_dim, activation=F.relu, dropout=0.0):
        super(Angle2DConv, self).__init__()
        self.feat_drop = nn.Dropout(dropout)
        # parameters for message construction
        self.G = nn.Linear(hidden_dim, hidden_dim, bias_attr=False)
        self.fc_src = DenseLayer(in_dim, hidden_dim, activation, bias=True)
        self.fc_dst = DenseLayer(in_dim, hidden_dim, activation, bias=True)
        self.fc_edge_update = DenseLayer(2 * hidden_dim, hidden_dim, activation, bias=True)

    def send_func(self, src_feat, dst_feat, edge_feat):
        return {"h": src_feat["h"] * edge_feat["angle"]}

    def recv_func(self, msg):
        return msg.reduce(msg["h"], pool_type="sum")

    def forward(self, graph, edge_feat, angle_feat):
        edge_h = self.feat_drop(edge_feat)
        angle_h = self.feat_drop(angle_feat)

        angle_h = self.G(angle_h)
        src_h = self.fc_src(edge_h)
        msg = graph.send(self.send_func,
                    src_feat={"h": src_h},
                    edge_feat={"angle": angle_h})
        rst = graph.recv(reduce_func=self.recv_func, msg=msg)

        dst_h = self.fc_dst(edge_h)
        m = paddle.concat([dst_h, rst], axis=-1)
        edge_h = self.fc_edge_update(m)
        return edge_h


class Dist2DConv(nn.Layer):
    def __init__(self, edge_in_dim, node_in_dim, hidden_dim, activation=F.relu, dropout=0.0):
        super(Dist2DConv, self).__init__()
        self.feat_drop = nn.Dropout(dropout)
        # parameters for message construction
        self.G = nn.Linear(hidden_dim, hidden_dim, bias_attr=False)
        self.fc_src = DenseLayer(edge_in_dim, hidden_dim, activation, bias=True)
        self.fc_dst = DenseLayer(node_in_dim, hidden_dim, activation, bias=True)
        self.fc_node_update = DenseLayer(2 * hidden_dim, hidden_dim, activation, bias=True)

    def send_func(self, src_feat, dst_feat, edge_feat):
        return {"h": src_feat["h"] * edge_feat["dist"]}

    def recv_func(self, msg):
        return msg.reduce(msg["h"], pool_type="sum")

    def forward(self, graph, edge_feat, node_feat, dist_feat):
        node_h = self.feat_drop(node_feat)
        edge_h = self.feat_drop(edge_feat)
        dist_h = self.feat_drop(dist_feat)

        dist_h = self.G(dist_h)
        src_h = self.fc_src(edge_h)
        msg = graph.send(self.send_func,
                    src_feat={"h": src_h},
                    edge_feat={"dist": dist_h})
        rst = graph.recv(reduce_func=self.recv_func, msg=msg)

        dst_h = self.fc_dst(node_h)
        m = paddle.concat([dst_h, rst], axis=-1)
        node_h = self.fc_node_update(m)
        return node_h


class Angle3DConv(nn.Layer):
    def __init__(self, in_dim, hidden_dim, num_angle, activation=F.relu, dropout=0.0):
        super(Angle3DConv, self).__init__()
        self.num_angle = num_angle
        self.hidden_dim = hidden_dim
        self.conv_layer = nn.LayerList()
        for i in range(num_angle):
            conv = Angle2DConv(in_dim, hidden_dim, activation, dropout)
            self.conv_layer.append(conv)
    
    def forward(self, graph_list, edge_feat, angle_feat_list):
        edge_h_list = []
        for k in range(self.num_angle):
            edge_h = self.conv_layer[k](graph_list[k], edge_feat, angle_feat_list[k])
            edge_h_list.append(edge_h)
        # edge_h = paddle.concat(edge_h_list, axis=-1)
        edge_h = paddle.stack(edge_h_list, axis=1)
        feat_max = paddle.max(edge_h, axis=1)
        feat_max = paddle.reshape(feat_max, [-1, 1, self.hidden_dim])
        edge_h = paddle.reshape(edge_h * feat_max, [-1, self.num_angle * self.hidden_dim])
        return edge_h


class Dist3DConv(nn.Layer):
    def __init__(self, edge_in_dim, node_in_dim, hidden_dim, num_dist, activation=F.relu, dropout=0.0):
        super(Dist3DConv, self).__init__()
        self.num_dist = num_dist
        self.conv_layer = nn.LayerList()
        for i in range(num_dist):
            conv = Dist2DConv(edge_in_dim, node_in_dim, hidden_dim, activation, dropout)
            self.conv_layer.append(conv)

    def get_dist_subgraphs(self, graph, dist_inds):
        subg_edge_list = []
        if self.num_dist == 1:
            subg_eids = paddle.greater_equal(dist_inds, paddle.to_tensor(0.)).nonzero().squeeze()
            subg_edge_list.append(subg_eids)
        elif self.num_dist == 2:
            subg_eids = paddle.equal(dist_inds, paddle.to_tensor(0.)).nonzero().squeeze()
            subg_edge_list.append(subg_eids)
            subg_eids = paddle.greater_equal(dist_inds, paddle.to_tensor(1.)).nonzero().squeeze()
            subg_edge_list.append(subg_eids)
        else:
            for k in range(self.num_dist):
                subg_edge_list.append(paddle.equal(dist_inds,  paddle.to_tensor(float(k))).nonzero().squeeze())

    def forward(self, graph_list, edge_feat, node_feat, dist_feat_list):
        edge_h_list = []
        for k in range(self.num_dist):
            edge_h = self.conv_layer[k](graph_list[k], edge_feat, node_feat, dist_feat_list[k])
            edge_h_list.append(edge_h)
        edge_h = paddle.concat(edge_h_list, axis=-1)
        return edge_h


class AttentivePooling(nn.Layer):
    def __init__(self, in_dim, dropout):
        super(AttentivePooling, self).__init__()
        self.compute_logits = nn.Sequential(
            nn.Linear(2 * in_dim, 1),
            nn.LeakyReLU()
        )
        self.project_nodes = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim)
        )
        self.pool = pgl.nn.GraphPool(pool_type='sum')
        self.gru = nn.GRUCell(in_dim, in_dim)
    
    def broadcast_graph_feat(self, graph, feat):
        nids = graph._graph_node_index
        nids_ = paddle.concat([nids[1:], nids[-1:]])
        batch_num_nodes = (nids_-nids)[:-1]
        h_list = []
        for i, k in enumerate(batch_num_nodes):
            h_list += [feat[i].tile([k,1])]
        return paddle.concat(h_list)
    
    def forward(self, graph, node_feat, graph_feat):
        graph_feat_broad = self.broadcast_graph_feat(graph, F.relu(graph_feat))
        graph_node_feat = paddle.concat([graph_feat_broad, node_feat], axis=1)
        graph_node_feat = self.compute_logits(graph_node_feat)
        node_a = pgl.math.segment_softmax(graph_node_feat, graph.graph_node_id)
        node_h = self.project_nodes(node_feat)
        context = self.pool(graph, node_h * node_a) ## NOTE
        graph_h, _ = self.gru(context, graph_feat) ## NOTE
        return graph_h


class DistRBF(nn.Layer):
    def __init__(self, K, cut_r, requires_grad=False):
        super(DistRBF, self).__init__()
        self.K = K
        self.cut_r = cut_r
        # self.mu = self.create_parameter(paddle.linspace(math.exp(-cut_r), 1., K).unsqueeze(0))
        # self.beta = self.create_parameter(paddle.full((1, K), math.pow((2 / K) * (1 - math.exp(-cut_r)), -2)))
        self.mu = paddle.linspace(math.exp(-cut_r), 1., K).unsqueeze(0)
        self.beta = paddle.full((1, K), math.pow((2 / K) * (1 - math.exp(-cut_r)), -2))
    
    def forward(self, r):
        batch_size = r.size
        K = self.K
        ratio_r = r / self.cut_r
        phi = 1 - 6 * ratio_r.pow(5) + 15 * ratio_r.pow(4) - 10 * ratio_r.pow(3)
        phi = paddle.expand(phi, shape=[batch_size, K])
        local_r = paddle.expand(r, shape=[batch_size, K])
        g = phi * paddle.exp(-self.beta.expand([batch_size, K]) * (paddle.exp(-local_r) - self.mu.expand([batch_size, K]))**2)
        return g


class AngleRBF(nn.Layer):
    def __init__(self, K, requires_grad=False):
        super(AngleRBF, self).__init__()
        self.K = K
        self.mu = paddle.linspace(0., math.pi, K).unsqueeze(0)
        self.beta = paddle.full((1, K), math.pow((2 / K) * math.pi, -2))
    
    def forward(self, r):
        batch_size = r.size
        K = self.K
        local_r = paddle.expand(r, shape=[batch_size, K])
        g = paddle.exp(-self.beta.expand([batch_size, K]) * (local_r - self.mu.expand([batch_size, K]))**2)
        return g


class OutputLayer(nn.Layer):
    def __init__(self, in_dim, hidden_dims, num_pool, dropout_pool):
        super(OutputLayer, self).__init__()
        self.pool = pgl.nn.GraphPool(pool_type='sum')
        self.attn_pool_layers = nn.LayerList()
        self.mlp = nn.LayerList()
        for _ in range(num_pool):
            self.attn_pool_layers.append(AttentivePooling(in_dim, dropout=dropout_pool))
  
    def forward(self, graph, node_feat):
        graph_feat = self.pool(graph, node_feat)
        for attn_pool in self.attn_pool_layers:
            graph_feat = attn_pool(graph, node_feat, graph_feat)