
from .relation import MLP,ContextMLP, TaskAwareRelation

import paddle.nn as nn
import paddle
from pahelix.model_zoo.pretrain_gnns_model import PretrainGNNModel
import pgl.graph as G
import numpy as np

### embedding setting ###
num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class attention(nn.Layer):
    def __init__(self, dim):
        super(attention, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layers(x)
        x = self.softmax(paddle.transpose(x, [1, 0]))
        return x


### embedding class ###
class All_Embedding(nn.Layer):
    """
    Atom/Edge Encoder
    """
    def __init__(self, num_1, num_2, embed_dim):
        super(All_Embedding, self).__init__()

        self.embed_1 = nn.Embedding(num_1, embed_dim, weight_attr=nn.initializer.XavierUniform())
        self.embed_2 = nn.Embedding(num_2, embed_dim, weight_attr=nn.initializer.XavierUniform())

    def forward(self, features):

        out_embed = self.embed_1(features['feature'][:, 0]) + self.embed_2(features['feature'][:, 1])
        return out_embed

class ContextAwareRelationNet(nn.Layer):
    def __init__(self, args):
        super(ContextAwareRelationNet, self).__init__()
        self.mol_relation_type = args.rel_type
        self.rel_layer = args.rel_layer
        self.edge_type = args.rel_adj
        self.edge_dim = args.rel_edge
        self.edge_activation = args.rel_act

        config_encoder = {
            "atom_names": ["atomic_num", "chiral_tag"],
            "bond_names": ["bond_dir", "bond_type"],
    
            "residual": False,
            "dropout_rate": args.dropout,
            "gnn_type": "gin",
            "graph_norm": False,

            "embed_dim": args.emb_dim,
            "layer_num": args.enc_layer,
            "JK": args.JK,

            "readout": args.enc_pooling
        }
        self.mol_encoder = PretrainGNNModel(config_encoder)
        self.mol_encoder.atom_embedding = All_Embedding(num_atom_type, num_chirality_tag, args.emb_dim)
        for i in range(args.enc_layer):
            self.mol_encoder.bond_embedding_list[i] = All_Embedding(num_bond_type, num_bond_direction, args.emb_dim)
        if args.pretrained:
            model_file = args.pretrained_weight_path
            print('load pretrained model from', model_file)
            params = paddle.load(model_file)
            for i in params:
                params[i] = paddle.to_tensor(params[i])
            self.mol_encoder.load_dict(params)

        if self.mol_relation_type not in ['par', 'no_rel','no_rel_no_w']:
            self.encode_projection = MLP(inp_dim=args.emb_dim, hidden_dim=args.map_dim, num_layers=args.map_layer,
                                     batch_norm=args.batch_norm,dropout=args.map_dropout)
        else:
            self.encode_projection = ContextMLP(inp_dim=args.emb_dim, hidden_dim=args.map_dim, num_layers=args.map_layer,
                                     batch_norm=args.batch_norm,dropout=args.map_dropout,
                                     pre_fc=args.map_pre_fc,ctx_head=args.ctx_head)

        if self.mol_relation_type == 'par':
            inp_dim = args.map_dim
            self.adapt_relation = TaskAwareRelation(inp_dim=inp_dim, hidden_dim=args.rel_hidden_dim,
                                                    num_layers=args.rel_layer, edge_n_layer=args.rel_edge_layer,
                                                    top_k=args.rel_k, res_alpha=args.rel_res,
                                                    batch_norm=args.batch_norm,edge_dim=args.rel_edge, adj_type=args.rel_adj,
                                                    activation=args.rel_act, node_concat=args.rel_node_concat,dropout=args.rel_dropout,
                                                    pre_dropout=args.rel_dropout2)
        else:
            self.adapt_relation = MLP(inp_dim=args.map_dim, hidden_dim=2, num_layers=1)

    def to_one_hot(self,class_idx, num_classes=2):
        return paddle.eye(num_classes)[class_idx]

    def label2edge(self, label, mask_diag=True):
        # get size
        num_samples = label.shape[1]
        # reshape
        label_i = paddle.transpose(paddle.expand(label,[num_samples,label.shape[0],label.shape[1]]),[1,2,0])
        label_j = label_i.transpose((0, 2, 1))
        # compute edge
        edge = paddle.cast(paddle.equal(label_i, label_j),'float32')

        # expand
        edge = edge.unsqueeze(1)
        if self.edge_type == 'dist':
            edge = 1 - edge
        if self.edge_dim == 2:
            edge = paddle.concat([edge, 1 - edge], 1)

        if mask_diag:
            diag_mask = 1.0 - paddle.expand(paddle.eye(edge.shape[2]),[edge.shape[0],self.edge_dim,edge.shape[2],edge.shape[2]])
            edge=edge*diag_mask
        if self.edge_activation == 'softmax':
            edge = edge / edge.sum(-1).unsqueeze(-1)
        return edge

    def relation_forward(self, s_emb, q_emb, s_label=None, q_pred_adj=False,return_adj=False,return_emb=False):
        if self.mol_relation_type == 'par':
            if not return_emb:
                s_logits, q_logits, adj = self.adapt_relation(s_emb, q_emb,return_adj=return_adj,return_emb=return_emb)
            else:
                s_logits, q_logits, adj, s_rel_emb, q_rel_emb = self.adapt_relation(s_emb, q_emb,return_adj=return_adj,return_emb=return_emb)
            if q_pred_adj:
                q_sim = adj[-1][:, 0, -1, :-1]
                q_logits = q_sim @ self.to_one_hot(s_label)
        else:
            s_logits = self.adapt_relation(s_emb)
            q_logits = self.adapt_relation(q_emb)
            adj = None
        if not return_emb:
            return s_logits, q_logits, adj
        else:
            return s_logits, q_logits, adj, s_rel_emb, q_rel_emb

    def forward(self, s_data, q_data, s_label=None, q_pred_adj=False):
        s_data.tensor()
        q_data.tensor()
        s_node_emb, s_emb = self.mol_encoder(s_data)
        q_node_emb, q_emb = self.mol_encoder(q_data)
        if self.mol_relation_type!='par':
            s_emb_map = self.encode_projection(s_emb)
            q_emb_map = self.encode_projection(q_emb)
        else:
            s_emb_map,q_emb_map = self.encode_projection(s_emb,q_emb)

        s_logits, q_logits, adj = self.relation_forward(s_emb_map, q_emb_map, s_label, q_pred_adj=q_pred_adj)

        return s_logits, q_logits, adj, s_node_emb

    def forward_one_batch(self, data):
        graph_emb, node_emb = self.mol_encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        logits = self.adapt_relation(graph_emb)
        return logits, node_emb

    def forward_query_list(self, s_data, q_data_list, s_label=None, q_pred_adj=False):
        s_emb, _ = self.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, s_data.batch)
        q_emb_list = [self.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, q_data.batch)[0] for q_data in
                      q_data_list]

        if self.mol_relation_type!='par':
            s_emb_map = self.encode_projection(s_emb)
        else:
            s_emb_map = None

        q_logits_list, adj_list = [], []
        for q_emb in q_emb_list:
            if self.mol_relation_type!='par':
                q_emb_map = self.encode_projection(q_emb)
            else:
                s_emb_map,q_emb_map = self.encode_projection(s_emb,q_emb)
            s_logit, q_logit, adj = self.relation_forward(s_emb_map, q_emb_map, s_label, q_pred_adj=q_pred_adj)
            q_logits_list.append(q_logit.detach())
            if adj is not None:
                sim_adj = adj[-1][:,0].detach()
                q_adj = sim_adj[:,-1]
                adj_list.append(q_adj)

        q_logits = paddle.concat(q_logits_list, 0)
        adj_list = paddle.concat(adj_list, 0)
        return s_logit.detach(),q_logits, adj_list

    def forward_query_loader(self, s_data, q_loader, s_label=None, q_pred_adj=False):
        s_data.tensor()
        _, s_emb = self.mol_encoder(s_data)
        if self.mol_relation_type!='par':
            s_emb_map = self.encode_projection(s_emb)
        else:
            s_emb_map = None
        y_true_list=[]
        q_logits_list, adj_list = [], []
        for q_data in q_loader:
            y_true_list.append(paddle.to_tensor(np.stack([i.y[0] for i in q_data])))
            q_data = G.Graph.batch(q_data)
            q_data.tensor()
            _, q_emb = self.mol_encoder(q_data)
            if self.mol_relation_type!='par':
                q_emb_map = self.encode_projection(q_emb)
            else:
                s_emb_map,q_emb_map = self.encode_projection(s_emb,q_emb)
            s_logit, q_logit, adj = self.relation_forward(s_emb_map, q_emb_map, s_label, q_pred_adj=q_pred_adj)
            q_logits_list.append(q_logit)
            if adj is not None and self.mol_relation_type == 'par':
                sim_adj = adj[-1].detach()
                adj_list.append(sim_adj)

        q_logits = paddle.concat(q_logits_list, 0)
        y_true = paddle.concat(y_true_list, 0)
        sup_labels={'support':s_label,'query':y_true_list}
        return s_logit, q_logits, y_true,adj_list,sup_labels


    def forward_one_query_loader(self, data_loader,device='cpu'):
        y_true_list=[]
        graph_emb_list=[]
        for data in data_loader:
            data = data
            y_true_list.append(data.y)
            graph_emb,_ = self.mol_encoder(data.x, data.edge_index, data.edge_attr, data.batch)
            graph_emb_list.append(graph_emb)

        logits_list = [self.adapt_relation(graph_emb) for graph_emb in graph_emb_list]
        logits = paddle.concat(logits_list, 0)
        y_true = paddle.concat(y_true_list, 0)
        return logits,y_true