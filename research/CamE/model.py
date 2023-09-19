from paddle.nn.initializer import XavierNormal
import paddle.nn as nn
import paddle
import paddle.nn.functional as F


class exchange_bilinear(nn.Layer):
    def __init__(self,params):
        super(exchange_bilinear,self).__init__()
        self.p = params
        self.output_dim = self.p.fusion_dim
        self.Hiddendim = self.p.fusion_dim

        self.weight_P = paddle.create_parameter((self.Hiddendim, self.output_dim),dtype='float32')
        self.weight_U = paddle.create_parameter((self.p.fusion_dim, self.Hiddendim),dtype='float32')
        self.weight_V = paddle.create_parameter((self.p.fusion_dim, self.Hiddendim),dtype='float32')
        self.bias = paddle.create_parameter((1, self.output_dim),dtype='float32')
        self.a = nn.Sigmoid()
        XavierNormal(self.weight_P)
        XavierNormal(self.weight_U)
        XavierNormal(self.weight_V)
        XavierNormal(self.bias)

    def exchange_channel(self,x,y):
        x_change = paddle.where(F.layer_norm(x,x.shape[1:]) >= self.p.threshold, x, y)
        y_change = paddle.where(F.layer_norm(y,y.shape[1:]) >= self.p.threshold, y, x)
        return x_change,y_change

    def forward(self,x,y):
        
        x,y = self.exchange_channel(x,y)
        output = self.a(paddle.matmul(x,self.weight_U)) * self.a(paddle.matmul(y,self.weight_V))
        output =  paddle.matmul(output,self.weight_P) + self.bias
        return output




class Fusion(nn.Layer):
    def __init__(self,params,device):
        super(Fusion, self).__init__()
        self.p = params
        self.num_heads = self.p.num_head
        self.output_dim = self.p.fusion_dim
        self.Hiddendim = self.p.fusion_dim

        self.coatt1 = multihead_tca(self.p,self.p.fusion_dim, self.p.fusion_dim, self.num_heads)
        self.coatt2 = multihead_tca(self.p,self.p.fusion_dim, self.p.fusion_dim, self.num_heads)
        self.coatt3 = multihead_tca(self.p,self.p.fusion_dim, self.p.fusion_dim, self.num_heads)


        self.EX = exchange_bilinear(self.p)


        self.weight_P = paddle.create_parameter((self.Hiddendim, self.output_dim),dtype='float32')
        self.weight_1 = paddle.create_parameter((self.p.fusion_dim, self.Hiddendim),dtype='float32')
        self.weight_2 = paddle.create_parameter((self.p.fusion_dim, self.Hiddendim),dtype='float32')
        self.weight_3 = paddle.create_parameter((self.p.fusion_dim, self.Hiddendim),dtype='float32')
        self.bias = paddle.create_parameter((1, self.output_dim),dtype='float32')

        self.a = nn.Sigmoid()

        XavierNormal(self.weight_P)
        XavierNormal(self.weight_1)
        XavierNormal(self.weight_2)
        XavierNormal(self.weight_3)
        XavierNormal(self.bias)

    def forward(self,x1,x2,x3):
        x12, x21 = self.coatt1(x1, x2)
        x13, x31 = self.coatt2(x1, x3)
        x23, x32 = self.coatt3(x2, x3) 

        out1 = self.EX(x12, x21)
        out2 = self.EX(x13, x31)
        out3 = self.EX(x23, x32)

        output = self.a(paddle.matmul(out1,self.weight_1)) * self.a(paddle.matmul(out2,self.weight_2)) * self.a(paddle.matmul(out3,self.weight_3))
        output =  paddle.matmul(output,self.weight_P) + self.bias
        return output



class multihead_tca(nn.Layer):
    def __init__(self,params,dim_h,dim_r,num_heads):
        super(multihead_tca,self).__init__()
        self.p = params
        self.num_heads = num_heads

        self.mh = nn.LayerList([TCA_operator(params,dim_h,dim_r) for i in range(num_heads)])

        self.temperture = paddle.create_parameter((1, 1),dtype='float32')
        XavierNormal(self.temperture)

        self.linear_h = nn.Linear(num_heads*dim_h,dim_h)
        self.linear_r = nn.Linear(num_heads*dim_r,dim_r)
        self.a = nn.LeakyReLU(0.1)
    def forward(self,h,r):
        heads_h = []
        heads_r = []
        for i in range(len(self.mh)):
            h_tca,r_tca = self.mh[i](h,r,self.temperture*self.p.interval*(i+1))
            heads_h.append(h_tca)
            heads_r.append(r_tca)
        
        heads_h = paddle.concat(heads_h,axis=1)
        heads_r = paddle.concat(heads_r,axis=1)

        heads_h = self.a(self.linear_h(heads_h))
        heads_r = self.a(self.linear_r(heads_r))
        return heads_h,heads_r


class TCA_operator(nn.Layer):
    def __init__(self,params,dim_h,dim_r):
        super(TCA_operator, self).__init__()
        self.p = params
        self.dim_h = dim_h
        self.dim_r = dim_r
        self.W_co_q = nn.Linear(dim_h, dim_h)
        self.W_co_d = nn.Linear(dim_r, dim_r)
        self.W_se_q = nn.Linear(dim_h, dim_h)
        self.W_se_d = nn.Linear(dim_r, dim_r)
        self.sig = nn.Sigmoid()

    def forward(self,h,r,temperture):

        Q = self.sig(self.W_co_q(h))
        D = self.sig(self.W_co_d(r))
        Q_se = self.sig(self.W_se_q(h))
        D_se = self.sig(self.W_se_d(r))

        #co affinity matrix
        M_co = paddle.matmul(Q.unsqueeze(2), D.unsqueeze(1)) / temperture #math.sqrt(self.dim_h)  # 3B x d1 x d2
        h_co = paddle.matmul(h.unsqueeze(1), F.softmax(M_co, axis=1)).squeeze(1) 
        r_co = paddle.matmul(F.softmax(M_co, axis=2), r.unsqueeze(2)).squeeze(2)

        #self affinity matrix
        M_se_q = paddle.matmul(Q_se.unsqueeze(2), D.unsqueeze(1)) / temperture
        h_se = paddle.matmul(h.unsqueeze(1), F.softmax(M_se_q, axis=1)).squeeze(1)

        M_se_d = paddle.matmul(Q.unsqueeze(2), D_se.unsqueeze(1)) / temperture
        r_se = paddle.matmul(h.unsqueeze(1), F.softmax(M_se_d, axis=1)).squeeze(1)

        h_tca = h_co+h_se
        r_tca = r_co+r_se

        return h_tca,r_tca


class Conv_Fc(nn.Layer):
    def __init__(self,params,channels,linear_dim):
        super(Conv_Fc, self).__init__()
        self.in_channels = channels
        self.p = params
        self.linear_dim = linear_dim
        self.input_drop = paddle.nn.Dropout(0.2)
        self.hidden_drop = paddle.nn.Dropout(0.5)
        # self.feature_map_drop = paddle.nn.functional.dropout2d(0.5)

        self.bn0 = paddle.nn.BatchNorm2D(self.in_channels)
        self.bn1 = paddle.nn.BatchNorm2D(self.p.num_filt)
        self.bn2 = paddle.nn.BatchNorm1D(self.p.embed_dim)
        paddding = self.p.ker_sz//2
        self.conv1 = nn.Conv2D(self.in_channels, self.p.num_filt, (self.p.ker_sz, self.p.ker_sz), 1, paddding)
        self.fc = nn.Linear(self.linear_dim, self.p.embed_dim)

        self.a = nn.LeakyReLU(0.1)

    def forward(self,x):
        x = self.bn0(x)
        x = self.input_drop(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.a(x)
        x = paddle.nn.functional.dropout2d(x,p=0.5)
        x = x.reshape((-1, self.linear_dim))
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = self.a(x)

        return x


class coatt(nn.Layer):
    def __init__(self, params,ent2id,device,ent_mm_emb , smiles_emb):
        super(coatt, self).__init__()
        self.p = params
        self.ent_embed = paddle.nn.Embedding(self.p.num_ent, self.p.embed_dim, padding_idx=None)
        self.rel_embed = paddle.nn.Embedding(self.p.num_rel * 2, self.p.rel_dim, padding_idx=None)
        self.bceloss = paddle.nn.BCELoss()
        self.bias = paddle.create_parameter((self.p.num_ent,),dtype='float32')

        self.device = device
        self.num_heads = self.p.num_head

        self.ent_mm_emb, self.smiles_emb = ent_mm_emb, smiles_emb

        self.ent_mm_mapping = nn.LayerList([nn.Linear(768,self.p.fusion_dim),
                                             nn.Linear(300,self.p.fusion_dim),
                                             nn.Linear(self.p.embed_dim,self.p.fusion_dim)])
        self.rel_mapping = nn.Linear(self.p.rel_dim, self.p.fusion_dim)

        self.coatt_h_r = multihead_tca(self.p,self.p.embed_dim, self.p.rel_dim, self.num_heads)
        self.coatt_text_r = multihead_tca(self.p,self.p.fusion_dim, self.p.fusion_dim, self.num_heads)
        self.coatt_smile_r = multihead_tca(self.p,self.p.fusion_dim, self.p.fusion_dim, self.num_heads)

        self.fc_text_r = nn.Linear(2*self.p.fusion_dim, self.p.fusion_dim)
        self.fc_smile_r = nn.Linear(2*self.p.fusion_dim, self.p.fusion_dim)

        self.fusion = Fusion(params,device)
        self.in_channels_1 = 3
        self.in_channels_2 = 2

        self.conv_fc1 = Conv_Fc(self.p,self.in_channels_1,self.p.num_filt * self.p.fusion_dim)
        self.conv_fc4 = Conv_Fc(self.p, self.in_channels_2,self.p.num_filt * (self.p.embed_dim+self.p.rel_dim))

        self.a = nn.LeakyReLU(0.1)
        self.init_network()
        self.sig = nn.Sigmoid()
    def init_network(self):
        for m in self.sublayers():
            if isinstance(m, nn.Linear):
                XavierNormal(m.weight)
            elif isinstance(m, nn.Conv2D):
                XavierNormal(m.weight)
    def loss(self, pred, true_label=None, sub_samp=None):
        loss = self.bceloss(pred, true_label)
        return loss
    def forward(self, sub, rel, neg_ents,strategy='one_to_x'):
        sub_emb = self.ent_embed(sub)
        rel_emb = self.rel_embed(rel)

        text = self.ent_mm_emb[sub]
        smile = self.smiles_emb[sub]
        

        text = self.ent_mm_mapping[0](text)
        smile = self.ent_mm_mapping[1](smile)
        structure = self.ent_mm_mapping[2](sub_emb)

        fusion = self.fusion(text,smile,structure)

        mm_rel = self.rel_mapping(rel_emb)
        h_att, r_att = self.coatt_h_r(sub_emb,rel_emb)

        text_rel,rel_text = self.coatt_text_r(text,mm_rel)
        smile_rel,rel_smile = self.coatt_smile_r(smile, mm_rel)
        t_r = self.fc_text_r(paddle.concat([text_rel, rel_text], axis=1))
        t_r = self.a(t_r)
        s_r = self.fc_smile_r(paddle.concat([smile_rel, rel_smile], axis=1))
        s_r = self.a(s_r)

        x_1 = paddle.stack([fusion,t_r,s_r] ,axis=1)
        
        x_1 = paddle.reshape(x_1,[sub.shape[0], self.in_channels_1, 20, self.p.fusion_dim//20])
        x_1 = self.conv_fc1(x_1)

        x_4 = paddle.stack([paddle.concat([sub_emb, rel_emb], axis=1),paddle.concat([h_att, r_att],axis=1)], axis=1)
        x_4 = paddle.reshape(x_4,[sub.shape[0], self.in_channels_2, 20, (self.p.embed_dim+self.p.rel_dim)//20])
        x_4 = self.conv_fc4(x_4)

        x = x_1 + x_4
        
        if strategy == 'one_to_n':
            x = paddle.matmul(x, self.ent_embed.weight.transpose([1,0]))
            x += self.bias.expand_as(x)
        else:
            x = paddle.mul((x).unsqueeze(1), self.ent_embed(neg_ents)).sum(axis=-1)
            x += self.bias[neg_ents]

        pred = nn.functional.sigmoid(x)
        return pred

