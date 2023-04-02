import paddle
from paddle import nn
import paddle.nn.functional as F
from utils import _norm_no_nan
import pgl
import pgl.math as math


class GVP(nn.Layer):
    '''
    Paddle version of the GVP proposed by https://github.com/drorlab/gvp-pytorch
    '''

    def __init__(self, in_dims, out_dims, h_dim=None, activations=(F.relu, F.sigmoid)):
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = Linear(self.vi, self.h_dim, bias_attr=False)
            self.ws = Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = Linear(self.h_dim, self.vo, bias_attr=False)
        else:
            self.ws = Linear(self.si, self.so)
        
        self.scalar_act, self.vector_act = activations
        
    def forward(self, x):
        if self.vi:
            s, v = x
            v = paddle.transpose(v, [0, 2, 1])
            vh = self.wh(v)
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(paddle.concat([s, vn], -1))
            if self.vo:
                v = self.wv(vh)
                v = paddle.transpose(v, [0, 2, 1])
                if self.vector_act:
                    v = v * self.vector_act(_norm_no_nan(v, axis=-1, keepdim=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = paddle.zeros(paddle.shape(s)[0], self.vo, 3)
        if self.scalar_act:
            s = self.scalar_act(s)
        
        return (s, v) if self.vo else s


class RR_VPConv(nn.Layer):

    def __init__(self, in_dims, out_dims, activations=(F.relu, F.sigmoid)):
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        
        self.vp_layer = GVP((in_dims[0], in_dims[1]),
                            out_dims, h_dim=out_dims[1], activations=activations)

    def forward(self, graph, x, local_sys):

        src_feat = {'s_f': x[0], 'v_f':x[1]}
        dst_feat = {'local_sys': local_sys}

        msg = graph.send(self.send_func, src_feat=src_feat, dst_feat=dst_feat)

        n_s_f = graph.recv(self.s_recv_func, msg)
        n_v_f = graph.recv(self.v_recv_func, msg)
        n_v_f = paddle.reshape(n_v_f, [n_v_f.shape[0], -1, 3])
        
        return (n_s_f, n_v_f)

    def send_func(self, src_feat, dst_feat, edge_feat):
        local_sys = dst_feat['local_sys']
        v_f_local = src_feat['v_f'] @ local_sys
        s_f = src_feat['s_f']

        x = self.vp_layer((s_f, v_f_local))

        s, v = x[0], x[1] @ paddle.transpose(local_sys, [0, 2, 1])

        v = paddle.reshape(v, [v.shape[0], -1]) ## Only 2D tensors are accepted

        return {'s': s, 'v': v}

    def s_recv_func(self, msg):
        return msg.reduce_sum(msg['s'])

    def v_recv_func(self, msg):
        return msg.reduce_sum(msg['v'])
    

def Linear(in_features, out_features, weight_attr=None, bias_attr=None, name=None):

    k = (1 / in_features) ** 0.5
    if weight_attr is None:
        weight_attr = paddle.ParamAttr(
            # name="weight",
            initializer=paddle.nn.initializer.Uniform(low=-k, high=k))
    
    if bias_attr:
        bias_attr = paddle.ParamAttr(
            # name="bias",
            initializer=paddle.nn.initializer.Uniform(low=-k, high=k))
    
    return nn.Linear(in_features, out_features, weight_attr=weight_attr, bias_attr=bias_attr, name=name)
