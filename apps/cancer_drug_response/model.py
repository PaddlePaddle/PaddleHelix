import paddle
from paddle import Model
from paddle.nn import Linear, Conv2D, Conv1D, MaxPool1D, MaxPool2D, AvgPool1D
from paddle.nn import Dropout, Flatten, BatchNorm1D, BatchNorm
import paddle.nn.functional as F

import pgl


class CDRModel(paddle.nn.Layer):
    """
    CDR model: a hybrid graph convolutional network consisting of a uniform graph
    convolutional network and multiple sub-networks.
    """

    def __init__(self, args):
        super(CDRModel, self).__init__()
        self.args = args
        self.use_mut = args.use_mut
        self.use_gexp = args.use_gexp
        self.use_methy = args.use_methy
        self.units_list = args.units_list  # [256, 256, 256, 100]
        self.gnn_type = args.gnn_type  # 'gcn'
        self.act = args.act
        self.layer_num = args.layer_num  # 4
        self.pool_type = args.pool_type  # 'max'

        self.gnn_layers = paddle.nn.LayerList()
        self.bns = paddle.nn.LayerList()
        self.dropout = paddle.nn.LayerList()
        if self.gnn_type == 'gcn':
            for layer_id in range(self.layer_num):
                self.gnn_layers.append(pgl.nn.GCNConv(
                    self._get_in_size(layer_id),
                    self.units_list[layer_id],
                    activation=self.act))
                bn = BatchNorm1D(self.units_list[layer_id], data_format='NC')
                self.bns.append(bn)
                dp = Dropout(0.2)
                self.dropout.append(dp)
            self.graph_pooling = pgl.nn.GraphPool(self.pool_type)
            self.linear = Linear(self.units_list[self.layer_num - 1] + 300, 300)

        elif self.gnn_type == 'gin':
            for layer_id in range(self.layer_num):
                self.gnn_layers.append(pgl.nn.GINConv(
                    self._get_in_size(layer_id),
                    self.units_list[layer_id],
                    activation=self.act))
                dp = Dropout(0.2)
                self.dropout.append(dp)
            self.graph_pooling = pgl.nn.GraphPool(self.pool_type)
            self.linear = Linear(self.units_list[self.layer_num - 1] + 300, 300)

        elif self.gnn_type == 'graphsage':
            for layer_id in range(self.layer_num):
                self.gnn_layers.append(pgl.nn.GraphSageConv(
                    self._get_in_size(layer_id),
                    self.units_list[layer_id]))
                dp = Dropout(0.2)
                self.dropout.append(dp)
            self.graph_pooling = pgl.nn.GraphPool(self.pool_type)
            self.linear = Linear(self.units_list[self.layer_num - 1] + 300, 300)

        self.MP2 = MaxPool2D(kernel_size=(1, 5), data_format='NHWC')
        self.MP3 = MaxPool2D(kernel_size=(1, 10), data_format='NHWC')
        self.MP4 = MaxPool2D(kernel_size=(1, 2), data_format='NHWC')
        self.MP5 = MaxPool2D(kernel_size=(1, 3), data_format='NHWC')
        self.MP6 = MaxPool2D(kernel_size=(1, 3), data_format='NHWC')

        self.Conv1 = Conv2D(in_channels=1,
                            out_channels=50,
                            kernel_size=(1, 700),
                            stride=(1, 5),
                            padding='valid',
                            data_format='NHWC'
                            )
        self.Conv2 = Conv2D(in_channels=50,
                            out_channels=30,
                            kernel_size=(1, 5),
                            stride=(1, 2),
                            padding='valid',
                            data_format='NHWC'
                            )
        self.Conv3 = Conv2D(in_channels=1,
                            out_channels=30,
                            kernel_size=(1, 150),
                            stride=(1, 1),
                            padding='VALID',
                            data_format='NHWC'
                            )
        self.Conv4 = Conv2D(in_channels=30,
                            out_channels=10,
                            kernel_size=(1, 5),
                            stride=(1, 1),
                            padding='VALID',
                            data_format='NHWC'
                            )
        self.Conv5 = Conv2D(in_channels=10,
                            out_channels=5,
                            kernel_size=(1, 5),
                            stride=(1, 1),
                            padding='VALID',
                            data_format='NHWC'
                            )

        self.fc1 = Linear(2010, 100)
        self.fc2 = Linear(697, 256)
        self.fc3 = Linear(256, 100)
        self.fc4 = Linear(808, 256)
        self.fc5 = Linear(256, 100)
        self.fc6 = Linear(30, 1)

        self.tanhs = paddle.nn.LayerList([paddle.nn.Tanh() for _ in range(8)])
        self.relus = paddle.nn.LayerList([paddle.nn.ReLU() for _ in range(8)])
        self.dropout1 = paddle.nn.LayerList([Dropout(0.1) for _ in range(8)])
        self.dropout2 = Dropout(0.2)

        self.flat = paddle.nn.LayerList([Flatten() for _ in range(5)])

    def _get_in_size(self, layer_id, gat_heads=None):
        in_size = 75
        gat_heads = 1 if gat_heads is None else gat_heads
        if layer_id > 0:
            in_size = self.units_list[layer_id - 1] * gat_heads
        return in_size

    def forward(self, inputs):
        graph, feat, mutation_input, gexpr_input, methy_input = inputs[0], inputs[0].node_feat['nfeat'], inputs[1], \
                                                                inputs[2], inputs[3]

        # drug feature
        feat_list = [feat]
        for i in range(self.layer_num):
            h = self.gnn_layers[i](graph, feat_list[i])
            h = self.dropout[i](h)
            feat_list.append(h)
        x_drug = self.graph_pooling(graph, h)

        # mutation feature
        x_mut = self.Conv1(mutation_input)  # 6795
        x_mut = self.tanhs[0](x_mut)
        x_mut = self.MP2(x_mut)
        x_mut = self.Conv2(x_mut)  # 678
        x_mut = self.relus[0](x_mut)
        x_mut = self.MP3(x_mut)
        x_mut = self.flat[1](x_mut)  # 2010
        x_mut = self.fc1(x_mut)
        x_mut = self.relus[1](x_mut)
        x_mut = self.dropout1[3](x_mut)

        # gene expression feature
        x_gexpr = self.fc2(gexpr_input)
        x_gexpr = self.tanhs[1](x_gexpr)
        x_gexpr = self.dropout1[4](x_gexpr)
        x_gexpr = self.fc3(x_gexpr)
        x_gexpr = self.relus[2](x_gexpr)

        # methylation feature
        x_methy = self.fc4(methy_input)
        x_methy = self.tanhs[2](x_methy)
        x_methy = self.dropout1[5](x_methy)
        x_methy = self.fc5(x_methy)
        x_methy = self.relus[3](x_methy)

        x = x_drug
        if self.use_mut:
            x = paddle.concat([x, x_mut], axis=-1)
        if self.use_gexp:
            x = paddle.concat([x, x_gexpr], axis=-1)
        if self.use_methy:
            x = paddle.concat([x, x_methy], axis=-1)

        x = self.linear(x)
        x = self.tanhs[3](x)
        x = self.dropout1[6](x)
        x = paddle.unsqueeze(x, -1)
        x = paddle.unsqueeze(x, 1)
        x = self.Conv3(x)
        x = self.relus[4](x)
        x = self.MP4(x)
        x = self.Conv4(x)
        x = self.relus[5](x)
        x = self.MP5(x)
        x = self.Conv5(x)
        x = self.relus[6](x)
        x = self.MP6(x)
        x = self.dropout1[7](x)
        x = self.flat[2](x)
        x = self.dropout2(x)
        output = self.fc6(x)

        return output
