from pkgutil import ImpImporter
import pgl
import paddle
import math
import numpy as np

class DualDataLoader(object):
    def __init__(self, data_2d, data_3d, batch_size, shuffle=False):
        ''' Load data
        '''
        self.data_2d = data_2d
        self.data_3d = data_3d
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.iter = 0
        self.steps = math.ceil(len(data_2d)/batch_size)
        self.indices = np.arange(len(self.data_2d), dtype=np.int64)
        if self.shuffle:
            np.random.shuffle(self.indices)
        assert len(self.data_2d) == len(self.data_3d)

    def __len__(self):
        return self.steps
        
    def next_batch(self):
        ''' generate a sequential batch
        '''
        if self.iter < len(self.data_2d):
            indices = self.indices[self.iter:(self.iter + self.batch_size)]
            self.iter += self.batch_size
        else:
            if self.shuffle:
                np.random.shuffle(self.indices)
            indices = self.indices[:self.batch_size]
            self.iter = self.batch_size
        return self._generate_batch(indices)
    
    def _generate_batch(self, indices):
        data_2d_batch = [self.data_2d[i] for i in indices]
        data_3d_batch = [self.data_3d[i] for i in indices]

        a2a_gs, e2a_gs, e2e_gs, labels = map(list, zip(*data_2d_batch))
        a2a_g = pgl.Graph.batch(a2a_gs).tensor()
        e2a_g = pgl.BiGraph.batch(e2a_gs).tensor()
        e2e_g = pgl.Graph.batch(e2e_gs).tensor()
        graph_2d = a2a_g, e2a_g, e2e_g
        y_2d = paddle.to_tensor(np.array(labels), dtype='float32')

        a2a_gs, e2a_gs, e2e_gs, labels = map(list, zip(*data_3d_batch))
        a2a_g = pgl.Graph.batch(a2a_gs).tensor()
        e2a_g = [pgl.BiGraph.batch([g[i] for g in e2a_gs]).tensor() for i in range(len(e2a_gs[0]))]
        e2e_g = [pgl.Graph.batch([g[i] for g in e2e_gs]).tensor() for i in range(len(e2e_gs[0]))]
        graph_3d = a2a_g, e2a_g, e2e_g
        y_3d = paddle.to_tensor(np.array(labels), dtype='float32')

        assert (y_2d - y_3d).sum().tolist()[0] == 0
        return graph_2d, graph_3d, y_3d