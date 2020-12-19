"""
This file implement the graph dataloader.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import sys
import time
import argparse
import numpy as np
import collections
import copy

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as fl
import pgl
from pgl.utils import mp_reader
from pgl.utils.logger import log


def batch_iter(data, batch_size, fid, num_workers):
    """node_batch_iter
    """
    size = len(data)
    perm = np.arange(size)
    np.random.shuffle(perm)
    start = 0
    cc = 0
    while start < size:
        index = perm[start:start + batch_size]
        start += batch_size
        cc += 1
        if cc % num_workers != fid:
            continue
        yield data[index]


def scan_batch_iter(data, batch_size, fid, num_workers):
    """scan_batch_iter
    """
    batch = []
    cc = 0
    for line_example in data.scan():
        cc += 1
        if cc % num_workers != fid:
            continue
        batch.append(line_example)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if len(batch) > 0:
        yield batch


class GraphDataloader(object):
    """Graph Dataloader
    """

    def __init__(
            self,
            dataset,
            batch_size,
            seed=0,
            num_workers=1,
            buf_size=1000,
            shuffle=True, ):

        self.shuffle = shuffle
        self.seed = seed
        self.num_workers = num_workers
        self.buf_size = buf_size
        self.batch_size = batch_size
        self.dataset = dataset

    def batch_combine_edges(self, n2e_feed_list):
        """new n2e_feed
        return joint edges and joint edges distance
        """
        srcs_list = []
        dsts_list = []
        edges_dist_list = []
        nidx_list = []
        eidx_list = []
        node_lod = [0]
        edge_lod = [0]
        num_cumul = 0
        for edges, edges_dist, num_n, num_e in n2e_feed_list:
            edges = copy.deepcopy(edges)
            edges += num_cumul
            srcs_list.append(edges[:, 0])
            dsts_list.append(edges[:, 1])
            edges_dist_list.append(edges_dist)
            node_lod += [node_lod[-1] + num_n]
            edge_lod += [edge_lod[-1] + num_e]
            nidx_list += range(num_cumul, num_cumul + num_n)
            eidx_list += range(num_cumul + num_n, num_cumul + num_n + num_e)
            num_cumul += num_n + num_e
        
        join_srcs = np.hstack(srcs_list).astype('int32') # int
        join_dsts = np.hstack(dsts_list).astype('int32') # int
        join_edges_dist = np.vstack(edges_dist_list)
        join_nidxs = np.array(nidx_list).astype('int32') # int
        join_eidxs = np.array(eidx_list).astype('int32') # int
        join_node_lod = np.array(node_lod).astype('int32') # int
        join_edge_lod = np.array(edge_lod).astype('int32') # int
        return join_srcs, join_dsts, join_edges_dist, join_nidxs, join_eidxs, join_node_lod, join_edge_lod


    def batch_fn(self, batch_examples):
        """ batch_fn batch producer"""
        e2n_graphs = [b[0][0] for b in batch_examples]
        e2e_graphs = [b[0][1] for b in batch_examples]
        pk_values = [b[1] for b in batch_examples]
        n2e_feed_list = [b[2] for b in batch_examples]
        join_e2n_graph = pgl.graph.MultiGraph(e2n_graphs)
        join_e2e_graph = pgl.graph.MultiGraph(e2e_graphs)
        join_srcs, join_dsts, join_edges_dist, join_nidxs, join_eidxs, join_nlod, join_elod = self.batch_combine_edges(n2e_feed_list)
        pk_values = np.array(pk_values, dtype="float32").reshape(-1, 1)
        return join_e2n_graph, join_e2e_graph, join_edges_dist, join_nidxs, join_eidxs, join_nlod, join_elod, join_srcs, join_dsts, pk_values
        #  feed_dict = self.graph_wrapper.to_feed(join_graph)
        #  raise NotImplementedError("No defined Batch Fn")

    def batch_iter(self, fid):
        """batch_iter"""
        if self.shuffle:
            for batch in batch_iter(self, self.batch_size, fid,
                                    self.num_workers):
                yield batch
        else:
            for batch in scan_batch_iter(self, self.batch_size, fid,
                                         self.num_workers):
                yield batch

    def __len__(self):
        """__len__"""
        return len(self.dataset)

    def __getitem__(self, idx):
        """__getitem__"""
        if isinstance(idx, collections.Iterable):
            return [self[bidx] for bidx in idx]
        else:
            return self.dataset[idx]

    def __iter__(self):
        """__iter__"""

        def worker(filter_id):
            def func_run():
                for batch_examples in self.batch_iter(filter_id):
                    batch_dict = self.batch_fn(batch_examples)
                    yield batch_dict

            return func_run

        if self.num_workers == 1:
            r = paddle.reader.buffered(worker(0), self.buf_size)
        else:
            worker_pool = [worker(wid) for wid in range(self.num_workers)]
            worker = mp_reader.multiprocess_reader(
                worker_pool, use_pipe=True, queue_size=1000)
            r = paddle.reader.buffered(worker, self.buf_size)

        for batch in r():
            yield batch

    def scan(self):
        """scan"""
        for example in self.dataset:
            yield example
