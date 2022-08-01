#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Communication group manager
"""

import numpy as np
from paddle import distributed as dist

def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(numerator, denominator)

class SingletonCommunicationGroup(object):
    """ A singleton communication group for bp, dap, ddp hybrid parallel. """
    def __init__(self):
        self.initialized = False

    def init_group(self, bp_degree=1, dap_degree=1, dap_comm_sync=True):
        """ init the hybrid parallel, it will auto calculate ddp_degree using bp_degree, dap_degree and world_size """
        assert self.initialized == False, "Communication group is already initialized!"
        # check valid config
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        inner_degree = bp_degree * dap_degree
        ensure_divisibility(world_size, bp_degree)
        ensure_divisibility(world_size, dap_degree)
        ensure_divisibility(world_size, inner_degree)

        self.dp_degree = world_size // inner_degree
        self.bp_degree = bp_degree
        self.dap_degree = dap_degree
        self.dap_comm_sync = dap_comm_sync

        if dist.parallel._global_parallel_env is not None:
            dist.init_parallel_env()

        arr = np.arange(0, world_size).reshape(self.dp_degree, self.dap_degree, self.bp_degree)

        # build bp group
        bp_arr = arr.transpose((0, 1, 2)).reshape(-1, self.bp_degree)
        for i in range(world_size // self.bp_degree):
            ranks = list(bp_arr[i])
            group = dist.new_group(ranks)
            print('> bp ranks:', ranks, 'bp group:', group)
            if rank in ranks:
                self.bp_group = group

        # build dap group
        dap_arr = arr.transpose((0, 2, 1)).reshape(-1, self.dap_degree)
        for i in range(world_size // self.dap_degree):
            ranks = list(dap_arr[i])
            group = dist.new_group(ranks)
            print('> dap ranks:', ranks, 'dap group:', group)
            if rank in ranks:
                self.dap_group = group

        # build dp group
        dp_arr = arr.transpose((1, 2, 0)).reshape(-1, self.dp_degree)
        for i in range(world_size // self.dp_degree):
            ranks = list(dp_arr[i])
            group = dist.new_group(ranks)
            print('> dp ranks:', ranks, 'dp group:', group)
            if rank in ranks:
                self.dp_group = group

        self.initialized = True
        if dist.get_rank() == 0:
            print('> initialize branch parallel with size {}'.format(self.bp_degree))
            print('> initialize dynamic axial parallel with size {}'.format(self.dap_degree))
            print('> initialize data parallel with size {}'.format(self.dp_degree))

    def dap_is_comm_sync(self):
        """ get dap whether use sync or async communication """
        return self.dap_comm_sync

    def bp_is_initialized(self):
        """ get bp commnication group whether is initialized """
        return self.initialized

    def dap_is_initialized(self):
        """ get dap commnication group whether is initialized """
        return self.initialized

    def dp_is_initialized(self):
        """ get dp commnication group whether is initialized """
        return self.initialized

    def is_initialized(self):
        """ get hybird commnication group whether is initialized """
        return self.initialized

    def get_bp_group(self):
        """ get bp commnication group """
        assert self.initialized == True, "bp group is not initialized!"
        return self.bp_group

    def get_bp_rank(self):
        """ get bp rank id in global group """
        if not self.initialized:
            return 0
        return self.bp_group.rank

    def get_bp_rank_in_group(self):
        """ get bp rank id in bp group """
        if not self.initialized:
            return -1
        rank = dist.get_rank()
        return self.bp_group.get_group_rank(rank)

    def get_bp_world_size(self):
        """ get bp world size in bp group """
        if not self.initialized:
            return 1
        return self.bp_group.nranks

    def get_dap_group(self):
        """ get dap commnication group """
        assert self.initialized == True, "dap group is not initialized!"
        return self.dap_group

    def get_dap_rank(self):
        """ get dap rank id in global group """
        if not self.initialized:
            return 0
        return self.dap_group.rank

    def get_dap_rank_in_group(self):
        """ get dap rank id in dap group """
        if not self.initialized:
            return -1
        rank = dist.get_rank()
        return self.dap_group.get_group_rank(rank)

    def get_dap_world_size(self):
        """ get dap world size in dap group """
        if not self.initialized:
            return 1
        return self.dap_group.nranks

    def get_dp_group(self):
        """ get ddp commnication group """
        assert self.initialized == True, "dp group is not initialized!"
        return self.dp_group

    def get_dp_rank(self):
        """ get ddp rank id in global group """
        if not self.initialized:
            return 0
        return self.dp_group.rank

    def get_dp_rank_in_group(self):
        """ get ddp rank id in ddp group """
        if not self.initialized:
            return -1
        rank = dist.get_rank()
        return self.dp_group.get_group_rank(rank)

    def get_dp_world_size(self):
        """ get ddp world size in ddp group """
        if not self.initialized:
            return 1
        return self.dp_group.nranks


scg = SingletonCommunicationGroup()
