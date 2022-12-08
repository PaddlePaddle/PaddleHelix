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

import numpy as np
import random
import paddle

from ppfleetx.distributed.protein_folding import dp
from ppfleetx.distributed.protein_folding.scg import scg

def init_seed(seed):
    """ set seed for reproduct results"""
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def init_distributed_env(args):
    dp_rank = 0 # ID for current device in distributed data parallel collective communication group
    dp_nranks = 1 # The number of devices in distributed data parallel collective communication group
    if args.distributed:
        # init bp, dap, dp hybrid distributed environment
        scg.init_process_group(parallel_degree=[('dp', None), ('dap', args.dap_degree), ('bp', args.bp_degree)])

        dp_nranks = dp.get_world_size()
        dp_rank = dp.get_rank_in_group() if dp_nranks > 1 else 0

        if args.bp_degree > 1 or args.dap_degree > 1:
            assert args.seed is not None, "BP and DAP should be set seed!"

    return dp_rank, dp_nranks