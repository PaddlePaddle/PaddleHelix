#   Copyright (c) 2024 PaddleHelix Authors. All Rights Reserved.
#
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# International License (the "License");  you may not use this file  except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils."""

from typing import Tuple

def get_custom_amp_list() -> Tuple[set, set]:
    """get custom amp list"""

    black_list = {
        # "layer_norm",
        "reduce_sum"
    }
    white_list = {
        "one_hot", 
        "concat",
        "dropout_nd",
        "einsum",
        "elementwise_add",
        "elementwise_div",
        "elementwise_mul",
        "elementwise_sub",
        "fill_any_like",
        "fill_constant",
        "fused_gate_attention",
        "fused_gemm_epilogue",
        "gather",
        "gaussian_random",
        "layer_norm",
        "log_softmax",
        "matmul_v2",
        "p_norm",
        "py_layer",
        "relu",
        "scale",
        "sigmoid",
        "slice",
        "softmax",
        "softplus",
        "split",
        "split_with_num",
        "sqrt",
        "square",
        "stack",
        "sum",
        "transpose2",
        "unsqueeze2",
        "unstack",
        "where"
    }
    return black_list, white_list

