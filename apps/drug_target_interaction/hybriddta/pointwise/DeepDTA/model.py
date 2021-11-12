#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

"""DeepDTA backbone model."""

import paddle
from paddle import nn
import paddle.nn.functional as F
import numpy as np

# Set seed for reproduction
paddle.seed(10)


class DeepdtaModel(nn.Layer):
    """DeepDTA model.

    Args:
        input_d: Input drug.
        input_p: Input target.
    
    Returns:
        res: Prediction results.
    """
    def __init__(self, max_d=100, max_p=1000, n_filter=5, embed_dim=128, p_features=25, d_features=64, d_filter=4, p_filter=8, num_filters=32):
        super(DeepdtaModel,self).__init__()
        # Basic config
        self.max_drug = max_d
        self.max_protein = max_p
        self.embed_dim = embed_dim
        self.BN = nn.BatchNorm(1024)
        self.dropout = nn.Dropout(p=0.1)
        self.relu =nn.ReLU()
        self.p_embedding = nn.Embedding(p_features, embed_dim)
        self.d_embedding = nn.Embedding(d_features, embed_dim)
        # Protein CNN
        self.p_conv = nn.Sequential(
            nn.Conv1D(in_channels=self.embed_dim, out_channels=num_filters, kernel_size=p_filter),
            nn.ReLU(),

            nn.Conv1D(in_channels=num_filters, out_channels=num_filters*2, kernel_size=p_filter),
            nn.ReLU(),

            nn.Conv1D(in_channels=num_filters*2, out_channels=num_filters*3, kernel_size=p_filter),
            nn.ReLU(),
            
            nn.MaxPool1D(kernel_size=1000-3*p_filter+3)
        )
        # Drug CNN
        self.d_conv = nn.Sequential(
            nn.Conv1D(in_channels=self.embed_dim, out_channels=num_filters, kernel_size=d_filter),
            nn.ReLU(),

            nn.Conv1D(in_channels=num_filters, out_channels=num_filters*2, kernel_size=d_filter),
            nn.ReLU(),

            nn.Conv1D(in_channels=num_filters*2, out_channels=num_filters*3, kernel_size=d_filter),
            nn.ReLU(),

            nn.MaxPool1D(kernel_size=100-3*d_filter+3)
        )
        #Decoder
        self.fc1 = nn.Linear(192,1024)
        self.fc2 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024,512)
        self.fc4 = nn.Linear(512,1)
    
    def forward(self, input_d, input_p):
        # Drug embedding
        d_emb = self.d_embedding(input_d)
        d_emb = paddle.transpose(d_emb, perm=[0,2,1])
        d_emb = self.d_conv(d_emb)
        
        # Protein embedding
        p_emb = self.d_embedding(input_p)
        p_emb = paddle.transpose(p_emb, perm=[0,2,1])
        p_emb = self.p_conv(p_emb)
        
        # Concatenate protein and drug
        x = paddle.concat(x=[d_emb,p_emb],axis=1)
        x = paddle.squeeze(x, axis = 2)

        # FC layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.relu(x)

        res = self.fc4(x)

        return res