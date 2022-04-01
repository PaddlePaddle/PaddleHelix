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
drug drug synergy model based on transformer encoder.
"""
import numpy as np
import paddle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from paddle.optimizer import Adam
from paddle.nn import TransformerEncoder, TransformerEncoderLayer
import pgl


class TSNet(paddle.nn.Layer):
    def __init__(self, num_drug_feat=78, 
                        num_L_feat=978, 
                        num_cell_feat=954, 
                        num_drug_out=128, 
                        coarsed_heads=4, 
                        fined_heads=4,
                        coarse_hidd=64,
                        fine_hidd=64,
                        dropout=0.2):
        super().__init__()
        self.num_drug_out = num_drug_out
        self.coarsed_heads = coarsed_heads
        self.fined_heads = fined_heads
        self.coarse_hidd = coarse_hidd
        self.fine_hidd = fine_hidd
        self.dropout = dropout
        self.relu = paddle.nn.ReLU()

        #drug smiles process
        self.drug1_conv1 = pgl.nn.GCNConv(num_drug_feat, 512)
        self.drug1_conv3 = pgl.nn.GCNConv(512, self.num_drug_out)

        self.drug2_conv1 = pgl.nn.GCNConv(num_drug_feat, 512)
        self.drug2_conv3 = pgl.nn.GCNConv(512, self.num_drug_out)

        self.drug_pool = pgl.nn.pool.GraphPool('max')

        #the coarsed-grained branch
        self.cell_redu = paddle.nn.Sequential(
                         paddle.nn.Linear(num_cell_feat, 2048),
                         paddle.nn.ReLU(),
                         paddle.nn.Linear(2048, 512),
                         paddle.nn.ReLU(),
                         paddle.nn.Linear(512, self.num_drug_out))
        coarse_attn = TransformerEncoderLayer(self.num_drug_out, nhead=self.coarsed_heads, dim_feedforward=self.coarse_hidd, dropout=self.dropout, normalize_before=True, activation='relu')
        self.coarse_layer = TransformerEncoder(coarse_attn, num_layers=2)

        #fined-grained branch
        fine_attn = TransformerEncoderLayer(self.num_drug_out, nhead=self.fined_heads, dim_feedforward=self.fine_hidd, dropout=self.dropout, normalize_before=True, activation='relu')
        self.fine_layer = TransformerEncoder(fine_attn, num_layers=2)


        self.fc = paddle.nn.Sequential(
                  paddle.nn.Flatten(),
                  paddle.nn.Linear(1181 * 128, 512), 
                  paddle.nn.ReLU(),
                  paddle.nn.Linear(512, 2)
                )

    def forward(self, dg1, dg2, mask1, mask2, ccle, lincs, batch_size):
        drug1 = self.drug1_conv1(dg1, dg1.node_feat['node_feat'].astype('float32'))
        drug1 = self.relu(drug1)
        drug1 = self.drug1_conv3(dg1, drug1)
        drug1 = self.relu(drug1)
        dp1 = self.drug_pool(dg1, drug1)

        drug2 = self.drug2_conv1(dg2, dg2.node_feat['node_feat'].astype('float32'))
        drug2 = self.relu(drug2)
        drug2 = self.drug2_conv3(dg2, drug2)
        drug2 = self.relu(drug2)
        dp2 = self.drug_pool(dg2, drug2) 
        
        ccle = paddle.nn.functional.normalize(ccle, 2, 1)

        #2-branch transformer
        ccle_redu = self.cell_redu(ccle)
        #print(dp1.shape, dp2.shape, ccle_redu.shape) # all shapes = [batch, 128]
        coarse_input = paddle.concat([dp1, dp2, ccle_redu], axis=-1) #[batch_size, 128*3]
        coarse_enc = paddle.reshape(coarse_input, [-1, 3, self.num_drug_out]) #[batch, 3, 128]
        #print(coarse_input.shape, coarse_enc.shape)
        coarse_out = self.coarse_layer(coarse_enc)  #[batch_size, 3, 128] donot need attention mask cuz all dimensions are equal
        #print(coarse_out.numpy())
        #########fined attention
        #drug1 shape: [batch_size*100, 128]
        #drug2 shape: [batch_size*100, 128]
        #lincs shape: [978, 128]
        lincs = paddle.expand(lincs, shape=[batch_size, 978, 128])
        lincs_batch = paddle.reshape(lincs, [batch_size * 978, 128])
        fine_input = paddle.concat([drug1, drug2, lincs_batch], axis=0) # [batch_size*(100+100+978), 128]
        #drug mask shape: [batch_size, 100]
        lincs_mask = paddle.to_tensor(np.ones((batch_size, 978)), 'int64')
        mask_input = paddle.concat([mask1, mask2, lincs_mask], axis=-1) # [batch_size, 1178]
        attn_mask = paddle.unsqueeze(mask_input, axis=[1, 2]) # [batch_size, 1, 1, 1178]
        
        fine_enc = paddle.reshape(fine_input, [batch_size, -1, self.num_drug_out])
        #[batch_size, 1178, 128], [batch_size, 1, 1, 1178]
        fine_out = self.fine_layer(fine_enc, attn_mask) # [batch_size, 1178, 128]
        #concat and predict
        combined = paddle.concat([coarse_out, fine_out], axis=1) # shape [batch_size, 1178+3,128]
        
        out = self.fc(combined)

        return out