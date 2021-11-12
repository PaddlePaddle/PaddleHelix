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

"""Calculate length of each group in dataset."""

import pandas as pd


def get_kiba_len():
    # Get length of validation set
    for cv in ["CV1", "CV2", "CV3", "CV4", "CV5"]:
        df = pd.read_csv("../Data/KIBA/"+cv+"/"+cv+"_KIBA_unseenP_seenD_val.csv")
        df = df.groupby(['Target ID']).size().reset_index(name = 'counts')
        f = open("../Data/KIBA/"+cv+"/"+cv+"_val.txt",'a')
        for i in df['counts'].values:
            f.write(str(i) + "\n")


    # Get length of testing set
    df = pd.read_csv("../Data/KIBA/test_KIBA_unseenP_seenD.csv")
    df = df.groupby(['Target ID']).size().reset_index(name = 'counts')
    f = open("../Data/KIBA/kiba_len.txt",'a')
    for i in df['counts'].values:
        f.write(str(i) + "\n")