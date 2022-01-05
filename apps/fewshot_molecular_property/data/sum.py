import sys
import os
from glob import glob
import paddle

import numpy as np
from itertools import compress

name = 'toxcast'

files = glob(name+'/new/*/processed/geometric_data_processed.pt')
files=sorted(files,key= lambda x: int(x.split('/')[-3]))
print('files:',len(files))
lable_list = []
for i in range(len(files)):
    lable_list.append([0,0])

del_list=[]
for file in files:
    data,slices = paddle.load(file)
    y=data.y
    neg=int((y==0).sum())
    pos = int((y == 1).sum())
    ids = int(file.split('/')[-3])-1
    lable_list[ids]=[neg,pos]
    print(ids,neg,pos)
    if pos<15 or neg<15 or (pos+neg)<150:
        del_list.append(ids)

print(lable_list)
print('del list')
print(del_list)
