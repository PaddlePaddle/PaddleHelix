import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import numpy as np
from itertools import compress

import random

import json

name = 'toxcast' # tox21


f = open(os.path.join(BASE_DIR, '{}/raw/{}.csv'.format(name,name)), 'r').readlines()[1:]
np.random.shuffle(f)


if __name__ == "__main__":
    tasks = {}
    
    # Below needs to be modified according to different original datasets
    for index, line in enumerate(f):
        line=line.strip()
        l = line.split(",")
        size=len(l)
        if size<2:
            continue
        '''
        toxcast, sider -> smi = l[0]; for i in range(1, size) 
        tox 21 -> smi = l[-1]; for i in range(12):
        muv -> smi = l[-1]; for i in range(17):
        '''
        smi = l[0] # modify to data
        for i in range(1, size):
            cur_item = l[i].strip()
            if i not in tasks:
                tasks[i] = [[],[]]
            if cur_item == "0.0" or cur_item == "0" or cur_item==0:
                tasks[i][0].append(smi)
            elif cur_item == "1.0" or cur_item == "1" or cur_item==1:
                tasks[i][1].append(smi)
    #until here

    cnt_tasks=[]
    for i in tasks:
        root = name + "/new/" + str(i)
        os.makedirs(root, exist_ok=True)
        os.makedirs(root + "/raw", exist_ok=True)
        os.makedirs(root + "/processed", exist_ok=True)

        file = open(root + "/raw/" + name + ".json", "w")
        file.write(json.dumps(tasks[i]))
        file.close()
        print('task:',i,len(tasks[i][0]), len(tasks[i][1]))
        cnt_tasks.append([len(tasks[i][0]), len(tasks[i][1])])
    print(cnt_tasks)
    
