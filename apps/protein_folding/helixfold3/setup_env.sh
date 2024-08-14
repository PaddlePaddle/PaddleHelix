#!/bin/bash

ENV_NAME='helixfold'
CUDA=12.0 #TODO

# Install py env
conda update -qy conda \
    && conda create -n ${ENV_NAME} -y -c conda-forge \
      cudatoolkit \
      pip \
      python=3.9

conda install -y -c bioconda aria2 hmmer==3.3.2 kalign2==2.04 hhsuite==3.3.0 -n ${ENV_NAME}
conda install -y -c conda-forge openbabel -n ${ENV_NAME}

/opt/conda/envs/${ENV_NAME}/bin/python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
/opt/conda/envs/${ENV_NAME}/bin/python -m pip install --upgrade pip \
    && /opt/conda/envs/${ENV_NAME}/bin/python -m pip install -r requirements.txt

