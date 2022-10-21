# GEM-2: Next Generation Molecular Property Prediction Network with Many-body and Full-range Interaction Modeling
Molecular property prediction is a fundamental task in the drug and material industries. Physically, the properties of a molecule are determined by its own electronic structure, which can be exactly described by the Schrödinger equation. However, solving the Schrödinger equation for most molecules is extremely challenging due to long-range interactions in the behavior of a quantum many-body system. While deep learning methods have proven to be effective in molecular property prediction, we design a novel method, namely GEM-2, which comprehensively considers both the long-range and many-body interactions in molecules. GEM-2 consists of two interacted tracks: an atom-level track modeling both the local and global correlation between any two atoms, and a pair-level track modeling the correlation between all atom pairs, which embed information between any 3 or 4 atoms. Extensive experiments demonstrated the superiority of GEM-2 over multiple baseline methods in quantum chemistry and drug discovery tasks.

A preprint version of our work can be found [here](https://arxiv.org/abs/2208.05863).

# Installation guide
## Prerequisites

* OS support: Linux
* Python version: 3.6, 3.7, 3.8

## Dependencies

| name         | version |
| ------------ | ---- |
| numpy        | - |
| pandas       | - |
| paddlepaddle | \>=2.0.0 |
| rdkit-pypi   | - |
| sklearn      | - |

# Usage

Firstly, download or clone the lastest github repository:

    git clone https://github.com/PaddlePaddle/PaddleHelix.git
    git checkout dev
    cd apps/pretrained_compound/ChemRL/GEM-2

# Data
You can download the PCQM4Mv2 dataset from ogb website: 
    
    https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip

# Processed Data
You can download the processed PCQM4Mv2 dataset with rdkit generated 3d information from:
    https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/compound_datasets/pcqm4mv2_gem2.tgz
And then use tar to unzip the data.
```bash
  mkdir -p ../data
  tar xzf pcqm4mv2_gem2.tgz -C ../data
```

# How to run
## Introduction to related configs
You can adjsut the json files in the config folder to  change the training settings.
### dataset_config
- `data_dir`: where the data located
- `task_names`: the name of the label column in the datafile

### model_config
- `model`: model related information, like the channel size, dropout
- `data`: data transform setting



### train_config
- `lr`: learning rate
- `warmup_step`: the step to warm up learning rate to lr
- `mid_step`: steps before learning rate decay

## Start training 

    sh scripts/train.sh

The models will be saved under `./model`.

It will take around 60 mintues to finish one epoch on 16 A100 cards with total batch size of 512.

## Run inference
To reproduce the result from the ogb leaderboard, you can download the checkponit from:
    https://baidu-nlp.bj.bcebos.com/PaddleHelix/models/molecular_modeling/gem2_l12_c256.pdparams
Then put it under the local `./model` folder and run the inference command:
    sh scripts/inference.sh


## Citing this work

If you use the code or data in this repos, please cite:

```bibtex
@article{liu2022gem-2,
  title={GEM-2: Next Generation Molecular Property Prediction Network with Many-body and Full-range Interaction Modeling
},
  author={Liu, Lihang and He, Donglong and Fang, Xiaomin and Zhang, Shanzhuo and Wang, Fan and He, Jingzhou and Wu, Hua},
  journal={arXiv preprint arXiv:2208.05863},
  year={2022}
}
```
