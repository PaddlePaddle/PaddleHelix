# ChemRL-GEM: Geometry Enhanced Molecular Representation Learning for Property Prediction

# Background
Recent advances in graph neural networks (GNNs) have shown great promise in applying GNNs for molecular representation learning. However, existing GNNs usually treat molecules as topological graph data without fully utilizing the molecular geometry information, which is one of the most critical factors for determining molecular physical, chemical, and biological properties. 

To this end, we propose a novel **G**eometry **E**nhanced **M**olecular representation learning method (GEM):

* At first, we design a geometry-based GNN architecture (GeoGNN) that simultaneously models atoms, bonds, and bond angles in a molecule. 
* Moreover, on top of the devised GNN architecture, we propose several novel geometry-level self-supervised learning strategies to learn spatial knowledge by utilizing the local and global molecular 3D structures.


# Installation guide
## Prerequisites

* OS support: Linux
* Python version: 3.6, 3.7, 3.8

## Dependencies

| name         | version |
| ------------ | ---- |
| numpy        | - |
| pandas       | - |
| networkx     | - |
| paddlepaddle | \>=2.0.0 |
| pgl          | \>=2.1.5 |
| rdkit-pypi   | - |
| sklearn      | - |

('-' means no specific version requirement for that package)

# Usage

Firstly, download or clone the lastest github repository:

    git clone https://github.com/PaddlePaddle/PaddleHelix.git
    git checkout dev
    cd apps/pretrained_compound/ChemRL/GEM

## Pretraining
Use the following command to download the demo data which is a tiny subset [Zinc Dataset](https://zinc.docking.org/) and run pretrain tasks.

    sh scripts/pretrain.sh

Note that the data preprocessing step will be time-consuming since it requires running MMFF optimization for all molecules. The demo data will take several hours to finish in a single V100 GPU card. The pretrained model will be save under `./pretrain_models`.

We also provide our pretrained model [here](https://baidu-nlp.bj.bcebos.com/PaddleHelix/pretrained_models/compound/pretrain_models-chemrl_gem.tgz) for reproducing the downstream finetuning results. Also, the pretrained model can be used for other molecular property prediction tasks.

## Downstream finetuning
After the pretraining, the downstream tasks can use the pretrained model as initialization. 

Firstly, download the pretrained model from the previous step:

    wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/pretrained_models/compound/pretrain_models-chemrl_gem.tgz
    tar xzf pretrain_models-chemrl_gem.tgz

Download the downstream molecular property prediction datasets from [MoleculeNet](http://moleculenet.ai/), including classification tasks and regression tasks:

    wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/compound_datasets/chemrl_downstream_datasets.tgz
    tar xzf chemrl_downstream_datasets.tgz
    
Run downstream finetuning and the final results will be saved under `./log/pretrain-$dataset/final_result`. 

    # classification tasks
    sh scripts/finetune_class.sh
    # regression tasks
    sh scripts/finetune_regr.sh

The whole finetuning process for all datasets requires 1-2 days in a single V100 GPU card.

