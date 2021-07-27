# Implementation of ChemRL-GEM: Geometry Enhanced Molecular Representation Learning for Property Prediction

## Pretraining
Use the following command to run pretrain tasks on the zinc dataset. 

    sh scripts/pretrain.sh

Note that the data preprocessing step will be time-consuming since it requires running MMFF optimization for all molecules. 

We also provide the pretrained parameters [here](https://tbd.com) if you want to run the downstream tasks directly.

## Downstream finetuning
After the pretraining, the downstream tasks can use the pretrained parameters as initialization. Use the following command to run downstream regression tasks to reproduce the reported results.

    sh scripts/finetune_regr.sh
