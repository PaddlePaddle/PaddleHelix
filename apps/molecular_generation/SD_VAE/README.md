# SD VAE

## Background
Deep generative models are rapidly becoming popular tools for generating new molecules and optimizing the chemical properties. In this work, we will introduce a VAE model based on the grammar and semantic of molecular sequence - SD VAE.

## Instructions
This repository will give you the instruction of training a SD VAE model.

## Data Link

You can download the data from [datalink] (https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/molecular_generation/data_SD_VAE.tgz).

1. create a folder './data'
2. unzip the file into './data'


data (project root)
|__  data_SD_VAE
|__  |__ context_free_grammars
|__  |__ zinc


## Data pre-processing
Before training/evaluation, we need to cook the raw txt dataset. 

    cd data_preprocessing
    
    python make_dataset_parallel.py \
    -info_fold ../data/data_SD_VAE/context_free_grammars \
        -grammar_file ../data/data_SD_VAE/context_free_grammars/mol_zinc.grammar \
        -smiles_file ../data/data_SD_VAE/zinc/250k_rndm_zinc_drugs_clean.smi 
        
        
    python dump_cfg_trees.py \
    -info_fold ../data/data_SD_VAE/context_free_grammars \
        -grammar_file ../data/data_SD_VAE/context_free_grammars/mol_zinc.grammar \
        -smiles_file ../data/data_SD_VAE/zinc/250k_rndm_zinc_drugs_clean.smi 
        

The above two scripts will compile the txt data into binary file and cfg dump, correspondingly.

## Training
    
#### Model config
The model config is the parameters used for building the model graph. They are saved in the file: model_config.json

    "latent_dim":the hidden size of latent space
    "max_decode_steps": maximum steps for making decoding decisions
    "eps_std": the standard deviation used in reparameterization tric
    "encoder_type": the type of encoder
    "rnn_type": The RNN type


#### Training setting
In order to train the model, we need to set the training parameters. The dafault paramaters are saved in file: args.py

    -loss_type : the type of loss
    -num_epochs : number of epochs
    -batch_size : minibatch size
    -learning_rate : learning_rate
    -kl_coeff : coefficient for kl divergence used in vae
    -clip_grad : clip gradients to this value


To run the trianing scripts:

    CUDA_VISIBLE_DEVICES=0 python train_zinc.py \
    -mode='gpu' \


#### Sampling results from prior
valid: 0.41
unique@1000: 1.0
unique@10000: 1.0
IntDiv: 0.92
IntDiv2: 0.81
Filters: 0.35


## Reference
[1] @misc{dai2018syntaxdirected,
      title={Syntax-Directed Variational Autoencoder for Structured Data}, 
      author={Hanjun Dai and Yingtao Tian and Bo Dai and Steven Skiena and Le Song},
      year={2018},
      eprint={1802.08786},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}