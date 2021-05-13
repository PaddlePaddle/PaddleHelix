# Sequence VAE

## Background
Deep generative models are rapidly becoming popular tools for generating new molecules and optimizing the chemical properties. In this work, we will introduce a VAE model based on molecular sequence - Sequence VAE.

## Instructions
This repository will give you the instruction of training a sequence VAE model and sampling new molecules from the pretrained model.

### How to get ?
#### Model link:
You can download the pre-trained model.

#### Data link
We use the data from ZINC Clean Leads collection [1].

You can download the data from [datalink] (https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/molecular_generation/zinc_moses.tgz).

1. create a folder './data'
2. unzip the file into './data'

data
|-- zinc_mose
|   |-- train.csv
|   |-- test.csv


### Training Models

#### Model config
The model config is the parameters used for building the model graph. They are saved in the file: model_config.json


  # encoding
  "max_length":the maximun length of the inout sequence
    "q_cell": the RNN cell type of encoding network
    "q_bidir": if it's bidirectional RNN or not
    "q_d_h": the hidden size of encoding RNN
    "q_n_layers": the layer numbers of encoding RNN
    "q_dropout": the drop out rate of encoding RNN
    
    # decoding
    "d_cell": the RNN cell type of decoding network
    "d_n_layers": the layer numbers of decoding RNN
    "d_dropout": the drop out rate of decoding RNN
    "d_z": the hidden size of latent space
    "d_d_h":the hidden size of decoding RNN
    "freeze_embeddings": if freeze the embedding layer
    
#### Training setting
In order to train the model, we need to set the training parameters. The dafault paramaters are saved in file: args.py

  
    # Train
    '--n_epoch': number of trianing epoch, default=1000
    '--n_batch': number of bach size, default=1000
    '--lr_start': 'Initial lr value, default=3 * 1e-4
    
    # kl annealing
    '--kl_start': Epoch to start change kl weight from, default=0
    '--kl_w_start': Initial kl weight value, default=0
    '--kl_w_end': Maximum kl weight value, default=0.05
    
To run the trianing scripts:

```

CUDA_VISIBLE_DEVICES=0 python trainer.py \

--device='gpu' \

--dataset_dir='./data/zinc_moses/train.csv' \

--model_config='model_config.json' \

--model_save='./results/train_models/' \

--config_save='./results/config/' \
```

#### Sampling results from prior

Valid: 0.9765
Novelty: 0.731
Unique@1k: 0.993
Filters: 0.853
IntDiv: 0.846

## Reference

[1] @article{polykovskiy2020molecular,
  title={Molecular sets (MOSES): a benchmarking platform for molecular generation models},
  author={Polykovskiy, Daniil and Zhebrak, Alexander and Sanchez-Lengeling, Benjamin and Golovanov, Sergey and Tatanov, Oktai and Belyaev, Stanislav and Kurbanov, Rauf and Artamonov, Aleksey and Aladinskiy, Vladimir and Veselov, Mark and others},
  journal={Frontiers in pharmacology},
  volume={11},
  year={2020},
  publisher={Frontiers Media SA}
}