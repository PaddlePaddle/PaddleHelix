# Sequence VAE

## Background

## Instructions

### How to get ?
#### Model link:
You can download the model or train it by yourself.

#### Data link
You can download the data from ...

### Training Models

#### Model config
The model config is the parameters used for building the model graph. They are saved in the file:  model_parameters.json


	# encoding
	"max_length":the maximun length of the inout sequence,
    "q_cell": the RNN cell type of encoding network,
    "q_bidir": if it's bidirectional RNN or not,
    "q_d_h": the hidden size of encoding RNN,
    "q_n_layers": the layer numbers of encoding RNN,
    "q_dropout": the drop out rate of encoding RNN,
    
    # decoding
    "d_cell": the RNN cell type of decoding network,
    "d_n_layers": the layer numbers of decoding RNN,
    "d_dropout": the drop out rate of decoding RNN,
    "d_z": the hidden size of latent space,
    "d_d_h":the hidden size of decoding RNN,
    "freeze_embeddings":0
    
#### Training setting
In order to train the model, we need to set the training parameters. The dafault paramaters are saved in file: args.py

  
    # Train
	'--n_epoch': number of trianing epoch, default=1000, 
    '--n_batch': number of bach size, default=1000
    '--lr_start': 'Initial lr value, default=3 * 1e-4
    
    # kl annealing
    '--kl_start': Epoch to start change kl weight from, default=0,
    '--kl_w_start': Initial kl weight value, default=0,
    '--kl_w_end': Maximum kl weight value, default=0.05
    
To run the trianing scripts:

```

CUDA_VISIBLE_DEVICES=0 python trainer.py \

--device='gpu' \

--dataset_dir='./data/zinc_moses/train.csv' \

--model_config='model_parameters.json' \

--model_save='./results/train_models/' \

--config_save='./results/config/' \
```

#### Sampling from prior

| Model  |  Valid	| Novelty	| Unique@1k	| Filters |	IntDiv	| IntDiv2|
| Seq-VAE | 0.9765 | 0.731	| 0.993 |	0.853	| 0.846|

## Reference

@article{polykovskiy2020molecular,
  title={Molecular sets (MOSES): a benchmarking platform for molecular generation models},
  author={Polykovskiy, Daniil and Zhebrak, Alexander and Sanchez-Lengeling, Benjamin and Golovanov, Sergey and Tatanov, Oktai and Belyaev, Stanislav and Kurbanov, Rauf and Artamonov, Aleksey and Aladinskiy, Vladimir and Veselov, Mark and others},
  journal={Frontiers in pharmacology},
  volume={11},
  year={2020},
  publisher={Frontiers Media SA}
}





