# Junction Tree Variational Autoencoder

[中文版本](./README_cn.md) [English Version](./README.md)

* [Background](#Background)
* [Instructions](#Instructions)
    * [How to get ?](#How-to-get-?)
        * [Model link](#Model-link)
        * [Data link](#Data-link)
    * [Training and Evaluation](#Training-and-evaluation)
* [Reference](#Reference)
    * [Papers](#Papers)
    * [Data](#Data)

## Background
Implementation of Junction Tree Variational Autoencoder (https://arxiv.org/abs/1802.04364)

## Instructions

### How to get ?

#### Model link
You can download our pretrained [model](https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/molecular_generation/vae_models.tgz) or train it by yourself.

#### Data link
You can download the dataset from the [link](https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/molecular_generation/zinc.tgz) we provided and perform the corresponding preprocessing for your use. It is recommended to unzip the dataset and put it in the data/directory. 
If you need to train on your own dataset, execute the following commands to generate the dictionary file of the dataset :    

```bash 
python -m src.mol_tree \
    --train_path dataset_file_path \
    --vocab_file vocab_file_save_path
```

### Training and Evaluation
Preprocessing:
```bash 
python preprocess.py \
    --train data/zinc/250k_rndm_zinc_drugs_clean_sorted.smi \
    --save_dir zinc_processed \
    --split 100 \
    --num_workers 8
```


Training:
```bash
CUDA_VISIBLE_DEVICES=0 python vae_train.py \
        --train zinc_processed \
        --vocab data/zinc/vocab.txt \
        --config configs/config.json \
        --save_dir vae_models \
        --num_workers 1 \
        --epoch 50 \
        --batch_size 32 \
        --use_gpu True 
```
We provide a configuration 'configs/config.json' for users to initialize the neural network.
Description of training parameters:

`beta`: the initial KL regularization weight (beta).   
`warmup`: warmup means that beta will not increase within first `warmup` training steps.     
`step_beta`: beta will increase by `step_beta` every `kl_anneal_iter` training steps.    
`kl_anneal_iter`:  beta will update every `kl_anneal_iter` training steps.   
`max_beta`: the maximum value of beta.   
`save_dir`: the model will be saved in save_dir/.   


Testing:
```bash
python sample.py \
        --nsample 10000 \
        --vocab data/zinc/vocab.txt \
        --model vae_models/model.iter-441000 \
        --config configs/config.json \
        --output sampling_output.txt
```
The sampling result of model.iter-422000 is as follows:
```bash
valid,1.0
unique@1000,1.0
unique@10000,0.9997
IntDiv,0.8701593437246322
IntDiv2,0.8646974999795127
Filters,0.6084
Novelty,0.9998999699909973
```
Since we didn't split the test set from the dataset, we didn't do the evaluation related to the test set in Moses Benchmark.

Fine-tuning:
```bash
CUDA_VISIBLE_DEVICES=0 python vae_train.py \
        --train zinc_processed \
        --vocab data/zinc/vocab.txt \
        --config configs/config.json \
        --save_dir vae_models \
        --num_workers 1 \
        --epoch 50 \
        --batch_size 32 \
        --use_gpu True \
        --load_epoch 441000
```


## Reference
### papers
**Junction Tree Variational Autoencoder**
> @article{Jin2018,
  author = {Jin, Wengong and Barzilay, Regina and Jaakkola, Tommi},
  title = {{Junction Tree Variational Autoencoder for Molecular Graph Generation}},
  url = {http://arxiv.org/abs/1802.04364},
  journal={ICML 2018},
  year = {2018}
}
> @article{polykovskiy2020molecular,
      title={Molecular Sets (MOSES): A Benchmarking Platform for Molecular Generation Models}, 
      author={Daniil Polykovskiy and Alexander Zhebrak and Benjamin Sanchez-Lengeling and Sergey Golovanov and Oktai Tatanov and Stanislav Belyaev and Rauf Kurbanov and Aleksey Artamonov and Vladimir Aladinskiy and Mark Veselov and Artur Kadurin and Simon Johansson and Hongming Chen and Sergey Nikolenko and Alan Aspuru-Guzik and Alex Zhavoronkov},
      year={2020},
      eprint={1811.12823},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

### Data
The training data  were randomly selected from the ZINC dataset.
**ZINC15(Pre-training):**
> @article{doi:10.1021/ci3001277,
    annote = {PMID: 22587354},
    author = {Irwin, John J and Sterling, Teague and Mysinger, Michael M and Bolstad, Erin S and Coleman, Ryan G},
    doi = {10.1021/ci3001277},
    journal = {Journal of Chemical Information and Modeling},
    number = {7},
    pages = {1757--1768},
    title = {{ZINC: A Free Tool to Discover Chemistry for Biology}},
    url = {https://doi.org/10.1021/ci3001277},
    volume = {52},
    year = {2012}
}

