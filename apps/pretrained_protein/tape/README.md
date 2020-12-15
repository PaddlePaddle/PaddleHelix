# Protein Sequence Pretraining (TAPE)

[中文版本](./README.ch.md) [English Version](./README.en.md)

* [Background](#background)
* [Instructions](#instructions)
    * [Training Models](#training-models)
        * [Multi-cpu Training / Single-gpu Training](#multi-cpu-training-/-single-gpu-training)
        * [Multi-gpu Training](#multi-gpu-training)
    * [Evaluating Models](#evaluating-models)
    * [Model Inference](#model-inference)
    * [Sequence Models](#sequence-models)
        * [Transformer](#transformer)
        * [LSTM](#lstm)
        * [ResNet](#resnet)
        * [Other Parameters](#other-parameters)
    * [Protein Related Tasks](#protein-related-tasks)
        * [Pretraining Tasks](#pretraining-tasks)
            * [Pfam](#pfam)
        * [Supervised Tasks](#supervised-tasks)
            * [Secondary Structure](#secondary-structure)
            * [Remote Homology](#remote-homology)
            * [Fluorescence](#fluorescence)
            * [Stability](#stability)
    * [Warm Start / Finetuning](#warm-start-/-finetuning)
    * [Complete Example](#complete-example)
* [Data](#data)
* [Pre-trained Models](#pre-trained-models)
* [Reference](#reference)
    * [Paper-related](#paper-related)
    * [Data-related](#data-related)

## Background
In recent years, with sequencing technology development, the protein sequence database scale has significantly increased. However, the cost of obtaining labeled protein sequences is still very high, as it requires biological experiments. Besides, due to the inadequate number of labeled samples, the model has a high probability of overfitting the data. Borrowing the ideas from natural language processing (NLP),  we can pre-train numerous unlabeled sequences by self-supervised learning. In this way, we can extract useful biological information from proteins and transfer them to other tagged tasks to make these tasks training faster and more stable convergence. These instructions refer to the work of paper TAPE, providing the model implementation of Transformer, LSTM, and ResNet.

## Instructions

### Training Models
We offer multiple training methods:

- Multi-cpu training.
- Single-gpu training.
- Multi-gpu training.

#### Multi-cpu Training / Single-gpu Training
The example of using CPU training with multiple threads / GPU training with a single card is shown as follows:
```bash
python train.py \
        --train_data ./train_data # Directory of training data for training models, including multiple training files. \
        --test_data ./test_data # Directory of test data for evaluating models, including multiple test files. If test_data is not defined, we will not evaluate the model during the training process. \
        --lr 0.0001 # Basic learning rate. \
        --thread_num 8 # The number of threads to be used in CPU training. \
        --warmup_steps 0 # Warmup steps. When warmup_steps=0, the model uses a constant learning rate. When warmup_steps>0, the model uses Noam Decay. \
        --batch_size 128 # The upper bound of the batch size. As the lengths of the proteins may be large, the model dynamically adjusts the batch size according to its length. \
        --model_dir ./models # The directory of the saved models. \
        --use_cuda # Whether use cuda for training. \
        ... # Model parameter settings and task parameter settings will be introduced in the following chapters.
```

#### Multi-gpu Training
We use paddle.distributed.launch for multi-gpu training and parameter "--distributed" should be added. Other parameters are consistent with multi-cpu training / single-gpu training. An example of multi-gpu training is shown as follows:

```bash
python -m paddle.distributed.launch --log_dir log_dir train.py # Specify the log directory by "--log_dir" \
        --train_data ./train_data # Directory of training data for training models, including multiple training files. \
        --test_data ./test_data # Directory of test data for evaluating models, including multiple test files. If test_data is not defined, we will not evaluate the model during the training process. \
        --lr 0.0001 # Basic learning rate. \
        --warmup_steps 0 # Warmup steps. When warmup_steps=0, the model uses a constant learning rate. When warmup_steps>0, the model uses Noam Decay.\
        --batch_size 128 # The upper bound of the batch size. As the lengths of the proteins may be large, the model dynamically adjusts the batch size according to its length. \
        --model_dir ./models # The directory of the saved models. \
        --use_cuda # Only support gpu for now. \
        --distributed # Distributed training. \
        ... # Model parameter settings and task parameter settings will be introduced in the following chapters.
```

### Evaluating Models
The model evaluation is similar to the model training method. Currently, only multi-cpu and single-gpu evaluation are supported.
```bash
python eval.py \
        --data ./test_data # Directory of test data for evaluating models, including multiple test files. \
        --batch_size 128 # The upper bound of the batch size. As the lengths of the proteins may be large, the model dynamically adjusts the batch size according to its length. \
        --model ./model # The model to be evaluated. \
        --use_cuda # Whether the model runs in CPU or GPU. \
        ... # Model parameter settings and task parameter settings will be introduced in the following chapters.
```

### Model Inference
The model inference is similar to the model evaluation method. Currently, only multi-cpu and single-gpu prediction are supported.
```bash
cat predict_file | # The file that contains amino acid sequence. \
python predict.py \
        --batch_size 128 # The upper bound of the batch size. As the lengths of the proteins may be large, the model dynamically adjusts the batch size according to its length. \
        --model ./model # The model. \
        --use_cuda # Whether the model runs in CPU or GPU. \
        ... # Model parameter settings and task parameter settings will be introduced in the following chapters.
```

### Sequence Models
We provides models Transformer, LSTM, and ResNet. The model related parameters should be included in "--model_config".  We set the model_type (transformer, lstm, resnet) by setting "model _type" in "model_config".

```bash
python train.py \
        ... # The way to set training parameters has been introduced. \
        --model_config ./transformer_config # The configuration file of the model, organized by json format. \
        ... # Task parameter settings will be introduced in the following chapters.
```

#### Transformer
Transformer is often used in semantic modeling of natural language processing. To use transformer, you need to set the following parameters:

- hidden_size: The hidden size of transformer.
- layer_num: The number of layers in transformer.
- head_num: The number of headers in each layer.

For details of transformer, please refer to the following papers:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)


#### LSTM
We use multilayer bidirectional LSTM. To use LSTM, we need to set the following parameters:

- hidden_size: The hidden size of LSTM.
- layer_num: The number of layers.

LSTM can refer to the following papers:

- [Long Short Term Memory](http://didawiki.di.unipi.it/lib/exe/fetch.php/magistraleinformatica/aa2/lstm.pdf)


#### ResNet
We use multilayer ResNet. Using ResNet, we need to set the following parameters:

- hidden_size: The hidden size of ResNet.
- layer_num: The number of layers.
- filter_num: The number of filter of the convolution layers.

ResNet can refer to the following papers:

- [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)


#### Other Parameters
Other parameters can be set in the model to avoid over-fitting or excessive parameter value.

- dropout: The dropout ratio of model parameters.
- weight_decay: Parameter decay ratio, used to avoid excessive parameter values.

### Protein Related Tasks
Referring to the paper tape, we reproduced the following tasks using PaddleHelix.
#### Pretraining Tasks
##### Pfam
Dataset *pfam* contains 30 million protein sequences, which can be used for pretraining models. We should set the following parameters in the `model_config`.
```bash
...
task: "pretrain",
...
```

#### Supervised Tasks
#### Secondary Structure
Dataset *secondary structure* consists of two sequence annotation tasks, a 3-category annotation task, and an 8-category annotation task. We should set the following parameters in the `model_config`.
```bash
...
task: "seq_classification",
class_num: 3,
label_name: "labels3",
...
```

##### Evaluation Results
The fine-tuning models' results are shown as follows.

Three-way Accuracy：

| Model         | CB513     | CASP12    | TS115 |   
| :--:          | :--:      | :--:      | :--:  |
| Transformer   | 0.741     | 0.686     | 0.779 |
| LSTM          | 0.727     | 0.718     | 0.763 |
| ResNet        | 0.724     | 0.704     | 0.748 |

Eight-way Accuracy：

| Model         | CB513     | CASP12    | TS115 |   
| :--:          | :--:      | :--:      | :--:  |
| Transformer   | 0.595     | 0.566     | 0.655 |
| LSTM          | 0.580     | 0.581     | 0.640 |
| ResNet        | 0.582     | 0.565     | 0.630 |

#### Remote Homology
Remote homology is a classification task with 1195 classes. We should set the following parameters in the `model_config`.
```bash
...
task: "classification",
class_num: 1195,
label_name: "labels",
...
```

##### Evaluation Results
The fine-tuning models' results are shown as follows. 

Accuracy：

| Model         | Fold  | Superfamily   | Family    |   
| :--:          | :--:  | :--:          | :--:      |
| Transformer   | 0.143 | 0.291         | 0.851     |
| LSTM          |       |               |           |
| ResNet        |       |               |           |


#### Fluorescence
*Fluorescence* is a regression task.  We should set the following parameters in the `model_config`.
```bash
...
task: "regression",
label_name: "labels",
...
```

##### Evaluation Results
The fine-tuning models' results are shown as follows.

Spearman：

| Model         | Test      |
| :--:          | :--:      |
| Transformer   | 0.678     | 
| LSTM          | 0.676     | 
| ResNet        | 0.684     |
#### Stability
*Stability* is a regression task. We should set the following parameters in the `model_config`.
```bash
...
task: "regression",
label_name: "labels",
...
```

##### Evaluation Results
The fine-tuning models' results are shown as follows.

Spearman：

| Model         | Test      |
| :--:          | :--:      |
| Transformer   | 0.749     | 
| LSTM          | 0.724     | 
| ResNet        | 0.746     |


### Warm Start / Finetuning
We can set the parameter "--init_model " to initialize the model or finetune the supervised tasks during the training process.
```bash
python train.py \
        ... \
        --init_model ./init_model # Directory of the initialization model. If this parameter is unset, the model is randomly initialized. \
        ... 
```

### Complete Example
We provide multiple training and evaluation examples in the folder *demos*. Here is a pretraining example of the Transformer.
```bash
#!/bin/bash

source ~/.bashrc

batch_size="256"
lr="0.001"
thread_num="8" # thread_num is for cpu, please set CUDA_VISIBLE_DEVICES for gpu
warmup_steps="0"
model_type="transformer" # candidate model_types: transformer, lstm, resnet
task="pfam" # candidate tasks: pfam, secondary_structure, remote_homology, fluorescence, stability
model_config="./${model_type}_${task}_config.json"
model_dir="./models"
use_cuda="true" # candidates: true/false
train_data="./toy_data/${task}/npz/train"
test_data="./toy_data/${task}/npz/valid"
distributed="false" # candidates: true/false

if [ "${distributed}" == "true" ]; then
    if [ "${use_cuda}" == "true" ]; then
        export FLAGS_sync_nccl_allreduce=1
        export FLAGS_fuse_parameter_memory_size=64
        export CUDA_VISIBLE_DEVICES="0,1"

        python -m paddle.distributed.launch \
            --log_dir log_dirs \
            ../train.py \
                --train_data ${train_data} \
                --test_data ${test_data} \
                --lr ${lr} \
                --thread_num ${thread_num} \
                --warmup_steps ${warmup_steps} \
                --batch_size ${batch_size} \
                --model_type ${model_type} \
                --model_config ${model_config} \
                --model_dir ${model_dir} \
                --use_cuda \
                --distributed
    else
        echo "Only gpu is supported for distributed mode at present."
    fi
else
    if [ "${use_cuda}" == "true" ]; then
        export CUDA_VISIBLE_DEVICES="2"
        python ../train.py \
                --train_data ${train_data} \
                --test_data ${test_data} \
                --lr ${lr} \
                --thread_num ${thread_num} \
                --warmup_steps ${warmup_steps} \
                --batch_size ${batch_size} \
                --model_type ${model_type} \
                --model_config ${model_config} \
                --model_dir ${model_dir} \
                --use_cuda
    else
        python ../train.py \
                --train_data ${train_data} \
                --test_data ${test_data} \
                --lr ${lr} \
                --thread_num ${thread_num} \
                --warmup_steps ${warmup_steps} \
                --batch_size ${batch_size} \
                --model_type ${model_type} \
                --model_config ${model_config} \
                --model_dir ${model_dir}
    fi
fi

```

Following shows the demo of model_config.
```bash
{
    "model_name": "secondary_structure",
    "task": "seq_classification",
    "class_num": 3,
    "label_name": "labels3",

    "hidden_size": 512,
    "layer_num": 12,
    "head_num": 8,

    "comment": "The following hyper-parameters are optional.",
    "dropout": 0.1,
    "weight_decay": 0.01
}
```

## Data
The datasets can be downloaded from the following urls:
pfam: [raw](https://baidu-nlp.bj.bcebos.com/PaddleHelix%2Fdatasets%2Fprotein_datasets%2Fpfam.npz.tgz), [npz](https://baidu-nlp.bj.bcebos.com/PaddleHelix%2Fdatasets%2Fprotein_datasets%2Fpfam.npz.tgz)
secondary structure: [all](https://baidu-nlp.bj.bcebos.com/PaddleHelix%2Fdatasets%2Fprotein_datasets%2Fsecondary_structure.tgz)
remote homology: [all](https://baidu-nlp.bj.bcebos.com/PaddleHelix%2Fdatasets%2Fprotein_datasets%2Fremote_homology.tgz)
fluorescence: [all](https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/protein_datasets/fluorescence.tgz)
stability: [all](https://baidu-nlp.bj.bcebos.com/PaddleHelix%2Fdatasets%2Fprotein_datasets%2Fstability.tgz)

## Pre-trained Models
The pre-trained models can be downloaded from the following urls:
Transformer: [model](https://baidu-nlp.bj.bcebos.com/PaddleHelix%2Fpretrained_models%2Fprotein%2Ftape_transformer.tgz)
LSTM: [model](https://baidu-nlp.bj.bcebos.com/PaddleHelix%2Fpretrained_models%2Fprotein%2Ftape_lstm.tgz)
ResNet: [model](https://baidu-nlp.bj.bcebos.com/PaddleHelix%2Fpretrained_models%2Fprotein%2Ftape_resnet.tgz)

## Reference
### Paper-related
We mainly refer to paper *TAPE*. The way we train the models and the hyper-parameters might be different.

**TAPE:**
> @inproceedings{tape2019,
author = {Rao, Roshan and Bhattacharya, Nicholas and Thomas, Neil and Duan, Yan and Chen, Xi and Canny, John and Abbeel, Pieter and Song, Yun S},
title = {Evaluating Protein Transfer Learning with TAPE},
booktitle = {Advances in Neural Information Processing Systems}
year = {2019}
}

**Transformer**
>@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in neural information processing systems},
  pages={5998--6008},
  year={2017}
}
@article{devlin2018bert,
  title={Bert: Pre-training of deep bidirectional transformers for language understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}

**LSTM**
>@article{hochreiter1997long,
  title={Long short-term memory},
  author={Hochreiter, Sepp and Schmidhuber, J{\"u}rgen},
  journal={Neural computation},
  volume={9},
  number={8},
  pages={1735--1780},
  year={1997},
  publisher={MIT Press}
}

**ResNet**
>@article{szegedy2016inception,
  title={Inception-v4, inception-resnet and the impact of residual connections on learning},
  author={Szegedy, Christian and Ioffe, Sergey and Vanhoucke, Vincent and Alemi, Alex},
  journal={arXiv preprint arXiv:1602.07261},
  year={2016}
}

### Data-related
We further process the data in paper *TAPE* to train the models.

**Pfam (Pretraining):**
>@article{pfam,
author = {El-Gebali, Sara and Mistry, Jaina and Bateman, Alex and Eddy, Sean R and Luciani, Aur{\'{e}}lien and Potter, Simon C and Qureshi, Matloob and Richardson, Lorna J and Salazar, Gustavo A and Smart, Alfredo and Sonnhammer, Erik L L and Hirsh, Layla and Paladin, Lisanna and Piovesan, Damiano and Tosatto, Silvio C E and Finn, Robert D},
doi = {10.1093/nar/gky995},
file = {::},
issn = {0305-1048},
journal = {Nucleic Acids Research},
keywords = {community,protein domains,tandem repeat sequences},
number = {D1},
pages = {D427--D432},
publisher = {Narnia},
title = {{The Pfam protein families database in 2019}},
url = {https://academic.oup.com/nar/article/47/D1/D427/5144153},
volume = {47},
year = {2019}
}

**SCOPe: (Remote Homology and Contact)**
>@article{scop,
  title={SCOPe: Structural Classification of Proteins—extended, integrating SCOP and ASTRAL data and classification of new structures},
  author={Fox, Naomi K and Brenner, Steven E and Chandonia, John-Marc},
  journal={Nucleic acids research},
  volume={42},
  number={D1},
  pages={D304--D309},
  year={2013},
  publisher={Oxford University Press}
}

**PDB: (Secondary Structure and Contact)**
>@article{pdb,
  title={The protein data bank},
  author={Berman, Helen M and Westbrook, John and Feng, Zukang and Gilliland, Gary and Bhat, Talapady N and Weissig, Helge and Shindyalov, Ilya N and Bourne, Philip E},
  journal={Nucleic acids research},
  volume={28},
  number={1},
  pages={235--242},
  year={2000},
  publisher={Oxford University Press}
}

**CASP12: (Secondary Structure and Contact)**
>@article{casp,
author = {Moult, John and Fidelis, Krzysztof and Kryshtafovych, Andriy and Schwede, Torsten and Tramontano, Anna},
doi = {10.1002/prot.25415},
issn = {08873585},
journal = {Proteins: Structure, Function, and Bioinformatics},
keywords = {CASP,community wide experiment,protein structure prediction},
pages = {7--15},
publisher = {John Wiley {\&} Sons, Ltd},
title = {{Critical assessment of methods of protein structure prediction (CASP)-Round XII}},
url = {http://doi.wiley.com/10.1002/prot.25415},
volume = {86},
year = {2018}
}

**NetSurfP2.0: (Secondary Structure)**
>@article{netsurfp,
  title={NetSurfP-2.0: Improved prediction of protein structural features by integrated deep learning},
  author={Klausen, Michael Schantz and Jespersen, Martin Closter and Nielsen, Henrik and Jensen, Kamilla Kjaergaard and Jurtz, Vanessa Isabell and Soenderby, Casper Kaae and Sommer, Morten Otto Alexander and Winther, Ole and Nielsen, Morten and Petersen, Bent and others},
  journal={Proteins: Structure, Function, and Bioinformatics},
  year={2019},
  publisher={Wiley Online Library}
}

**ProteinNet: (Contact)**
>@article{proteinnet,
  title={ProteinNet: a standardized data set for machine learning of protein structure},
  author={AlQuraishi, Mohammed},
  journal={arXiv preprint arXiv:1902.00249},
  year={2019}
}

**Fluorescence:**
>@article{sarkisyan2016,
  title={Local fitness landscape of the green fluorescent protein},
  author={Sarkisyan, Karen S and Bolotin, Dmitry A and Meer, Margarita V and Usmanova, Dinara R and Mishin, Alexander S and Sharonov, George V and Ivankov, Dmitry N and Bozhanova, Nina G and Baranov, Mikhail S and Soylemez, Onuralp and others},
  journal={Nature},
  volume={533},
  number={7603},
  pages={397},
  year={2016},
  publisher={Nature Publishing Group}
}

**Stability:**
>@article{rocklin2017,
  title={Global analysis of protein folding using massively parallel design, synthesis, and testing},
  author={Rocklin, Gabriel J and Chidyausiku, Tamuka M and Goreshnik, Inna and Ford, Alex and Houliston, Scott and Lemak, Alexander and Carter, Lauren and Ravichandran, Rashmi and Mulligan, Vikram K and Chevalier, Aaron and others},
  journal={Science},
  volume={357},
  number={6347},
  pages={168--175},
  year={2017},
  publisher={American Association for the Advancement of Science}
}
