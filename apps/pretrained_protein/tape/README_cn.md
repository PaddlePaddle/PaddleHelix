# TAPE

[中文版本](./README_ch.md) [English Version](./README.md)

- [TAPE](#tape)
  - [背景介绍](#背景介绍)
  - [使用说明](#使用说明)
    - [模型训练](#模型训练)
      - [CPU多线程/GPU单卡训练](#cpu多线程gpu单卡训练)
      - [GPU多机多卡训练](#gpu多机多卡训练)
    - [模型评估](#模型评估)
    - [模型预测](#模型预测)
    - [序列模型](#序列模型)
      - [Transformer](#transformer)
      - [LSTM](#lstm)
      - [ResNet](#resnet)
      - [其他参数](#其他参数)
    - [蛋白质相关任务](#蛋白质相关任务)
      - [预训练任务](#预训练任务)
        - [Pfam](#pfam)
      - [下游任务](#下游任务)
      - [Secondary Structure](#secondary-structure)
        - [评估结果](#评估结果)
      - [Remote Homology](#remote-homology)
        - [评估结果](#评估结果-1)
      - [Fluorescence](#fluorescence)
        - [评估结果](#评估结果-2)
      - [Stability](#stability)
        - [评估结果](#评估结果-3)
    - [热启动/Finetuning](#热启动finetuning)
    - [完整样例](#完整样例)
  - [数据](#数据)
  - [预训练模型](#预训练模型)
  - [引用](#引用)
    - [论文相关](#论文相关)
    - [数据相关](#数据相关)

## 背景介绍
近年来，随着测序技术的发展，蛋白质序列的数据库的规模大幅度增长。然而，带标签的蛋白质序列的获取代价依然十分昂贵，因为它们因为它们需要通过通过生物实验才能获取。此外，由于带标签的样本量非常不充足，模型很容易对数据过拟合(overfit)。借鉴自然语言处理(Natural Language Processing, NLP)中对大量未标记的序列使用自监督学习(self-supervised learning)的方式进行预训练(pre-training)，从而提取蛋白质中有用的生物学信息，并将这些信息迁移到其他带标签的任务，使得这些任务训练更快更稳定的收敛。本篇蛋白质预训练模型参考论文TAPE，提供Transformer，LSTM和ResNet的模型实现。

## 使用说明

### 模型训练

我们提供多种训练方式：

- CPU多线程训练。
- GPU单卡训练。
- GPU多机多卡训练。

#### CPU多线程/GPU单卡训练
使用CPU多线程训练/GPU单卡训练的例子及相关参数解释如下：
```bash
python train.py \
        --train_data ./train_data # 训练数据目录，包含多个训练数据文件，用于训练模型。\
        --valid_data ./valid_data # 验证数据目录，包含多个验证数据文件，用于在训练过程中评估模型。\
        --lr 0.0001 # 基准的学习率。 \
        --use_cuda # 是否使用GPU训练。 \
        ... # 设定模型参数和任务参数，将在后续章节介绍。
```

#### GPU多机多卡训练
使用paddle的分布式训练paddle.distributed.launch来启动任务，此外使用分布式训练需要添加参数"--distributed"，其他参数与CPU多线程训练/GPU单卡训练一致。使用GPU多机多卡训练的例子及相关超参数解释如下：

```bash
python -m paddle.distributed.launch --log_dir log_dir train.py # paddle分布式训练通过log_dir指定运行日志的目录 \
        --train_data ./train_data # 训练数据目录，包含多个训练数据文件，用于训练模型。\
        --valid_data ./valid_data # 验证数据目录，包含多个验证数据文件，用于在训练过程中评估模型。\
        --lr 0.0001 # 基准的学习率。 \
        --use_cuda # 模型在cpu还是gpu中运行。目前分布式版本只支持gpu。 \
        --distributed # 分布式运行。 \
        ... # 设定模型参数和任务参数，将在后续章节介绍。
```

### 模型评估
模型评估和模型训练方式类似，目前只支持CPU多线程/GPU单卡评估。
```bash
python eval.py \
        --eval_data ./eval_data # 测试数据目录，包含多个测试数据文件，用于评估模型。 \
        --eval_model ./model # 待评估的模型。 \
        --use_cuda # 模型在cpu还是gpu中运行。 \
        ... # 设定模型参数和任务参数，将在后续章节介绍。
```

### 模型预测
模型预测和模型评估方式类似，目前只支持CPU多线程/GPU单卡评估。
```bash
cat predict_file | # 预测的文件，每一行是一个氨基酸序列。氨基酸都是用单字母表示。 \
python predict.py \
        --batch_size 128 # batch大小的上界。由于蛋白质序列的长度差异比较大，模型根据序列的长短动态调整batch大小。 \
        --model ./model # 待评估的模型。 \
        --use_cuda # 模型在cpu还是gpu中运行。 \
        ... # 设定模型参数和任务参数，将在后续章节介绍。
```

### 序列模型
我们提供了Transformer，LSTM和ResNet三种序列模型。我们通过参数model_config设定对应模型的超参数。model_config中通过设置model_type指定模型类型(transformer, lstm, resnet)。

```bash
python train.py \
        ... # 设定训练参数，已在上面章节介绍。 \
        --model_config ./transformer_config # 模型配置文件，使用json方式组织，通过配置文件设置模型相关参数。 \
        ... # 设定任务参数，将在后续章节介绍。
```

#### Transformer
Transformer常用在自然语言处理的语义建模中。使用Transformer需要设定以下超参数：

- hidden_size: transformer的hidden_size。
- layer_num: transformer的层数。
- head_num: transformer每一层的header数量。

Transformer的详细介绍可参考以下文章：

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)


#### LSTM
我们使用多层双向LSTM。使用LSTM我们需要设定以下超参数：

- hidden_size: lstm的hidden_size。
- layer_num: lstm的层数。

LSTM可以参考以下文章：

- [Long Short Term Memory](http://didawiki.di.unipi.it/lib/exe/fetch.php/magistraleinformatica/aa2/lstm.pdf)


#### ResNet
我们使用多层ResNet。使用ResNet我们需要设定以下超参数：

- hidden_size: Resnet的hidden_size。
- layer_num: resnet的层数。
- filter_num: 卷积层的filter数量。

ResNet可以参考以下文章：

- [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)


#### 其他参数
模型配置文件还可以设置其他参数避免过拟合和模型参数值过大。

- dropout: 模型参数dropout的比例。
- weight_decay: 参数衰减比例，用于避免参数值过大。

### 蛋白质相关任务
参考论文TAPE，我们使用PaddleHelix复现了以下任务。
#### 预训练任务
##### Pfam
Pfam包含三千万蛋白质序列，可用于预训练模型。在模型配置中需要设定一下参数：
```bash
...
task: "pretrain",
...
```

#### 下游任务
#### Secondary Structure
Secondary structure包含两个序列标注任务，一个3分类任务和一个8分类任务。在模型配置中需要设定一下参数：
```bash
...
task: "seq_classification",
class_num: 3,
label_name: "labels3",
...
```

##### 评估结果
使用pfam任务预训练的模型对下游任务finetuning后的结果如下表。在模型配置中需要设定一下参数：

Three-way Accuracy：

| Model         | CB513     | CASP12    | TS115 |   
| :--:          | :--:      | :--:      | :--:  |
| Transformer   | 0.732     | 0.717     | 0.771 |
| LSTM          | 0.758     | 0.711     | 0.785 |
| ResNet        | 0.747     | 0.713     | 0.777 |

Eight-way Accuracy：

| Model         | CB513     | CASP12    | TS115 |   
| :--:          | :--:      | :--:      | :--:  |
| Transformer   | 0.586     | 0.584     | 0.649 |
| LSTM          | 0.617     | 0.589     | 0.667 |
| ResNet        | 0.606     | 0.587     | 0.661 |

#### Remote Homology
Remote homology是一个多分类任务，包含1195个类。在模型配置中需要设定一下参数：
```bash
...
task: "classification",
class_num: 1195,
label_name: "labels",
...
```

##### 评估结果
使用pfam任务预训练的模型对下游任务finetuning后的结果如下表。

Accuracy：

| Model         | Fold  | Superfamily   | Family    |   
| :--:          | :--:  | :--:          | :--:      |
| Transformer   | 0.247 | 0.355         | 0.886     |
| LSTM          | 0.219 | 0.314         | 0.820     |
| ResNet        | 0.153 | 0.136         | 0.535     |


#### Fluorescence
Fluorescence是一个回归任务。在模型配置中需要设定一下参数：
```bash
...
task: "regression",
label_name: "labels",
...
```

##### 评估结果
使用pfam任务预训练的模型对下游任务finetuning后的结果如下表。

Spearman：

| Model         | Test      |
| :--:          | :--:      |
| Transformer   | 0.680     | 
| LSTM          | 0.533     | 
| ResNet        | 0.573     |

#### Stability
Stability是一个回归任务。在模型配置中需要设定一下参数：
```bash
...
task: "regression",
label_name: "labels",
...
```

##### 评估结果
使用pfam任务预训练的模型对下游任务finetuning后的结果如下表。

Spearman：

| Model         | Test      |
| :--:          | :--:      |
| Transformer   | 0.807     | 
| LSTM          | 0.785     | 
| ResNet        | 0.792     |

### 热启动/Finetuning
在训练时，通过设置参数"--init_model"设置初始化模型，用于热启动训练模型，或finetune下游任务。
```bash
python train.py \
        ... \
        --init_model ./init_model # 初始化模型目录。如果不设定该参数，则模型冷启动训练。 \
        --hot_start "hot_start" # 热启动训练设置成"hot_start", finetune下游任务设置成"finetune" \
        ... 
```

### 完整样例
我们在demos文件夹内提供多个训练和评估样例。以下是transformer预训练的样例。
```bash

#!/bin/bash
model_type="transformer" # candidate model_types: transformer, lstm, resnet
task="secondary_structure" # candidate tasks: pfam, secondary_structure, remote_homology, fluorescence, stability
model_config="./configs/${model_type}_${task}_config.json"
train_data="./secondary_structure_toy_data/"
valid_data="./secondary_structure_toy_data/"

export PYTHONPATH="../../../"
python train.py \
        --train_data ${train_data} \
        --valid_data ${valid_data} \
        --model_config ${model_config} \

```

以下是模型配置的完整样例：
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

## 数据
数据集可以从以下链接获取：
pfam: [raw](https://baidu-nlp.bj.bcebos.com/PaddleHelix%2Fdatasets%2Fprotein_datasets%2Fpfam.npz.tgz), [npz](https://baidu-nlp.bj.bcebos.com/PaddleHelix%2Fdatasets%2Fprotein_datasets%2Fpfam.npz.tgz)
secondary structure: [all](https://baidu-nlp.bj.bcebos.com/PaddleHelix%2Fdatasets%2Fprotein_datasets%2Fsecondary_structure.tgz)
remote homology: [all](https://baidu-nlp.bj.bcebos.com/PaddleHelix%2Fdatasets%2Fprotein_datasets%2Fremote_homology.tgz)
fluorescence: [all](https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/protein_datasets/fluorescence.tgz)
stability: [all](https://baidu-nlp.bj.bcebos.com/PaddleHelix%2Fdatasets%2Fprotein_datasets%2Fstability.tgz)

## 预训练模型
预训练模型可以从以下链接获取：
Transformer: [model](https://baidu-nlp.bj.bcebos.com/PaddleHelix/pretrained_models/protein/tape_transformer_pretrain.pdparams)
LSTM: [model](https://baidu-nlp.bj.bcebos.com/PaddleHelix/pretrained_models/protein/tape_lstm_pretrain.pdparams)
ResNet: [model](https://baidu-nlp.bj.bcebos.com/PaddleHelix/pretrained_models/protein/tape_resnet_pretrain.pdparam)

## 引用
### 论文相关
本篇蛋白质预训练方法主要参考论文**TAPE**，部分训练方式，训练超参数略有不同。

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

### 数据相关
本篇蛋白质预训练方法使用论文**TAPE**中的数据集进行进一步处理。

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
