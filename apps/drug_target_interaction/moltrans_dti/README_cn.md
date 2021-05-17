# MolTrans

[中文版本](./README_cn.md) [English Version](./README.md)

* [背景介绍](#背景介绍)
* [数据集](#数据集)
    * [DAVIS](#davis)
    * [KIBA](#kiba)
    * [BioSNAP](#biosnap)
    * [BindingDB](#bindingdb)
    * [ChEMBL](#chembl)
* [使用说明](#使用说明)
    * [环境安装](#环境安装)
    * [参数设置](#参数设置)
    * [训练与评估](#训练与评估)
* [引用](#引用)

## 背景介绍

在计算药物发现的进程中，药物和靶点蛋白亲和性（DTI）的预测是至关重要的一环。然而，由于在大量的药理化合物中进行实验搜索非常耗时且十分昂贵，这并不是一件轻而易举的事情。近年来，深度学习在DTI预测任务中取得了快速发展。在所有这些方法中，与SOTA基准相比，MolTrans在DTI分类和回归任务上均表现出令人印象深刻的性能。它利用增强型Transformer编码器来捕获从海量未标注的生物医学数据中提取的子结构之间的丰富语义关系。此外，它利用受知识启发的子结构模式挖掘算法和交互建模模块来进行更精确和具解释性的DTI预测任务。

## 数据集

为了能顺利进行后续实验，首先需要下载所有相关的数据集并放置在`/apps/drug_target_interaction/moltrans_dti/`路径下。如果电脑上没有安装`wget`，可以将下面链接复制到浏览器直接下载。

```sh
cd /apps/drug_target_interaction/moltrans_dti/
wget "https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/dti_datasets/dti_dataset.tgz" --no-check-certificate
tar -zxvf "dti_dataset.tgz"
```

数据集下载完成后，`dataset`目录如下所示：

```txt
dataset
├── classification
│   ├── BindingDB
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── val.csv
│   ├── BIOSNAP
│   │   ├── full_data
│   │   │   ├── test.csv
│   │   │   ├── train.csv
│   │   │   └── val.csv
│   │   ├── missing_data
│   │   │   ├── 70
│   │   │   ├── 80
│   │   │   ├── 90
│   │   │   └── 95
│   │   ├── unseen_drug
│   │   │   ├── test.csv
│   │   │   ├── train.csv
│   │   │   └── val.csv
│   │   └── unseen_protein
│   │       ├── test.csv
│   │       ├── train.csv
│   │       └── val.csv
│   └── DAVIS
│       ├── test.csv
│       ├── train.csv
│       └── val.csv
└── regression
    ├── benchmark
    │   ├── DAVIStest
    │   └── KIBAtest
    ├── BindingDB
    │   ├── bindingDB cleanup.R
    │   ├── BindingDB_Kd.txt
    │   ├── BindingDB_SMILES_new.txt
    │   ├── BindingDB_SMILES.txt
    │   ├── BindingDB_Target_Sequence_new.txt
    │   └── BindingDB_Target_Sequence.txt
    ├── ChEMBL
    │   ├── Chem_Affinity.txt
    │   ├── Chem_Kd_nM.txt
    │   ├── Chem_SMILES_only.txt
    │   ├── Chem_SMILES.txt
    │   ├── ChEMBL cleanup.R
    │   └── ChEMBL_Target_Sequence.txt
    ├── DAVIS
    │   ├── affinity.txt
    │   ├── SMILES.txt
    │   └── target_seq.txt
    └── KIBA
        ├── affinity.txt
        ├── SMILES.txt
        └── target_seq.txt
```

在MolTrans原工作中，只包含了分类任务。我们提供了DTI分类和回归任务的数据集与代码。

### DAVIS

DAVIS包含72种激酶抑制剂与442种激酶的亲和性数据，以Kd常数（平衡解离常数）衡量，该数据集覆盖了人类催化蛋白激酶组的80％以上。 Kd值越小，药物和其靶标蛋白之间的亲和性越大。

### KIBA

KIBA包含2,116种药物和229种靶标蛋白及其亲和性数据。不同于DAVIS，部分药物和靶标蛋白没有亲和性指标。另外，KIBA中使用KIBA分数作为亲和性的评估指标。KIBA分数提供了一种基于统计分析的归一化不同亲和性指标（Ki、Kd、IC50）的方法。

### BioSNAP

BioSNAP包含许多现成的大型生物医学网络，可用于方法开发，算法评估，基准测试和网络分析。 它是一个多样化的生物医学网络集合，包括蛋白质-蛋白质相互作用网络，单细胞相似性网络，药物-药物相互作用网络等。

### BindingDB

BindingDB是一个开源的数据库，其中包含约20,000种通过实验得到的蛋白质-配体复合物及其亲和性数据，其中包含110个靶标蛋白质（包括同工型和突变变体）和约11,000个小分子配体。

### ChEMBL

ChEMBL是一个开源的大规模生物活性数据库。该数据库中包含超过160万种不同的化合物结构，以及用超过120万种测定方法得到的约1,400万个亲和性数据。这些测定方法定位到约11,000个靶标蛋白，其中包括9,052种蛋白质（其中4,255种属于人类）。

## 使用说明

### 环境安装

在运行实验之前，还需要安装所有在`/apps/drug_target_interaction/moltrans_dti/requirement.txt/`文件中需求的包和工具。

| 名字         | 版本 |
| ------------ | ---- |
| numpy        | - |
| pandas       | - |
| paddlepaddle | \>=2.0.0rc0 |
| subword-nmt  | \>=0.3.7 |
| scipy        | - |
| scikit-learn | - |
| visualdl     | - |
| PyYAML       | - |

('-' 代表没有版本要求)

### 参数设置

在`/apps/drug_target_interaction/moltrans_dti/`目录下的Python脚本`double_towers.py`描述了MolTrans模型的详细结构。模型参数存储在`config.json`文件内，参数设置上最接近原论文的是：

```txt
    "drug_max_seq": 50,               # Max length of drug sequence
    "target_max_seq": 545,            # Max length of protein sequence
    "emb_size": 384,                  # Embedding size
    "input_drug_dim": 23532,          # Length of drug vocabulary
    "input_target_dim": 16693,        # Length of protein vocabulary
    "interm_size": 1536,              # Latent size
    "num_attention_heads": 12,        # Number of attention heads
    "flatten_dim": 81750,             # Flatten size 
    "layer_size": 2,                  # Layer size of transformer blocks
    "dropout_ratio": 0.1,             # Dropout rate
    "attention_dropout_ratio": 0.1,   # Dropout rate within attention
    "hidden_dropout_ratio": 0.1       # Dropout rate within hidden states
```

### 训练与评估

对于**分类**任务，可以直接运行`train_cls.py`。如果想尝试其他分类任务的数据集，可以直接在`cls_davis`、`cls_biosnap`、`cls_bindingdb`中选择一个。带默认参数的基本用法如下：

```sh
CUDA_VISIBLE_DEVICES=0 python train_cls.py --batchsize 64 --epochs 200 --lr 5e-4 --dataset cls_davis
```

进行评估时，我们使用AUROC作为分类任务的指标。AUROC越大，模型的预测性能越好。在DAVIS上不同方法的实验结果如下：

| Methods        |  AUROC      |
| :--:           |  :--:       |
| LR             | 0.835±0.010 |
| DNN            | 0.864±0.009 |
| GNN-CPI        | 0.840±0.012 |
| DeepDTI        | 0.861±0.002 |
| DeepDTA        | 0.880±0.007 |
| DeepConv-DTI   | 0.884±0.008 |
| MolTrans       | 0.907±0.002 |
| Ours(MolTrans) | 0.912±0.002 |

对于**回归**任务，可以直接运行`train_reg.py`。如果想尝试其他回归任务的数据集，可以直接在`raw_chembl_pkd`、`raw_chembl_kd`、`raw_bindingdb_kd`、`raw_davis`、`raw_kiba`、`benchmark_davis`、`benchmark_kiba`中选择一个。带默认参数的基本用法如下：

```sh
CUDA_VISIBLE_DEVICES=0 python train_reg.py --batchsize 64 --epochs 200 --lr 5e-4 --dataset benchmark_davis
```

进行评估时，我们使用MSE和一致性指数（CI）作为回归任务的指标。MSE越小，CI越大，模型的预测性能越好。

在DAVIS上不同方法的实验结果如下：

| Methods           | MSE        | CI        |
| :--:              | :--:       | :--:      |
| WideDTA           | 0.262      | 0.886     |
| GraphDTA_GIN      | 0.229      | 0.893     |
| GraphDTA+pretrain | 0.225      | 0.899     |
| DGraphDTA         | 0.202      | 0.904     |
| Ours(MolTrans)    | 0.199      | 0.901     |

在KIBA上不同方法的实验结果如下：

| Methods           | MSE        | CI        |
| :--:              | :--:       | :--:      |
| WideDTA           | 0.179      | 0.875     |
| GraphDTA_GIN      | 0.139      | 0.891     |
| GraphDTA+pretrain | 0.135      | 0.896     |
| DGraphDTA         | 0.126      | 0.904     |
| Ours(MolTrans)    | 0.132      | 0.898     |

## 引用

**MolTrans**
```
@article{10.1093/bioinformatics/btaa880,
    author = {Huang, Kexin and Xiao, Cao and Glass, Lucas M and Sun, Jimeng},
    title = {MolTrans: Molecular Interaction Transformer for drug–target interaction prediction},
    journal = {Bioinformatics},
    volume = {37},
    number = {6},
    pages = {830-836},
    year = {2020},
    month = {10},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btaa880},
    url = {https://doi.org/10.1093/bioinformatics/btaa880}
}
```

**DAVIS**
```
@article{10.1038/nbt.1990,
    author = {Mindy I Davis, Jeremy P Hunt, Sanna Herrgard, Pietro Ciceri, Lisa M Wodicka, Gabriel Pallares, Michael Hocker, Daniel K Treiber and Patrick P Zarrinkar},
    title = {Comprehensive analysis of kinase inhibitor selectivity},
    journal = {Nature Biotechnology},
    year = {2011},
    url = {https://doi.org/10.1038/nbt.1990}
}
```

**KIBA**
```
@article{doi:10.1021/ci400709d,
    author = {Tang, Jing and Szwajda, Agnieszka and Shakyawar, Sushil and Xu, Tao and Hintsanen, Petteri and Wennerberg, Krister and Aittokallio, Tero},
    title = {Making Sense of Large-Scale Kinase Inhibitor Bioactivity Data Sets: A Comparative and Integrative Analysis},
    journal = {Journal of Chemical Information and Modeling},
    volume = {54},
    number = {3},
    pages = {735-743},
    year = {2014},
    doi = {10.1021/ci400709d},
    note = {PMID: 24521231},
    url = {https://doi.org/10.1021/ci400709d}
}
```

**BioSNAP**
```
@misc{biosnapnets,
    author = {Marinka Zitnik, Rok Sosi\v{c}, Sagar Maheshwari, and Jure Leskovec},
    title = {BioSNAP Datasets: Stanford Biomedical Network Dataset Collection},
    month = aug,
    year = 2018,
    url = {http://snap.stanford.edu/biodata}
}
```

**BindingDB**
```
@article{10.1093/nar/gkv1072,
    author = {Gilson, Michael K. and Liu, Tiqing and Baitaluk, Michael and Nicola, George and Hwang, Linda and Chong, Jenny},
    title = {BindingDB in 2015: A public database for medicinal chemistry, computational chemistry and systems pharmacology},
    journal = {Nucleic Acids Research},
    volume = {44},
    number = {D1},
    pages = {D1045-D1053},
    year = {2015},
    month = {10},
    issn = {0305-1048},
    doi = {10.1093/nar/gkv1072},
    url = {https://doi.org/10.1093/nar/gkv1072}
}
```

**ChEMBL**
```
@article{10.1093/nar/gkw1074,
    author = {Gaulton, Anna and Hersey, Anne and Nowotka, Michał and Bento, A. Patrícia and Chambers, Jon and Mendez, David and Mutowo, Prudence and Atkinson, Francis and Bellis, Louisa J. and Cibrián-Uhalte, Elena and Davies, Mark and Dedman, Nathan and Karlsson, Anneli and Magariños, María Paula and Overington, John P. and Papadatos, George and Smit, Ines and Leach, Andrew R.},
    title = {The ChEMBL database in 2017},
    journal = {Nucleic Acids Research},
    volume = {45},
    number = {D1},
    pages = {D945-D954},
    year = {2016},
    month = {11},
    doi = {10.1093/nar/gkw1074},
    url = {https://doi.org/10.1093/nar/gkw1074}
}
```