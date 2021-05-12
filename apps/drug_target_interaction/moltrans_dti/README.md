# MolTrans

[中文版本](./README_cn.md) [English Version](./README.md)

* [Background](#background)
* [Datasets](#datasets)
    * [DAVIS](#davis)
    * [KIBA](#kiba)
    * [BioSNAP](#biosnap)
    * [BindingDB](#bindingdb)
    * [ChEMBL](#chembl)
* [Instructions](#instructions)
    * [Dependencies installation](#dependencies-installation)
    * [Model configuration](#model-configuration)
    * [Training and evaluation](#training-and-evaluation)
* [Reference](#reference)

## Background

In the process of in-silico drug discovery, drug target interaction(DTI) prediction plays a fundamental role. However, it is not an easy task due to time-consuming and costly experimental search over large drug compound space. Recent years have witnessed rapid progress for deep learning in DTI predictions. Among all of the methods, MolTrans shows impressive performance on both classification and regression tasks in DTI compared to state-of-the-art baselines. It leverages an augmented Transformer encoder to capture abundant semantic relations among substructures extracted from massive unlabeled biomedical data. Besides, it makes use of knowledge inspired substructural pattern mining algorithm and interaction modeling module for more accurate and interpretable DTI prediction.

## Datasets

In order to run the experiments, you need to download all related datasets and put them under `/apps/drug_target_interaction/moltrans_dti/`. If you don't have `wget`, you could also copy the url below into your web browser to download them.

```sh
cd /apps/drug_target_interaction/moltrans_dti/
wget "https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/dti_datasets/dti_dataset.tgz" --no-check-certificate
tar -zxvf "dti_dataset.tgz"
```

After downloading the datasets, the `dataset` folder looks like:

```txt
dataset
|-- classification
|   |-- BindingDB
|   |   |-- test.csv
|   |   |-- train.csv
|   |   `-- val.csv
|   |-- BIOSNAP
|   |   |-- full_data
|   |   |   |-- test.csv
|   |   |   |-- train.csv
|   |   |   `-- val.csv
|   |   |-- missing_data
|   |   |   |-- 70
|   |   |   |-- 80
|   |   |   |-- 90
|   |   |   `-- 95
|   |   |-- unseen_drug
|   |   |   |-- test.csv
|   |   |   |-- train.csv
|   |   |   `-- val.csv
|   |   `-- unseen_protein
|   |       |-- test.csv
|   |       |-- train.csv
|   |       `-- val.csv
|   `-- DAVIS
|       |-- test.csv
|       |-- train.csv
|       `-- val.csv
`-- regression
    |-- benchmark
    |   |-- DAVIStest
    |   `-- KIBAtest
    |-- BindingDB
    |   |-- bindingDB cleanup.R
    |   |-- BindingDB_Kd.txt
    |   |-- BindingDB_SMILES_new.txt
    |   |-- BindingDB_SMILES.txt
    |   |-- BindingDB_Target_Sequence_new.txt
    |   `-- BindingDB_Target_Sequence.txt
    |-- ChEMBL
    |   |-- Chem_Affinity.txt
    |   |-- Chem_Kd_nM.txt
    |   |-- Chem_SMILES_only.txt
    |   |-- Chem_SMILES.txt
    |   |-- ChEMBL cleanup.R
    |   `-- ChEMBL_Target_Sequence.txt
    |-- DAVIS
    |   |-- affinity.txt
    |   |-- SMILES.txt
    |   `-- target_seq.txt
    `-- KIBA
        |-- affinity.txt
        |-- SMILES.txt
        `-- target_seq.txt
```

In the original work of MolTrans, it only contains classification task. Here we provide datasets for both classification and regression tasks of DTI.

### DAVIS

DAVIS contains the binding affinities of 72 kinase inhibitors with 442 kinases covering >80% of the human catalytic protein kinome, measured as Kd constant (equilibrium dissociation constant). The smaller the Kd value, the greater the binding affinity of the drug for its target.

### KIBA

KIBA contains the binding affinities for 2,116 drugs and 229 targets. Comparing to DAVIS, some drug-target pairs do not have affinity labels. Moreover, the affinity in KIBA is measured as KIBA scores, which were constructed to optimize the consistency among Ki, Kd, and IC50 by utilizing the statistical information they contained.

### BioSNAP

BioSNAP contains many large biomedical networks that are ready-to-use for method development, algorithm evaluation, benchmarking, and network science analyses. It is a collection diverse biomedical networks, including protein-protein interaction networks, single-cell similarity networks, drug-drug interaction networks.

### BindingDB

BindingDB is a publicly accessible database containing approximately 20,000 experimentally determined binding affinities of protein-ligand complexes, for 110 protein targets including isoforms and mutational variants, and approximately 11,000 small molecule ligands.

### ChEMBL

ChEMBL is an open large-scale bioactivity database. In total, there are >1.6 million distinct compound structures represented in the database, with 14 million activity values from >1.2 million assays. These assays are mapped to ∼11 000 targets, including 9052 proteins (of which 4255 are human).

## Instructions

### Dependencies installation

Before playing, you need to install all dependencies and packages indicated in `/apps/drug_target_interaction/moltrans_dti/requirement.txt/`.

| name         | version |
| ------------ | ---- |
| numpy        | - |
| pandas       | - |
| paddlepaddle | \>=2.0.0rc0 |
| subword-nmt  | \>=0.3.7 |
| scipy        | - |
| scikit-learn | - |
| visualdl     | - |
| PyYAML       | - |

('-' means no specific version requirement for that package)

### Model configuration

The script `double_towers.py` under `/apps/drug_target_interaction/moltrans_dti/` describes the detailed structure of MolTrans model. The model configurations, which is in `config.json`, mostly match the original paper are following:

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

### Training and evaluation

For **classification** task, you can just run the script `train_cls.py`. If you want to try other datasets, just select one from `cls_davis`, `cls_biosnap`, `cls_bindingdb`. The basic usage with default settings is following:

```sh
CUDA_VISIBLE_DEVICES=0 python train_cls.py --batchsize 64 --epochs 200 --lr 5e-4 --dataset cls_davis
```

For evaluation, we use AUROC as a standard metric for the classification task. The larger AUROC, the better. The comparisons of different methods on DAVIS are following:

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

For **regression** task, you can just run the script `train_reg.py`. If you want to try other datasets, just select one from `raw_chembl_pkd`, `raw_chembl_kd`, `raw_bindingdb_kd`, `raw_davis`, `raw_kiba`, `benchmark_davis`, `benchmark_kiba`. The basic usage with default settings is following:

```sh
CUDA_VISIBLE_DEVICES=0 python train_reg.py --batchsize 64 --epochs 200 --lr 5e-4 --dataset benchmark_davis
```

For evaluation, we use MSE and concordance index(CI) as standard metrics for the regression task. The smaller MSE, the better. While the larger CI, the better. 

The comparisons of different methods on DAVIS are following:

| Methods           | MSE        | CI        |
| :--:              | :--:       | :--:      |
| WideDTA           | 0.262      | 0.886     |
| GraphDTA_GIN      | 0.229      | 0.893     |
| GraphDTA+pretrain | 0.225      | 0.899     |
| DGraphDTA         | 0.202      | 0.904     |
| Ours(MolTrans)    | 0.199      | 0.901     |

The comparisons of different methods on KIBA are following:

| Methods           | MSE        | CI        |
| :--:              | :--:       | :--:      |
| WideDTA           | 0.179      | 0.875     |
| GraphDTA_GIN      | 0.139      | 0.891     |
| GraphDTA+pretrain | 0.135      | 0.896     |
| DGraphDTA         | 0.126      | 0.904     |
| Ours(MolTrans)    | 0.132      | 0.898     |

## Reference

**MolTrans**
>
@article{10.1093/bioinformatics/btaa880,
    author = {Huang, Kexin and Xiao, Cao and Glass, Lucas M and Sun, Jimeng},
    title = "{MolTrans: Molecular Interaction Transformer for drug–target interaction prediction}",
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

**DAVIS**
>
@article{10.1038/nbt.1990,
    author = {Mindy I Davis, Jeremy P Hunt, Sanna Herrgard, Pietro Ciceri, Lisa M Wodicka, Gabriel Pallares, Michael Hocker, Daniel K Treiber and Patrick P Zarrinkar},
    title = "{Comprehensive analysis of kinase inhibitor selectivity}",
    journal = {Nature Biotechnology},
    year = {2011},
    url = {https://doi.org/10.1038/nbt.1990}
}

**KIBA**
>
@article{doi:10.1021/ci400709d,
    author = {Tang, Jing and Szwajda, Agnieszka and Shakyawar, Sushil and Xu, Tao and Hintsanen, Petteri and Wennerberg, Krister and Aittokallio, Tero},
    title = "{Making Sense of Large-Scale Kinase Inhibitor Bioactivity Data Sets: A Comparative and Integrative Analysis}",
    journal = {Journal of Chemical Information and Modeling},
    volume = {54},
    number = {3},
    pages = {735-743},
    year = {2014},
    doi = {10.1021/ci400709d},
    note = {PMID: 24521231},
    url = {https://doi.org/10.1021/ci400709d}
}

**BioSNAP**
>
@misc{biosnapnets,
    author = {Marinka Zitnik, Rok Sosi\v{c}, Sagar Maheshwari, and Jure Leskovec},
    title = "{{BioSNAP Datasets}: {Stanford} Biomedical Network Dataset Collection}",
    howpublished = {\url{http://snap.stanford.edu/biodata}},
    month = aug,
    year = 2018
}

**BindingDB**
>
@article{10.1093/nar/gkv1072,
    author = {Gilson, Michael K. and Liu, Tiqing and Baitaluk, Michael and Nicola, George and Hwang, Linda and Chong, Jenny},
    title = "{BindingDB in 2015: A public database for medicinal chemistry, computational chemistry and systems pharmacology}",
    journal = {Nucleic Acids Research},
    volume = {44},
    number = {D1},
    pages = {D1045-D1053},
    year = {2015},
    month = {10},
    issn = {0305-1048},
    doi = {10.1093/nar/gkv1072},
    url = {https://doi.org/10.1093/nar/gkv1072},
}

**ChEMBL**
>
@article{10.1093/nar/gkw1074,
    author = {Gaulton, Anna and Hersey, Anne and Nowotka, Michał and Bento, A. Patrícia and Chambers, Jon and Mendez, David and Mutowo, Prudence and Atkinson, Francis and Bellis, Louisa J. and Cibrián-Uhalte, Elena and Davies, Mark and Dedman, Nathan and Karlsson, Anneli and Magariños, María Paula and Overington, John P. and Papadatos, George and Smit, Ines and Leach, Andrew R.},
    title = "{The ChEMBL database in 2017}",
    journal = {Nucleic Acids Research},
    volume = {45},
    number = {D1},
    pages = {D945-D954},
    year = {2016},
    month = {11},
    doi = {10.1093/nar/gkw1074},
    url = {https://doi.org/10.1093/nar/gkw1074}
}