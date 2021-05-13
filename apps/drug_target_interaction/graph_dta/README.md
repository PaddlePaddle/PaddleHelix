# GraphDTA

[中文版本](./README_cn.md) [English Version](./README.md)

* [Background](#background)
* [Datasets](#datasets)
    * [Davis](#davis)
    * [Kiba](#kiba)
* [Instructions](#instructions)
    * [Configuration](#configuration)
    * [Training and Evaluation](#train-and-evaluation)
* [Reference](#reference)

## Background
Knowing which proteins are targeted by which drugs is very useful for new drug design, drug repurposing etc. GraphDTA is a model which represents drugs as graphs and uses graph neural networks to predict drug-target affinity.

## Datasets

First, let us create a dataset root folder `data` under this application folder.

```sh
mkdir -p data && cd data
```

### Davis

Davis contains the binding affinities for all pairs of 72 drugs and 442 targets, measured as Kd constant (equilibrium dissociation constant). The smaller the Kd value, the greater the binding affinity of the drug for its target. You can download and uncompress this dataset using following command:

```sh
wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/dti_datasets/davis_v1.tgz -O davis.tgz
tar -zxvf davis.tgz
```

### Kiba

Kiba contains the binding affinity for 2,116 drugs and 229 targets. Comparing to Davis, some drug-target pairs do not have affinity labels. Moreover, the affinity in Kiba is measured as KIBA scores, which were constructed to optimize the consistency between Ki, Kd, and IC50 by utilizing the statistical information they contained. You can download and uncompress this dataset using following command:

```sh
wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/dti_datasets/kiba_v1.tgz -O kiba.tgz
tar -zxvf kiba.tgz
```

Then, you can redirect to this application folder and follow instructions to finish next steps.

After downloaed these datasets, the `data` folder looks like:

```txt
data
├── davis
│   ├── folds
│   │   ├── test_fold_setting1.txt
│   │   └── train_fold_setting1.txt
│   ├── ligands_can.txt
│   ├── processed
│   │   ├── test
│   │   │   └── davis_test.npz
│   │   └── train
│   │       └── davis_train.npz
│   ├── proteins.txt
│   └── Y
├── davis.tgz
├── kiba
│   ├── folds
│   │   ├── test_fold_setting1.txt
│   │   └── train_fold_setting1.txt
│   ├── ligands_can.txt
│   ├── processed
│   │   ├── test
│   │   │   └── kiba_test.npz
│   │   └── train
│   │       └── kiba_train.npz
│   ├── proteins.txt
│   └── Y
└── kiba.tgz
```

## Instructions

### Configuration

The script `scripts/train.py` is the entry for GraphDTA model. It creats `DTAModel` from `src/model.py`. Since the GraphDTA processes protein sequences into new sequences with fixed length, the model configurations mostly match the original paper are:

* `model_configs/fix_prot_len_gat_config.json` (GAT)
* `model_configs/fix_prot_len_gat_gcn_config.json` (GAT-GCN)
* `model_configs/fix_prot_len_gcn_config.json` (GCN)
* `model_configs/fix_prot_len_gin_config.json` (GIN)

### Training and Evaluation

For convenience, we provide a shell script `scripts/train.sh` for easy experiments.
Its usage is:

```sh
sh scripts/train.sh DATASET YOU_CONFIG_JSON [EXTRA-ARGS]
```

For example, to train the GIN model on Davis dataset, just execute:

```sh
sh scripts/train.sh davis fix_prot_len_gin_config.json
```

Notice that if you want to train the GIN model on Kiba dataset, you need to use KIBA label, instead of default Kd label, so execute:

```sh
sh scripts/train.sh kiba fix_prot_len_gin_config.json --use_kiba_label
```

For evaluation, we use MSE as a standard metric for the regression task. Besides, concordance index (CI) is an another metric. The smaller MSE, the better. While, the larger CI, the better.

Evaluation results on Davis:

| Methods      |  MSE       | CI        |
| :--:         | :--:       | :--:      |
| GCN          | 0.251      | 0.888     |
| GAT_GCN      | 0.244      | 0.885     |
| GAT          | 0.250      | 0.887     |
| GIN          | 0.242      | 0.889     |

Evaluation results on Kiba:

| Methods      |  MSE       | CI        |
| :--:         | :--:       | :--:      |
| GCN          | 0.179      | 0.880     |
| GAT_GCN      | 0.142      | 0.895     |
| GAT          | 0.192      | 0.867     |
| GIN          | 0.177      | 0.878     |

## Reference

**GraphDTA**
> @article{nguyen2020graphdta,
  title={GraphDTA: Predicting drug-target binding affinity with graph neural networks},
  author={Thin Nguyen, Hang Le, Thomas P. Quinn, Tri Nguyen, Thuc Duy Le, and Svetha Venkatesh},
  journal={Bioinformatics},
  year={2020},
  url={https://doi.org/10.1093/bioinformatics/btaa921}
}

**Davis**
>@article{davis2011natbiotech,
  title={Comprehensive analysis of kinase inhibitor selectivity},
  author={Mindy I Davis, Jeremy P Hunt, Sanna Herrgard, Pietro Ciceri, Lisa M Wodicka, Gabriel Pallares, Michael Hocker, Daniel K Treiber and Patrick P Zarrinkar},
  journal={Nature Biotechnology},
  year={2011},
  url={https://doi.org/10.1038/nbt.1990}
}

**Kiba**
>@article{tang2014kiba,
  title={Making sense of large-scale kinase inhibitor bioactivity data sets: a comparative and integrative analysis},
  author={Jing Tang 1, Agnieszka Szwajda, Sushil Shakyawar, Tao Xu, Petteri Hintsanen, Krister Wennerberg, Tero Aittokallio},
  journal={J Chem Inf Model},
  year={2014},
  url={https://pubs.acs.org/doi/10.1021/ci400709d}
}
