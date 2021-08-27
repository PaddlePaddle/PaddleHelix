# GraphDTA

[中文版本](./README_cn.md) [English Version](./README.md)

* [背景介绍](#背景介绍)
* [数据](#数据)
    * [Davis](#davis)
    * [Kiba](#kiba)
* [使用说明](#使用说明)
    * [参数设置](#参数设置)
    * [训练与评估](#训练与评估)
* [引用](#引用)

## 背景介绍

了解哪些药物对靶标蛋白起作用对于新药研发、老药新用都很有帮助。GraphDTA模型将药物分子表示为图数据，然后使用图神经网络预测药物和靶标蛋白的亲和性。

## 数据集

首先，在该应用目录下创建一个`data`子目录，作为数据集的root目录。

```sh
mkdir -p data && cd data
```

### Davis

Davis数据集包含了72种药物和442种靶标蛋白任意之间的Kd值（平衡解离常数）。Kd值越小，说明药物和靶标蛋白之间的亲和性越高。

执行下面的命令即可下载并解压Davis数据集：

```sh
wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/dti_datasets/davis_v1.tgz -O davis.tgz
tar -zxvf davis.tgz
```

### Kiba

Kiba数据集包含了2,116种药物和229种靶标蛋白，不同于Davis数据集，部分药物和靶标蛋白直接没有亲和性指标。另外，Kiba使用KIBA分数作为亲和性的评估指标。KIBA分数提供了一种基于统计分析的归一化不同亲和性指标（Ki, Kd, IC50）的方法。

执行下面的命令即可下载并解压Kiba数据集：

```sh
wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/dti_datasets/kiba_v1.tgz -O kiba.tgz
tar -zxvf kiba.tgz
```

下载完成后，`data`目录看起来是这样的：

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

## 使用说明

### 参数设置

Python脚本`scripts/train.py`是GraphDTA模型的入口，它创建了`src/model.py`中的`DTAModel`模型，并完成训练和评估等功能。由于GraphDTA将蛋白质序列转换为定长的新序列，参数设置上最接近原论文的是：

* `model_configs/fix_prot_len_gat_config.json` (GAT)
* `model_configs/fix_prot_len_gat_gcn_config.json` (GAT-GCN)
* `model_configs/fix_prot_len_gcn_config.json` (GCN)
* `model_configs/fix_prot_len_gin_config.json` (GIN)

### 训练与评估

为了方便实验，我们提供了shell脚本`scripts/train.sh`来运行实验，它的使用方法是：

```sh
./scripts/train.sh DATASET YOU_CONFIG_JSON [EXTRA-ARGS]
```

例如，要在Davis数据集上训练GIN模型，只需要执行：

```sh
./scripts/train.sh davis model_configs/fix_prot_len_gin_config.json
```

需要注意的是，在Kiba数据集上训练GIN模型时，由于数据集使用了KIBA分数作为指标，而非默认的Kd指标，运行脚本时需要加上额外参数：

```sh
./scripts/train.sh kiba model_configs/fix_prot_len_gin_config.json --use_kiba_label
```

进行评估时，我们使用回归任务中标准的均方误差MSE作为指标，除此之外，引入一致性指数CI作为新指标。均方误差越小，一致性指数越大，模型的预测性能越好。

Davis数据集上的效果：

| Methods      |  MSE       | CI        |
| :--:         | :--:       | :--:      |
| GCN          | 0.251      | 0.888     |
| GAT_GCN      | 0.244      | 0.885     |
| GAT          | 0.250      | 0.887     |
| GIN          | 0.242      | 0.889     |

Kiba数据集上的效果：

| Methods      |  MSE       | CI        |
| :--:         | :--:       | :--:      |
| GCN          | 0.179      | 0.880     |
| GAT_GCN      | 0.142      | 0.895     |
| GAT          | 0.192      | 0.867     |
| GIN          | 0.177      | 0.878     |

## 引用

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
