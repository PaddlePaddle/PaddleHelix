# InfoGraph

[中文版本](./README.cn.md) [English Version](./README.md)

* [背景介绍](#背景介绍)
* [使用说明](#使用说明)
    * [参数设置](#参数设置)
    * [训练与评估](#训练与评估)
* [数据集](#数据集)
    * [MUTAG](#mutag)
    * [PTC-MR](#ptc-mr)
* [引用](#引用)

## 背景介绍

整张图的表征对现实中的很多任务都至关重要，比如分子的性质预测。InfoGraph是目前对整个图特征表征学习模型，它通过最大图的整体表征与不同子结构表征之间的信息熵，从而实现无监督的表征学习。

## 使用说明

### 参数设置

Python脚本`unsupervised_pretrain.py`是InfoGraph模型的入口，它需要的超参数配置请参见`demos/unsupervised_pretrain_config.json`.

### 训练与评估

为了方便实验，我们提供了shell脚本`demos/unsupervised_pretrain.sh`实现一体化的训练以及评估学习到的图表征。在运行该脚本前，需要配置实验重复次数，数据集的根目录，以及超参配置的路径。

```sh
runs=3
root="/path/to/datasets"
config="unsupervised_pretrain_config.json"
```

评估结果：

| Datasets      | paper        | our run-1 | our run-2 | our run-3 |
| :--:          | :--:         | :--:      | :--:      | :--:      |
| MUTAG         | 89.01+/-1.13 | 91.43     | 90.20     | 90.45     |
| PTC-MR        | 61.64+/-1.43 | 60.59     | 64.09     | 60.28     |

## 数据集

### MUTAG

MUTAG是图学习中的常用数据集，它包含了188个测试了对细菌诱变效应的有机分子，可以看做是一个图的二分类问题。假设所以下载的数据集都放在`data`目录下，则可以使用下面方法下载MUTAG数据集：

```sh
cd data
mkdir -p mutag/raw && cd mutag/raw
wget ftp://ftp.ics.uci.edu/pub/baldig/learning/mutag/mutag_188_data.can
wget ftp://ftp.ics.uci.edu/pub/baldig/learning/mutag/mutag_188_target.txt
```

### PTC-MR

PTC-MR是一个包含344个分子的二分类数据集，它标记了这些化合物对啮齿动物的致癌性。假设所以下载的数据集都放在`data`目录下，则可以使用下面方法下载PTC-MR数据集：

```sh
cd data
mkdir -p ptc_mr/raw && cd ptc_mr/raw
wget ftp://ftp.ics.uci.edu:21/pub/baldig/learning/ptc/ptc_MR_data.can
wget ftp://ftp.ics.uci.edu:21/pub/baldig/learning/ptc/ptc_MR_target.txt
```

## 引用

**InfoGraph**
> @inproceedings{sun2019infograph,
  title={InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization},
  author={Sun, Fan-Yun and Hoffman, Jordan and Verma, Vikas and Tang, Jian},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
