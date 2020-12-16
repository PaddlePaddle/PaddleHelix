[English](README.md) | 简体中文

<p align="center">
<img src="./.github/paddlehelix_logo.png" align="middle"
</p>

------
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleHelix.svg)](https://github.com/PaddlePaddle/PaddleHelix/releases)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)

PaddleHelix（螺旋桨）是一个基于机器学习的生物计算工具集，致力于加速如下领域的进展：
> * 疫苗设计
> * 新药发现
> * 精准医疗

## 特色

* **高性能**：提供了 LinearRNA 系列高性能算法助力 mRNA 疫苗设计。例如，LinearFold 和 LinearParition 能够迅速准确定位能量较低 RNA 二级结构，性能相比传统方法提升数百甚至上千倍。
<p align="center">
<img src="./.github/LinearRNA.jpg" align="middle"
</p>

* 由大规模 **表示预训练** 和 **迁移学习** 支撑的生物计算工具：随着自监督学习用于分子表示训练的进展，为样本量非常稀少的很多生物计算任务带来了全新的突破，这些任务包括分子性质预测，药物-靶点相互作用，蛋白质-蛋白质相互作用，RNA-RNA 相互作用，蛋白质折叠，RNA 折叠等等领域。螺旋桨广泛提供了业界最领先的表示学习方法和模型，使得开发者可以基于大规模模型快速切入需求的任务，站在巨人的肩膀上。
<p align="center">
<img src="./.github/paddlehelix_features.jpg" align="middle"
</p>

* 简单易用的 API 接口：螺旋桨提供了生物计算中常用的模型结构和预训练模型，用户可以用非常简单的接口调起这些模型，快速组建自己的网络和系统。
----

## 安装

### 操作系统支持

Windows，Linux 以及 OSX

### Python 版本

Python 3.6, 3.7

### 包依赖

('-'代表对应的依赖没有具体的版本要求)


|  名字 |版本 |
|  ----  | ----  |
| numpy | - |
| pandas | - |
|networkx|-|
|paddlepaddle|\>=2.0.0rc0|
|pgl|\>=1.2.0|
|rdkit|-|


### 安装命令

因为 PaddleHelix 安装包的依赖有最新版的 paddlepaddle（2.0.0rc0 或以上），以及无法直接使用 `pip` 命令直接安装的 rdkit，因此我们建议创建一个新的 conda 环境来运行代码，具体命令如下：

* 如果你之前从来没有使用过 conda，可以参考这个网页来安装 conda:

   https://docs.conda.io/projects/conda/en/latest/user-guide/install/

* 在安装完 conda 之后, 可以开始创建一个新的 conda 环境:

```bash
conda create -n paddlehelix python=3.7
```

* 使用如下命令激活 conda 环境:

```bash
conda activate paddlehelix
```

* 在安装 PaddleHelix 之前，首先需要使用 conda 安装 rdkit：
```bash
conda install -c conda-forge rdkit
```
* rdkit 安装完成之后，使用 pip 命令安装 PaddleHelix
```bash
pip install paddlehelix
```

* 等待 PaddleHelix 安装完成！

* 如果想要退出当前 conda 环境，可以使用下列命令：

```bash
conda deactivate
```
----
## 文档

### 教学
* 我们提供了大量的[教学实例](./tutorials)以方便开发者快速了解和使用该框架
* PaddleHelix 基于[飞桨（PaddlePaddle）](https://github.com/paddlepaddle/paddle)开源深度学习框架实现，该框架在性能表现上尤其出色。

### 使用示例
* [表示学习 - 化合物](./apps/pretrained_compound)
* [表示学习 - 蛋白质](./apps/pretrained_protein)
* [药物-分子作用预测](./apps/drug_target_interaction)
* [LinearRNA](./c/pahelix/toolkit/linear_rna)

### [The API reference](https://readthedocs.org/projects/paddlehelix/)
