[English](README.md) | 简体中文

<p align="center">
<img src="./.github/飞桨-螺旋桨_logo.png" align="middle" height="75%" width="75%" />
</p>

------
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleHelix.svg)](https://github.com/PaddlePaddle/PaddleHelix/releases)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)

螺旋桨（PaddleHelix）是一个基于机器学习的生物计算工具集，致力于加速如下领域的进展：
> * 疫苗设计
> * 新药发现
> * 精准医疗

## 特色

* **高性能**：提供了 LinearRNA 系列高性能算法助力 mRNA 疫苗设计。例如，LinearFold 和 LinearParition 能够迅速准确定位能量较低 RNA 二级结构，性能相比传统方法提升数百甚至上千倍。
<p align="center">
<img src="./.github/LinearRNA.jpg" align="middle" />
</p>

* 由大规模 **表示预训练** 和 **迁移学习** 支撑的生物计算工具：随着自监督学习用于分子表示训练的进展，为样本量非常稀少的很多生物计算任务带来了全新的突破，这些任务包括分子性质预测，药物-靶点相互作用，蛋白质-蛋白质相互作用，RNA-RNA 相互作用，蛋白质折叠，RNA 折叠等等领域。螺旋桨广泛提供了业界最领先的表示学习方法和模型，使得开发者可以基于大规模模型快速切入需求的任务，站在巨人的肩膀上。
<p align="center">
<img src="./.github/paddlehelix_features.jpg" align="middle" />
</p>

* 简单易用的 API 接口：螺旋桨提供了生物计算中常用的模型结构和预训练模型，用户可以用非常简单的接口调起这些模型，快速组建自己的网络和系统。
----

## 安装
详细的安装指引和环境配置请查阅[这里](./installation_guide_cn.md)。

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

### API 文档
* 如果你对 PaddleHelix 的详细接口感兴趣，请查阅 [API 文档](https://readthedocs.org/projects/paddlehelix/)。
