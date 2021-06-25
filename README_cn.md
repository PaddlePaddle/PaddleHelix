[English](README.md) | 简体中文

<p align="center">
<img src="./.github/飞桨-螺旋桨_logo.png" align="middle" height="75%" width="75%" />
</p>

------
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleHelix.svg)](https://github.com/PaddlePaddle/PaddleHelix/releases)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)

## 最新消息
`2021.06.17` 螺旋桨团队在[OGB-LCS KDD Cup 2021 PCQM4M-LSC track](https://ogb.stanford.edu/kddcup2021/results/)比赛中赢得了亚军。该项比赛预测使用DFT计算的分子HOMO-LUMO的能量差。请参阅[解决方案](./competition/kddcup2021-PCQM4M-LSC)获得更多的细节。.

`2021.05.20` 螺旋桨v1.0正式版发布。 1)将模型全面从静态图升级到动态图; 2) 添加更多应用: 分子生成和药物联用.

`2021.03.15` 螺旋桨团队在权威图榜单[OGB](https://ogb.stanford.edu/docs/leader_graphprop/)的ogbg-molhiv和ogbg-molpcba任务上取得第一名。这两项任务均是预测小分子的属性。

---

## 简介
螺旋桨（PaddleHelix）是一个生物计算工具集，是用机器学习的方法，特别是深度神经网络，致力于促进以下领域的发展：

* **新药发现**. Provide 1) Large-scale pre-training models: compounds and proteins; 2) Various applications: molecular property prediction, drug-target affinity prediction, and molecular generation.
* **苗设计**. Provide RNA design algorithms, including LinearFold and LinearPartition.
* **精准医疗**. Provide application of drug-drug synergy.

<p align="center">
<img src=".github/PaddleHelix_Structure.png" align="middle" heigh="80%" width="80%" />
</p>

---
## 资源
### 安装
详细的安装指引和环境配置请查阅[这里](./installation_guide_cn.md)。

### 教学示例
* 我们提供了大量的[教学示例](./tutorials)以方便开发者快速了解和使用该框架
* PaddleHelix基于[飞桨（PaddlePaddle）](https://github.com/paddlepaddle/paddle)开源深度学习框架实现，该框架在性能表现上尤其出色。

### 使用示例
* [表示学习 - 化合物](./apps/pretrained_compound/README_cn.md)
* [表示学习 - 蛋白质](./apps/pretrained_protein/README_cn.md)
* [药物-分子作用预测](./apps/drug_target_interaction/README_cn.md)
* [分子生成](./apps/molecular_generation/README_cn.md)
* [药物联用](./apps/drug_drug_synergy/README_cn.md)
* [LinearRNA](./c/pahelix/toolkit/linear_rna/README_cn.md)

### 开发者指南
* 如果你需要修改PaddleHelix的源代码，请查阅我们提供的[开发者指南](./developer_guide_cn.md)。
* 如果你对PaddleHelix的详细接口感兴趣，请查阅[API 文档](https://paddlehelix.readthedocs.io/en/dev/)。

### 欢迎加入我们
我们正在招聘对人工智能驱动的药物设计感兴趣的机器学习研究人员/工程师或生物信息/计算化学相关研究人员。
我们的工作地点在中国深圳/上海。
请把简历寄到wangfan04@baidu.com 或者fangxiaomin01@baidu.com。
