[English](README.md) | 简体中文

<p align="center">
<img src="./.github/飞桨-螺旋桨_logo.png" align="middle" height="75%" width="75%" />
</p>

------
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleHelix.svg)](https://github.com/PaddlePaddle/PaddleHelix/releases)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)

## 最新消息
`2022.12.08` 论文"HelixMO: Sample-Efficient Molecular Optimization in Scene-Sensitive Latent Space"被**BIBM 2022**接收。详情参见[链接1](https://www.computer.org/csdl/proceedings-article/bibm/2022/09995561/1JC23yWxizC)或[链接2](https://aps.arxiv.org/abs/2112.00905)去获得更多信息。也欢迎到我们的服务平台[PaddleHelix](https://paddlehelix.baidu.com/app/drug/drugdesign/forecast)试用药物设计服务.

`2022.08.11` 螺旋桨团队开源了HelixGEM-2的代码, 它是一个全新的基于长程多体建模的小分子属性预测框架，并在OGB [PCQM4Mv2](https://ogb.stanford.edu/docs/lsc/leaderboards/) 排行榜取得第一的成绩。详情参见 [论文](https://arxiv.org/abs/2208.05863) 和 [代码](./apps/pretrained_compound/ChemRL/GEM-2)。

`2022.07.29` 螺旋桨团队开源了HelixFold-Single的代码，HelixFold-Single是一个**不依赖于MSA的**蛋白质结构预测流程，仅仅需要一级序列作为输入就可以提供**秒级别的蛋白质结构预测**。详情参见[论文](https://arxiv.org/abs/2207.13921)和[代码](./apps/protein_folding/helixfold-single)。欢迎到[PaddleHelix网站](https://paddlehelix.baidu.com/app/drug/protein-single/forecast
)去试用结构预测的在线服务。

`2022.07.18` 螺旋桨团队全面开源HelixFold训练和推理代码，**完整训练天数从11天优化至5.12天**。详情参见[论文](https://arxiv.org/abs/2207.05477)和[代码](./apps/protein_folding/helixfold)。

`2022.07.07` 论文"BatchDTA: implicit batch alignment enhances deep learning-based drug–target affinity estimation"发表于期刊**Briefings in Bioinformatics**。详情参见[论文](https://academic.oup.com/bib/advance-article-abstract/doi/10.1093/bib/bbac260/6632927)和[代码](./apps/drug_target_interaction/batchdta)。

`2022.05.24` 论文"HelixADMET: a robust and endpoint extensible ADMET system incorporating self-supervised knowledge transfer"发表于期刊**Bioinformatics**. 详情参见[论文](https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/btac342/6590643)。

`2022.02.07` 论文"Geometry-enhanced molecular representation learning for property prediction"发表于期刊**Nature Machine Intelligence**。详情参见[论文](https://www.nature.com/articles/s42256-021-00438-4)和[代码](./apps/pretrained_compound/ChemRL/GEM)。

<details>
<summary>更多信息...</summary>

`2022.01.07` 螺旋桨团队开源基于PaddlePaddle深度学习框架的[AlphaFold 2](https://doi.org/10.1038/s41586-021-03819-2)蛋白质结构预测模型推理实现，详见[HelixFold](./apps/protein_folding/helixfold)。

`2021.11.23` 论文"Multimodal Pre-Training Model for Sequence-based Prediction of Protein-Protein Interaction"被[MLCB 2021](https://sites.google.com/cs.washington.edu/mlcb2021/home)接收. 详细信息请参见[论文](https://arxiv.org/abs/2112.04814)和[代码](https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/protein_protein_interaction).

`2021.10.25` 论文"Docking-based Virtual Screening with Multi-Task Learning"被[BIBM 2021](https://ieeebibm.org/BIBM2021/)接收.

`2021.09.29` 论文"Property-Aware Relation Networks for Few-shot Molecular Property Prediction"被[NeurIPS 2021](https://papers.nips.cc/paper/2021/hash/91bc333f6967019ac47b49ca0f2fa757-Abstract.html)接收为Spotlight Paper。代码细节请参见[PAR](./apps/fewshot_molecular_property).

`2021.07.29` 螺旋桨团队基于3D空间结构的化合物预训练模型，充分利用海量的无标注的化合物3D信息。请参阅[GEM](./apps/pretrained_compound/ChemRL/GEM)获取更多的细节。

`2021.06.17` 螺旋桨团队在[OGB-LCS KDD Cup 2021 PCQM4M-LSC track](https://ogb.stanford.edu/kddcup2021/results/)比赛中赢得了亚军。该项比赛预测使用DFT计算的分子HOMO-LUMO的能量差。请参阅[解决方案](./competition/kddcup2021-PCQM4M-LSC)获得更多的细节。.

`2021.05.20` 螺旋桨v1.0正式版发布。 1)将模型全面从静态图升级到动态图; 2) 添加更多应用: 分子生成和药物联用.

`2021.05.18` 论文"Structure-aware Interactive Graph Neural Networks for the Prediction of Protein-Ligand Binding Affinity"被[KDD 2021](https://kdd.org/kdd2021/accepted-papers/index)接收。代码参见[这里](./apps/drug_target_interaction/sign).

`2021.03.15` 螺旋桨团队在权威图榜单[OGB](https://ogb.stanford.edu/docs/leader_graphprop/)的ogbg-molhiv和ogbg-molpcba任务上取得第一名。这两项任务均是预测小分子的属性。
 </details>

---

## 简介
螺旋桨（PaddleHelix）是一个生物计算工具集，是用机器学习的方法，特别是深度神经网络，致力于促进以下领域的发展：

* **新药发现**。提供1)大规模预训练模型:化合物和蛋白质; 2)多种应用:分子属性预测,药物靶点亲和力预测,和分子生成。
* **疫苗设计**。提供RNA设计算法,包括LinearFold和LinearPartition。
* **精准医疗**。提供药物联用的应用。

<p align="center">
<img src=".github/PaddleHelix_Structure.png" align="middle" heigh="70%" width="70%" />
</p>

---
## 项目资源
### 计算平台
[PaddleHelix平台](https://paddlehelix.baidu.com/)提供AI+生物计算能力，满足新药研发、疫苗设计、精准医疗场景的AI需求。

### 安装指南
螺旋桨是一个基于高性能机器学习工具[PaddlePaddle飞桨](https://github.com/paddlepaddle/paddle)的生物计算开源工具库。详细的安装和环境配置指引请查阅[这里](./installation_guide_cn.md)。

### 教学示例
我们提供了大量的[教学示例](./tutorials/README_cn.md)以方便开发者快速了解和使用该框架：
* **Drug Discovery**
  - [化合物表示和属性预测](./tutorials/compound_property_prediction_tutorial_cn.ipynb)
  - [蛋白质表示和属性预测](./tutorials/protein_pretrain_and_property_prediction_tutorial_cn.ipynb)
  - [药物-分子作用预测: GraphDTA](./tutorials/drug_target_interaction_graphdta_tutorial_cn.ipynb), [MolTrans](./tutorials/drug_target_interaction_moltrans_tutorial_cn.ipynb)
  - [分子生成](./tutorials/molecular_generation_tutorial_cn.ipynb)
* **Vaccine Design**
  - [RNA结构预测](./tutorials/linearrna_tutorial_cn.ipynb)

### 使用示例
我们也提供了多个算法的[代码和使用示例](./apps/README_cn.md):
* **预训练**
  - [表示学习 - 化合物](./apps/pretrained_compound/README_cn.md)
  - [表示学习 - 蛋白质](./apps/pretrained_protein/README_cn.md)
* **新药发现和精准医疗**
  - [药物-分子作用预测](./apps/drug_target_interaction/README_cn.md)
  - [分子生成](./apps/molecular_generation/README_cn.md)
  - [药物联用](./apps/drug_drug_synergy/README_cn.md)
  - [小样本分子性质预测](./apps/fewshot_molecular_property.md)
* **疫苗设计**
  - [LinearRNA](./c/pahelix/toolkit/linear_rna/README_cn.md)
* **蛋白质结构预测**
  - [HelixFold](./apps/protein_folding/helixfold)
  - [HelixFold-Single](./apps/protein_folding/helixfold-single)

### 比赛解决方案
螺旋桨团队参加了多项生物计算相关的赛事，相关解决方案可以参阅[这里](./competition).

### 开发者指南
* 如果你需要基于螺旋桨的源代码进行新功能的开发，请查阅我们提供的[开发者指南](./developer_guide_cn.md)。
* 如果你想知道螺旋桨各种接口的详情，请查阅[API文档](https://paddlehelix.readthedocs.io/en/dev/)。

------

## 欢迎加入我们
我们正在招聘对人工智能驱动的药物设计感兴趣的机器学习研究人员/工程师或生物信息/计算化学相关研究人员。
我们的工作地点在中国深圳/上海。
请把简历寄到wangfan04@baidu.com 或者fangxiaomin01@baidu.com。
