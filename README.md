English | [简体中文](README_cn.md)

<p align="center">
<img src="./.github/paddlehelix_logo.png" align="middle" heigh="90%" width="90%" />
</p>

------
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleHelix.svg)](https://github.com/PaddlePaddle/PaddleHelix/releases)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)

## Latest News
`2021.06.17` PaddleHelix team won the 2nd place in the [OGB-LCS KDD Cup 2021 PCQM4M-LSC track](https://ogb.stanford.edu/kddcup2021/results/), predicting DFT-calculated HOMO-LUMO energy gap of molecules. Please refer to [the solution](https://github.com/PaddlePaddle/PaddleHelix/tree/dev/competition/kddcup2021-PCQM4M-LSC) for more details.

`2021.05.20` PaddleHelix v1.0 released. 1) Update from static framework to dynamic framework; 2) Add new applications: molecular generation and drug-drug synergy.

---

## Introduction
PaddleHelix is a bio-computing platform, taking advantage of machine learning approach, especially deep neural networks, for facilitating the development of the following areas:
* **Drug Discovery** Provide 1) Large-scale pre-training models: compounds and proteins; 2) Various applications: molecular property prediction, drug-target affinity prediction, and molecular generation.
* **Vaccine Design** Provide 1) Large-scale pre-training models: compounds and proteins; 2) Various applications: molecular property prediction, drug-target affinity prediction, and molecular generation.
* **Precision Medicine** Provide application of drug-drug synergy.

----

## Installation

The installation prerequisites and guide can be found [here](./installation_guide.md).

----

## Documentation

### Tutorials
* We provide abundant [tutorials](./tutorials) to help you navigate the repository and start quickly.
* PaddleHelix is based on [PaddlePaddle](https://github.com/paddlepaddle/paddle), a high-performance Parallelized Deep Learning Platform.

### Examples
* [Representation Learning - Compounds](./apps/pretrained_compound)
* [Representation Learning - Proteins](./apps/pretrained_protein)
* [Drug-Target Interaction](./apps/drug_target_interaction)
* [Molecular Generation](./apps/molecular_generation)
* [Drug Drug Synergy](./apps/drug_drug_synergy)
* [LinearRNA](./c/pahelix/toolkit/linear_rna)

### The API reference
* Detailed API reference of PaddleHelix can be found [here](https://paddlehelix.readthedocs.io/en/dev/).

### Guide for developers
* If you need help in modifying the source code of PaddleHelix, please see our [Guide for developers](./developer_guide.md).

### Welcome to join us
We are looking for machine learning researchers / engineers or bioinformatics / computational chemistry researchers interested in AI-driven drug design.
We base in Shenzhen or Shanghai, China.
Please send the resumes to wangfan04@baidu.com or fangxiaomin01@baidu.com.
