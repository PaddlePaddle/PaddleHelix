English | [简体中文](README_cn.md)

<p align="center">
<img src="./.github/paddlehelix_logo.png" align="middle" heigh="90%" width="90%" />
</p>

------
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleHelix.svg)](https://github.com/PaddlePaddle/PaddleHelix/releases)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)

PaddleHelix is a machine-learning-based bio-computing framework aiming at facilitating the development of the following areas:
> * Vaccine design
> * Drug discovery
> * Precision medicine

## Features
* High Efficency: We provide LinearRNA, a highly efficient toolkit for mRNA vaccine development. LinearFold & LinearParitition achieve O(n) complexity in RNA-folding prediction, which is hundreds of times faster than traditional folding techniques.
<p align="center">
<img src="./.github/LinearRNA.jpg" align="middle" />
</p>

* Large-scale Representation Learning and Transfer Learning: Self-supervised learning for molecule representations offers prospects of a breakthrough in tasks with limited annotation, including drug profiling, drug-target interaction, protein-protein interaction, RNA-RNA interaction, protein folding, RNA folding, and molecule design. PaddleHelix implements a variety of representation learning algorithms and state-of-the-art large-scale pre-trained models to help developers to start from "the shoulders of giants" quickly.
<p align="center">
<img src="./.github/paddlehelix_features.jpg" align="middle" />
</p>

* Easy-to-use APIs: PaddleHelix provides frequently used structures and pre-trained models. You can easily use those components to build up your models and systems.

## Installation

The installation prerequisites and guide can be found [here](./installation_guide.md).

## Documentation

### Tutorials
* We provide abundant [tutorials](./tutorials) to help you navigate the directory and start quickly.
* PaddleHelix is based on [PaddlePaddle](https://github.com/paddlepaddle/paddle), a high-performance Parallelized Deep Learning Platform.

### Examples
* [Representation Learning - Compounds](./apps/pretrained_compound)
* [Representation Learning - Proteins](./apps/pretrained_protein)
* [Drug-Target Interaction](./apps/drug_target_interaction)
* [LinearRNA](./c/pahelix/toolkit/linear_rna)

### The API reference
* Detailed API reference of PaddleHelix can be found [here](https://paddlehelix.readthedocs.io/en/dev/).
