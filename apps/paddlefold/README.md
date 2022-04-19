# PaddleFold 

Reproduction of [AlphaFold 2](https://doi.org/10.1038/s41586-021-03819-2) with [PaddlePaddle](https://github.com/paddlepaddle/paddle).

PaddleFold currently provides a PaddlePaddle implementation of the AlphaFold training/inference pipeline in DCU, as well as the inference pipeline in GPU. We try to reproduce all the training details as stated in the AF2 paper. As for the inference pipeline, we reproduces exactly the same results as the original open source inference code (v2.0.1) including recycle and ensembling.

The installation prerequisites are difference for the training/inference pipeline in DCU and the inference pipeline in GPU. 
The following links provide detailed instructures to run PaddleFold.

* [Training Pipeline in DCU](README_DCU.md)
* [Inference Pipeline in GPU](README_inference.md)

## Copyright

PaddleFold code is licensed under the Apache 2.0 License, which is same as AlphaFold. However, we use the AlphaFold parameters pretrained by DeepMind, which are made available for non-commercial use only under the terms of the CC BY-NC 4.0 license.
