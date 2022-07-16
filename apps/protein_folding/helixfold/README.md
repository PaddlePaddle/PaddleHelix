# HelixFold 

Reproduction of [AlphaFold 2](https://doi.org/10.1038/s41586-021-03819-2) with [PaddlePaddle](https://github.com/paddlepaddle/paddle).

HelixFold currently provides a PaddlePaddle implementation of the AlphaFold training/inference pipeline in GPU and DCU. We try to reproduce all the training details as stated in the AF2 paper. As for the inference pipeline, we reproduce exactly the same results as the original open source inference code (v2.0.1) including recycle and ensembling.

The installation prerequisites are different for the training/inference pipeline in GPU and DCU. The following links provide detailed instructures to run HelixFold.

* [Training Pipeline in GPU](README_train.md)
* [Training Pipeline in DCU](README_DCU.md)
* [Inference Pipeline in GPU](README_inference.md)

## Copyright

HelixFold code is licensed under the Apache 2.0 License, which is same as AlphaFold. However, we use the AlphaFold parameters pretrained by DeepMind, which are made available for non-commercial use only under the terms of the CC BY-NC 4.0 license.

## Reference

[1] Jumper J, Evans R, Pritzel A, et al. Highly accurate protein structure prediction with AlphaFold[J]. Nature, 2021, 596(7873): 583-589.
