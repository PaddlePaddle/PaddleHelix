# HelixFold Training

We implement [AlphaFold2](https://doi.org/10.1038/s41586-021-03819-2) using [PaddlePaddle](https://github.com/paddlepaddle/paddle), namely HelixFold, to improve training and inference speed and reduce memory consumption. The performance is improved by operator fusion, tensor fusion, and hybrid parallelism computation, while the memory is optimized through Recompute, BFloat16, and memory read/write in-place. Compared with the original AlphaFold2 (implemented with Jax) and OpenFold (implemented with PyTorch), HelixFold needs only 7.5 days to complete the full end-to-end training and only 5.3 days when using hybrid parallelism, while both AlphaFold2 and OpenFold take about 11 days. HelixFold saves 1x training time. We verified that HelixFold's accuracy could be on par with AlphaFold2 on the CASP14 and CAMEO datasets.

## Environment

To reproduce the results reported in our paper, specific environment settings are required as below. 

- python：3.7
- cuda: 11.2
- cudnn: 8.10.1
- nccl: 2.12.12

## Installation

We provide a PaddlePaddle `dev` package and a script `setup_env` that setups a `conda ` environment and installs all dependencies. Within the directory of `helixfold`, run:

```bash
wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/HelixFold/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
sh setup_env
conda activate helixfold # activate the conda environment
```

## Usage

After installing all the above required dependencies, you can have a try with running `gpu_train.sh`.

```bash
sh gpu_train.sh
```