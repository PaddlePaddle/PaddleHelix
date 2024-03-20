# HelixFold Training

We implement [AlphaFold2](https://doi.org/10.1038/s41586-021-03819-2) using [PaddlePaddle](https://github.com/paddlepaddle/paddle), namely [HelixFold](https://arxiv.org/abs/2207.05477), to improve training and inference speed and reduce memory consumption. The performance is improved by operator fusion, tensor fusion, and hybrid parallelism computation, while the memory is optimized through Recompute, BFloat16, and memory read/write in-place. Compared with the original AlphaFold2 (implemented with Jax), OpenFold and Uni-Fold (implemented with PyTorch), HelixFold needs only 5.12 days to complete the full end-to-end training and only 2.89 days when using hybrid parallelism, while both AlphaFold2 and OpenFold take about 11 days. HelixFold saves 3x training time. We verified that HelixFold's accuracy could be on par with AlphaFold2 on the CASP14 and CAMEO datasets.

## Environment

To reproduce the results reported in this repo, specific environment settings are required as below. 

- python: 3.7
- cuda: 11.6
- cudnn: 8.4.0
- nccl: 2.14.3

## Installation

PaddlePaddle `dev` package is required to run HelixFold. Script `setup_env` is used to setup the `conda` environment, installing all dependencies. Locate to the directory of `helixfold` and run:
```bash
wget https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.6-cudnn8.4.0-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post116-cp37-cp37m-linux_x86_64.whl
sh setup_env
conda activate helixfold # activate the conda environment
```
Note: If you have a different version of python3 and cuda, please refer to [here](https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html) for the compatible PaddlePaddle `dev` package.

In order to run scripts with DAP/BP/DP-DAP-BP mode, you also need to install `ppfleetx`. Please refer to [here](https://github.com/PaddlePaddle/PaddleFleetX/tree/develop/projects/protein_folding) for more details.
```bash
wget https://paddle-qa.bj.bcebos.com/PaddleFleetX/ppfleetx-0.0.0-py3-none-any.whl
python -m pip install ppfleetx-0.0.0-py3-none-any.whl  	   # install ppfleetx
```

## Download Demo Dataset and Evaluation Tools

In order to facilitate the users to get start quickly, a demo dataset is provided to test the training pipeline. Locate to the directory of `helixfold` and run:
```bash
wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/HelixFold/data.tar.gz
tar -zxvf data.tar.gz
```

To evaluate the accuracy of the trained model, evaluation tools `lddt` and `tm-score` are required for evaluation.
```bash
wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/HelixFold/lddt
wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/HelixFold/tm_score
mkdir tools && mv lddt tm_score tools && chmod +x ./tools/lddt && chmod +x ./tools/tm_score
```

## Usage

After installing all the above required dependencies and downloading the demo dataset, you can have a try by running `gpu_train.sh`. We provide multiple training modes in one script, which includes intial training and finetune modes on `single node, single GPU`, `single node, multiple GPUs` and `multiple nodes, multiple GPUs`. Note that you need to set `PADDLE_NNODES=number of devices` and `PADDLE_MASTER="xxx.xxx.xxx.xxx:port"` according to your network environment. The details of each parameter are included in the script `gpu_train.sh`.

```bash
bash gpu_train.sh [demo_initial_N1C1, demo_finetune_N1C1, demo_initial_N1C8, demo_finetune_N1C8, demo_initial_N8C64, demo_finetune_N8C64, demo_initial_N8C64_dp16_bp2_dap2, demo_initial_N8C64_dp32_bp1_dap2, demo_initial_N8C64_dp32_bp2_dap1]
```

Following are few examples:

**DP mode**

* Train on a single node with 1 GPU in initial training mode:
```bash
bash gpu_train.sh demo_initial_N1C1
```

* Train on a single node with 8 GPUs in finetune mode:
```bash
bash gpu_train.sh demo_finetune_N1C8
```

* Train on 8 nodes with 64 GPUs in initial training mode:
```bash
bash gpu_train.sh demo_initial_N8C64
```

**DAP/BP/DP-DAP-BP mode**

* Train on 8 nodes with 64 GPUs, DP=16, BP=2, DAP=2 in initial training mode:
```bash
bash gpu_train.sh demo_initial_N8C64_dp16_bp2_dap2
```

* Train on 8 nodes with 64 GPUs, DP=32, BP=1, DAP=2 in initial training mode:
```bash
bash gpu_train.sh demo_initial_N8C64_dp32_bp1_dap2
```

* Train on 8 nodes with 64 GPUs, DP=32, BP=2, DAP=1 in initial training mode:
```bash
bash gpu_train.sh demo_initial_N8C64_dp32_bp2_dap1
```

## Copyright

HelixFold code is licensed under the Apache 2.0 License, which is same as AlphaFold. However, the HelixFold parameters are made available for non-commercial use only under the terms of the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
