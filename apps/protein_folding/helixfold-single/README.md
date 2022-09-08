# HelixFold-Single Inference

AI-based protein structure prediction pipelines, such as AlphaFold2, have achieved near-experimental accuracy. These advanced pipelines mainly rely on Multiple Sequence Alignments (MSAs) and templates as inputs to learn the co-evolution information from the homologous sequences. Nonetheless, searching MSAs and templates from protein databases is time-consuming, usually taking dozens of minutes. Consequently, we attempt to explore the limits of fast protein structure prediction by using only primary sequences of proteins. **HelixFold-Single** is proposed to combine a large-scale protein language model with the superior geometric learning capability of AlphaFold2. Our proposed method, HelixFold-Single, first pre-trains a large-scale protein language model (PLM) with thousands of millions of primary sequences utilizing the self-supervised learning paradigm, which will be used as an alternative to MSAs and templates for learning the co-evolution information. Then, by combining the pre-trained PLM and the essential components of AlphaFold2, we obtain an end-to-end differentiable model to predict the 3D coordinates of atoms from only the primary sequence. 

## Online Service
For those who want to try out our model without any installation, we also provide an online interface [PaddleHelix HelixFold-Single Forecast](https://paddlehelix.baidu.com/app/drug/protein-single/forecast) through web service.

## Environment
To reproduce the results reported in our paper, specific environment settings are required as below.

- python: 3.7
- cuda: 11.2
- cudnn: 8.10.1
- nccl: 2.12.12


## Installation
Except those listed in the `requirements.txt`, PaddlePaddle `dev` package is required to run HelixFold.
Visit [here](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html) to 
install PaddlePaddle `dev`. Also, we provide a package here if your machine environment is Nvidia A100 with
cuda=11.2.

```bash
python -m pip install -r requirements.txt
wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/HelixFold/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
python -m pip install paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
```

## Download the Trained Model
Here we provide the trained model that can be used to reproduce the results of our paper.

```bash
wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/HelixFold-Single/helixfold-single.pdparams
```

## Usage
To run the inference, what you need is a fasta file and the pre-downloaded trained model:

```bash
python helixfold_single_inference.py \
        --init_model=./helixfold-single.pdparams \
        --fasta_file=data/7O9F_B.fasta \
        --output_dir="./output" 
```

- `init_model`: the trained model.
- `fasta_file`: the fasta_file file which contains the protein sequence to be predicted.

The output is organized as：

    ./output
        unrelaxed.pdb

where `unrelaxed.pdb` is the predicted pdb file.

## Citing this work

If you use the code or data in this repos, please cite:

```bibtex
@article{fang2022helixfold_single,
  title={HelixFold-Single: MSA-free Protein Structure Prediction by Using Protein Language Model as an Alternative},
  author={Fang, Xiaomin and Wang, Fan and Liu, Lihang and He, Jingzhou and Lin, Dayong and Xiang, Yingfei and Zhang, Xiaonan and Wu, Hua and Li, Hui and Song, Le},
  journal={arXiv preprint arXiv:2207.13921},
  year={2022}
}
```
