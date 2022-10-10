# STR

[English Version](./README.md)
* [Background](#background)
* [Datasets](#datasets)
    * [CCLE](#ccle)
    * [GDSC](#gdsc)
* [Usage](#Usage)
    * [Dependencies](#Dependencies) 
    * [Preprocessing](Preprocessing)
    * [Training](#Training) 
    * [Testing](#Testing) 
    
* [Reference](#reference)

## Background

Accurate prediction of cancer drug response (CDR) is challenging due to the uncertainty of drug efficacy and heterogeneity of cancer patients.  Precise identification of CDR is crucial in both guiding anti-cancer drug design and understanding cancer biology. This is a model which integrates multi-omics profiles of cancer cells and explores intrinsic chemical structures of drugs for predicting CDR.
Here is the code for paper : STR: A Substructure Sensitive Transformer Network for Predicting Cancer Drugs Response

## Datasets
The CCLE and GDSC data can be download and uncompress dataset using following command:
```sh
wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/drug_response/str_data.tar && tar -xvf str_data.tar
cd data
```
Your can use process_data.py to generate your own dataset. TCGA and other dataset can also be used.

### CCLE

```
The three following CCLE files will in located in `./data/CCLE` folder.

1.genomic mutation matrix where each column denotes mutation locus and each row denotes a cell line

2.gene expression matrix where each column denotes a coding gene and each row denotes a cell line

3.DNA methylation matrix where each column denotes a methylation locus and each row denotes a cell line

4.cell line annotations file.

```

### GDSC
```
1. Genomics of Drug Sensitivity in Cancer file.
2. drug list, smiles and drug info discription file.
```

## Usage

1. Dependencies
```
python >= 3.7
paddlepaddle >= 2.0.0rc0
paddlehelix
pgl
rdkit
sklearn
```

```sh
python -m pip install paddlepaddle-gpu==2.3.2.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install paddlehelix
pip install pgl
```
2. Preprocessing

```sh
python process_data.py --save_dir processed_data --split_ratio 0.8
```
3. Training

```sh
CUDA_VISIBLE_DEVICES=0 python main.py --batch_size 256 --mode train --task STR_01 --beta 0.4 --data_path ./data/processed_data/ --split_mode drug --model STR
```

4. Testing

```sh
CUDA_VISIBLE_DEVICES=0 python main.py --task STR_01 --mode test --beta 0.4 --data_path ./data/processed_inference/ --split_mode mix --model STR
```
## Reference


**TCR**

>@article{gao2022tcr,
  title={TCR: A Transformer Based Deep Network for Predicting Cancer Drugs Response},
  author={Gao, Jie and Hu, Jing and Sun, Wanqing and Shen, Yili and Zhang, Xiaonan and Fang, Xiaomin and Wang, Fan and Zhao, Guodong},
  journal={arXiv preprint arXiv:2207.04457},
  year={2022}
}

**DeepCDR**
>@article{nguyen2020graphdta,title={DeepCDR: a hybrid graph convolutional network for predicting cancer drug response},author={Qiao Liu, Zhiqiang Hu, Rui Jiang, Mu Zhou},journal={Bioinformatics},year={2020},url={https://doi.org/10.1093/bioinformatics/btaa822}}
