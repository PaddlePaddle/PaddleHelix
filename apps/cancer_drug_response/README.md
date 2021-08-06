# DeepCDR

[English Version](./README.md)

* [Background](#background)
* [Datasets](#datasets)
    * [CCLE](#ccle)
    * [GDSC](#gdsc)
* [Instructions](#instructions)
    * [Data Preparation](#data-preparation) 
    * [Training and Evaluation](#train-and-evaluation)
* [Reference](#reference)

## Background

Accurate prediction of cancer drug response (CDR) is challenging due to the uncertainty of drug efficacy and heterogeneity of cancer patients.  Precise identification of CDR is crucial in both guiding anti- cancer drug design and understanding cancer biology. DeepCDR is a model which integrates multi-omics profiles of cancer cells and explores intrinsic chemical structures of drugs for predicting CDR.

## Datasets

First, create a dataset root folder `data` under this application folder.

```sh
mkdir -p data && cd data
```

### CCLE
Download and uncompress CCLE dataset using following command:
```sh
wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/drug_response/CCLE.tar && tar -xvf CCLE.tar
```
The three following CCLE files will in located in `./data/CCLE` folder.

`genomic_mutation_34673_demap_features.csv` -- genomic mutation matrix where each column denotes mutation locus and each row denotes a cell line

`genomic_expression_561celllines_697genes_demap_features.csv` -- gene expression matrix where each column denotes a coding gene and each row denotes a cell line

`genomic_methylation_561celllines_808genes_demap_features.csv` -- DNA methylation matrix where each column denotes a methylation locus and each row denotes a cell line



### GDSC

Download and uncompress GDSC dataset using following command:

```sh
wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/drug_response/GDSC.tar && tar -xvf GDSC.tar
```
The two following CCLE files will be located in `./data/GDSC` folder.

`1.Drug_listMon Jun 24 09_00_55 2019.csv` -- drug list

`223drugs_pubchem_smiles.txt` -- drug information with pubchem ID and SMILES



Then, you can redirect to this application folder and follow instructions to finish next steps.

After downloading these datasets, the `data` folder looks like:

```txt
data
├── CCLE
│   ├── genomic_mutation_34673_demap_features.csv   
│   ├── genomic_expression_561celllines_697genes_demap_features.csv
│   ├── genomic_methylation_561celllines_808genes_demap_features.csv   
│   ├── Cell_lines_annotations_20181226.txt
│   └── GDSC_IC50.csv
|
├── CCLE.tar
├── GDSC
│   ├── 1.Drug_listMon Jun 24 09_00_55 2019.csv
│   └── 223drugs_pubchem_smiles.txt
└── GDSC.tar
```

## Instructions

### Data Preparation

The script `process_data.py` is the entry for data preprocessing. It creats `train_data_split_ratio.npz` and `test_data_ratio.npz` under `./data/processed/`, which can be trained directly. 


### Training and Evaluation

The script `train.py` is the entry for CDR model's training and evaluating. It creats `CDRModel` from `model.py`. And the best model params are saved in `./best_model/`


Below are the detailed explanations of model parameters and an example of `train.py`'s use: 

```data_path```: The path you load data.First you need to download the datasets per the guidence above, It is recommended to untar the datasets and put it in the data folder under the root directory, if not, please create a new data folder.  
```output_path```: The path model results be saved in.  
```batch_size```: Batch size of the model, at training phase, the default value will be 64.  
```use_cuda```: Using GPU if used.  
```device```: GPU device number, use with ```use_cuda```.  
```layer_num```: Layer nums of convolutional graph.  
```units_list```: List of hidden size of each layer.  
```gnn_type```: Three choices of convolutional graphs are presented, which are GCN, GIN and GraphSage.  
```pool_type```: Graph pooling type.  
```epoch_num```: Epochs to train the model.  
```lr```: Learning rate to train the model.   

```sh
CUDA_VISIBLE_DEVICES=0 python pretrain_attrmask.py \
        --data_path='./data/processed/' \ 
        --output_path='./best_model/' \
        --batch_size=64 \
        --use_cuda \
        --device=0 \
        --layer_num=4 \
        --units_list=[256,256,256,100] \
        --gnn_type='gcn' \
        --pool_type='max' \
        --epoch_num=500 \
        --lr=1e-4 \  
```

Or, for convenience, use
```sh
test -d ./data/processed && python train.py || (python process_data.py && python train.py)
```
to cover data preparation and simply start training.



## Reference

**DeepCDR**
>@article{nguyen2020graphdta,title={DeepCDR: a hybrid graph convolutional network for predicting cancer drug response},author={Qiao Liu, Zhiqiang Hu, Rui Jiang, Mu Zhou},journal={Bioinformatics},year={2020},url={https://doi.org/10.1093/bioinformatics/btaa822}}



