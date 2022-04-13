# BatchDTA


## Backgrounds

Candidate compounds with high binding affinities toward a target protein are likely to be developed as drugs. Deep neural networks (DNNs) have attracted increasing attention for drug-target affinity (DTA) estimation owning to their efficiency. However, the negative impact of batch effects caused by measure metrics, system technologies, and other assay information is seldom discussed when training a DNN model for DTA. Suffering from the data deviation caused by batch effects, the DNN models can only be trained on a small amount of "clean" data. Thus, it is challenging for them to provide precise and consistent estimations. 

We design a batch-sensitive training framework, namely BatchDTA, to train the DNN models. BatchDTA implicitly aligns multiple batches toward the same protein, alleviating the impact of the batch effects on the DNN models. Extensive experiments demonstrate that BatchDTA facilitates four mainstream DNN models (DeepDTA, GraphDTA_GCN, GraphDTA_GATGCN, MolTrans) to enhance the ability and robustness on multiple DTA datasets. The average concordance index (CI) of the DNN models achieves a relative improvement of 4.0%. BatchDTA can also be applied to the fused data collected from multiple sources to achieve further improvement.

## Dependencies

- python >= 3.7
- torch
- paddlepaddle >= 2.0.0rc0
- rdkit 
- sklearn


## Datasets

We provide the benchmark dataset Davis and KIBA with the 5-fold cross-validation of training set and the independant test set. We split based on the unseen proten sequence. We also provide the BindingDB dataset with the four subsets of indicators KD, KI, IC50 and EC50. Each subset is splitted as training/validation/test sets with ratio 8:1:1.

The processed datasets can be downloaded from [here](https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/dti_datasets/BatchDTA_processed_data.zip). Before running the scripts, please uncompress and put the downloaded directory with data files under `/apps/drug_target_interaction/batchdta/`.


## How to run

### BatchDTA
Considering the size of training set after making the pairs, we speed up the training process by using multiple GPUs distributively.

#### DeepDTA
```bash
cd ./pairwise/DeepDTA
```
##### run the training script for Davis or KIBA with cross-validation
```bash
python run_pairwise_deepdta_CV.py --data_path '../../Data/'  --dataset 'DAVIS' --is_mixed False 
```
##### run the training script for BindingDB 
```bash
python run_pairwise_deepdta_bindingDB.py --data_path '../../Data/'  "--is_mixed" False
```

#### GraphDTA
```bash
cd ./pairwise/GraphDTA
```
##### run the training script for Davis or KIBA with cross-validation
```bash
python -m torch.distributed.launch run_mixed_run_pairwise_GraphDTA_CV.py --data_path '../../Data/'  --dataset 'DAVIS' --is_mixed False 
```
##### run the training script for BindingDB 
```bash
python -m torch.distributed.launch run_pairwise_GraphDTA_BindingDB.py --data_path '../../Data/'  "--is_mixed" False
```

#### Moltrans
```bash
cd ./pairwise/Moltrans
```
##### run the training script for Davis or KIBA with cross-validation 
```bash
python run_pairwise_Moltrans_CV.py --data_path '../../Data/'  --dataset 'DAVIS' --is_mixed False 
```
##### run the training script for BindingDB 
```bash
python run_pairwise_Moltrans_bindingDB.py --data_path '../../Data/'  "--is_mixed" False
```


### Baseline
We reproduce and provide all the baseline backbone models as following.

#### DeepDTA
```bash
cd ./pointwise/DeepDTA
```
##### run the training script for Davis with cross-validation
```bash
CUDA_VISIBLE_DEVICES=0 python train_davis.py --batchsize 256 --epochs 100 --rounds 1 --lr 1e-3
```
##### run the training script for KIBA with cross-validation
```bash
CUDA_VISIBLE_DEVICES=0 python train_kiba.py --batchsize 256 --epochs 200 --rounds 1 --lr 1e-3
```
##### run the training script for BindingDB 
```bash
CUDA_VISIBLE_DEVICES=0 python train_bindingdb.py --batchsize 256 --epochs 50 --rounds 1 --lr 1e-3
```

#### GraphDTA
```bash
cd ./pointwise/GraphDTA
```
##### run the training script for Davis with cross-validation
```bash
python train_davis.py --batchsize 512 --epochs 100 --rounds 1 --lr 5e-4 --cudanum 0 --model 2
```
##### run the training script for KIBA with cross-validation
```bash
python train_kiba.py --batchsize 512 --epochs 200 --rounds 1 --lr 5e-4 --cudanum 0 --model 2
```
##### run the training script for BindingDB 
```bash
python train_bindingdb.py --batchsize 512 --epochs 50 --rounds 1 --lr 5e-4 --cudanum 0 --model 2
```

#### Moltrans
```bash
cd ./pointwise/Moltrans
```
##### run the training script for Davis with cross-validation
```bash
CUDA_VISIBLE_DEVICES=0 python train_davis.py --batchsize 64 --epochs 100 --rounds 1 --lr 5e-4
```
##### run the training script for KIBA with cross-validation
```bash
CUDA_VISIBLE_DEVICES=0 python train_kiba.py --batchsize 64 --epochs 200 --rounds 1 --lr 5e-4
```
##### run the training script for BindingDB 
```bash
CUDA_VISIBLE_DEVICES=0 python train_bindingdb.py --batchsize 64 --epochs 50 --rounds 1 --lr 5e-4
```



## Reference

**DAVIS**
```bibtex
@article{10.1038/nbt.1990,
    author = {Mindy I Davis, Jeremy P Hunt, Sanna Herrgard, Pietro Ciceri, Lisa M Wodicka, Gabriel Pallares, Michael Hocker, Daniel K Treiber and Patrick P Zarrinkar},
    title = {Comprehensive analysis of kinase inhibitor selectivity},
    journal = {Nature Biotechnology},
    year = {2011},
    url = {https://doi.org/10.1038/nbt.1990}
}
```

**KIBA**
```bibtex
@article{doi:10.1021/ci400709d,
    author = {Tang, Jing and Szwajda, Agnieszka and Shakyawar, Sushil and Xu, Tao and Hintsanen, Petteri and Wennerberg, Krister and Aittokallio, Tero},
    title = {Making Sense of Large-Scale Kinase Inhibitor Bioactivity Data Sets: A Comparative and Integrative Analysis},
    journal = {Journal of Chemical Information and Modeling},
    volume = {54},
    number = {3},
    pages = {735-743},
    year = {2014},
    doi = {10.1021/ci400709d},
    note = {PMID: 24521231},
    url = {https://doi.org/10.1021/ci400709d}
}
```

**BindingDB**
```bibtex
@article{10.1093/nar/gkv1072,
    author = {Gilson, Michael K. and Liu, Tiqing and Baitaluk, Michael and Nicola, George and Hwang, Linda and Chong, Jenny},
    title = {BindingDB in 2015: A public database for medicinal chemistry, computational chemistry and systems pharmacology},
    journal = {Nucleic Acids Research},
    volume = {44},
    number = {D1},
    pages = {D1045-D1053},
    year = {2015},
    month = {10},
    issn = {0305-1048},
    doi = {10.1093/nar/gkv1072},
    url = {https://doi.org/10.1093/nar/gkv1072}
}
```

**DeepDTA**
```bibtex
@article{ozturk2018deepdta,
  title={DeepDTA: deep drug--target binding affinity prediction},
  author={{\"O}zt{\"u}rk, Hakime and {\"O}zg{\"u}r, Arzucan and Ozkirimli, Elif},
  journal={Bioinformatics},
  volume={34},
  number={17},
  pages={i821--i829},
  year={2018},
  publisher={Oxford University Press}
}
```

**GraphDTA**
```bibtex
@article{nguyen2021graphdta,
  title={GraphDTA: Predicting drug--target binding affinity with graph neural networks},
  author={Nguyen, Thin and Le, Hang and Quinn, Thomas P and Nguyen, Tri and Le, Thuc Duy and Venkatesh, Svetha},
  journal={Bioinformatics},
  volume={37},
  number={8},
  pages={1140--1147},
  year={2021},
  publisher={Oxford University Press}
}
```

**MolTrans**
```bibtex
@article{huang2021moltrans,
  title={MolTrans: Molecular Interaction Transformer for drug--target interaction prediction},
  author={Huang, Kexin and Xiao, Cao and Glass, Lucas M and Sun, Jimeng},
  journal={Bioinformatics},
  volume={37},
  number={6},
  pages={830--836},
  year={2021},
  publisher={Oxford University Press}
}
```
