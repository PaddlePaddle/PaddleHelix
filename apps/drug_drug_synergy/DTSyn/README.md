# DTSyn(Dual-Transformer neural network predicting Synergistic pairs)

[中文版本](./README_cn.md) [English Version](./README.md)

* [Background](#background)
* [Dataset](#dataset)
    * [ddi](#ddi)
    * [lincs](#lincs)
    * [rna](#rna)
* [Example](#example)
    * [training and evaluation](#training&evaluation)
* [Reference](#reference)

## background
Drug combinations, compared to monotherapies, have the potential to increase efficacy, reduce host toxicity and overcome drug resistance. However, screening novel synergistic drug pairs is challenging due to the enormous number of potential combination space. Further, lacking the understanding of mechanism of action (MoA) also limits the application of drug combinations. Our model utilizes different granularity level transformers to capture biological interactions from different dimensions.

## dataset
drug combinations can be stored in  directory `data`.
### training data
```sh
cd data && "wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/drug_synergy_datasets/DTSyn.tgz" && tar xzvf DTSyn.tgz
```

## usage
We use `main.py` for illustration,
the cmdline is as follows:
```
CUDA_VISIBLE_DEVICES=0 python3 main.py 
                         --ddi ./data/ddi.csv
                         --lincs ./data//gene_vector.csv
                         --rna ./data/rna.csv
                         --epochs 150  
```
 
## Reference
**DTSyn**
> @article{jing2022DTSyn,
  title={DTSyn: a dual-transformer-based neural network to predict synergistic drug combinations},
  author={Jing Hu, Jie Gao, Xiaomin Fang, Zijing Liu, Fan Wang, Weili Huang, Hua Wu, Guodong Zhao},
  journal={preprint on bioRxiv}
}
}