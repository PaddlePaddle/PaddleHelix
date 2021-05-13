# DDs(Drug Drug synergy)

[中文版本](./README_cn.md) [English Version](./README.md)

* [Background](#background)
* [Datasets](#datasets)
    * [ddi](#ddi)
    * [dti](#dti)
    * [ppi](#ppi)
    * [features](#features)
* [Instructions](#instructions)
    * [Training and Evaluation](#train-and-evaluation)
* [Reference](#reference)

## Background

Drug combinations, also known as combinatorial therapy, are frequently prescribed to treat patients with complex disease. Graph convolutional network(GCN) can be used to predict drug-drug synergy by intergrating multiple biological networks.

## Datasets
Drug-drug synergy information and drug physi-chemical features can be put under `data` folder. Then let us create `ddi`, `dti` and `ppi` folder under `data` folder.
### ddi

```sh
cd data/ddi && wget "http://www.bioinf.jku.at/software/DeepSynergy/labels.csv"
```

### dti

### ppi 


## Instructions
For illustration, we provide a python script `train.py`.
Its usage is:
```
CUDA_VISIBLE_DEVICES=0 python3 train.py 
                         --ddi ./data/ddi/DDs.csv
                         --dti ./data/dti/drug_protein_links.tsv
                         --ppi ./data/ppi/protein_protein_links.txt
                         --d_feat ./data/all_drugs_name.fet
                         --epochs 10
                         --num_graph 10
                         --sub_neighbours 10 10
                         --cuda   
```
Notice that if you only have CPU machine, just remove `--cuda`. 

## Reference
**RGCN**
> @article{jiang2020deep,
  title={Deep graph embedding for prioritizing synergistic anticancer drug combinations},
  author={Jiang, Peiran and Huang, Shujun and Fu, Zhenyuan and Sun, Zexuan and Lakowski, Ted M and Hu, Pingzhao},
  journal={Computational and structural biotechnology journal},
  volume={18},
  pages={427--438},
  year={2020},
  publisher={Elsevier}
}