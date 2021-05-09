# DDs(Drug Drug synergy)

[中文版本](./README_cn.md) [English Version](./README.md)

* [背景介绍](#背景介绍)
* [数据集](#数据集)
    * [ddi](#ddi)
    * [dti](#dti)
    * [ppi](#ppi)
    * [特征集](#特征集)
* [使用说明](#使用说明)
    * [训练与评估](#训练与评估)
* [引用](#引用)

## 背景
药物联用，也叫做协同治疗，通常在应对复杂疾病时使用。而图神经网络能够结合多种生物学网络从而来预测药物的协同作用。
## 数据集
药物协同的分值文件和药物的理化特征信息文件在 `data` 文件夹下. 首先在`data` 文件夹下创建`ddi`, `dti`和`ppi`文件夹。
### ddi
```sh
cd data/ddi && wget "http://www.bioinf.jku.at/software/DeepSynergy/labels.csv"
```
### dti
### ppi

## 使用说明
为了方便展示，我们构建了一个脚本， `train.py`.
用法如下:
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
请注意，如果训练环境没有GPU，去掉`--cuda`即可。 
## 引用
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