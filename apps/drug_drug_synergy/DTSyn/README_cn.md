# DTSyn(Dual-Transformer neural network predicting Synergistic pairs)

[中文版本](./README_cn.md) [English Version](./README.md)

* [背景介绍](#背景介绍)
* [数据集](#数据集)
    * [ddi](#ddi)
    * [lincs](#lincs)
    * [rna](#rna)
* [使用说明](#使用说明)
    * [训练与评估](#训练与评估)
* [引用](#引用)

## 背景
药物联用可以解决单药使用面料的耐药，毒副作用过大等问题。当前双药联合使用同时还面临着组合爆炸，机理不明确等问题。本模型通过借助transformer结构从不同粒度出发捕获不同角度的生物学互作信息。
## 数据集
药物协同的分值文件放在 `data` 文件夹下。
### 训练集
```sh
cd data && "wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/drug_synergy_datasets/DTSyn.tgz" && tar xzvf DTSyn.tgz
```

## 使用说明
为了方便展示，我们构建了一个脚本， `main.py`
用法如下:
```
CUDA_VISIBLE_DEVICES=0 python3 main.py 
                         --ddi ./data/ddi.csv
                         --lincs ./data//gene_vector.csv
                         --rna ./data/rna.csv
                         --epochs 150  
```
 
## 引用
**DTSyn**
> @article{jing2022DTSyn,
  title={DTSyn: a dual-transformer-based neural network to predict synergistic drug combinations},
  author={Jing Hu, Jie Gao, Xiaomin Fang, Zijing Liu, Fan Wang, Weili Huang, Hua Wu, Guodong Zhao},
  journal={preprint on bioRxiv}
}