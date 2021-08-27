# S-MAN: Spatial-aware Molecule Graph Attention Network

[中文版本](./SMAN-README.CN.md) [English Version](./SMAN-README.md)

S-MAN是一个用来预测药物-靶点反应亲和性的深度学习框架，这是基于PaddlePaddle和Paddle Graph Learning (PGL)的实现代码。该方法的论文参见：[Distance-aware Molecule Graph Attention Network for Drug-Target Binding Affinity Prediction](https://arxiv.org/abs/2012.09624).

### 数据集
PDBbind数据集可以从这个[网址](http://pdbbind-cn.org/download.php)进行下载.
下载数据集之后，你在运行模型之前需要先生成蛋白质-配体的分子图和节点特征。处理好的蛋白质-配体的分子图和节点特征也可以在这里下载:[蛋白质-配体数据](https://baidu-nlp.bj.bcebos.com/PaddleHelix%2Fdatasets%2Fdti_datasets%2Fsman-data.tgz)

你也可以下面的命令对PDBbind数据集进行预处理，从而生成蛋白质-配体的分子图和节点特征。
```
python preprocess.py --data_path YOUR_DATASET_PATH --dataset_name v2016_LPHIN3f5t_Sp --output_path YOUR_OUTPUT_PATH --cutoff 5
```
说明: cutoff参数是构建图的距离切割阈值。



### 包依赖

- networkx >= 2.1
- paddlepaddle >=  1.8.4
- pgl >= 1.1.0
- openbabel == 3.1.1 (可选的)
### 运行说明

例如，可以通过运行如下命令使用PBDbind数据集来训练模型：
```
python train.py --lr_d --data_path YOUR_DATA_PATH --dataset v2016_LPHIN3f5t_Sp --save_path MODEL_SAVE_PATH --gpu YOUR_DEVICE
```
你也可以通过下面的命令加载保存的模型进行测试：
```
python test.py --data_path YOUR_DATA_PATH --dataset v2016_LPHIN3f5t_Sp --model_path YOUR_MODEL_PATH --gpu YOUR_DEVICE
```
#### 超参数

- dataset: 数据集名称
- num_layers: 模型GNN层的数目
- dist_dim: 对原子距离进行划分的维度
- lr: 学习率
- lr_d: 使用学习率衰减
- drop: dropout比例

- data_path: 数据集文件夹路径
- save_path: 保存训练模型的路径（例如, ./runs)
- model_path: 在测试阶段要加载的模型路径 (例如, ./runs/SMAN)

### 参考文献

**S-MAN**
> @article{zhou2020distance,
  title={Distance-aware Molecule Graph Attention Network for Drug-Target Binding Affinity Prediction},
  author={Zhou, Jingbo and Li, Shuangli and Huang, Liang and Xiong, Haoyi and Wang, Fan and Xu, Tong and Xiong, Hui and Dou, Dejing},
  journal={arXiv preprint arXiv:2012.09624},
  year={2020},
  url={https://arxiv.org/abs/2012.09624}
}
	

