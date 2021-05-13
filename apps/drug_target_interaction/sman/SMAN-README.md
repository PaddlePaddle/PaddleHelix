# S-MAN: Spatial-aware Molecule Graph Attention Network

[中文版本](./SMAN-README.CN.md) [English Version](./SMAN-README.md)

DiStance-aware Molecule Graph Attention Network (S-MAN) is a novel deep learning framework to predict drug-target binding affinity(DTA). This is the implementation of S-MAN based on PaddlePaddle and PGL. The paper about this implementation is: [Distance-aware Molecule Graph Attention Network for Drug-Target Binding Affinity Prediction](https://arxiv.org/abs/2012.09624).

### Datasets

The PDBbind dataset can be downloaded [here](http://pdbbind-cn.org/download.php).
After downloading the data, you should first preprocess dataset to generate the protein-ligand graph and features.
The preprocessed Protein-Ligand graph and feature can be downloaded here: [Protein-Ligand dataset](https://baidu-nlp.bj.bcebos.com/PaddleHelix%2Fdatasets%2Fdti_datasets%2Fsman-data.tgz).


You can also run the following commnd to preprocess the PDBbind dataset to generate the protein-ligand graph and features. 
```
python preprocess.py --data_path YOUR_DATASET_PATH --dataset_name v2016_LPHIN3f5t_Sp --output_path YOUR_OUTPUT_PATH --cutoff 5
```
PS: cutoff is the threshold of distance cutoff between atoms.

### Dependencies

- networkx >= 2.1
- paddlepaddle >=  1.8.4
- pgl >= 1.1.0
- openbabel == 3.1.1 (optional)
### How to run

For examples, use gpu to train S-MAN on PDBbind dataset.
```
python train.py --lr_d --data_path YOUR_DATA_PATH --dataset v2016_LPHIN3f5t_Sp --save_path MODEL_SAVE_PATH --gpu YOUR_DEVICE
```
You can also test the saved models as follows:
```
python test.py --data_path YOUR_DATA_PATH --dataset v2016_LPHIN3f5t_Sp --model_path YOUR_MODEL_PATH --gpu YOUR_DEVICE
```
#### Hyperparameters

- dataset: name of dataset
- num_layers: number of GNN layers
- dist_dim: dimensions or buckets of distance spliting
- lr: learning rate
- lr_d: use learning rate decay
- drop: dropout ratio

- data_path: file path of dataset
- save_path: the path to save model （e.g, ./runs)
- model_path: file path of the saved model (e.g, ./runs/SMAN)


### Reference

**S-MAN**
> @article{zhou2020distance,
  title={Distance-aware Molecule Graph Attention Network for Drug-Target Binding Affinity Prediction},
  author={Zhou, Jingbo and Li, Shuangli and Huang, Liang and Xiong, Haoyi and Wang, Fan and Xu, Tong and Xiong, Hui and Dou, Dejing},
  journal={arXiv preprint arXiv:2012.09624},
  year={2020},
  url={https://arxiv.org/abs/2012.09624}
}
	