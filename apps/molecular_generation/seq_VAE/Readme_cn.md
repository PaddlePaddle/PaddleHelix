# 序列 VAE

## 背景
深度生成模型目前是对生成新分子和优化其化学属性的流行模型。在这项工作中，我们将会介绍一个分子序列的的VAE生成模型 - 序列 VAE。


## 指导
这个代码库将会给出训练一个序列VAE模型的指导。


### 数据链接
下载训练数据ZINC Clean Leads[1]: (https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/molecular_generation/zinc_moses.tgz).

1. 创建一个文件夹 './data'
2. 把文件解压到文件夹 './data'

data
|-- zinc_mose
|   |-- train.csv
|   |-- test.csv


### 训练模型

#### 模型设置
模型设置将会设置建造模型所用的参数，它们保存在：model_config.json

  # 编码器
  "max_length":the maximun length of the inout sequence
    "q_cell": the RNN cell type of encoding network
    "q_bidir": if it's bidirectional RNN or not
    "q_d_h": the hidden size of encoding RNN
    "q_n_layers": the layer numbers of encoding RNN
    "q_dropout": the drop out rate of encoding RNN
    
    # 解码器
    "d_cell": the RNN cell type of decoding network
    "d_n_layers": the layer numbers of decoding RNN
    "d_dropout": the drop out rate of decoding RNN
    "d_z": the hidden size of latent space
    "d_d_h":the hidden size of decoding RNN
    "freeze_embeddings": if freeze the embedding layer
    
#### Training setting
为了训练模型，我们需要先设置模型的参数。默认参数值保存在文件：args.py

  
    # Train
    '--n_epoch': number of trianing epoch, default=1000
    '--n_batch': number of bach size, default=1000
    '--lr_start': 'Initial lr value, default=3 * 1e-4
    
    # kl annealing
    '--kl_start': Epoch to start change kl weight from, default=0
    '--kl_w_start': Initial kl weight value, default=0
    '--kl_w_end': Maximum kl weight value, default=0.05
    
训练模型

```

CUDA_VISIBLE_DEVICES=0 python trainer.py \

--device='gpu' \

--dataset_dir='./data/zinc_moses/train.csv' \

--model_config='model_config.json' \

--model_save='./results/train_models/' \

--config_save='./results/config/' \
```


#### 下载训练好的模型
你可以从以下链接下载已经预先训练好的模型 (https://baidu-nlp.bj.bcebos.com/PaddleHelix/models/molecular_generation/seq_VAE_model.tgz).


#### 从正态分布先验中采样的结果

Valid: 0.9765

Novelty: 0.731

Unique@1k: 0.993

Filters: 0.853

IntDiv: 0.846

## 参考文献

[1] @article{polykovskiy2020molecular,
  title={Molecular sets (MOSES): a benchmarking platform for molecular generation models},
  author={Polykovskiy, Daniil and Zhebrak, Alexander and Sanchez-Lengeling, Benjamin and Golovanov, Sergey and Tatanov, Oktai and Belyaev, Stanislav and Kurbanov, Rauf and Artamonov, Aleksey and Aladinskiy, Vladimir and Veselov, Mark and others},
  journal={Frontiers in pharmacology},
  volume={11},
  year={2020},
  publisher={Frontiers Media SA}
}