# SD VAE

## 背景
深度生成模型目前是对生成新分子和优化其化学属性的流行模型。在这项工作中，我们将会介绍一个基于分子序列语法和语义的VAE生成模型 - SD VAE。

## 指导
这个代码库将会给你训练SD VAE模型的指导。

## 数据

你可以从以下链接下载数据 [datalink] (https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/molecular_generation/data_SD_VAE.tgz).

1. 创建一个文件夹 './data'
2. 把下载的文件解压到 './data'

文件夹结构：

|__ data (project root)

|__ |__  data_SD_VAE

|__ |__  |__ context_free_grammars

|__ |__  |__ zinc


## 数据预处理
在训练和评估模型之间, 我们需要对数据进行预处理，以提取语法和语义信息：

    cd data_preprocessing
    
    python make_dataset_parallel.py \
    -info_fold ../data/data_SD_VAE/context_free_grammars \
        -grammar_file ../data/data_SD_VAE/context_free_grammars/mol_zinc.grammar \
        -smiles_file ../data/data_SD_VAE/zinc/250k_rndm_zinc_drugs_clean.smi 
        
        
    python dump_cfg_trees.py \
    -info_fold ../data/data_SD_VAE/context_free_grammars \
        -grammar_file ../data/data_SD_VAE/context_free_grammars/mol_zinc.grammar \
        -smiles_file ../data/data_SD_VAE/zinc/250k_rndm_zinc_drugs_clean.smi 
        

上面的两个文件将会把txt数据分别转化成二进制和cfg dump文件。

## 模型训练
    
#### 模型设置
模型设置将会设置建造模型所用的参数，它们保存在：model_config.json

    "latent_dim":the hidden size of latent space
    "max_decode_steps": maximum steps for making decoding decisions
    "eps_std": the standard deviation used in reparameterization tric
    "encoder_type": the type of encoder
    "rnn_type": The RNN type


#### 训练设置
为了训练模型，我们需要先设置模型的参数。默认参数值保存在文件：./mol_common/cmd_args.py

    -loss_type : the type of loss
    -num_epochs : number of epochs
    -batch_size : minibatch size
    -learning_rate : learning_rate
    -kl_coeff : coefficient for kl divergence used in vae
    -clip_grad : clip gradients to this value


训练模型：

    CUDA_VISIBLE_DEVICES=0 python train_zinc.py \
    -mode='gpu' \

#### 下载训练好的模型
你可以从以下链接下载已经预先训练好的模型 (https://baidu-nlp.bj.bcebos.com/PaddleHelix/models/molecular_generation/SD_VAE_model.tgz).

解压文件， 然后把模型保存在 './model' 文件夹，格式如下:

|__  model

|__ |__ train_model_epoch499


#### 模型采样
从正态先验中采样：

    python sample_prior.py \
      -info_fold ../data/data_SD_VAE/context_free_grammars  \
      -grammar_file ../data/data_SD_VAE/context_free_grammars/mol_zinc.grammar \
      -model_config ../model_config.json \
      -saved_model ../model/train_model_epoch499


从参考序列中重构：

    python reconstruct_zinc.py  \
      -info_fold ../data/data_SD_VAE/context_free_grammars \        
      -grammar_file ../data/data_SD_VAE/context_free_grammars/mol_zinc.grammar \      
      -model_config ../model_config.json \       
      -saved_model ../model/train_model_epoch499 \
      -smiles_file ../data/data_SD_VAE/zinc/250k_rndm_zinc_drugs_clean.smi 


##### 正态先验采样的结果
valid: 0.49

unique@100: 1.0

unique@1000: 1.0

IntDiv: 0.92

IntDiv2: 0.82

Filters: 0.30

##### 重构结果
accuracy: 0.92



## 参考文献
[1] @misc{dai2018syntaxdirected,
      title={Syntax-Directed Variational Autoencoder for Structured Data}, 
      author={Hanjun Dai and Yingtao Tian and Bo Dai and Steven Skiena and Le Song},
      year={2018},
      eprint={1802.08786},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}