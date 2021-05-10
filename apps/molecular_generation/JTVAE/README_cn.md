# Junction Tree Variational Autoencoder

[中文版本](./README_cn.md) [English Version](./README.md)

* [背景介绍](#背景介绍)
* [使用说明](#使用说明)
    *  [下载链接](#下载链接)
        * [模型地址](#模型地址)
        * [数据地址](#数据地址)
    * [训练与评估](#训练与评估)
* [引用](#引用)
    * [论文相关](#论文相关)
    * [数据相关](#数据相关)

## 背景介绍
Junction Tree Variational Autoencoder的实现 https://arxiv.org/abs/1802.04364

## 使用说明

### 下载链接

#### 模型地址
您可以使用我们已经预训练好的[模型](https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/molecular_generation/vae_models.tgz)，也可以选择自己训练。

#### 数据地址
我们提供[训练数据集](https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/molecular_generation/zinc.tgz)。建议解压数据集并将其放入data/目录下。
若您需要在自己的数据集上训练，请执行以下命令，以生成该数据集的字典文件：
```bash 
python -m src.mol_tree \
    --train_path dataset_file_path \
    --vocab_file vocab_file_save_path
```

#### 训练与评估
数据预处理：
```bash 
python preprocess.py \
    --train data/zinc/250k_rndm_zinc_drugs_clean_sorted.smi \
    --save_dir zinc_processed \
    --split 100 \
    --num_workers 8
```

训练：
```bash
CUDA_VISIBLE_DEVICES=0 python vae_train.py \
        --train zinc_processed \
        --vocab data/zinc/vocab.txt \
        --config configs/config.json \
        --save_dir vae_models \
        --num_workers 2 \
        --epoch 50 \
        --batch_size 32 \
        --use_gpu True 
```
我们提供了一个配置`configs/config.json`方便用户初始化神经网络。
训练参数:
`beta`： KL regularization的系数。
`warmup`： `beta`在模型训练`warmup`个step后开始更新。
`step_beta`： `beta` 每经过`kl_anneal_iter`个step更新的增量。
`kl_anneal_iter`： `beta` 每经过`kl_anneal_iter`个step更新。
`max_beta`： `beta`的最大值。
`save_dir`： 模型保存的目录。

测试：
```bash
python sample.py \
        --nsample 10000 \
        --vocab data/zinc/vocab.txt \
        --model vae_models/model.iter-422000 \
        --config configs/config.json \
        --output sampling_output.txt
```
从model.iter-422000的采样结果如下：
```bash
valid,1.0
unique@1000,1.0
unique@10000,0.9997
IntDiv,0.8701593437246322
IntDiv2,0.8646974999795127
Filters,0.6084
Novelty,0.9998999699909973
```
由于我们没有从数据集中split测试集，所以没有做moses benchmark中测试集相关的测评。    

Fine-tuning:
```bash
CUDA_VISIBLE_DEVICES=0 python vae_train.py \
        --train zinc_processed \
        --vocab data/zinc/vocab.txt \
        --config configs/config.json \
        --save_dir vae_models \
        --num_workers 2 \
        --epoch 50 \
        --batch_size 32 \
        --use_gpu True \
        --load_epoch 422000
```


## 引用
### 论文相关
**Junction Tree Variational Autoencoder**
> @article{Jin2018,
  author = {Jin, Wengong and Barzilay, Regina and Jaakkola, Tommi},
  title = {{Junction Tree Variational Autoencoder for Molecular Graph Generation}},
  url = {http://arxiv.org/abs/1802.04364},
  journal={ICML 2018},
  year = {2018}
}
> @article{polykovskiy2020molecular,
      title={Molecular Sets (MOSES): A Benchmarking Platform for Molecular Generation Models}, 
      author={Daniil Polykovskiy and Alexander Zhebrak and Benjamin Sanchez-Lengeling and Sergey Golovanov and Oktai Tatanov and Stanislav Belyaev and Rauf Kurbanov and Aleksey Artamonov and Vladimir Aladinskiy and Mark Veselov and Artur Kadurin and Simon Johansson and Hongming Chen and Sergey Nikolenko and Alan Aspuru-Guzik and Alex Zhavoronkov},
      year={2020},
      eprint={1811.12823},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
### 数据相关
数据从zinc数据集中随机选择：
**ZINC15(Pre-training):**
> @article{doi:10.1021/ci3001277,
    annote = {PMID: 22587354},
    author = {Irwin, John J and Sterling, Teague and Mysinger, Michael M and Bolstad, Erin S and Coleman, Ryan G},
    doi = {10.1021/ci3001277},
    journal = {Journal of Chemical Information and Modeling},
    number = {7},
    pages = {1757--1768},
    title = {{ZINC: A Free Tool to Discover Chemistry for Biology}},
    url = {https://doi.org/10.1021/ci3001277},
    volume = {52},
    year = {2012}
}


