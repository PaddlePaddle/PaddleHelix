# README_cn.md

# PretrainGNNs

[中文版本](./README_cn.md) [English Version](./README.md)

* [背景介绍](#背景介绍)
* [使用说明](#使用说明)
   *  [下载链接](#下载链接)
      * [模型地址](#模型地址)
      * [数据地址](#数据地址)
    * [模型训练](#模型训练)
    * [模型微调](#模型微调)
    * [图网络模型](#序列模型)
        * [GIN](#GIN)
        * [GAT](#GAT)
        * [GCN](#GCN)
        * [其他参数](#其他参数)
    * [化合物相关任务](#化合物相关任务)
        * [预训练任务](#预训练任务) 
           * [Pre-training datasets](#Pre-training-datasets)
            * [Node-level](#Node-level)
            * [Graph-level](#Graph-level)
        
        * [下游任务](#下游任务)
            * [Chemical molecular properties prediction](#Chemical-molecular-properties-prediction)
            *  [Downstream classification datasets](#Downstream-classification-datasets)
            *  [Fine-tuning](#Fine-tuning)
            
        * [评估结果](#评估结果)
* [数据](#数据)
   * [数据获取地址](#数据获取地址)
   * [数据介绍](#数据介绍)


* [引用](#引用)
    * [论文相关](#论文相关)
    * [数据相关](#数据相关)

## 背景介绍
在近些年来，深度学习在各个领域都取得了很好的成果，但是在分子信息学和药物研发领域内依旧有一些限制。而药物研发是一个比较昂贵并且耗时的过程，中间对成药性化合物的筛选是最需要提高效率的，早期大多用传统的机器学习方法来预测物化性质，而图形具有不规则的形状和大小，节点上也没有空间顺序，节点的邻居也与所处的位置有关，因此分子结构数据可以被看做图形来处理，图网络的应用开发也逐渐被重视起来。但是在实际训练的过程中可能会缺少特定任务的标签，测试集与训练集分布不同，因此本篇主要是采取在数据丰富的相关任务上预训练模型，在节点级别和整图级别进行预训练，再对下游任务进行微调的策略。本篇预训练模型参考论文[《Strategies for Pre-training Graph Neural Networks》](https://openreview.net/pdf?id=HJlWWJSFDH)，提供了GIN、GAT、GCN等模型进行实现。

## 使用说明

### 下载链接

#### 模型地址
这里我们提供已经已经预训练好的好的模型获取[地址](https://baidu-nlp.bj.bcebos.com/PaddleHelix/pretrained_models/compound/pretrain_gnns_attr_super.tgz)供您使用，您也可以选择自己训练。

#### 数据地址
您可以选择从我们提供的[网址](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip)上下载数据集然后进行相应的预处理来供您使用。建议解压数据集并将其放入根目录下的data文件夹中，如果没有请新建一个data文件夹。

    # cd to pahelix folder
    mkdir -p data
    cd data
    wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
    unzip chem_dataset.zip

### 模型训练

我们提供的预训练策略的训练方式分为两个方面，首先是在节点级别的预训练，一共有两种方法，其次是整图的监督预训练策略，在具体实验的过程中，你可以选择先在节点级别进行预训练，再在整图级别上进行图级别的预训练，具体模型结构图如下：

   ![图片](https://agroup-bos-bj.cdn.bcebos.com/bj-136829c31a8edcaa1800c88bdb02038cfb1630e6)


 以下为对具体三种策略进行验证的代码示例：

```bash 
pretrain_attrmask.py          # 节点级别的属性遮蔽预训练文件
pretrain_contextpred.py       # 节点级别的上下文预测的预训练文件
pretrain_supervised.py        # 整图级别的预训练文件
```
 - 以     pretrain_attrmask.py为例训练的相关参数解释如下:

`use_cuda` :  是否使用GPU

`lr` : 基准学习率, 这里用的是0.001

`batch_size` :  batch大小，训练阶段为256

`max_epoch` : 最大训练步数可以自己选择，但attrmask每一个epoch预估耗时15分钟左右，可以根据算力进行设置.

`data_path` : 加载数据的路径。首先需要从我们提供的链接中下载数据集，建议解压数据集并将其放入根目录下的data文件夹中，如果没有请新建一个data文件夹。

`init_model` :  init_model在这里是指加入预训练策略的模型，参考[路径](https://baidu-nlp.bj.bcebos.com/PaddleHelix/pretrained_models/compound/pretrain_gnns_attr_super.tgz)，你也可以根据需要自己调整。

`model_config` : 模型配置文件，关于配gnn_model的参数选择文件，可参考gnn_model.json.

`dropout_rate` : 模型随机丢弃的概率大小，在这里是0.5

`model_dir` : 模型的存放地址  

```bash
CUDA_VISIBLE_DEVICES=0 python pretrain_attrmask.py \
        --use_cuda \ 
        --batch_size=256 \ 
        --max_epoch=100 \ 
        --data_path=../../../data/chem_dataset/zinc_standard_agent/raw \
        --model_config=gnn_model.json \
        --model_dir=../../../output/pretrain_gnns/pretrain_attrmask        
```
 -  在这里我们提供了几种直接运行shell脚本的示例，你可以在这些脚本里更改你的模型配置参数：

```bash 
sh scripts/pretrain_attrmask.sh         #运行脚本，用此方法来执行上面的py文件，具体路径可根据需要更改。
sh scripts/pretrain_contextpred.sh      #运行脚本，用此方法来执行上面的py文件，具体路径可根据需要更改。
sh scripts/pretrain_supervised.sh       #运行脚本，用此方法来执行上面的py文件，具体路径可根据需要更改。
```

### 模型微调
模型微调和模型训练方式类似，具体的相关参数解释与上面的类似，目前是在8个数据集上进行下游任务的微调。init_model就是从上面下载的已经训练好的模型，放在对应的文件夹下以便进行finetune。
```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
        --use_cuda \
        --batch_size=128 \ 
        --max_epoch=4 \ 
        --dataset_name=tox21 \ 
        --data_path=../../../data/chem_dataset/tox21/raw \
        --model_config=gnn_model.json \  
        --init_model= ../../pretrain_gnns_attr_super \
        --model_dir=../../../output/pretrain_gnns/finetune/tox21
```
-  在这里我们提供了一个直接运行shell脚本的示例，你可以在这个脚本里更改你的模型配置参数：
```bash 
   sh scripts/local_finetune.sh   #运行脚本，用此方法来执行上面对应的py文件，具体路径可根据需要更改。
```


### 图网络模型
我们提供了GIN、GCN和GAT三三种图网络模型，以下为使用模型的具体介绍：

#### GIN
Graph Isomorphism Network (GIN) 图同构网络，使用递归迭代的方式对图中的节点特征按照边的结构进行聚合来进行计算，并且同构图处理后的图特证应该相同，对非同构图处理后的图特证应该不同。使用GIN需要设定以下超参数：

- hidden_size: GIN的hidden_size。
- embed_dim:GIN的嵌入向量。
- layer_num: GIN的层数。

GIN的详细介绍可参考以下文章：

- [How Powerful are Graph Neural Networks？](https://arxiv.org/pdf/1810.00826.pdf)


#### GAT
我们使用图注意力网络Graph Attention Network，它不依赖于完整的图结构，只依赖于边，采用Attention机制，可以为不同的邻居节点分配不同的权重。使用GAT我们需要设定以下超参数：

- hidden_size: GAT的hidden_size。
- embed_dim:GAT的嵌入向量。
- layer_num:  GAT的层数。

GAT可以参考以下文章：

- [Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf)

#### GCN
在这里我们使用多层GCN。使用GCN我们需要设定以下超参数：

- hidden_size: GCN的hidden_size。
- embed_dim:GCN的嵌入向量。
- layer_num:  GCN的层数。

GCN可以参考以下文章：

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf)


#### 其他参数
模型还可设置其他参数避免过拟合和模型参数值过大，以及调整运行速度。

- dropout: 模型参数dropout的比例。
- residual：模型是否使用残差网络。
- graph_norm：模型是否加入norm操作。
- layer_norm:模型是否选择layer_norm归一化操作。
- pool_type：GNN模型最后选用什么样的池化操作

### 化合物相关任务

参考论文[Pretrain-GNN](https://openreview.net/pdf?id=HJlWWJSFDH)，我们使用PaddleHelix复现了预训练任务和相关的下游任务。
#### 预训练任务

##### Pre-training datasets
 -  Node-level：使用的是从ZINC15数据库中采样的200万个未标记分子进行节点级自我监督的预训练。
 - Graph-level： 对于图级多任务监督式预训练，我们使用预处理的ChEMBL数据集，其中包含456K分子以及1310种多样且广泛的生化分析。

##### Node-level：Self-supervised pre-training

 在每个目录中，我们有两种方法来训练GNN。这会将预训练的模型加载到`model_dir`中，将所得的预训练模型log文件保存。

对于GNN的节点级预训练，我们的方法是先使用容易得到的unlabeled数据，并使用自然图分布来捕获图中的特定领域知识/规则。 接下来，再用两种self-supervised的方法：context prediction和attribute masking。 

 - Context prediction
    - 用子图来预测其周围的图结构，找到每个节点的邻域图和上下文图，用辅助的GNN把上下文编码成固定向量，再通过负采样来学习主GNN和上下文GNN，再用来预训练模型。
```bash
CUDA_VISIBLE_DEVICES=0 python pretrain_contextpred.py \
        --use_cuda \ 
        --batch_size=256 \ 
        --max_epoch=100 \ 
        --data_path=../../../data/chem_dataset/zinc_standard_agent/raw \
        --model_config=gnn_model.json \
        --model_dir= ../../../output/pretrain_gnns/pretrain_contextpred
```
 - Attribute masking
    - 通过学习分布在图结构上的节点/边属性的规律性来捕获领域知识，屏蔽node/edge属性，让GNN根据相邻结构预测这些属性。
```bash
CUDA_VISIBLE_DEVICES=0 python pretrain_attrmask.py \
        --use_cuda \ 
        --batch_size=256 \ 
        --max_epoch=100 \ 
        --data_path=../../../data/chem_dataset/zinc_standard_agent/raw \
        --model_config=gnn_model.json \
        --model_dir= ../../../output/pretrain_gnns/pretrain_attrmask
```

##### Graph-level ：Supervised pre-training

图级别的预训练是在节点级别预训练的基础上完成的。

 - 图级别多任务监督预训练
    - 首先先在单个节点级别上对GNN进行正则化，也就是上面的两个策略执行完之后再加在supervised，再在整个图上进行多任务的监督预训练，来预测各个图的不同监督标签集。

```bash
CUDA_VISIBLE_DEVICES=0 python pretrain_supervised.py \
        --use_cuda \ 
        --batch_size=256 \ 
        --max_epoch=100 \ 
        --data_path=../../../data/chem_dataset/chembl_filtered/raw \
        --model_config=gnn_model.json \
        --init_model=../../../output/pretrain_gnns/pretrain_attrmask \
        --model_dir=../../../output/pretrain_gnns/pretrain_supervised
```

这会将预训练的模型加载到`model_dir`中，使用监督式预训练进一步对其进行预训练，然后将所得的预训练模型log文件保存。



#### 下游任务
##### Chemical molecular properties prediction
   化学分子性质预测主要包含对已经预训练好的模型拿来进行finetune，下游任务主要是在图表示上添加线性分类器，来预测下游图形标签。然后再以端到端的方式进行微调。

##### Downstream classification datasets

选用的是从[MoleculeNet](http://moleculenet.ai/datasets-1)中包含的8个大的二分类数据集。

##### Fine-tuning

在每个目录中，我们提供三种训练GNN的方法，将使用下游任务数据集来微调 `model_dir`中指定的预训练模型，并且保存微调的结果。
```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
        --use_cuda \
        --batch_size=128 \ 
        --max_epoch=4 \ 
        --dataset_name=tox21 \  
        --data_path=../../../data/chem_dataset/tox21/raw \
        --model_config=gnn_model.json \  
        --init_model=../../pretrain_gnns_attr_super \
        --model_dir=../../../output/pretrain_gnns/finetune/tox21
```

#### 评估结果

使用图级别多任务监督预训练的模型对下游任务finetuning后的结果如下表，是八个二分类任务：

![图片](https://agroup-bos-bj.cdn.bcebos.com/bj-65af0e23c7f4a691db22fa2f7ee855ccf2f71087)



## 数据

### 数据获取地址

  您可以选择从我们提供的[网址](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip)上下载数据集然后进行相应的预处理来供您使用。

### 数据介绍
本篇化合物预训练方法使用论文[**Pretrain-GNN**](https://openreview.net/pdf?id=HJlWWJSFDH)中的数据集进行进一步处理。

 - BACE
    - 介绍：
       - BACE数据集提供了一组人β分泌酶1抑制剂（BACE-1）的定量（IC50）和定性（二进制标记）结合结果。 所有数据均为过去十年间科学文献中报道的实验值，其中一些具有详细的晶体结构。 提供了152种化合物的2D结构和性质。
    - 输入：
       - 数据文件包含一个csv表，其中使用了以下列：
         - “mol” :分子结构的SMILES表示
    -  特性：
       -  ”pIC50” : IC50结合亲和力的负对数
       - “class” : 抑制剂的二元标签 
       - Valid ratio: 1.0
       - Task evaluated: 1/1
   
 - BBBP
    -  介绍：
       - 血脑屏障渗透（Blood-brain barrier penetration）数据集是从对屏障渗透性建模和预测的研究中提取的。 作为分隔循环血液和大脑细胞外液的膜，血脑屏障可以阻止大多数药物，激素和神经递质。 因此，在针对中枢神经系统的药物开发中，屏障的渗透形成了长期存在的问题。 该数据集包括针对2000多种化合物的渗透性特性的二进制标记。
    -  输入：
       - 数据文件包含一个csv表，其中使用了以下列：
         - Num-编号
         - “名称”: 化合物的名称(药物之类的）
         - “SMILES” : 分子结构的SMILES表示
    - 特性：
       - “ p_np” : 渗透/不渗透的二进制标签
       - Valid ratio: 1.0
       - Task evaluated: 1/1      


 - Clintox
    - 介绍：
       - ClinTox数据集比较了FDA批准的药物和由于毒性原因而在临床试验中失败的药物。该数据集包括具有已知化学结构的1491种化合物的两个分类任务：（1）临床试验毒性（或无毒性）和（2）FDA批准状态。
       - FDA批准的药物清单是从SWEETLEAD数据库编制的，而由于毒性原因而在临床试验中失败的药物清单是从ClinicalTrials.gov（AACT）的汇总分析数据库编制的。

    -  输入：
       - 数据文件包含一个csv表，其中使用了以下列：
         -  “SMILES” : 分子结构的SMILES表示
    - 特性：
       -   “ FDA_APPROVED” : FDA批准状态   1是批准，0是不批准
       -  “ CT_TOX” : 临床试验结果  0表示没有毒，1是有毒
       -  Valid ratio: 1.0
       - Task evaluated: 2/2   
   
 - HIV
    - 介绍：
       - HIV数据集由药物治疗计划（DTP）AIDS抗病毒筛选引入，该筛选测试了40,000多种化合物抑制HIV复制的能力。 对筛选结果进行了评估，并将其分为三类：确认为无效（CI），确认为活跃（CA）和确认为中等活跃（CM）。 我们进一步结合了后两个标签，使其成为非活：动（CI）和活动（CA和CM）之间的分类任务。

    -  输入：
       - 数据文件包含一个csv表，其中使用了以下列：
         -   “SMILES” : 分子结构的SMILES表示
    - 特性：
       -   “ACTIVITY” : 用于筛选结果的三类标签：CI / CM / CA
       -  “HIV_active” - 筛查结果的二进制标签：1（CA / CM）和0（CI）
       - Valid ratio: 1.0 
       - Task evaluated: 1/1

 - MUV

    - 介绍：
       - 最大无偏验证（The Maximum Unbiased Validation）组是通过应用精确的最近邻分析从PubChem BioAssay中选择的基准数据集。 MUV数据集包含针对90,000种化合物的17个具有挑战性的任务，是专门为验证虚拟筛选技术而设计的。
    - 输入：
       - 数据文件包含一个csv表，其中使用了以下列：
         - “ mol_id” : 该化合物的PubChem CID
         -  “SMILES” : 分子结构的SMILES表示
    - 特性：
       -  “ MUV-XXX”-生物测定的测量结果（有效/无效）：测量指标
       -  Valid ratio: 0.155、0.160
       - Task evaluated: 15/17、16/17

 - SIDER
    - 介绍：
       - 副作用资源（SIDER）是市售药物和药物不良反应（ADR）的数据库。 DeepChem的SIDER数据集版本根据对1427种批准药物进行的MedDRA分类，将药物副作用分为27种系统器官类别。
    - 输入：
       - 数据文件包含一个csv表，其中使用了以下列：
         - “SMILES” : 分子结构的SMILES表示；
    - 特性：
       - “肝胆疾病”〜“伤害，中毒和程序并发症”-记录的药物副作用
       -   Valid ratio: 1.0
       -  Task evaluated: 27/27
 - Tox21
    - 介绍：
       - “ 21世纪的毒理学”（Tox21）计划创建了一个测量化合物毒性的公共数据库，该数据库已在2014年Tox21数据挑战赛中使用。 该数据集包含针对12种不同靶标上的8k化合物的定性毒性测量，包括核受体和应激反应途径。

    - 输入：
       - 数据文件包含一个csv表，其中使用了以下列：
         - “SMILES”-分子结构的SMILES表示
    - 特性：
       - “ NR-XXX” : 核受体信号传导生物测定结果
       - “ SR-XXX” : 压力反应生物测定结果
       -  Valid ratio: 0.751、0.760
       - Task evaluated: 12/12

 - Toxcast
    - 介绍：
       - ToxCast(toxicity forecaster)是来自与Tox21相同的计划的扩展数据收集，可基于体外高通量筛选为大型化合物库提供毒理学数据。 处理后的集合包括对8k化合物进行的600多次实验的定性结果。
    - 输入：
       - 数据文件包含一个csv表，其中使用了以下列：
         -  “SMILES” : 分子结构的SMILES表示
    - 特性：
       - “ ACEA_T47D_80hr_Negative”〜“ Tanguay_ZF_120hpf_YSE_up” : 生物测定结果
       -  Valid ratio: 0.234、0.268
       - Task evaluated: 610/617



## 引用
### 论文相关
本篇化合物预训练方法主要参考论文[**Pretrain-GNN**](https://openreview.net/pdf?id=HJlWWJSFDH)，部分训练方式，训练超参数略有不同。

**Pretrain-GNN:**
>@article{hu2019strategies,
  title={Strategies for Pre-training Graph Neural Networks},
  author={Hu, Weihua and Liu, Bowen and Gomes, Joseph and Zitnik, Marinka and Liang, Percy and Pande, Vijay and Leskovec, Jure},
  journal={arXiv preprint arXiv:1905.12265},
  year={2019}
}

**InfoGraph**
>@article{sun2019infograph,
  title={Infograph: Unsupervised and semi-supervised graph-level representation learning via mutual information maximization},
  author={Sun, Fan-Yun and Hoffmann, Jordan and Verma, Vikas and Tang, Jian},
  journal={arXiv preprint arXiv:1908.01000},
  year={2019}
}

**GIN**
>@article{xu2018powerful,
  title={How powerful are graph neural networks?},
  author={Xu, Keyulu and Hu, Weihua and Leskovec, Jure and Jegelka, Stefanie},
  journal={arXiv preprint arXiv:1810.00826},
  year={2018}
}

**GAT**
>@article{velivckovic2017graph,
  title={Graph attention networks},
  author={Veli{\v{c}}kovi{\'c}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Lio, Pietro and Bengio, Yoshua},
  journal={arXiv preprint arXiv:1710.10903},
  year={2017}
}

**GCN**
>@article{kipf2016semi,
  title={Semi-supervised classification with graph convolutional networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1609.02907},
  year={2016}
}

### 数据相关
数据是从[MoleculeNet](http://moleculenet.ai/datasets-1)中挑选的部分数据集：

**ZINC15(Pre-training):**
>@article{sterling2015zinc,
  title={ZINC 15--ligand discovery for everyone},
  author={Sterling, Teague and Irwin, John J},
  journal={Journal of chemical information and modeling},
  volume={55},
  number={11},
  pages={2324--2337},
  year={2015},
  publisher={ACS Publications}
}

**ChEMBL(Pre-training):**
>@article{bento2014chembl,
  title={The ChEMBL bioactivity database: an update},
  author={Bento, A Patr{\'\i}cia and Gaulton, Anna and Hersey, Anne and Bellis, Louisa J and Chambers, Jon and Davies, Mark and Kr{\"u}ger, Felix A and Light, Yvonne and Mak, Lora and McGlinchey, Shaun and others},
  journal={Nucleic acids research},
  volume={42},
  number={D1},
  pages={D1083--D1090},
  year={2014},
  publisher={Narnia}
}

**BACE:**
>@article{john2003human,
  title={Human $\beta$-secretase (BACE) and BACE inhibitors},
  author={John, Varghese and Beck, James P and Bienkowski, Michael J and Sinha, Sukanto and Heinrikson, Robert L},
  journal={Journal of medicinal chemistry},
  volume={46},
  number={22},
  pages={4625--4630},
  year={2003},
  publisher={ACS Publications}
}
**BBBP:**
>@article{martins2012bayesian,
  title={A Bayesian approach to in silico blood-brain barrier penetration modeling},
  author={Martins, Ines Filipa and Teixeira, Ana L and Pinheiro, Luis and Falcao, Andre O},
  journal={Journal of chemical information and modeling},
  volume={52},
  number={6},
  pages={1686--1697},
  year={2012},
  publisher={ACS Publications}
}

**ClinTox:**
>@article{gayvert2016data,
  title={A data-driven approach to predicting successes and failures of clinical trials},
  author={Gayvert, Kaitlyn M and Madhukar, Neel S and Elemento, Olivier},
  journal={Cell chemical biology},
  volume={23},
  number={10},
  pages={1294--1301},
  year={2016},
  publisher={Elsevier}
}

**HIV:**
>@inproceedings{kramer2001molecular,
  title={Molecular feature mining in HIV data},
  author={Kramer, Stefan and De Raedt, Luc and Helma, Christoph},
  booktitle={Proceedings of the seventh ACM SIGKDD international conference on Knowledge discovery and data mining},
  pages={136--143},
  year={2001}
}

**MUV:**
>@article{rohrer2009maximum,
  title={Maximum unbiased validation (MUV) data sets for virtual screening based on PubChem bioactivity data},
  author={Rohrer, Sebastian G and Baumann, Knut},
  journal={Journal of chemical information and modeling},
  volume={49},
  number={2},
  pages={169--184},
  year={2009},
  publisher={ACS Publications}
}

**SIDER:**
>@article{kuhn2016sider,
  title={The SIDER database of drugs and side effects},
  author={Kuhn, Michael and Letunic, Ivica and Jensen, Lars Juhl and Bork, Peer},
  journal={Nucleic acids research},
  volume={44},
  number={D1},
  pages={D1075--D1079},
  year={2016},
  publisher={Oxford University Press}
}

**Tox21:**
>@article{capuzzi2016qsar,
  title={QSAR modeling of Tox21 challenge stress response and nuclear receptor signaling toxicity assays},
  author={Capuzzi, Stephen J and Politi, Regina and Isayev, Olexandr and Farag, Sherif and Tropsha, Alexander},
  journal={Frontiers in Environmental Science},
  volume={4},
  pages={3},
  year={2016},
  publisher={Frontiers}
}

**ToxCast:**
>@article{richard2016toxcast,
  title={ToxCast chemical landscape: paving the road to 21st century toxicology},
  author={Richard, Ann M and Judson, Richard S and Houck, Keith A and Grulke, Christopher M and Volarath, Patra and Thillainadarajah, Inthirany and Yang, Chihae and Rathman, James and Martin, Matthew T and Wambaugh, John F and others},
  journal={Chemical research in toxicology},
  volume={29},
  number={8},
  pages={1225--1251},
  year={2016},
  publisher={ACS Publications}
}
