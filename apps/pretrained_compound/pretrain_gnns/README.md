# README.en.md

[中文版本](./README.ch.md) [English Version](./README.en.md)

 * [Background](#Background)
 * [Instructions](#Instructions)
	* [Training Models](#Training-Models)
    * [Evaluating Models](#Evaluating-Models)
    * [GNN Models](#GNN-Models)
        * [GIN](#gin)
        * [GAT](#gat)
        * [GCN](#gcn)
        *  [GraphSAGE](#graphsage)
        * [Other Parameters](#Other-Parameters)
    * [Compound Related Tasks](#Compound-Related-Tasks)
        * [Pretraining Tasks](#Pretraining-Tasks)
            * [Pre-training datasets](#Pre-training-datasets)
            * [Node-level](#node-level)
            * [Graph-level](#graph-level)
            
        * [Downstream Tasks](#Downstream-Tasks)
            * [Chemical molecular properties prediction](#chemical-molecular-properties-prediction)
            *  [Downstream classification datasets](#Downstream-classification-datasets)
            * [Fine-tuning](#fine-tuning)
        
	 *  [Evaluating results](#Evaluating-results)

    * [Warm Start / Finetuning](#Warm-Start-Finetuning)

* [Data](#Data)
		*  [How to get ?](#How-to-get-?)
		*  [Datasets introduction](#Datasets-introduction)
* [Pre-trained Models](#Pre-trained-Models)
		* [How to get ?](#How-to-get-?)
*  [Q&A](#q&a)
* [Reference](#Reference)
	    * [Paper-related](#Paper-related)
	    * [Data-related](#Data-related)

## Background
In recent years, deep learning has achieved good results in various fields, but there are still some limitations in the fields of molecular informatics and drug development.  However, drug development is a relatively expensive and time-consuming process. The screening of pharmaceutical compounds in the middle of the process is in need for efficiency improving. In the early days, traditional machine learning methods were used to predict physical and chemical properties, and the graphs have irregular shapes and sizes.  There is no spatial order on the nodes, and the neighbors of the nodes are also related to their positions. Therefore, molecular structure data can be treated as graphs, and the application development of graph networks is gradually being valued.  However, in the actual training process, the model's performance is limited by  missing labels, and different distributions between the training and testing set. Therefore, this article mainly adopts pre-training models on data-rich related tasks, and pre-training at the node level and the entire image level.  , And then fine-tune the downstream tasks.  This pre-training model refers to the paper "Strategies for Pre-training Graph Neural Networks", which provides GIN, GAT, GCN, Graphsage and other models for implementation.
Therefore, we implement the model mentioned in "Strategies for Pre-training Graph Neural Networks" to mitigate this issue. The model is firstly pre-trained on the data-rich related tasks, on both node level and graph level. Then the pre-trained model is fine-tuned for the downstream tasks. As for the implementation details, we provide GIN,GAT,GCN, Graphsage and other models implementation of the model.
## Instructions

### Training Models

The training methods of the pre-training strategy we provide are divided into two aspects. The first is the pre-training at the node level. There are two methods. The second is the supervised pre-training strategy for the whole image. You can choose during the specific experiment.  Perform pre-training at the node level first, and then perform the pre-training at the graph level at the entire graph level, as follows:
![图片](https://agroup-bos-bj.cdn.bcebos.com/bj-136829c31a8edcaa1800c88bdb02038cfb1630e6)
Following are the examples:
```
    pretrain_attrmask.py		 \ #Node-level attribute masking pre-training file
	pretrain_contextpred.py      \ #Pre-training file for node-level context prediction
	pretrain_supervised.py       \ #Pre-training files at the entire image level
```
Using pretrain_attrmask.py as an example to show the usage of the model parameters:

`use_cuda` : Whether to use GPU

`lr` : The base learning rate

`batch_size` : Batch size of the model, at training phase, the default value will be 256

`max_epoch` : Max epochs to train the model, can be chosen according to the compute power (Train on a single Telsa V will take about 11 minutes to finish an epoch)

`train_data` : The path of the training data folder, containing multiple training data files

`test_data` : The path of the testing data folder, containing multiple testing data files. If not specified, the mode will not be evaluted during the process of the training.

`init_model` : init_model referes to the model without using the pre-training strategy

`model_config` : the path of the model config file, containing the parameters of the gnn model

`dropout_rate` : dropout rate of the model. 0, 0.2, 0.5 can be chosen

`model_dir` : the path to save the model

`log_dir` : the path to save the log files

`mask_ratio` : the ratio of masking

```bash
CUDA_VISIBLE_DEVICES=0 paddle2.0 -m paddle.distributed.launch pretrain_attrmask.py \
		--use_cuda \ 
		--batch_size=$batch_size \ 
		--max_epoch=$max_epoch \ 
		--lr=$lr \ 
		--train_data=$npz_root/train \  
		--test_data=$npz_root/test \ 
		--model_config=$model_config \ 
		--init_model=$init_model \ 
		--model_dir=../../model/ \ 
		--dropout_rate=$dropout_rate \
		--log_dir=../../log/ \
		--mask_ratio=0.15 \ 

```

We provide the shell scripts to run the python files directly, you can adjust the parameters in the scripts.

```bash
    sh local_pretrain_attrmask.sh       #run pretrain_attrmask.py with given parameters
    sh local_pretrain_contextpred.sh    #run pretrain_contextpred.py with given parameters
    sh local_pretrain_supervised.sh     #run pretrain_supervised.py with given parameters 
```

### Fine Tune

Fine tuning the model is similar to trainging the model. Parameters' definition is the same. 

```bash
CUDA_VISIBLE_DEVICES=$cuda_id paddle2.0 finetune.py \ 
                --use_cuda \ 
                --batch_size=$batch_size \ 
                --max_epoch=$max_epoch \ 
                --lr=$lr \ 
                --dataset_name=$dataset \ 
                --train_data=$data_root/train \
                --valid_data=$data_root/valid \ 
                --test_data=$data_root/test \ 
                --model_config=$model_config \ 
                --init_model=$init_model \ 
                --model_dir=../../model/$dataset-$cuda_id \ 
                --dropout_rate=$dropout_rate \ #
                --log_dir=../../log/$dataset-$cuda_id &> ../../log/$prefix-$cuda_id.log 
```

We proivde the shell script to run the fine tune file, you can adjust the parameters in the script.

```bash
sh local_finetune.sh                 #run fine_tune.py with given parameters 
```

### GNN Models
We provides models GIN、GCN、GAT and GraphSAGE。We use parameter "--model_type" to specific the model we want to train. The model related parameters should be included in "--model_param".

#### GIN
Graph Isomorphism Network (GIN) Graph Isomorphism Network It uses recursive iterative method to aggregate the node features in the graph according to the structure of the edges to calculate, and the graph characteristics after isomorphism graph processing should be the same, and non-isomorphism graph processing  The following figure special certificate should be different.  To use GIN, you need to set the following hyperparameters:

- hidden_size: The hidden size of GIN。
- embed_dim:The embedding size of GIN。
- lay_num: The number of layers in GIN。

```bash
python model.py \
        ... # Training parameter settings, which has already introduced in the previous sections. \
        --gnn_type gin \
        --model_param "hidden_size=256" "embed_dim=300" "num_layers=5" "dropout_rate=0.5" "encode_type=concat" \
        ... # Task parameter settings will be introduced in the following chapters.
```

For details of GIN, please refer to the following papers:

- [How Powerful are Graph Neural Networks？](https://arxiv.org/pdf/1810.00826.pdf)



#### GAT
We use the Graph Attention Network, which does not rely on the complete graph structure, but only on the edges. Using the Attention mechanism, different neighbor nodes can be assigned different weights.  To use GAT we need to set the following hyperparameters:

- hidden_size: The hidden size of GAT。
- embed_dim:The embedding size of GAT。
- layer_num:  The number of layers in GAT。


For details of GAT, please refer to the following papers:

- [Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf)

#### GCN
Here we use multi-layer GCN.  To use GCN we need to set the following hyperparameters:

- hidden_size: The hidden size of  GCN。
- embed_dim:The embedding size of GCN。
- layer_num:  The number of layers in  GCN。


For details of GCN, please refer to the following papers:

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf)

#### GraphSAGE
 GraphSAGE simultaneously uses the node feature information and structure information to obtain the mapping of Graph Embedding, and saves the mapping for generating embedding, which is more scalable, and is also more prominent for node classification and link prediction problems.  Here we use multi-layer GraphSAGE.  To use GraphSAGE we need to set the following hyperparameters:


- hidden_size: The hidden size of  GraphSAGE。
- embed_dim:The embedding size of GraphSAGE。
- layer_num:  The number of layers in   GraphSAGE。



For details of GraphSAGE, please refer to the following papers:

- [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)


#### Other Parameters
The model can also set other parameters to avoid over-fitting and excessive model parameter values, and to adjust the running speed.

- dropout: The dropout ratio of model parameters.
- residual：Whether the model uses residual network.
- graph_norm：Whether the model joins norm operation.
- layer_norm: Whether the model use layer_norm.
- pool_type：What pooling operation does the GNN model choose?


### Compound related tasks
Referring to the paper [[Pretrain-gnn](https://openreview.net/pdf?id=HJlWWJSFDH)](https://openreview.net/pdf?id=HJlWWJSFDH), we reproduced the following tasks using PaddleHelix.

#### Pretraining Tasks
##### Pre-training datasets      
 -  Node-level：Two million unlabeled molecules sampled from the ZINC15 database are used for node-level self-supervised pre-training.
 - Graph-level：For graph-level multi-task supervised pre-training, we use the pre-processed ChEMBL data set, which contains 456K molecules and 1310 diverse and extensive biochemical analyses.

##### Node-level : Self-supervised pre-training
 We have two methods to pretrain GNN, it will load the pretrained model to`model_dir`, and save the log file to `log_dir`.

For the node-level pre-training of GNN, our method is to first use the easily available unlabeled data, and use the natural graph distribution to capture the specific domain knowledge/rules in the graph.  Next, use two more self-supervised methods：context prediction and attribute masking。 

 - Context prediction
	 - Use subgraphs to predict the surrounding graph structure, find the neighborhood graph and context graph of each node, use auxiliary GNN to encode the context into a fixed vector, and then use negative sampling to learn the main GNN and context GNN, and then use it to predict  train the model.

```bash
paddle2.0 -m paddle.distributed.launch pretrain_contextpred.py\
        ... # Training parameter settings, which has already introduced in the previous sections. \
        --model_dir=../../model/pretrain_contextpred/$dataset \ # the pretrained model address
        --log_dir=../../logs/pretrain_contextpred/$dataset   \  # the log file address
        --task context prediction
```

 - Attribute masking
	 - The domain knowledge is captured by learning the regularity of the node/edge attributes distributed on the graph structure, the node/edge attributes are shielded, and the GNN can predict these attributes based on the adjacent structure.
	 
```bash
paddle2.0 -m paddle.distributed.launch pretrain_attrmask.py \
         ... # Training parameter settings, which has already introduced in the previous sections. \
        --model_dir=../../model/pretrain_contextpred/$dataset \ # the pretrained model address
        --log_dir=../../logs/pretrain_contextpred/$dataset   \  # the log file address
        --task attribute masking
```
##### Graph-level ：Supervised pre-training

The pre-training at the graph level is completed on the basis of the pre-training at the node level.

 - Graph-level multi-task supervised pre-training
	 - First, the GNN is regularized at the single node level, that is, after the above two strategies are executed, they are added to supervised, and then multi-task supervised pre-training is performed on the entire graph to predict the different supervised label sets of each graph  .
```bash
paddle2.0 pretrain_supervised.py \
        ... # Training parameter settings, which has already introduced in the previous sections. \
        --model_dir=../../model/pretrain_supervised/$dataset \ #choose to add which one pretrained model
        --log_dir=../../logs/pretrain_supervised/$dataset \ # log file
        --task supervised
```

It will load the pretrained model to `model_dir`,and use supervised pretraining to do further experiments, and it will save the pretrained model log file to `log_dir`.

 - Structural similarity prediction
	 - Model the structural similarity of two graphs, including graph editing distance modeling or predicting graph structure similarity.
##### Downstream tasks

#####Chemical molecular properties prediction

The prediction of chemical molecular properties mainly includes finetune on the pre-trained model, and the downstream task is mainly to add a linear classifier to the graph representation to predict the downstream graph label.  And then fine-tune it in an end-to-end manner.

##### Downstream classification binary datasets

The  8 classification  8 classification binary datasets we used are collected form[MoleculeNet](http://moleculenet.ai/datasets-1).

#####Fine-tuning
In each directory, we provide three methods for training GNN, which will use the downstream task data set to fine-tune the pre-trained model specified in `model_dir`.  The result of fine tuning will be saved in `log_dir`.

```bash
paddle2.0 finetune.py \
        ... # Training parameter settings, which has already introduced in the previous sections. \
        --model_dir=../../model/finetune/$dataset \ # choosen which model to use
        --log_dir=../../logs/finetune/$dataset # the log file
        --task finetune
```


##### Evaluation Results
The results of finetuning downstream tasks using the graph-level multi-task supervision pre-training model are as follows, which are eight binary classification tasks:
![图片](https://agroup-bos-bj.cdn.bcebos.com/bj-3deae92e534c38eeb847a9992b766171a9e45d28)

### Warm Start / Finetuning
We can set the parameter "--init_model " to initialize the model or finetune the supervised tasks during the training process.

```bash
python finetune.py \
        ... \
        --init_model ./init_model # Directory of the initialization model. If this parameter is unset, the model is randomly initialized. \
        ... 
```

## Data
**data's address**
You can choose to download the dataset from the [link](http://moleculenet.ai/datasets-1) provided by us and perform the corresponding preprocessing for your use. If you need a processed dataset, you can also contact us  .
### Data introduction
This compound pre-training method uses the data set in the paper [**Pretrain-GNN**](https://openreview.net/pdf?id=HJlWWJSFDH) for further processing.

 - BACE
	 - Introduction：
		 -  BACE dataset provides quantitative (IC50) and qualitative (binary label) binding results for a set of inhibitors of human β-secretase 1 (BACE-1). All data are experimental values reported in scientific literature over the past decade, some with detailed crystal structures available. A collection of 1522 compounds with their 2D structures and properties are provided.
	 - Input：
		 - The data file contains a csv table, in which columns below are used:
			 - “mol” - SMILES representation of the molecular structure
	 - Properties：
		 - ”pIC50” - Negative log of the IC50 binding affinity
		 - “class” - Binary labels for inhibitor
		 - Valid ratio: 1.0
		 - Task evaluated: 1/1
	 
	 
 - BBBP
	 -  Introduction：
		 - The Blood-brain barrier penetration (BBBP) dataset is extracted from a study on the modeling and prediction of the barrier permeability. As a membrane separating circulating blood and brain extracellular fluid, the blood-brain barrier blocks most drugs, hormones and neurotransmitters. Thus penetration of the barrier forms a long-standing issue in development of drugs targeting central nervous system. This dataset includes binary labels for over 2000 compounds on their permeability properties.

	 -  Input：
		 - The data file contains a csv table, in which columns below are used:
			 - Num:number
			 - ”name” - Name of the compound
			 - “smiles” - SMILES representation of the molecular structure
	 - Properties：
		 - ”p_np” - Binary labels for penetration/non-penetration
		 - Valid ratio: 1.0
		 - Task evaluated: 1/1		

 - Clintox
	 - Introduction：
		 - The ClinTox dataset compares drugs approved by the FDA and drugs that have failed clinical trials for toxicity reasons. The dataset includes two classification tasks for 1491 drug compounds with known chemical structures: (1) clinical trial toxicity (or absence of toxicity) and (2) FDA approval status. List of FDA-approved drugs are compiled from the SWEETLEAD database, and list of drugs that failed clinical trials for toxicity reasons are compiled from the Aggregate Analysis of ClinicalTrials.gov(AACT) database.
	 -  Input：
		 - The data file contains a csv table, in which columns below are used:
			 - “smiles” - SMILES representation of the molecular structure
	 - Properties：
		 -  ”FDA_APPROVED” - FDA approval status
		 - “CT_TOX” - Clinical trial results
		 - Valid ratio: 1.0
		 - Task evaluated: 2/2		 
		
 - HIV
	 - Introduction：
		 - The HIV dataset was introduced by the Drug Therapeutics Program (DTP) AIDS Antiviral Screen, which tested the ability to inhibit HIV replication for over 40,000 compounds. Screening results were evaluated and placed into three categories: confirmed inactive (CI),confirmed active (CA) and confirmed moderately active (CM). We further combine the latter two labels, making it a classification task between inactive (CI) and active (CA and CM).

	 -  Input：
		 - The data file contains a csv table, in which columns below are used:
			 -  “smiles” - SMILES representation of the molecular structure
	 - Properties：
		 -  ”activity” - Three-class labels for screening results: CI/CM/CA
		 - “HIV_active” - Binary labels for screening results: 1 (CA/CM) and 0 (CI)
		 - Valid ratio: 1.0 
		 - Task evaluated: 1/1
 - MUV

	 - Introduction：
		 - The Maximum Unbiased Validation (MUV) group is a benchmark dataset selected from PubChem BioAssay by applying a refined nearest neighbor analysis. The MUV dataset contains 17 challenging tasks for around 90,000 compounds and is specifically designed for validation of virtual screening techniques.

	 - Input：
		 - The data file contains a csv table, in which columns below are used:
			 - ”mol_id” - PubChem CID of the compound
			 -  “smiles” - SMILES representation of the molecular structure.
	 - Properties：
		 -  ”MUV-XXX” - Measured results (Active/Inactive) for bioassays.
		 - Valid ratio: 0.155、0.160
		 - Task evaluated: 15/17、16/17

 - SIDER
	 - Introduction：
		 - The Side Effect Resource (SIDER) is a database of marketed drugs and adverse drug reactions (ADR). The version of the SIDER dataset in DeepChem has grouped drug side effects into 27 system organ classes following MedDRA classifications measured for 1427 approved drugs.
	 
	 - Input：
		 - The data file contains a csv table, in which columns below are used:
			 - “smiles” - SMILES representation of the molecular structure
	 - Properties：
		 - ”Hepatobiliary disorders” ~ “Injury, poisoning and procedural complications” - Recorded side effects for the drug
		 - Valid ratio: 1.0
		 - Task evaluated: 27/27
 - Tox21
	 - Introduction：
		 - The “Toxicology in the 21st Century” (Tox21) initiative created a public database measuring toxicity of compounds, which has been used in the 2014 Tox21 Data Challenge. This dataset contains qualitative toxicity measurements for 8k compounds on 12 different targets, including nuclear receptors and stress response pathways.

	 - Input：
		 - The data file contains a csv table, in which columns below are used:
			 - “smiles” - SMILES representation of the molecular structure
	 - Properties：
		 - ”NR-XXX” - Nuclear receptor signaling bioassays results
		 - “SR-XXX” - Stress response bioassays results
		 - Valid ratio: 0.751、0.760
		 - Task evaluated: 12/12
		 - please refer to the links at https://tripod.nih.gov/tox21/challenge/data.jsp for details.

 - Toxcast
	 - Introduction：
		 - ToxCast is an extended data collection from the same initiative as Tox21, providing toxicology data for a large library of compounds based on in vitro high-throughput screening. The processed collection includes qualitative results of over 600 experiments on 8k compounds.

	 - Input：
		 - The data file contains a csv table, in which columns below are used
			 - “smiles” - SMILES representation of the molecular structure
	 - Properties：
		 - ”ACEA_T47D_80hr_Negative” ~ “Tanguay_ZF_120hpf_YSE_up” - Bioassays results
		 - Valid ratio: 0.234、0.268
		 - Task evaluated: 610/617
		 

##Pre-trained Models
**TO DO：Provide pre-trained models' address**

## Q&A
- Q1: Must the hyperparameter configuration during pre-training be the same as that of finetune?
       - The valid ratio of each data set is different, and the size of the data set is also inconsistent. You can choose different configuration files according to different data sets.
- Q2: Is it too long during pre-training?
       - Change the size of your max_epoch in the shell script to reduce the corresponding value.
       - Increase batch_size
## Reference
### Paper-related
We mainly refer to paper **Pretrain-GNN** The way we train the models and the hyper-parameters might be different.

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

**GraphSAGE**
>@inproceedings{hamilton2017inductive,
  title={Inductive representation learning on large graphs},
  author={Hamilton, Will and Ying, Zhitao and Leskovec, Jure},
  booktitle={Advances in neural information processing systems},
  pages={1024--1034},
  year={2017}
}


### Data-related
 This compound pre-training method uses the data set in the paper **Pretrain-GNN** for further processing.

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
