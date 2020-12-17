# LinearRNA: 线性时间RNA结构分析算法

[中文版本](./README_cn.md) [English Version](./README.md)

* [背景介绍](#背景介绍)
* [LinearFold调用](#linearFold调用)
    * [机器学习模型](#机器学习模型)
    * [热力学模型](#热力学模型)
    * [参数说明](#参数说明)
    * [返回值](#返回值)
    * [运行实例](#运行实例)
        * [二级结构预测（无约束条件）](#二级结构预测（无约束条件）)
        * [二级结构预测（有约束条件）](#二级结构预测（有约束条件）)
    * [数据](#数据)
        * [ArchiveII数据集](#ArchiveII数据集)
        * [RNAcentral数据集](#RNAcentral数据集)
    * [baselines系统](#baselines系统)
        * [RNAfold](#RNAfold)
        * [CONTRAfold](#CONTRAfold)
    * [相关论文](#相关论文)
* [LinearPartition调用](#linearPartition调用)
    * [机器学习模型](#机器学习模型)
    * [热力学模型](#热力学模型)
    * [参数说明](#参数说明)
    * [返回值](#返回值)
    * [运行实例](#运行实例)
    * [数据](#数据)
    * [baselines系统](#baselines系统)




## 背景介绍
百度免费开放LinearFold算法，可将RNA二级结构预测的时间大大降低，LinearFold论文已经在计算生物学顶级会议ISMB及生物信息学权威杂志Bioinformatics 上发表。论文链接请见：[LinearFold: linear-time approximate RNA folding by 5'-to-3' dynamic programming and beam search](http://academic.oup.com/bioinformatics/article/35/14/i295/5529205)。

传统上，RNA二级结构预测采用自底向上的动态规划（DP），是一种三次方时间复杂度的算法，也就是说，如果序列长度翻一倍的话，就要付出 8 倍的计算时间，这对于 RNA 病毒基因组这样的超长序列（例如艾滋病毒有约1万个碱基，埃博拉病毒有约2万个碱基）需要很长的等待时间。LinearFold创造性的将传统算法中自底向上的动态规划改为从左到右的方式，并利用”beam pruning“的思想，只保留分数较高的中间状态，从而大大减小了搜索空间。

LinearFold能够在线性时间内预测RNA二级结构，在长序列RNA上的预测速度远远大于传统算法。其中，LinearFold能够将新冠病毒全基因组序列（约30,000 nt）二级结构预测时间从55分钟降低到27秒，速度提升120倍。同时LinearFold在预测精度上相比传统算法也有提升。尤其对于长序列RNA二级结构（如16S和23S rRNA二级结构）和长碱基对（相距500+ nt）预测上，LinearFold预测精度有显著地提升。

2020年，百度再次发表世界最快RNA配分方程和碱基对概率预测算法LinearPartition。该算法功能更加强大，可以模拟RNA序列在平衡态时成千上万种不同结构的分布，并预测碱基对概率矩阵。LinearPartition算法同样被ISMB顶会接收并在Bioinformatics杂志上发表，论文链接请见：[LinearPartition: linear-time approximation of RNA folding partition function and base-pairing probabilities](https://academic.oup.com/bioinformatics/article/36/Supplement_1/i258/5870487)。

## LinearFold调用
### 机器学习模型
```bash
linear_fold_c(rna_sequence, beam_size = 100, use_constraints = False, constraint = "", no_sharp_turn = True)
```
### 热力学模型
```bash
linear_fold_v(rna_sequence, beam_size = 100, use_constraints = False, constraint = "", no_sharp_turn = True)
```
### 参数说明
- rna_sequence: string, 需要预测结构的RNA sequence
- beam_size: int (optional), 控制beam pruning size的参数，默认值为100。该参数越大，则预测速度越慢，而与精确搜索相比近似效果越好;
- use_constraints: bool (optional), 在预测二级结构时增加约束条件, 默认值时False。为True时, constraint参数需要提供约束序列;
- constraint: string (optional), 二级结构预测约束条件, 默认为空。当提供约束序列时, use_constraints参数需要设置为True。该约束须与输入的RNA序列长度相同，每个点位可以指定“? . ( )”四种符号中的一种，其中“?”表示该点位无限制，“.”表示该点位必须是unpaired，“(”与“)”表示该点位必须是paired。注意“(”与“)”必须数量相等，即相互匹配。具体操作请参考运行实例。
- no_sharp_turn: bool (optional), 不允许在预测的hairpin结构中出现sharp turn, 默认为True。
### 返回值
- tuple(string, double): 返回一个二元组, 第一个位置是结构序列, 第二个位置是结构的folding free energy
### 运行示例
#### 二级结构预测（无约束条件）
```bash
python
>>> import pahelix.toolkit.linear_rna as linear_rna
>>> linear_rna.linear_fold_c("GGGCUCGUAGAUCAGCGGUAGAUCGCUUCCUUCGCAAGGAAGCCCUGGGUUCAAAUCCCAGCGAGUCCACCA")
('(((((((..((((.......))))(((((((.....))))))).(((((.......))))))))))))....', 13.974767487496138)
>>> linear_rna.linear_fold_v("GGGCUCGUAGAUCAGCGGUAGAUCGCUUCCUUCGCAAGGAAGCCCUGGGUUCAAAUCCCAGCGAGUCCACCA")
('(((((((..((((.......))))(((((((.....))))))).(((((.......))))))))))))....', -31.5)
```
#### 二级结构预测（有约束条件）
```bash
>>> input_sequence = "AACUCCGCCAGGCCUGGAAGGGAGCAACGGUAGUGACACUCUCUGUGUGCGUAGGUUGCCUAGCUACCAUUU"
>>> constraint = "??(???(??????)?(????????)???(??????(???????)?)???????????)??.???????????"
>>> linear_rna.linear_fold_c(input_sequence, use_constraints = True, constraint = constraint)
('..(.(((......)((........))(((......(.......).))).....))..)..............', -27.328358240425587)
>>> linear_rna.linear_fold_v(input_sequence, use_constraints = True, constraint = constraint)
('..(.(((......)((........))(((......(.......).))).....))..)..............', 13.4)
```

### 数据
LinearFold论文中我们使用了两个公开数据集：ArchivieII数据集和RNAcentral数据集。
#### ArchiveII数据集
[ArchiveII数据集](http://rna.urmc.rochester.edu/pub/archiveII.tar.gz)包含了3,857 RNA序列及其二级结构（实验得到），涵盖了9个不同的RNA家族，最长序列长度为2,968 nt。LinearFold论文中ArchiveII数据集被用来验证算法效率和预测准确度。具体细节请参考LinearFold论文。
#### RNAcentral数据集
[RNAcentral数据集](https://rnacentral.org/)包含了海量的人类已知的 RNA序列（没有二级结构）。数据集中最长序列长度为244,296 nt。LinearFold论文中RNAcentral数据集被用来验证算法效率和内存使用情况。具体细节请参考LinearFold论文。

### baselines系统
LinearFold与当前两个主流的RNA二级结构预测算法（系统）进行了对比，分别是Vienna RNAfold和CONTRAfold。
#### RNAfold
[Vienna RNAfold](https://www.tbi.univie.ac.at/RNA/)是目前用户量最大的RNA结构分析平台，由奥地利维也纳大学开发。它使用热力学模型作为RNA结构预测模型，并采用自底向上的动态规划算法作为二级结构预测算法。该算法复杂度随着序列长度的增加成三次方比例增加，也就是说，输入序列长度增加1倍，运行时间变为原来的8倍。
#### CONTRAfold
[CONTRAfold](http://contra.stanford.edu/)是第一个较为成功的基于机器学习的RNA二级结构预测系统。它采用CRF的方法，学习模型参数的权重，从而得到一个data-driven的RNA结构预测模型。从算法角度，CONTRAfold也仍然采用了三次方的动态规划算法。

## 相关论文
**ViennaRNA Package 2.0**
> @article{lorenz+:2011,
  title={ViennaRNA Package 2.0},
  author={Lorenz, Ronny and others},
  journal={Alg.~Mol.~Biol.},
  volume={6},
  number={1},
  pages={1},
  year={2011},
  publisher={BioMed Central}
}

**CONTRAfold**
>@article{do2006contrafold,
  title={CONTRAfold: RNA secondary structure prediction without physics-based models},
  author={Do, Chuong B and Woods, Daniel A and Batzoglou, Serafim},
  journal={Bioinformatics},
  volume={22},
  number={14},
  pages={e90--e98},
  year={2006},
  publisher={Oxford University Press}
}

## LinearPartition调用
### 机器学习模型
```bash
linear_partition_c(rna_sequence, beam_size = 100, bp_cutoff = 0.0, no_sharpe_turn = True)
```
### 热力学模型
```bash
linear_partition_v(rna_sequence, beam_size = 100, bp_cutoff = 0.0, no_sharpe_turn = True)
```
### 参数说明
- rna_sequence: string, 需要计算配分函数和碱基对概率的RNA sequence
- beam_size: int (optional), 控制beam pruning size的参数，默认值为100。该参数越大，则预测速度越慢，而与精确搜索相比近似效果越好;
- bp_cutoff: double (optinal), 只输出概率大于等于bp_cutoff的碱基对及其概率, 0 <= pf_cutoff <= 1, 默认为0.0; 
- no_sharp_turn: bool (optional), 不允许在预测的hairpin结构中出现sharp turn, 默认为True。
### 返回值
- tuple(string, list): 返回一个二元组, 第一个位置是配分函数值, 第二个位置是存有碱基对及其概率的列表
### 运行示例
```bash
python
>>> import pahelix.toolkit.linear_rna as linear_rna
>>> linear_rna.linear_partition_c("UGAGUUCUCGAUCUCUAAAAUCG", bp_cutoff = 0.2)
(0.64, [(4, 13, 2.0071e-01), (10, 22, 2.4662e-01), (11, 21, 2.4573e-01), (12, 20, 2.0927e-01)])
>>> linear_rna.linear_partition("UGAGUUCUCGAUCUCUAAAAUCG", energy_model = "v", bp_cutoff = 0.2) // V model
(-1.96, [(2, 15, 8.3313e-01), (3, 14, 8.3655e-01), (4, 13, 8.3554e-01)])
```

### 数据
LinearPartition论文中我们同样使用了两个公开数据集：ArchivieII数据集和RNAcentral数据集，详细说明请见上文LinearFold数据部分。

### baselines系统
LinearPartition同样与当前两个主流的RNA二级结构预测算法（系统）Vienna RNAfold和CONTRAfold进行了对比，详细说明请见上文LinearFold baselines系统部分。


