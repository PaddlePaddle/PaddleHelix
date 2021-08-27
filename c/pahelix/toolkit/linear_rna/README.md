# LinearRNA: linear-time RNA structures analysis algorithms 

[中文版本](./README_cn.md) [English Version](./README.md)

* [Background](#background)
* [LinearFold Function](#linearfold-function)
    * [Machine learning-based model](#machine-learning-based-model)
    * [Thermodynamic model](#thermodynamic-model)
    * [Parameters](#parameters)
    * [Return](#return)
    * [Examples](#examples)
        * [secondary structure prediction(without constarints)](#secondary-structure-prediction(without-constarints))
        * [secondary structure prediction(with constarints)](#secondary-structure-prediction(with-constarints))
    * [dataset](#dataset)
        * [ArchiveII dataset](#archiveii-dataset)
        * [RNAcentral dataset](#rnacentral-dataset)
    * [baselines](#baselines)
        * [RNAfold](#rnafold)
        * [CONTRAfold](#contrafold)
    * [Relative papers](#relative-papers)
* [LinearPartition Function](#linearpartition-function)
    * [Machine learning-based model](#machine-learning-based-model)
    * [Thermodynamic model](#thermodynamic-model)
    * [Parameters](#parameters)
    * [Return](#return)
    * [Examples](#examples)
    * [dataset](#dataset)
    * [baselines](#baselines)


## Background
Baidu freely releases LinearFold algorithm, which can significantly reduce the runtime of RNA secondary structure prediction. The LinearFold paper has been accepted by the ISMB, the top-level computational biology conference, and published on Bioinformatics. 
Please check the paper: [LinearFold: linear-time approximate RNA folding by 5'-to-3' dynamic programming and beam search](http://academic.oup.com/bioinformatics/article/35/14/i295/5529205). 

Traditionally, RNA secondary structure prediction uses a bottom-up dynamic programming (DP), and scales cubically with the sequence length, which means that if the sequence length is doubled, the calculation time will cost 8 times. This requires a long runntime for extremely long sequences such as RNA virus genomes (for example, HIV has about 10,000 bases and the Ebola virus has about 20,000 bases). LinearFold creatively changed the bottom-up dynamic programming in the traditional algorithm to a left-to-right approach, and used the "beam pruning" idea to keep only the intermediate states with higher scores, thereby greatly reducing the search space and running time.

LinearFold can predict RNA secondary structures in linear time, and is astonishingly faster on longer sequences than traditional algorithms. Specifically, LinearFold can reduce the runtime of the secondary structure prediction for the new full-length coronavirus (about 30,000 bases) from 55 minutes to 27 seconds, which is 120 times faster. Meanwhile, LinearFold has improved prediction accuracy compared to traditional algorithms. Especially for longer RNA sequences (such as 16S and 23S rRNA) and long-distance base pairs (500+ bases apart), LinearFold's prediction accuracy has been significantly improved.

In 2020, Baidu released the fastest algorithm for partition function and base pair probabilities calculation. This algorithm is more powerful to assemble difference structures in equilibrium and predict the base pair probability matrix. The LinearParition paper has been accepted by ISMB and published on Bioinformatics. Please check the paper: [LinearPartition: linear-time approximation of RNA folding partition function and base-pairing probabilities](https://academic.oup.com/bioinformatics/article/36/Supplement_1/i258/5870487).

## LinearFold Function
### Machine learning-based model
```bash
linear_fold_c(rna_sequence, beam_size = 100, use_constraints = False, constraint = "", no_sharp_turn = True)
```
### Thermodynamic model
```bash
linear_fold_v(rna_sequence, beam_size = 100, use_constraints = False, constraint = "", no_sharp_turn = True)
```
### Parameters
- rna_sequence: the input RNA sequence to predict the secondary structure;
- beam_size: int (optional, default 100), set 0 to turn off the beam pruning;
- use_constraints: bool (optional, default False), enable adding constraints when predicting structures;
- constraint: string (optional, default ""), the constraint sequence. It works when the parameter use_constraints is Ture. The constraint - sequence should have the same length as the RNA sequence. "? . ( )" indicates a position for which the proper matching is unknown, unpaired, left or right parenthesis respectively. The parentheses must be well-balanced and non-crossing.
- no_sharp_turn: bool (optional, default True), disable sharpturn in prediction.
### Return
- tuple(string, float): return a tuple including the predicted structures and the folding free energy.
### Examples
#### secondary structure prediction(without constarints)
```bash
python
>>> import pahelix.toolkit.linear_rna as linear_rna
>>> linear_rna.linear_fold_c("GGGCUCGUAGAUCAGCGGUAGAUCGCUUCCUUCGCAAGGAAGCCCUGGGUUCAAAUCCCAGCGAGUCCACCA")
('(((((((..((((.......))))(((((((.....))))))).(((((.......))))))))))))....', 13.974767487496138)
>>> linear_rna.linear_fold_v("GGGCUCGUAGAUCAGCGGUAGAUCGCUUCCUUCGCAAGGAAGCCCUGGGUUCAAAUCCCAGCGAGUCCACCA")
('(((((((..((((.......))))(((((((.....))))))).(((((.......))))))))))))....', -31.5)
```
#### secondary structure prediction(with constarints)
```bash
>>> input_sequence = "AACUCCGCCAGGCCUGGAAGGGAGCAACGGUAGUGACACUCUCUGUGUGCGUAGGUUGCCUAGCUACCAUUU"
>>> constraint = "??(???(??????)?(????????)???(??????(???????)?)???????????)??.???????????"
>>> linear_rna.linear_fold_c(input_sequence, use_constraints = True, constraint = constraint)
('..(.(((......)((........))(((......(.......).))).....))..)..............', -27.328358240425587)
>>> linear_rna.linear_fold_v(input_sequence, use_constraints = True, constraint = constraint)
('..(.(((......)((........))(((......(.......).))).....))..)..............', 13.4)
```
### dataset
In LinearFold paper, we use two public datasets: ArchivieII and RNAcentral. 
#### ArchiveII dataset
[ArchiveII](http://rna.urmc.rochester.edu/pub/archiveII.tar.gz) includes nine different families and 3,857 RNA sequences with their secondary structures. We evaluate the effectiveness and accuracy of the LinearFold algorithm on the ArchiveII dataset.
Please check the paper for more details. 
#### RNAcentral dataset
[RNAcentral](https://rnacentral.org/) contains a large number of known RNA sequences without gold structures. The longest sequence is 244,296 bases. We use this dataset to assess the efficiency and scalability of LinearFold algorithm.  
### baselines
We compare LinearFold with two popular algorithms/softwares: Vienna RNAfold and CONTRAfold. 
#### RNAfold
[Vienna RNAfold](https://www.tbi.univie.ac.at/RNA/) is the most used RNA structure analysis platform, which is developed by the University of Vienna. They adopt the thermodynamic model and the bottom-up dynamic algorithm to predict secondary structures. 
The bottom-up algorithm scales cubically with the sequence length. In other words, if the sequence length doubles, the runtime grows up to eight times. 
#### CONTRAfold
[CONTRAfold](http://contra.stanford.edu/) is the first algorithm to use machine learning-based models to predict secondary structures. Specifically, it uses the CRF method to learn the weights of parameters and get a data-driven model. Basically, the CONTRAfold is still a cubic-time dynamic algorithm. 
## Relative papers
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

## LinearPartition Function
### Machine learning-based model
```bash
linear_partition_c(rna_sequence, beam_size = 100, bp_cutoff = 0.0, no_sharpe_turn = True)
```
### Thermodynamic model
```bash
linear_partition_v(rna_sequence, beam_size = 100, bp_cutoff = 0.0, no_sharpe_turn = True)
```
### Parameters
- rna_sequence: string, the input RNA sequence to calculate partition function and base pair probabities. 
- beam_size: int (optional, default 100),set 0 to turn off the beam pruning;
- bp_cutoff: double (optional, default 0.0), only output base pairs with correponding proabilities whose values larger than the bp_cutoff (between 0 and 1);
- no_sharp_turn: bool (optional, default True), enable sharpturn in prediction.
### Return
- tuple(string, list): ruturn a tuple consisting the partition function value, and a list of base pair probabilities
### Examples
```bash
python
>>> import pahelix.toolkit.linear_rna as linear_rna
>>> linear_rna.linear_partition_c("UGAGUUCUCGAUCUCUAAAAUCG", bp_cutoff = 0.2)
(0.64, [(4, 13, 2.0071e-01), (10, 22, 2.4662e-01), (11, 21, 2.4573e-01), (12, 20, 2.0927e-01)])
>>> linear_rna.linear_partition_v("UGAGUUCUCGAUCUCUAAAAUCG", bp_cutoff = 0.2)
(-1.96, [(2, 15, 8.3313e-01), (3, 14, 8.3655e-01), (4, 13, 8.3554e-01)])
```
### Datasets
In LinearPartition paper, we use the same datasets as LinearFold: ArchivieII and RNAcentral. 
### baselines
We compare LinearPartition with two popular RNA secondary structure prediction algorithms: Vienna RNAfold and CONTRAfold. Please check more details in LinearFold baselines. 

