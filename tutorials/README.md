# Backgrouds

Machine learning (ML), especially deep learning (DL) are playing increasingly important role in pharmaceutical industry and bio-informatics. For instance, deep learning based methodology are found to predict the drug-target interaction [^ozturk2018deepdta] [^nguyen2019graphdta] and molecule profiles [^daina2017swissadme] [^yang2019admetsar] [^hu2019strategies] to reasonable precision with quite low computational cost, while those properties can only be accessed through in vivo / in vitro experiments or computationally expensive simulations (molecular dyanmics simulation etc) before. As another example, folding biomacromolecules such as Protein and RNA in silico are becoming more likely to be accomplished with the help of deep neural models [^alquraishi2019alphafold] [^huang2019linearfold], too. The usage of deep learning can greatly improve the efficiency, and thus reduce the cost of drug discovery, vaccine design etc. In contrast to the powerful ability of deep learning metrics, a key challenge lying in utilizing them in drug industry is the contradiction between the demand of huge data for training and the limited annotated data. Recently, there are tremendous success in self-supervised learning in natural language processing [^devlin2018bert] and computer vision [^misra2020self], showing that large corpus of unlabeled data can be beneficial to learning universal tasks. In molecule representations, there are huge amount of unlabeled data including protein sequences (over 100 million) and compounds (over 50 million). 

**PaddleHelix** is a high-performance machine-learning-based bio-computing framework. It features large scale representation learning and easy-to-use APIs, providing pharmaceutical and biological researchers and engineers convenient access to the most up-to-date and state-of-the-art AI tools.

# Navigating PaddleHelix

# QuickStart
To start using PaddleHelix simply do:
```python
import pahelix as ph
```
To use LinearFold algorithm[^huang2019linearfold] to fold an RNA sequence, use the ribonucleic acid sequence as the input and run:
```python
ph.toolkit.linear_fold('ACAAGTCCCCAAAGGG...')
```


[^ozturk2018deepdta]: Hakime Öztürk, Arzucan Özgür, and Elif Ozkirimli.  Deepdta:  deep drug–target binding affinity prediction.Bioinformatics, 34(17):i821–i829, 2018

[^nguyen2019graphdta]: Thin Nguyen, Hang Le, and Svetha Venkatesh. Graphdta: prediction of drug–target binding affinity using graphconvolutional networks.BioRxiv, page 684662, 2019

[^daina2017swissadme]: Antoine Daina, Olivier Michielin, and Vincent Zoete. Swissadme: a free web tool to evaluate pharmacokinetics,drug-likeness and medicinal chemistry friendliness of small molecules.Scientific reports, 7:42717, 201

[^yang2019admetsar]: Hongbin Yang, Chaofeng Lou, Lixia Sun, Jie Li, Yingchun Cai, Zhuang Wang, Weihua Li, Guixia Liu, and YunTang. admetsar 2.0: web-service for prediction and optimization of chemical admet properties.Bioinformatics,35(6):1067–1069, 2019

[^hu2019strategies]: Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure Leskovec. Strategiesfor pre-training graph neural networks. InInternational Conference on Learning Representations, 2019.

[^alquraishi2019alphafold]: Mohammed AlQuraishi. Alphafold at casp13.Bioinformatics, 35(22):4862–4865, 2019

[^huang2019linearfold]: Liang Huang, He Zhang, Dezhong Deng, Kai Zhao, Kaibo Liu, David A Hendrix, and David H Mathews. Linear-fold: linear-time approximate rna folding by 5’-to-3’dynamic programming and beam search.Bioinformatics,35(14):i295–i304, 2019

[^devlin2018bert]: Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectionaltransformers for language understanding.arXiv preprint arXiv:1810.04805, 2018.

[^misra2020self]: Ishan Misra and Laurens van der Maaten.   Self-supervised learning of pretext-invariant representations.   InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6707–6717, 202
