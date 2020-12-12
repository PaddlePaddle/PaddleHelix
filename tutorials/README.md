# Backgrouds

Machine learning (ML), especially deep learning (DL) is playing an increasingly important role in the pharmaceutical industry and bio-informatics. For instance, the deep learning-based methodology is found to predict the [drug-target interaction](https://www.researchgate.net/publication/334088358_GraphDTA_prediction_of_drug-target_binding_affinity_using_graph_convolutional_networks) and [molecule profiles](https://pubmed.ncbi.nlm.nih.gov/30165565/) to reasonable precision with quite low computational cost, while those properties can only be accessed through in vivo / in vitro experiments or computationally expensive simulations (molecular dynamics simulation etc.) before. As another example, in-silico [RNA folding](https://www.researchgate.net/publication/344954534_LinearFold_Linear-Time_Prediction_of_RNA_Secondary_Structures) and [Protein folding](https://www.researchgate.net/publication/338619491_Improved_protein_structure_prediction_using_potentials_from_deep_learning) are becoming more likely to be accomplished with the help of deep neural models. The usage of machine learning and deep learning can greatly improve efficiency, and thus reduce the cost of drug discovery, vaccine design etc. In contrast to the powerful ability of deep learning metrics, a key challenge lying in utilizing them in the drug industry is the contradiction between the demand for huge data for training and the limited annotated data. Recently, there is a tremendous success in self-supervised learning in natural language processing and computer vision, showing that large corpus of unlabeled data can be beneficial to learning universal tasks. In molecule representations, there is a similar situation. We have large amount of unlabeled data, including protein sequences (over 100 million) and compounds (over 50 million) but relatively small annoted data.

**PaddleHelix** is a high-performance machine-learning-based bio-computing framework. It features large scale representation learning and easy-to-use APIs, providing pharmaceutical and biological researchers and engineers convenient access to the most up-to-date and state-of-the-art AI tools.

# Navigating PaddleHelix

# QuickStart
To start using PaddleHelix simply do:
```python
import pahelix as ph
```
To use [LinearFold](https://www.researchgate.net/publication/344954534_LinearFold_Linear-Time_Prediction_of_RNA_Secondary_Structures) algorithm to fold an RNA sequence, use the nucleotide sequence as the input and run:
```python
ph.toolkit.linear_fold('ACAAGTCCCCAAAGGG...')
```

...To be continued

# Advanced Tutorials
* [Predicting Drug-Target Interaction](drug_target_interaction_tutorial.ipynb)
* [Compound Representation Learning and Drug Profiling](compound_property_prediction_tutorial.ipynb)
* [Protein Representation Learning and Protein Property Prediction](protein_pretrain_and_property_prediction_tutorial.ipynb)


