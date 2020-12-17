*********
Tutorials
*********

.. contents:: Table of Contents

Backgrounds
===========

Machine learning (ML), especially deep learning (DL), is playing an increasingly important role in the pharmaceutical industry and bio-informatics. For instance, the DL-based methodology is found to predict the `drug target interaction <https://www.researchgate.net/publication/334088358_GraphDTA_prediction_of_drug-target_binding_affinity_using_graph_convolutional_networks>`_ and `molecule properties <https://pubmed.ncbi.nlm.nih.gov/30165565/>`_ with reasonable precision and quite low computational cost, while those properties can only be accessed through *in vivo* / *in vitro* experiments or computationally expensive simulations (molecular dynamics simulation etc.) before. As another example, *in silico* `RNA folding <https://www.researchgate.net/publication/344954534_LinearFold_Linear-Time_Prediction_of_RNA_Secondary_Structures>`_ and `protein folding <https://www.researchgate.net/publication/338619491_Improved_protein_structure_prediction_using_potentials_from_deep_learning>`_ are becoming more likely to be accomplished with the help of deep neural models. The usage of ML and DL can greatly improve efficiency, and thus reduce the cost of drug discovery, vaccine design, etc.

In contrast to the powerful ability of DL metrics, a key challenge lying in utilizing them in the drug industry is the contradiction between the demand for huge data for training and the limited annotated data. Recently, there is tremendous success in adopting self-supervised learning in natural language processing and computer vision, showing that a large corpus of unlabeled data can be beneficial to learning universal tasks. In molecule representations, there is a similar situation. We have large amount of unlabeled data, including protein sequences (over 100 million) and compounds (over 50 million) but relatively small annotated data. It is quite promising to adopt DL-based pretaining technique in the representation learning of chemical compounds, proteins, RNA, etc.

**PaddleHelix** is a high-performance ML-based bio-computing framework. It features large scale representation learning and easy-to-use APIs, providing pharmaceutical and biological researchers and engineers convenient access to the most up-to-date and state-of-the-art AI tools.

Navigating PaddleHelix
======================

.. image:: ../.github/PaddleHelix_Structure.png
   :align: center

Tutorials
=========

- `Predicting Drug Target Interaction <https://github.com/PaddlePaddle/PaddleHelix/blob/dev/tutorials/drug_target_interaction_tutorial.ipynb>`_

- `Compound Representation Learning and Property Prediction <https://github.com/PaddlePaddle/PaddleHelix/blob/dev/tutorials/compound_property_prediction_tutorial.ipynb>`_

- `Protein Representation Learning and Property Prediction <https://github.com/PaddlePaddle/PaddleHelix/blob/dev/tutorials/protein_pretrain_and_property_prediction_tutorial.ipynb>`_

- `Predicting RNA Secondary Structure <https://github.com/PaddlePaddle/PaddleHelix/blob/dev/tutorials/linearrna_tutorial.ipynb>`_

Run tutorials locally
=====================

The tutorials are written as **Jupyter** Notebooks and designed to be smoothly run on you own machine. If you don't have **Jupyter** installed, please refer to `here <https://jupyter.org/install>`_. And please also install PaddleHelix before proceeding (`instructions <https://github.com/PaddlePaddle/PaddleHelix/blob/dev/installation_guide.md>`_).

After the installation of **Jypyter**, please go through the following steps:

1. Clone this repository to your own machine

2. Change the working directory of your shell to ``path_to_your_repo/PaddleHelix/tutorials/``

3. Open ``Jupyter lab`` with the command `jupyter-lab`, wait for your web browser being called out

4. All the tutorials should be in the ``File Browser``, click and enjoy!

