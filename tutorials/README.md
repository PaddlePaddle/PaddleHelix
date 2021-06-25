English | [简体中文](README_cn.md)

# Introduction
Recently, increasing studies employ machine learning methods, especially deep neural networks, for DTA drug discovery due to their time efficiency compared with biological experiments and huge advances in various domains. We provide abundant tutorials to help you navigate the repository and start quickly.

* **Drug Discovery**
  - [Compound Representation Learning and Property Prediction](./compound_property_prediction_tutorial.ipynb)
  - [Protein Representation Learning and Property Prediction](./protein_pretrain_and_property_prediction_tutorial.ipynb)
  - Predicting Drug-Target Interaction: [GraphDTA](./drug_target_interaction_graphdta_tutorial.ipynb), [MolTrans](./drug_target_interaction_moltrans_tutorial.ipynb)
  - [Molecular Generation](./molecular_generation_tutorial.ipynb)
* **Vaccine Design**
  - [Predicting RNA Secondary Structure](./linearrna_tutorial.ipynb)

# Run tutorials locally

The tutorials are written as Jupyter Notebooks and designed to be smoothly run on you own machine. If you don't have Jupyter installed, please refer to [here](https://jupyter.org/install). And please also install PaddleHelix before proceeding ([instructions](../installation_guide.md)).

After the installation of Jupyter, please go through the following steps:

1. Clone this repository to your own machine
2. Change the working directory of your shell to "path_to_your_repo/PaddleHelix/tutorials/"
3. Open "Jupyter lab" with the command `jupyter-lab`, wait for your web browser being called out
4. All the tutorials should be in the "File Browser" now, click and enjoy!
