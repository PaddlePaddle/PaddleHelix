[English](README.md) | 简体中文

# 背景

最近，机器学习，特别是深度学习正在制药工业和生物信息学中发挥着越来越重要的作用。相比起传统的生物实验，通过机器学习和深度学习，我们可以大大提升研发的效率，从而降低药物发现、疫苗设计等工业应用的成本。我们提供了大量的教学示例以方便开发者快速了解和使用该框架：
* **Drug Discovery**
  - [化合物表示和属性预测](./compound_property_prediction_tutorial_cn.ipynb)
  - [蛋白质表示和属性预测](./protein_pretrain_and_property_prediction_tutorial_cn.ipynb)
  - Predicting Drug-Target Interaction: [GraphDTA](./drug_target_interaction_graphdta_tutorial_cn.ipynb), [MolTrans](./drug_target_interaction_moltrans_tutorial_cn.ipynb)
  - [分子生成](./molecular_generation_tutorial_cn.ipynb)
* **Vaccine Design**
  - [RNA结构预测](./linearrna_tutorial_cn.ipynb)

# 如何运行教学示例
我们的教程以`Jupyter Notebook`的形式编写，可以方便地在本地计算机上运行。在运行前，请先安装[Jupyter](https://jupyter.org/install)和[螺旋桨](../installation_guide_cn.md)）。

在正确安装Jupyter和螺旋桨以后，请按照以下的步骤来运行：
1. git 克隆 PaddleHelix 到你的本地；
2. 打开 shell 并切换路径到 "path_to_your_repo/PaddleHelix/tutorials/"；
3. 输入命令 `jupyter-lab` 以打开 "Jupyter lab"，等待一会儿，直到你的浏览器被唤起；
4. 所有的教程应该都列在 Jupyter 的 "文件浏览器" 中。

点击任意一个教程，开始吧！
