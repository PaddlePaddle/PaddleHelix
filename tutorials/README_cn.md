[English](README.md) | 简体中文

# 背景

最近，机器学习，特别是深度学习正在制药工业和生物信息学中发挥着越来越重要的作用。相比起传统的生物实验，通过机器学习和深度学习，我们可以大大提升研发的效率，从而降低药物发现、疫苗设计等工业应用的成本。


# PaddleHelix 导览
<p align="center">
<img src="../.github/PaddleHelix_Structure.png" align="middle" heigh="80%" width="80%" />
</p>

# 教程
* [药物-靶点相互作用预测：GraphDTA](drug_target_interaction_graphdta_tutorial_cn.ipynb), [MolTrans](drug_target_interaction_moltrans_tutorial_cn.ipynb)
* [化合物表示学习和性质预测](compound_property_prediction_tutorial_cn.ipynb)
* [蛋白质表示学习和性质预测](protein_pretrain_and_property_prediction_tutorial_cn.ipynb)
* [分子生成](molecular_generation_tutorial_cn.ipynb)
* [RNA二级结构预测](linearrna_tutorial_cn.ipynb)

# 在本地运行

我们的教程以 Jupyter Notebook 的形式编写，可以方便的在你的本地计算机上运行。如果你没有安装过 Jupyter，请看[这里](https://jupyter.org/install)。另外也请安装好 PaddleHelix（[教程](../installation_guide_cn.md)）。

安装好 Jupyter 之后，请按照以下的步骤来运行：

1. git 克隆 PaddleHelix 到你的本地
2. 打开 shell 并切换路径到 "path_to_your_repo/PaddleHelix/tutorials/"
3. 输入命令 `jupyter-lab` 以打开 "Jupyter lab"，等待一会儿，直到你的浏览器被唤起
4. 所有的教程应该都列在 Jupyter 的 "文件浏览器" 中，点击任意一个教程，开始吧！
