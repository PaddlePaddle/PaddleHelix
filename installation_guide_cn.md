# 安装指引

## 环境支持

* 操作系统支持：Windows，Linux 以及 OSX
* Python 版本：3.6, 3.7

## 包依赖

| 名字 | 版本 |
| ---- | ---- |
| numpy | - |
| pandas | - |
| networkx | - |
| paddlepaddle | \>=2.0.0rc0 |
| pgl | \>=1.2.0 |
| rdkit | - |

（“-” 代表没有版本要求）

## 安装步骤

因为 PaddleHelix 安装包依赖于最新版的 paddlepaddle（2.0.0rc0 或以上），以及无法直接使用 `pip` 命令直接安装的 rdkit，因此我们建议创建一个新的 conda 环境，具体步骤如下：

1. 如果你之前从来没有使用过 conda，可以参考这个网页来安装 conda：

   https://docs.conda.io/projects/conda/en/latest/user-guide/install/

2. 在安装完 conda 之后, 可以开始创建一个新的 conda 环境：

```bash
conda create -n paddlehelix python=3.7
```

3. 使用如下命令激活 conda 环境：

```bash
conda activate paddlehelix
```

4. 在安装 PaddleHelix 之前，首先需要使用 conda 安装 rdkit：
```bash
conda install -c conda-forge rdkit
```
5. 等待 rdkit 安装完成，之后使用 pip 命令安装 PaddleHelix
```bash
pip install paddlehelix
```

6. 等待 PaddleHelix 安装完成！

### 注意

如果想要退出当前 conda 环境，可以使用下列命令：
```bash
conda deactivate
```
