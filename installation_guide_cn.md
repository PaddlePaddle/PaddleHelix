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

4. 在安装 PaddleHelix 之前，首先需要使用 conda 安装 `rdkit`：
```bash
conda install -c conda-forge rdkit
```
5. 基于你对 CPU/GPU 版本的选择来安装 `paddle`:

请注意安装 **paddle2.0** 以上版本，方法参见 paddlepaddle [官方文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc1/install/index_cn.html)。

比如，你想在Linux系统上安装PaddlePaddle 2.0 GPU版本，你可以运行以下命令：

```bash
python -m pip install paddlepaddle-gpu==2.0.0rc1.post90 -f https://paddlepaddle.org.cn/whl/stable.html
```

6. 使用 pip 命令安装`PGL`:
```bash
pip insatll pgl
```

7. 使用 pip 命令安装 PaddleHelix
```bash
pip install paddlehelix
```

8. 等待 PaddleHelix 安装完成！

### 注意
运行完项目之后，如果想要退出当前 conda 环境，可以使用下列命令：
```bash
conda deactivate
```
