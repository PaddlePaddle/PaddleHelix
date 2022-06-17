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
| pgl | \>=2.1 |
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
5. 基于你想在 CPU 或者 GPU 上运行 PaddleHelix 来选择安装不同版本的 `paddlepaddle`:

   比如，你想使用 `paddlepaddle` 的 GPU 版本，你可以运行以下命令：

   ```bash
   python -m pip install paddlepaddle-gpu -f https://paddlepaddle.org.cn/whl/stable.html
   ```

   或者你想使用 `paddlepaddle` 的 CPU 版本，你可以运行以下命令：

   ```bash
   python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
   ```

   请注意安装 `paddlepaddle` **2.0** 以上版本，其他安装方法参见 `paddlepaddle` [官方文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc1/install/index_cn.html)。

6. 使用 pip 命令安装`PGL`:
```bash
pip install pgl
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

## 使用 Docker 快速体验

> Docker 是一种开源工具，用于在和系统本身环境相隔离的环境中构建、发布和运行各类应用程序。如果您没有 Docker 运行环境，请参考 [Docker 官网](https://www.docker.com/)进行安装，如果您准备使用 GPU 版本镜像，还需要提前安装好 [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)。 

我们提供了包含最新 PaddleHelix 代码的 docker 镜像，并预先安装好了所有的环境和库依赖，您只需要**拉取并运行 docker 镜像**，无需其他任何额外操作，即可开始享用 PaddleHelix 的所有功能。

在 [Docker Hub](https://hub.docker.com/repository/docker/paddlecloud/paddlehelix) 中获取这些镜像及相应的使用指南，包括 CPU、GPU、ROCm 版本。

如果您对自动化制作docker镜像感兴趣，或有自定义需求，请访问 [PaddlePaddle/PaddleCloud](https://github.com/PaddlePaddle/PaddleCloud/tree/main/tekton) 做进一步了解。
