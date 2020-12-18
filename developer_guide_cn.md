# 开发者指南

如果你需要修改 PaddleHelix 中的算法，则需要以开发者模式使用 PaddleHelix。PaddleHelix 的核心算法大部分以 Python 实现，但也包含 C++，所以不能简单的采用 `pip install --editable {pahelix_path}` 来进行开发。为了在你的本地计算机上开发 PaddleHelix，请阅读以下教程。

1. 首先请按照[安装指引](./installation_guide_cn.md)安装 PaddleHelix 的依赖（paddlepaddle>=2.0.0rc0, pgl>=1.2.0）。

2. 如果你之前已经使用 `pip install paddlehelix` 安装了 PaddleHelix 的发行包，请卸载：

    ```bash
    pip uninstall paddlehelix
    ```

3. 克隆 PaddleHelix 的源代码库到你的本地，假设路径为 “/path_to_your_repo/” ：

    ```bash
    git clone https://github.com/PaddlePaddle/PaddleHelix.git /path_to_your_repo/
    cd /path_to_your_repo/
    ```

4. 根据你要修改的算法的不同，请遵循 4.1 或 4.2 所示的流程：
    
    4.1. LinearRNA
            
    LinearRNA 的源代码位于 “./c/pahelix/toolkit/linear_rna/linear_rna”，您可以按照自己的需要调整其 C++ 源码。修改代码后，请回到项目根目录，调用下面的脚本重新编译（请保证环境中已经安装有 cmake >= 3.6 和 g++ >= 4.8）：

    ```bash
    sh scripts/prepare.sh
    sh scripts/build.sh
    ```

    编译成功后用以下命令即可正常 `import` LinearRNA:

    ```bash
    cd build
    python
    >>> import c.pahelix.toolkit.linear_rna.linear_rna as linear_rna
    ```

    4.2. 其他算法

    PaddleHelix 中除 LinearRNA 外，其他算法均以 Python 实现。如果你想要修改这些算法，可以在 “./pahelix” 路径下找到对应文件，之后将 “/path_to_your_repo/” 加入 Python 环境路径即可：

    ```python
    import sys
    sys.path.append('/path_to_your_repo/')
    import pahelix
    ```

如果仍有疑问或者建议, 欢迎提出 [issue](https://github.com/PaddlePaddle/PaddleHelix/issues)，我们会尽快回复。
