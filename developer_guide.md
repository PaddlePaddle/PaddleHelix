# Developer's guide

If you need to modify the algorithms/models in PaddleHelix, you have to switch to the developer mode. The core algorithms of PaddleHelix are most implemented in Python, but some also in C++, so you cannot develop PaddleHelix simply with `pip install --editable {pahelix_path}`. To develop on your machiine, please do the following:

1. Please following the [installation guide](./installation_guide.md) to install all dependencies of PaddleHelix (paddlepaddle>=2.0.0rc0, pgl>=1.2.0).

2. If you have installed distributed PaddleHelix with `pip install paddlehelix`, please uninstall it with:

    ```bash
    pip uninstall paddlehelix
    ```

3. Clone this repository to your local machine, supposed path at "/path_to_your_repo/":

    ```bash
    git clone https://github.com/PaddlePaddle/PaddleHelix.git /path_to_your_repo/
    cd /path_to_your_repo/
    ```

4. Depends on which model you'd like to modify, go to 4.1 or 4.2:

    4.1. LinearRNA
            
    The source code of LinearRNA is at "./c/pahelix/toolkit/linear_rna/linear_rna". You could modify it for your needs. Then remember to return to the root directory of the repository, call scripts below to re-compile (please ensure there are `cmake >= 3.6` and `g++ >= 4.8` on your machine):

    ```bash
    sh scripts/prepare.sh
    sh scripts/build.sh
    ```

    After a successful compilaiton, `import` LinearRNA as following:

    ```bash
    cd build
    python
    >>> import c.pahelix.toolkit.linear_rna.linear_rna as linear_rna
    ```

    4.2. Other algorithms

    Except LinearRNA, other algorithms in PaddleHelix are all implemented in Python. If you want to change these algorithms, just find corresponding files under path "./pahelix", then add "/path_to_your_repo/" to your Python environment path:

    ```python
    import sys
    sys.path.append('/path_to_your_repo/')
    import pahelix
    ```

If you have any question or suggestion, feel free to create an [issue](https://github.com/PaddlePaddle/PaddleHelix/issues). We will response as soon as possible.
