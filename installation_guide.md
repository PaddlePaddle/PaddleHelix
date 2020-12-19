# Installation guide

## Prerequisites

* OS support: Windows, Linux and OSX
* Python version: 3.6, 3.7

## Dependencies

| name         | version |
| ------------ | ---- |
| numpy        | - |
| pandas       | - |
| networkx     | - |
| paddlepaddle | \>=2.0.0rc0 |
| pgl          | \>=1.2.0 |
| rdkit        | - |
|sklearn|-|

('-' means no specific version requirement for that package)

## Instruction
Since PaddleHelix depends on the `paddlepaddle` of version 2.0.0rc0 or above, and `rdkit` cannot be installed directly using `pip`, we suggest using `conda` to create a new environment for the installation. Detailed instruction is shown below:

1. If you do not have conda installed, please check this website to get it first:

  https://docs.conda.io/projects/conda/en/latest/user-guide/install/

2. Create a new environment with conda:

```bash
conda create -n paddlehelix python=3.7  
```

3. Activate the environment which is just created:

```bash
conda activate paddlehelix
```

4. Install `rdkit` using conda:

```bash
conda install -c conda-forge rdkit
```
5. Install the right version of `paddlepaddle` depends on CPU/GPU you want to run PaddleHelix on.

    If you want to use the GPU version of `paddlepaddle`, run this:

    ```bash
    python -m pip install paddlepaddle-gpu==2.0.0rc1.post90 -f https://paddlepaddle.org.cn/whl/stable.html
    ```

    Or if you want to use the CPU version of `paddlepaddle`, run this:

    ```bash
    python -m pip install paddlepaddle==2.0.0rc1 -i https://mirror.baidu.com/pypi/simple
    ```

    Noting that the version of `paddlepaddle` should be higher than **2.0**.
    Check `paddlepaddle`'s [official document](https://www.paddlepaddle.org.cn/documentation/docs/en/2.0-rc1/install/index_en.html)
    for more installation guide.

6. Install `PGL` using pip:
   
```bash
pip install pgl
```

7. Install PaddleHelix using pip:

```bash
pip install paddlehelix
```

8. The installation is done!

### Note
After playing, if you want to deactivate the conda environment, do this:

```bash
conda deactivate
```
