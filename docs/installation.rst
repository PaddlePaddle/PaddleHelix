============
Installation
============

.. contents:: Table of Contents

Prerequisites
-------------

- OS support: Windows, Linux and OSX

- Python version: **3.6, 3.7**

Dependencies
------------

(``-`` means no specific version requirement for that package)

   +--------------+----------------+
   |     Name     |     Version    |
   +==============+================+
   |     numpy    |      ``-``     |
   +--------------+----------------+
   |     pandas   |      ``-``     |
   +--------------+----------------+
   |    networkx  |      ``-``     |
   +--------------+----------------+
   | paddlepaddle | ``>=2.0.0rc0`` |
   +--------------+----------------+
   |     pgl      |  ``>=1.2.0``   |
   +--------------+----------------+
   |     rdkit    |      ``-``     |
   +--------------+----------------+
   |    sklearn   |      ``-``     |
   +--------------+----------------+

Instruction
------------

Since PaddleHelix depends on the ``paddlepaddle`` of version 2.0.0rc0 or above, and ``rdkit`` cannot be installed directly using ``pip``, we suggest using ``conda`` to create a new environment for the installation. Detailed instruction is shown below:

- If you do not have ``conda`` installed, please `install`_ it at first:

.. _install: https://docs.conda.io/projects/conda/en/latest/user-guide/install/

- Create a new environment with ``conda``:

.. code:: console

   $ conda create -n paddlehelix python=3.7

- Activate the environment just created:

.. code:: console

   $ conda activate paddlehelix

- Install ``rdkit`` using ``conda``:

.. code:: console

   $ conda install -c conda-forge rdkit

- Install the right version of ``paddlepaddle`` depends on CPU/GPU you want to run PaddleHelix on.

1) If you want to use the GPU version of ``paddlepaddle``, run this:

.. code:: console

  $ python -m pip install paddlepaddle-gpu==2.0.0rc1.post90 -f https://paddlepaddle.org.cn/whl/stable.html

2) Or if you want to use the CPU version of ``paddlepaddle``, run this:

.. code:: console

  $ python -m pip install paddlepaddle==2.0.0rc1 -i https://mirror.baidu.com/pypi/simple

.. note:: The version of ``paddlepaddle`` should be higher than **2.0**. Check `paddlepaddle official document <https://www.paddlepaddle.org.cn/documentation/docs/en/2.0-rc1/install/index_en.html>`_ for more installation guide.

- Install ``pgl`` using ``pip``:

.. code:: console

   $ pip install pgl

- Install **PaddleHelix** using ``pip``:

.. code:: console

   $ pip install paddlehelix

- The installation is done!

.. note:: After playing, if you want to deactivate the ``conda`` environment, do this:

.. code:: console

   $ conda deactivate

