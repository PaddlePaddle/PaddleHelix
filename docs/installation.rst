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

(``-`` means that paddlehelix doesn't have a special version requirement for that package)

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

- If you do not have ``conda`` installed, please check this website to `install`_ it:

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

- Install ``paddle`` based on your choice of GPU/CPU version:

  Check `paddlepaddle official document <https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html>`_ to install **paddle2.0**.

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

