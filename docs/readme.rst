.. image:: ../.github/paddlehelix_logo.png
   :align: center

*****************************
Welcome to PaddleHelix Helper
*****************************

.. image:: https://travis-ci.org/readthedocs/sphinx_rtd_theme.svg?branch=master
   :target: https://github.com/PaddlePaddle/PaddleHelix/releases
   :alt: Release Version
.. image:: https://img.shields.io/badge/python-3.6+-orange.svg
   :alt: Python Version
.. image:: https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg
   :alt: OS
.. image:: https://readthedocs.org/projects/sphinx-rtd-theme/badge/?version=latest
   :target: https://paddlehelix.readthedocs.io/en/dev/
   :alt: Documentation Status


PaddleHelix is a machine-learning-based bio-computing framework aiming at facilitating the development of the following areas:
  * Vaccine design
  * Drug discovery
  * Precision medicine


Installation
============

OS support
----------

Windows, Linux and OSX

Python version
--------------

Python 3.6, 3.7

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

Since our package requires a paddlepaddle version of 2.0.0rc0 or above and the rdkit dependency cannot be installed directly installed using ``pip`` command, we suggest you use ``conda`` to create a new environment for our project. Detailed instructions are shown below:

- If you haven't used ``conda`` before, you can check this website to `install`_ it:

.. _install: https://docs.conda.io/projects/conda/en/latest/user-guide/install/

- After installing ``conda``, you can create a new ``conda`` envoronment:

.. code:: console

   $ conda create -n paddlehelix python=3.7

- To activate the environment, you can use this command:

.. code:: console

   $ conda activate paddlehelix

- Before installing the paddlhelix package, you should install the rdkit package using ``conda`` command:

.. code:: console

   $ conda install -c conda-forge rdkit

- Then you can install the paddlehelix package using the ``pip`` command:

.. code:: console

   $ pip install paddlehelix

- After installing the paddlehelix, you can run the code now.

- If you want to deactivate the ``conda`` environment, you can use this command:

.. code:: console

   $ conda deactivate


Tutorials
=========

- We provide abundant `tutorials`_ to navigate the directory and start quickly.

- PaddleHelix is based on `PaddlePaddle`_, a high-performance Parallelized Deep Learning Platform.

.. _tutorials: https://github.com/PaddlePaddle/PaddleHelix/tree/dev/tutorials
.. _PaddlePaddle: https://github.com/paddlepaddle/paddle

Features
========

- Highly Efficent: We provide LinearRNA - highly efficient toolkit for mRNA vaccine development. LinearFold & LinearParitition achieves O(n) complexity in RNA-folding prediction, which is hundreds of times faster than traditional folding techniques.

.. image:: ../.github/LinearRNA.jpg
   :align: center

- Large-scale Representation Learning and Transfer Learning: Self-supervised learning for molecule representations offers prospects of a breakthrough in tasks with limited annotation, including drug profiling, drug-target interaction, protein-protein interaction, RNA-RNA interaction, protein folding, RNA folding, and molecule design. PaddleHelix implements a variety of representation learning algorithms and state-of-the-art large-scale pre-trained models to help developers to start from "the shoulders of giants" quickly.

.. image:: ../.github/paddlehelix_features.jpg
   :align: center

- Easy-to-use APIs: PaddleHelix provide frequently used structures and pre-trained models. You can easily use those components to build up your models and systems.


Contribution
============

If you would like to develop and maintain PaddleHelix with us, please refer to our `GitHub repo`_.

.. _GitHub repo: https://github.com/PaddlePaddle/PaddleHelix



