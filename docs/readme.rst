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

Features
========

- High Efficency: We provide LinearRNA, a highly efficient toolkit for RNA structure prediction and analysis. LinearFold & LinearParitition achieves O(n) complexity in RNA-folding prediction, which is hundreds of times faster than traditional folding techniques.

.. image:: ../.github/LinearRNA.jpg
   :align: center

- Large-scale Representation Learning and Transfer Learning: Self-supervised learning for molecule representations offers prospects of a breakthrough in tasks with limited annotation, including drug profiling, drug-target interaction, protein-protein interaction, RNA-RNA interaction, protein folding, RNA folding, and molecule design. PaddleHelix implements a variety of representation learning algorithms and state-of-the-art large-scale pre-trained models to help developers to start from "the shoulders of giants" quickly.

.. image:: ../.github/paddlehelix_features.jpg
   :align: center

- Easy-to-use APIs: PaddleHelix provide frequently used structures and pre-trained models. You can easily use those components to build up your models and systems.


Installation
============

OS support
----------

Windows, Linux and OSX

Python version
--------------

Python **3.6, 3.7**

Dependencies
-------------------

- PaddlePaddle **>= 2.0.0rc0**
- pgl **>= 1.2.0**

Quick Start
-------------

- PaddleHelix can be installed directly with ``pip``:

.. code:: console

   $ pip install paddlehelix

- or install from source:

.. code:: console

   $ pip install --upgrade git+https://github.com/PaddlePaddle/PaddleHelix.git

.. note:: Please check our :doc:`/installation` part for full installation prerequisites and guide.


Tutorials
=========

- We provide abundant `tutorials`_ to navigate the directory and start quickly.

- PaddleHelix is based on `PaddlePaddle`_, a high-performance Parallelized Deep Learning Platform.

.. _tutorials: https://github.com/PaddlePaddle/PaddleHelix/tree/dev/tutorials
.. _PaddlePaddle: https://github.com/paddlepaddle/paddle


Examples
========

- `Representation Learning_Compounds <https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/pretrained_compound>`_

- `Representation Learning_Proteins <https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/pretrained_protein>`_

- `Drug Target Interaction <https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/drug_target_interaction>`_

- `LinearRNA <https://github.com/PaddlePaddle/PaddleHelix/tree/dev/c/pahelix/toolkit/linear_rna>`_


Contribution
============

If you would like to develop and maintain PaddleHelix with us, please refer to our `GitHub repo`_.

.. _GitHub repo: https://github.com/PaddlePaddle/PaddleHelix



