============
Installation
============

.. contents:: Table of Contents

OS support
----------

Windows, Linux and OSX

Python version
--------------

Python **3.6, 3.7**

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