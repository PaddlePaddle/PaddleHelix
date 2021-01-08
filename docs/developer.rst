====================
Guide for developers
====================

If you need to modify the algorithms/models in **PaddleHelix**, you have to switch to the developer mode. The core algorithms of **PaddleHelix** are mostly implemented in Python, but some also in C++, so you cannot develop **PaddleHelix** simply with ``pip install --editable {pahelix_path}``. To develop on your machine, please do the following:

- Please follow the :doc:`/installation` part to install all dependencies of **PaddleHelix** (``paddlepaddle >= 2.0.0rc0``, ``pgl >= 1.2.0``).

- If you have already installed distributed **PaddleHelix** with ``pip install paddlehelix``, please uninstall it with:

.. code:: console

   $ pip uninstall paddlehelix

- Clone this repository to your local machine, supposed path at `/path_to_your_repo/`:

.. code:: console

   $ git clone https://github.com/PaddlePaddle/PaddleHelix.git /path_to_your_repo/

   $ cd /path_to_your_repo/

- Depends on which model you'd like to modify, go to **LinearRNA** or **Other algorithms**:

  1)**LinearRNA**

  The source code of LinearRNA is at `./c/pahelix/toolkit/linear_rna/linear_rna`. You could modify it for your needs. Then remember to return to the root directory of the repository, run scripts below to re-compile (please ensure there are ``cmake >= 3.6`` and ``g++ >= 4.8`` on your machine):

.. code:: console

   $ sh scripts/prepare.sh

   $ sh scripts/build.sh

- After a successful compilaiton, `import` LinearRNA as following:

.. code:: console

   $ cd build

   $ python
   >>> import c.pahelix.toolkit.linear_rna.linear_rna as linear_rna

- Except LinearRNA, other algorithms in PaddleHelix are all implemented in Python.

  2)**Other algorithms**

  If you want to change these algorithms, just find and modify corresponding ``.py`` files under the path `./pahelix`, then add `/path_to_your_repo/` to your Python environment path:

.. code-block:: python

            import sys
            sys.path.append('/path_to_your_repo/')
            import pahelix

- If you have any question or suggestion, feel free to file on our `GitHub issue page <https://github.com/PaddlePaddle/PaddleHelix/issues>`_. We will response **ASAP**.

