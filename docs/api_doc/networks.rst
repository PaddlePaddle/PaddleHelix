****************
pahelix.networks
****************

.. contents:: Table of Contents

basic_block
===========

.. automodule:: pahelix.networks.basic_block
   :members: Activation, MLP
   :inherited-members:

compound_encoder
================

.. automodule:: pahelix.networks.compound_encoder
   :members: AtomEmbedding, BondEmbedding
   :inherited-members:

gnn_block
=========

.. automodule:: pahelix.networks.gnn_block
   :members: GraphNorm, MeanPool, GIN
   :inherited-members:

involution_block
================

.. automodule:: pahelix.networks.involution_block
   :members: Involution2D
   :inherited-members:

lstm_block
==========

.. automodule:: pahelix.networks.lstm_block
   :members: lstm_encoder
   :inherited-members:

optimizer
=========

.. automodule:: pahelix.networks.optimizer
   :members: AdamW
   :inherited-members:

pre_post_process
================

.. autofunction:: pahelix.networks.pre_post_process.pre_post_process_layer

resnet_block
============

.. automodule:: pahelix.networks.resnet_block
   :members: resnet_encoder
   :inherited-members:

transformer_block
=================

.. automodule:: pahelix.networks.transformer_block
   :members: multi_head_attention, positionwise_feed_forward, transformer_encoder_layer, transformer_encoder
   :inherited-members:

Helpful Link
============

Please refer to our `GitHub repo`_ to see the whole module.

.. _GitHub repo: https://github.com/PaddlePaddle/PaddleHelix/tree/dev/pahelix/networks