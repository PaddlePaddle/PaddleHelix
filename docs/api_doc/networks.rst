****************
pahelix.networks
****************

.. contents:: Table of Contents

gnn_block
=========

.. automodule:: pahelix.networks.gnn_block
   :members: gcn_layer, get_layer, gin_layer, mean_recv, sum_recv, max_recv
   :inherited-members:

lstm_block
==========

.. automodule:: pahelix.networks.lstm_block
   :members: lstm_encoder
   :inherited-members:

optimizer
=========

.. autoclass:: pahelix.networks.optimizer.AdamW
   :members: 
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