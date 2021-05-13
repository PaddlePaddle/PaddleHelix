*****************
pahelix.model_zoo
*****************

.. contents:: Table of Contents

pretrain_gnns_model
===================

.. automodule:: pahelix.model_zoo.pretrain_gnns_model
   :members: PretrainGNNModel, AttrmaskModel, SupervisedModel
   :inherited-members:

protein_sequence_model
======================

.. autoclass:: pahelix.model_zoo.protein_sequence_model
   :members: LstmEncoderModel, ResnetEncoderModel, TransformerEncoderModel, PretrainTaskModel, SeqClassificationTaskModel, ClassificationTaskModel, RegressionTaskModel, ProteinEncoderModel, ProteinModel, ProteinCriterion
   :inherited-members:

sd_vae_model
============

.. autoclass:: pahelix.model_zoo.sd_vae_model
   :members: StateDecoder, PerpCalculator, MyPerpLoss, CNNEncoder, MolVAE
   :inherited-members:

seq_vae_model
============

.. autoclass:: pahelix.model_zoo.seq_vae_model
   :members: VAE
   :inherited-members:

Helpful Link
============

Please refer to our `GitHub repo`_ to see the whole module.

.. _GitHub repo: https://github.com/PaddlePaddle/PaddleHelix/tree/dev/pahelix/model_zoo