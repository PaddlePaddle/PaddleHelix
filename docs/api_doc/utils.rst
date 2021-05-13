*************
pahelix.utils
*************

.. contents:: Table of Contents

basic_utils
===========

.. automodule:: pahelix.utils.basic_utils
   :members: mp_pool_map, load_json_config
   :inherited-members:

compound_tools
==============

.. automodule:: pahelix.utils.compound_tools
   :members: get_gasteiger_partial_charges, create_standardized_mol_id, check_smiles_validity, split_rdkit_mol_obj, get_largest_mol, rdchem_enum_to_list, CompoundKit, Compound3DKit, mol_to_graph_data, mol_to_md_graph_data, mol_to_polar_graph_data
   :inherited-members:

data_utils
==========

.. automodule:: pahelix.utils.data_utils
   :members: save_data_list_to_npz, load_npz_to_data_list, get_part_files
   :inherited-members:

language_model_tools
====================

.. automodule:: pahelix.utils.language_model_tools
   :members: apply_bert_mask
   :inherited-members:

protein_tools
=============

.. autoclass:: pahelix.utils.protein_tools.ProteinTokenizer
   :members:
   :inherited-members:

splitters
=========

.. automodule:: pahelix.utils.splitters

.. autoclass:: pahelix.utils.splitters.RandomSplitter
   :members:
   :inherited-members:

.. autoclass:: pahelix.utils.splitters.IndexSplitter
   :members:
   :inherited-members:

.. autoclass:: pahelix.utils.splitters.ScaffoldSplitter
   :members:
   :inherited-members:

.. autoclass:: pahelix.utils.splitters.RandomScaffoldSplitter
   :members:
   :inherited-members:

.. autofunction:: pahelix.utils.splitters.generate_scaffold

Helpful Link
============

Please refer to our `GitHub repo`_ to see the whole module.

.. _GitHub repo: https://github.com/PaddlePaddle/PaddleHelix/tree/dev/pahelix/utils

