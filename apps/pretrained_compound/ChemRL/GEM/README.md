# GEM: Geometry Enhanced Molecular Representation Learning for Property Prediction

## Background

Recent advances in graph neural networks (GNNs) have shown great promise in applying GNNs for molecular representation learning. However, existing GNNs usually treat molecules as topological graph data without fully utilizing the molecular geometry information, which is one of the most critical factors for determining molecular physical, chemical, and biological properties. 

To this end, we propose a novel **G**eometry **E**nhanced **M**olecular representation learning method (GEM):

- At first, we design a geometry-based GNN architecture (GeoGNN) that simultaneously models atoms, bonds, and bond angles in a molecule. 
- Moreover, on top of the devised GNN architecture, we propose several novel geometry-level self-supervised learning strategies to learn spatial knowledge by utilizing the local and global molecular 3D structures.


