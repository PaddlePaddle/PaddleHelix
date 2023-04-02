import os
import glob

from paddle.io import Dataset, DataLoader
import paddle
import numpy as np
import pgl
import numpy.linalg as LA
from sklearn.preprocessing import normalize

class GoTermDataset(Dataset):
    def __init__(
        self,
        prot_chain_list,
        n_feats,
        prot_chain_data_path,
        cmap_thresh,
        label_data_path,
        use_cache=False,
    ):
        prot_chain_list = set(prot_chain_list)
        prot_chain_data_dir = os.path.join(prot_chain_data_path, str(cmap_thresh))
        available_proteins = set(
            os.path.splitext(f)[0] for f in os.listdir(prot_chain_data_dir)
        )
        label_data = np.load(label_data_path, allow_pickle=True)
        available_protein_with_labels = set(label_data)

        valid_prots = (
            prot_chain_list & available_proteins & available_protein_with_labels
        )

        self.prot_chain_list = list(valid_prots)
        self.name = label_data["name"]  # This is a string
        self.label_counts = paddle.to_tensor(label_data["counts"], dtype="float32")
        self.labels = label_data
        self.n_labels = len(label_data["idx_goterm_map"].item())
        self.prot_chain_data_dir = prot_chain_data_dir
        self.n_feats = n_feats
        self.cmap_thresh = cmap_thresh
        self.use_cache = use_cache
        self.cache = {}

    def __len__(self):
        return len(self.prot_chain_list)

    def __getitem__(self, index):
        prot_chain_name = self.prot_chain_list[index]
        if prot_chain_name in self.cache:
            return self.cache[prot_chain_name]
        label_idx = self.labels[prot_chain_name].astype("int64")
        labels = np.zeros(self.n_labels)
        labels[label_idx] = 1.0

        prot_chain = np.load(
            os.path.join(self.prot_chain_data_dir + f"/{prot_chain_name}.npz"),
            allow_pickle=True,
        )

        n_seq = prot_chain['seq']
        n2n_edges  = prot_chain['n2n_edges']
        n_graph = pgl.Graph(n2n_edges, num_nodes=len(n_seq))
        n_graph.node_feat['seq'] = paddle.to_tensor(n_seq, dtype="int64")
        n_graph.node_feat['local_sys'] = paddle.to_tensor(prot_chain['local_sys'], 'float32')
        n_graph.node_feat['pos_in_chain'] = paddle.to_tensor(prot_chain['pos_in_chain'], 'int64')
        n_graph.node_feat['node_indx'] = paddle.arange(len(n_seq), dtype='int64')
        
        coords = prot_chain['coords']
        predecessor = np.arange(len(n_seq)) - 1
        predecessor = np.clip(predecessor, 0, predecessor.shape[0]-1)
        successor = np.arange(len(n_seq)) + 1
        successor = np.clip(successor, 0, successor.shape[0] - 1)
        forward_v = coords[successor] - coords
        forward_v = normalize(forward_v, axis=1)
        reverse_v = coords[predecessor] - coords
        reverse_v = normalize(reverse_v, axis=1)
        v_feats = np.stack([forward_v, reverse_v], axis=1)
        n_graph.node_feat['v_feats'] = paddle.to_tensor(v_feats, dtype='float32')
        
        coords = prot_chain['coords']
        neigh_v = coords[n2n_edges[:, 0]] - coords[n2n_edges[:, 1]]
        neigh_v = normalize(neigh_v, axis=1)

        v = prot_chain['cb_coords'] - prot_chain['coords']
        v = normalize(v)
        assert np.allclose((v * v).sum(-1), 1)

        cos_theta = (v[n2n_edges[:, 1]] * v[n2n_edges[:, 0]]).sum(axis=-1)
        assert np.allclose(cos_theta[cos_theta <= -1], -1) and np.allclose(cos_theta[cos_theta >= 1], 1), prot_chain_name
        assert len(n2n_edges) == len(cos_theta)

        cos_theta = np.clip(cos_theta, -1, 1)
        inter_ang = np.rad2deg(np.arccos(cos_theta))
        ang_dom = np.trunc(inter_ang * 2 / 180.0).astype(np.int64)
        ang_dom = np.clip(ang_dom, 0, 1)

        rr_graphs = []
        for i in range(2):
            mask = (ang_dom == i)
            rr_edges = n2n_edges[mask]
            rr_g = pgl.Graph(rr_edges, num_nodes=len(n_seq))
            rr_graphs.append(rr_g)

        out = n_graph, tuple(rr_graphs), labels

        if self.use_cache:
            self.cache[prot_chain_name] = out
        return out

    def collate(self, batch):
        n_graphs = []
        rr_graphs = []
        labels = []

        for i, (n_g, rr_g, l) in enumerate(batch):
            n_graphs.append(n_g)
            rr_graphs.append(rr_g)
            labels.append(l)

        new_batch = (
            pgl.Graph.batch(n_graphs).tensor(),
            [pgl.Graph.batch(g).tensor() for g in zip(*rr_graphs)],
            paddle.to_tensor(np.array(labels), dtype="float32"),
        )

        return new_batch


class GoTermDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, collate_fn=dataset.collate, **kwargs)
