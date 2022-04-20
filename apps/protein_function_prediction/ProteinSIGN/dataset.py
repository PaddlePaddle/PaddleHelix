import os
import glob

from paddle.io import Dataset, DataLoader
import paddle
import numpy as np
import pgl


class GoTermDataset(Dataset):
    def __init__(
        self,
        prot_chain_list,
        num_ang_dom,
        n_feats=26,
        prot_chain_data_path="data/chain_graphs",
        cmap_thresh=10,
        label_data_path="molecular_function.npz",
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
        self.num_ang_dom = num_ang_dom
        self.n_feats = n_feats
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

        ############################
        # build node to node graph #
        ############################
        n_seq = prot_chain["seq"]
        n2n_edges = prot_chain["n2n_edges"]
        n2n_dist = prot_chain["n2n_edge_dist"]
        num_nodes = len(n_seq)
        n2n_g = pgl.Graph(
            n2n_edges,
            num_nodes=num_nodes,
            node_feat={"seq": paddle.to_tensor(n_seq, dtype="int64")},
            edge_feat={"dist": paddle.to_tensor(n2n_dist, dtype="float32")},
        )

        ############################
        # build edge to edge graph #
        ############################
        e2e_edges = prot_chain["e2e_edges"]
        e2e_ang_dom = np.trunc(
            prot_chain["e2e_polar_ang"] * self.num_ang_dom / 180.0
        ).astype(np.int64)
        e2e_ang_dom = np.clip(e2e_ang_dom, 0, self.num_ang_dom - 1)
        e2e_graphs = []
        for i in range(self.num_ang_dom):
            mask = e2e_ang_dom == i
            s_e = e2e_edges[:, 0][mask]
            t_e = e2e_edges[:, 1][mask]
            edges_temp = np.stack([s_e, t_e], axis=1)
            angle_feats = prot_chain["e2e_polar_ang"][mask]
            e2e_g = pgl.Graph(
                edges_temp, num_nodes=len(n2n_edges), edge_feat={"angle": angle_feats}
            )
            e2e_graphs.append(e2e_g)

        out = n2n_g, tuple(e2e_graphs), labels

        if self.use_cache:
            self.cache[prot_chain_name] = out
        return out

    def collate(self, batch):
        n2n_graphs = []
        e2e_graphs = []
        labels = []

        for i, (n2n_g, e2e_g, l) in enumerate(batch):
            n2n_graphs.append(n2n_g)
            e2e_graphs.append(e2e_g)
            labels.append(l)

        new_batch = (
            pgl.Graph.batch(n2n_graphs).tensor(),
            [pgl.Graph.batch(dom_graph).tensor() for dom_graph in zip(*e2e_graphs)],
            paddle.to_tensor(np.array(labels), dtype="float32"),
        )

        return new_batch


class GoTermDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, collate_fn=dataset.collate, **kwargs)
