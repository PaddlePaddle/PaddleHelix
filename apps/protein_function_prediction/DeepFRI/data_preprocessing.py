import paddle

paddle.disable_static()
from pgl.utils.data import Dataset
import pgl
import numpy as np
import os
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(
        self,
        prot_chain_list,
        n_feats=26,
        padded_len=1000,
        prot_chain_data_path="data/chain_graphs",
        cmap_thresh=10,
        label_data_path="data/molecular_function.npz",
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
        self.padded_len = padded_len
        self.one_hot = np.eye(self.n_feats).astype("float32")

        self.use_cache = use_cache
        self.cache = {}
        # self.load_data()

    def load_data(self):
        self.cache = {}
        self.use_cache = True
        for chain_name in tqdm(self.prot_chain_list, desc="Loading data"):
            self.cache[chain_name] = self.lod_prot_chain(chain_name)

    def lod_prot_chain(self, prot_chain_name):
        if prot_chain_name in self.cache:
            return self.cache[prot_chain_name]
        label_idx = self.labels[prot_chain_name].astype("int64")
        labels = np.zeros(self.n_labels)
        labels[label_idx] = 1.0

        prot_chain = np.load(
            os.path.join(self.prot_chain_data_dir + f"/{prot_chain_name}.npz"),
            allow_pickle=True,
        )

        seq = prot_chain["seq"]
        edges = prot_chain["n2n_edges"]
        num_nodes = len(seq)

        n_self_loops = np.sum(edges[:, 0] == edges[:, 1])
        if n_self_loops == 0:
            node_id = np.arange(num_nodes, dtype="int64")
            self_loop = np.array([node_id, node_id]).T
            edges = np.concatenate([edges, self_loop])

        p_graph = pgl.Graph(
            edges,
            num_nodes=num_nodes,
            node_feat={"seq": paddle.to_tensor(seq, dtype="int64")},
        )
        padded_features = np.zeros((self.padded_len, self.n_feats)).astype("float32")
        seq_one_hot = self.one_hot[seq]
        padded_features[: seq.shape[0]] = seq_one_hot
        out = p_graph, padded_features, labels

        if self.use_cache:
            self.cache[prot_chain_name] = out
        return out

    def __len__(self):
        return len(self.prot_chain_list)

    def __getitem__(self, index):
        prot_chain_name = self.prot_chain_list[index]
        return self.lod_prot_chain(prot_chain_name)

    def collate_fn(self, batch):
        p_graphs = []
        padded_feats = []
        labels = []

        for i, (p_g, padded_f, l) in enumerate(batch):
            seq_valid_idx = (
                np.arange(p_g.num_nodes).astype("int64") + self.padded_len * i
            )  # TODO: Useful if language model is used for node embedding.
            p_g.node_feat["seq_valid_idx"] = paddle.to_tensor(
                seq_valid_idx, dtype="int64"
            )
            p_graphs.append(p_g)
            padded_feats.append(padded_f)
            labels.append(l)

        new_batch = (
            pgl.Graph.batch(p_graphs).tensor(),
            paddle.to_tensor(np.array(padded_feats), dtype="float32"),
            paddle.to_tensor(np.array(labels), dtype="float32"),
        )

        return new_batch
