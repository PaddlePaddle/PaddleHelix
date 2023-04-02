import os
import numpy as np
import paddle
import pgl
import pickle
from pgl.utils.data import Dataset as BaseDataset
from pgl.utils.data import Dataloader
from scipy.spatial import distance
from scipy.sparse import coo_matrix
from dataloader import DualDataLoader
from tqdm import tqdm

def cos_formula(a, b, c):
    res = (a**2 + b**2 - c**2) / (2 * a * b)
    res = -1. if res < -1. else res
    res = 1. if res > 1. else res
    return np.arccos(res)

class Subset(BaseDataset):
    """
    Subset of a dataset at specified indices.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        """getitem"""
        return self.dataset[self.indices[idx]]

    def __len__(self):
        """len"""
        return len(self.indices)

class DualSubset(BaseDataset):
    """
    Subset for 2D and 3D datasets.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        """getitem"""
        return self.dataset[self.indices[idx]]

    def __len__(self):
        """len"""
        return len(self.indices)

class MoleculeDataset(BaseDataset):
    def __init__(self, data_path, dataset, save_file=True):
        self.data_path = data_path
        self.dataset = dataset
        self.save_file = save_file
        self.labels = []
        self.a2a_graphs = []
        self.e2a_graphs = []
        self.e2e_graphs = []
        self.view = ""

    def __len__(self):
        """ Return the number of molecules. """
        return len(self.labels)

    def __getitem__(self, idx):
        """ Return graphs and label. """
        return self.a2a_graphs[idx], self.e2a_graphs[idx], self.e2e_graphs[idx], self.labels[idx]

    def has_cache(self):
        """ Check cache file."""
        self.graph_path = f'{self.data_path}/{self.dataset}/{self.dataset}_{self.view}_graph.pkl'
        return os.path.exists(self.graph_path)

    def save(self):
        """ Save the generated graphs. """
        print(f'Saving processed {self.view} molecular data...')
        graphs = [self.a2a_graphs, self.e2a_graphs, self.e2e_graphs]
        with open(self.graph_path, 'wb') as f:
            pickle.dump((graphs, self.labels), f)
    
    def load(self):
        """ Load the generated graphs. """
        print(f'Loading processed {self.view} view molecular graphs...')
        with open(self.graph_path, 'rb') as f:
            graphs, labels = pickle.load(f)
        return graphs, labels

    def load_data(self):
        """ Generate the graph for molecule. """
        if self.has_cache():
            graphs, labels = self.load()
            self.a2a_graphs, self.e2a_graphs, self.e2e_graphs = graphs
            self.labels = labels
            # self.labels = np.array(labels).reshape(-1, 1)
        else:
            print(f'Processing raw molecule data for {self.view} view graph...')
            file_name = os.path.join(self.data_path, f"{self.dataset}/{self.dataset}_{self.view}_processed.pkl")
            with open(file_name, 'rb') as f:
                data_mols, data_Y = pickle.load(f)

            for mol, y in tqdm(zip(data_mols, data_Y)):
                graphs = self.build_graph(mol)
                if graphs is None:
                    continue
                self.a2a_graphs.append(graphs[0])
                self.e2a_graphs.append(graphs[1])
                self.e2e_graphs.append(graphs[2])
                self.labels.append(y)

            self.labels = np.array(self.labels)
            if len(self.labels.shape) == 1:
                self.labels = self.labels.reshape(-1, 1)
            if self.save_file:
                self.save()

    def build_graph(self, mol):
        pass

class Molecule2DView(MoleculeDataset):
    def __init__(self, data_path, dataset, save_file=True):
        self.data_path = data_path
        self.dataset = dataset
        self.save_file = save_file

        self.view = "2d"
        self.labels = []
        self.a2a_graphs = []
        self.e2a_graphs = []
        self.e2e_graphs = []
        self.load_data()
        self.atom_feat_dim = self.a2a_graphs[0].node_feat['feat'].shape[-1]
        self.bond_feat_dim = self.e2e_graphs[0].node_feat['feat'].shape[-1]

    def build_graph(self, mol):
        num_atoms, atom_features, atom_2dcoords, bond_features = mol
        dist_mat = distance.cdist(atom_2dcoords, atom_2dcoords, 'euclidean')
        np.fill_diagonal(dist_mat, np.inf)
        if num_atoms == 1:
            return None
        if len(bond_features) == 0:
            print('NO BOND FEATURES,', num_atoms)
            return None

        dist_feats = []
        edge_feats = []
        a2a_edges = []
        indices = []
        # build directional graph
        for i in range(num_atoms):
            for j in range(num_atoms):
                ii, jj = min(i, j), max(i, j)
                bf = bond_features.get((ii, jj))
                if bf is None:
                    continue
                a2a_edges.append((i, j))
                dist_feats.append([dist_mat[i, j]])
                edge_feats.append(bf)
                indices.append([i, j])

        num_nodes = num_atoms
        num_edges = len(indices)

        # edge-to-node and node-to-edge graph
        assignment_e2a = np.zeros((num_edges, num_nodes), dtype=np.int64)
        assignment_a2e = np.zeros((num_nodes, num_edges), dtype=np.int64)
        for i, idx in enumerate(indices):
            assignment_e2a[i, idx[1]] = 1
            assignment_a2e[idx[0], i] = 1

        edge2node_graph = coo_matrix(assignment_e2a)
        node2edge_graph = coo_matrix(assignment_a2e)

        # edge-to-edge graph
        edge_graph_base = assignment_e2a @ assignment_a2e
        np.fill_diagonal(edge_graph_base, 0) # eliminate self connections
        edge_graph_base[range(num_edges), [indices.index([x[1],x[0]]) for x in indices]] = 0 # eliminate connections of the same edge

        x, y = np.where(edge_graph_base > 0)
        angle_feats = []
        for i in range(len(x)):
            body1 = indices[x[i]]
            body2 = indices[y[i]]
            a = dist_mat[body1[0], body1[1]]
            b = dist_mat[body2[0], body2[1]]
            c = dist_mat[body1[0], body2[1]]
            if a == 0 or b == 0:
                print(body1, body2)
                print('One distance is zero.')
                return None
            else:
                angle_feats.append(cos_formula(a, b, c))
        
        atom_features = np.array(atom_features)
        bond_features = np.array(edge_feats)
        # pgl graph
        # a2a_edges = list(zip(range(num_nodes), range(num_nodes)))
        # a2e_edges = list(zip(node2edge_graph.row, node2edge_graph.col))
        # a2e_graph = pgl.BiGraph(a2e_edges, src_num_nodes=num_nodes, dst_num_nodes=num_edges)
        e2a_edges = list(zip(edge2node_graph.row, edge2node_graph.col))
        e2e_edges = list(zip(x, y))
        e2a_graph = pgl.BiGraph(e2a_edges, src_num_nodes=num_edges, dst_num_nodes=num_nodes)
        # print(num_nodes, num_edges, angle_feats)
        # assert len(np.array(angle_feats).shape) == 2
        
        a2a_graph = pgl.Graph(a2a_edges, num_nodes=num_nodes, node_feat={"feat": atom_features}, edge_feat={"dist": dist_feats}) # dist_feats: (num_edges_of_node, )
        e2e_graph = pgl.Graph(e2e_edges, num_nodes=num_edges, node_feat={"feat": bond_features}, edge_feat={"angle": angle_feats}) # angle_feats: (num_edges_of_edge, )
        return a2a_graph, e2a_graph, e2e_graph
    
    
class Molecule3DView(MoleculeDataset):
    def __init__(self, data_path, dataset, cut_dist=4, num_angle=4, num_dist=None, save_file=True):
        self.data_path = data_path
        self.dataset = dataset
        self.save_file = save_file
        self.view = "3d"

        self.cut_dist = cut_dist
        self.num_dist = num_dist
        self.num_angle = num_angle
        if not self.num_dist:
            self.num_dist = 2 if cut_dist <= 4 else 4

        self.labels = []
        self.a2a_graphs = []
        self.e2a_graphs = []
        self.e2e_graphs = []
        self.load_data()
        self.atom_feat_dim = self.a2a_graphs[0].node_feat['feat'].shape[-1]
    
    def build_graph(self, mol):
        num_atoms, atom_features, atom_types, atom_3dcoords, bond_features = mol
        atom_features = np.array(atom_features)
        dist_mat = distance.cdist(atom_3dcoords, atom_3dcoords, 'euclidean')
        np.fill_diagonal(dist_mat, np.inf)
        if num_atoms == 1:
            return None
        if len(bond_features) == 0:
            print('NO BOND FEATURES,', num_atoms)
            return None

        # node-to-node graph
        num_nodes = num_atoms
        dist_graph_base = dist_mat.copy()
        dist_feats = dist_graph_base[dist_graph_base < self.cut_dist].reshape(-1,1)
        dist_graph_base[dist_graph_base >= self.cut_dist] = 0.
        atom_graph = coo_matrix(dist_graph_base)
        a2a_edges = list(zip(atom_graph.row, atom_graph.col))
        a2a_graph = pgl.Graph(a2a_edges, num_nodes=num_nodes, node_feat={"feat": atom_features}, edge_feat={"dist": dist_feats})

        # edge-to-node graph
        indices = []
        for i in range(num_atoms):
            for j in range(num_atoms):
                a = dist_mat[i, j]
                if a < self.cut_dist:
                    indices.append([i, j])

        num_edges = len(indices)
        assignment_e2a = np.zeros((num_edges, num_nodes), dtype=np.int64)
        assignment_a2e = np.zeros((num_nodes, num_edges), dtype=np.int64)
        for i, idx in enumerate(indices):
            assignment_e2a[i, idx[1]] = 1
            assignment_a2e[idx[0], i] = 1
        edge2node_graph = coo_matrix(assignment_e2a)
        e2a_edges = list(zip(edge2node_graph.row, edge2node_graph.col))

        e2a_graph_list = []
        if self.num_dist == 1:
            e2a_graph = pgl.BiGraph(e2a_edges, src_num_nodes=num_edges, dst_num_nodes=num_nodes, edge_feat={"dist": dist_feats})
            e2a_graph_list += [e2a_graph]
        else:
            dist_inds = np.clip(dist_feats, 1.0, self.cut_dist - 1e-8).astype(np.int64) - 1
            if self.num_dist == 2:
                inds = np.where(dist_inds == 0)[0]
                e2a_edges_sub = [e2a_edges[i] for i in inds]
                dist_feat_sub = dist_feats[inds]
                if len(e2a_edges_sub) == 0:
                    e2a_edges_sub = [(0,0)]
                    dist_feat_sub = dist_feats[[0]]
                e2a_graph = pgl.BiGraph(e2a_edges_sub, src_num_nodes=num_edges, dst_num_nodes=num_nodes, edge_feat={"dist": dist_feat_sub})
                e2a_graph_list += [e2a_graph]
                inds = np.where(dist_inds >= 1)[0]
                e2a_edges_sub = [e2a_edges[i] for i in inds]
                dist_feat_sub = dist_feats[inds]
                if len(e2a_edges_sub) == 0:
                    e2a_edges_sub = [(0,0)]
                    dist_feat_sub = dist_feats[[0]]
                e2a_graph = pgl.BiGraph(e2a_edges_sub, src_num_nodes=num_edges, dst_num_nodes=num_nodes, edge_feat={"dist": dist_feat_sub})
                e2a_graph_list += [e2a_graph]
            else:
                for k in range(self.num_dist):
                    inds = np.where(dist_inds == k)[0]
                    e2a_edges_sub = [e2a_edges[i] for i in inds]
                    dist_feat_sub = dist_feats[inds]
                    if len(e2a_edges_sub) == 0:
                        e2a_edges_sub = [(0,0)]
                        dist_feat_sub = dist_feats[[0]]
                    e2a_graph = pgl.BiGraph(e2a_edges_sub, src_num_nodes=num_edges, dst_num_nodes=num_nodes, edge_feat={"dist": dist_feat_sub})
                    e2a_graph_list += [e2a_graph]

        # edge-to-edge graphs
        edge_graph_base = assignment_e2a @ assignment_a2e
        np.fill_diagonal(edge_graph_base, 0) # eliminate self connections
        edge_graph_base[range(num_edges), [indices.index([x[1],x[0]]) for x in indices]] = 0
        x, y = np.where(edge_graph_base > 0)

        # calculate angle
        angle_feat = np.zeros_like(x, dtype=np.float32)
        for i in range(len(x)):
            body1 = indices[x[i]]
            body2 = indices[y[i]]
            a = dist_mat[body1[0], body1[1]]
            b = dist_mat[body2[0], body2[1]]
            c = dist_mat[body1[0], body2[1]]
            if a == 0 or b == 0:
                print(body1, body2)
                print('One distance is zero.')
                return None
            else:
                angle_feat[i] = cos_formula(a, b, c)

        # angle domain divisions
        unit = 180.0 / self.num_angle
        angle_index = (np.rad2deg(angle_feat) / unit).astype('int64')
        angle_index = np.clip(angle_index, 0, self.num_angle - 1)
        e2e_edges_list = [[] for _ in range(self.num_angle)]
        e2e_angle_list = [[] for _ in range(self.num_angle)]
        for i, (ind, radian) in enumerate(zip(angle_index, angle_feat)):
            e2e_edges_list[ind].append((x[i], y[i]))
            e2e_angle_list[ind].append(radian)

        e2e_graph_list = []
        for ind in range(self.num_angle):
            e2e_graph = pgl.Graph(e2e_edges_list[ind], num_nodes=num_edges, edge_feat={"angle": e2e_angle_list[ind]})
            e2e_graph_list.append(e2e_graph)
        
        return a2a_graph, e2a_graph_list, e2e_graph_list


def collate_fn(batch):
    a2a_gs, e2a_gs, e2e_gs, labels = map(list, zip(*batch))
    a2a_g = pgl.Graph.batch(a2a_gs).tensor()
    # e2a_g = pgl.BiGraph.batch(e2a_gs).tensor()
    if type(e2a_gs[0]) == list:
        e2a_g = [pgl.BiGraph.batch([g[i] for g in e2a_gs]).tensor() for i in range(len(e2a_gs[0]))]
    else:
        e2a_g = pgl.BiGraph.batch(e2a_gs).tensor()
    if type(e2e_gs[0]) == list:
        e2e_g = [pgl.Graph.batch([g[i] for g in e2e_gs]).tensor() for i in range(len(e2e_gs[0]))]
    else:
        e2e_g = pgl.Graph.batch(e2e_gs).tensor()
    labels = paddle.to_tensor(np.array(labels), dtype='float32')
    return a2a_g, e2a_g, e2e_g, labels