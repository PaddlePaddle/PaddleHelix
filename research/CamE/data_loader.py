import paddle
from paddle.io import Dataset
import numpy as np

class TrainDataset(Dataset):
    def __init__(self, triples, params):
        self.triples = triples
        self.p = params
        self.strategy = self.p.strategy
        self.entities = np.arange(self.p.num_ent, dtype=np.int32)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label = ele['triple'], ele['label']

        if self.strategy == 'one_to_n':
            y = paddle.zeros([self.p.num_ent], dtype=paddle.float32) + self.p.label_smoothing
            for e2 in label: y[e2] = 1.0
            return triple[0],triple[1],0,y

        elif self.strategy == 'one_to_x':
            neg_ent = self.get_neg_ent(triple, label)

            y = paddle.zeros((neg_ent.shape[0]), dtype=paddle.float32) + self.p.label_smoothing
            y[0] = 1.
            return triple[0],triple[1],neg_ent,y
        else:
            raise ValueError("Invalid strategy")

    def get_neg_ent(self, triple, label):
        if self.strategy == 'one_to_x':
            pos_obj = triple[2]
            mask = np.ones([self.p.num_ent], dtype=np.bool)
            mask[label] = 0

            neg_ent = np.random.choice(self.entities[mask], self.p.neg_num+1,replace=False)
            neg_ent[0] = pos_obj

        else: 
            raise ValueError("Invalid strategy")
        
        return neg_ent.astype(np.int64)


class TestDataset(Dataset):
    def __init__(self, triples, params):
        self.triples = triples
        self.p = params
    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label = ele['triple'], ele['label']
        label = self.get_label(label)
        return triple[0],triple[1],triple[2], label

    def get_label(self, label):
        y = paddle.zeros([self.p.num_ent], dtype=paddle.float32)
        for e2 in label: y[e2] = 1.0
        return y



import h5py
import numpy as np
import paddle

def drkg_multimodal_emb(ent2id, rel2id, device):
    ent_mm_emb = paddle.zeros([len(ent2id), 768], dtype='float32')
    Smiles_emb = paddle.zeros([len(ent2id), 300], dtype='float32')
    structure_ent_emb = paddle.zeros([len(ent2id), 500], dtype='float32')
    structure_rel_emb = paddle.zeros([len(rel2id), 500], dtype='float32')

    files = ['Anatomy', 'Gene', 'Disease', 'Compound', 'BiologicalProcess', 'CellularComponent', 'MolecularFunction', 'Pathway', 'PharmacologicClass', 'Side-effect', 'Symptom']
    for f_name in files:
        with h5py.File("data/drkg/multimodal_emb/" + f_name + ".h5", 'r') as f:
            for k in f.keys():
                v = np.array(f[k])
                if k in ent2id.keys():
                    ent_mm_emb[ent2id[k]] = paddle.to_tensor(v.sum(0, keepdims=False).astype('float32'))
    S_emb = np.load("data/drkg/multimodal_emb/mol_masking.npy")
    S_emb = paddle.to_tensor(S_emb.astype('float32'))


    with open("data/drkg/multimodal_emb/Compound_smiles.txt", 'r', encoding='utf-8') as f:
        index = 0
        for line in f.readlines():
            entname, smile = line.strip().split('\t')
            try:
                ID = ent2id['Compound::' + entname]
                Smiles_emb[ID] = S_emb[index]
                index += 1
            except:
                pass


    with h5py.File("data/drkg/multimodal_emb/structure_drkg_ent.h5", 'r') as f:
        for k in f.keys():
            if k not in ent2id.keys(): continue
            structure_ent_emb[ent2id[k]] = paddle.to_tensor(np.array(f[k]).astype('float32'))

    with h5py.File("data/drkg/multimodal_emb/structure_drkg_ent.h5", 'r') as f:
        for k in f.keys():
            if k not in rel2id.keys(): continue
            structure_rel_emb[rel2id[k]] = paddle.to_tensor(np.array(f[k])[:500].astype('float32'))

    return ent_mm_emb, Smiles_emb, structure_ent_emb, structure_rel_emb
    
