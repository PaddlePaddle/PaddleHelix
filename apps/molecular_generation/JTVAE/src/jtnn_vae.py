#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""jtvae"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import rdkit
import rdkit.Chem as Chem
import copy
import math
from src.nnutils import flatten_tensor, avg_pool
from src.jtnn_enc import JTNNEncoder
from src.jtnn_dec import JTNNDecoder
from src.mpn import MPN
from src.jtmpn import JTMPN
from src.chemutils import enum_assemble, set_atommap, copy_edit_mol, attach_mols



class JTNNVAE(nn.Layer):
    """JTVAE layer"""
    def __init__(self, vocab, hidden_size, latent_size, depthT, depthG):
        super(JTNNVAE, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.latent_size = latent_size = int(latent_size / 2)  

        self.jtnn = JTNNEncoder(hidden_size, depthT, nn.Embedding(vocab.size(), hidden_size))
        self.decoder = JTNNDecoder(vocab, hidden_size, latent_size, nn.Embedding(vocab.size(), hidden_size))

        self.jtmpn = JTMPN(hidden_size, depthG)
        self.mpn = MPN(hidden_size, depthG)

        self.A_assm = nn.Linear(latent_size, hidden_size, bias_attr=False)
        self.assm_loss = nn.CrossEntropyLoss(reduction='sum')

        self.T_mean = nn.Linear(hidden_size, latent_size)
        self.T_var = nn.Linear(hidden_size, latent_size)
        self.G_mean = nn.Linear(hidden_size, latent_size)
        self.G_var = nn.Linear(hidden_size, latent_size)

    def encode(self, jtenc_holder, mpn_holder):
        """Encode"""
        tree_vecs, tree_mess = self.jtnn(*jtenc_holder)
        mol_vecs = self.mpn(**mpn_holder)
        return tree_vecs, tree_mess, mol_vecs

    def encode_latent(self, jtenc_holder, mpn_holder):
        """Encode latent space"""
        tree_vecs, _ = self.jtnn(*jtenc_holder)
        mol_vecs = self.mpn(**mpn_holder)
        tree_mean = self.T_mean(tree_vecs)
        mol_mean = self.G_mean(mol_vecs)
        tree_var = -paddle.abs(self.T_var(tree_vecs))
        mol_var = -paddle.abs(self.G_var(mol_vecs))
        return paddle.concat([tree_mean, mol_mean], axis=1), paddle.concat([tree_var, mol_var], axis=1)

    def rsample(self, z_vecs, W_mean, W_var, esp=True):
        """
        Reparameterization trick
        Args:
            z_vecs(tensor): latent representation.
            W_mean(tensor): mean vector.
            W_var(tensor): variance vaector.
     
        Returns:
            z_vecs(tensor): resampled latent representation.
            kl_loss(tensor): kl loss.
        """
        batch_size = z_vecs.shape[0]
        z_mean = W_mean(z_vecs)
        z_log_var = -paddle.abs(W_var(z_vecs))  
        kl_loss = -0.5 * paddle.sum(1.0 + z_log_var - z_mean * z_mean - paddle.exp(z_log_var)) / batch_size
        z_mean_shape = paddle.to_tensor(z_mean.shape)
        epsilon = paddle.randn(z_mean_shape)
        if esp:
            z_vecs = z_mean + paddle.exp(z_log_var / 2) * epsilon
        else:
            z_vecs = z_mean + paddle.exp(z_log_var / 2) * 1
        return z_vecs, kl_loss

    def sample_prior(self, prob_decode=False):
        """       
        Sample a molecule from prior distribution.
        Args:
            prob_decode(bool): using bernoulli distribution in graph decode if prob_decode=true.
    
        Returns:
            a smiles.
        """
        z_tree = paddle.randn([1, self.latent_size])
        z_mol = paddle.randn([1, self.latent_size])
        return self.decode(z_tree, z_mol, prob_decode)

    def reconstruction(self, x_jtenc_holder, x_mpn_holder):
        """
        Reconstruct a molecule
        Args:
            x_jtenc_holder(tuple): (tree feature, message dict).
            x_mpn_holder(dict): graph feature.
        Returns:
            a smiles list.
        """
        x_tree_vecs, x_tree_mess, x_mol_vecs = self.encode(x_jtenc_holder, x_mpn_holder)
        z_tree_vecs, tree_kl = self.rsample(x_tree_vecs, self.T_mean, self.T_var)
        z_mol_vecs, mol_kl = self.rsample(x_mol_vecs, self.G_mean, self.G_var)
        res = []
        for i in range(z_tree_vecs.shape[0]):
            z_tree_vecs_i = z_tree_vecs[i].reshape([1, z_tree_vecs[i].shape[0]])
            z_mol_vecs_i = z_mol_vecs[i].reshape([1, z_mol_vecs[i].shape[0]])
            res.append(self.decode(z_tree_vecs_i, z_mol_vecs_i, False))
        return res

    def forward(self, batch, beta):
        """
        Forward
        Args:
            batch(dict): batch data.
            beta(float): KL regularization weight.
        Returns:
            a dict which contains total loss, kl divergence, word accuracy, topology accuracy and assembly accuracy.
        """
        x_batch = batch['tree_batch']
        x_jtenc_holder = batch['jtenc_holder']

        x_mpn_holder = batch['mpn_holder']
        x_jtmpn_holder = batch['jtmpn_holder']
        x_tree_vecs, x_tree_mess, x_mol_vecs = self.encode(x_jtenc_holder, x_mpn_holder)
        z_tree_vecs, tree_kl = self.rsample(x_tree_vecs, self.T_mean, self.T_var)
        z_mol_vecs, mol_kl = self.rsample(x_mol_vecs, self.G_mean, self.G_var)
        kl_div = tree_kl + mol_kl
        decoder_res = self.decoder(x_batch, z_tree_vecs)
        word_loss = decoder_res['pred_loss'] 
        topo_loss = decoder_res['stop_loss']
        word_acc = decoder_res['pred_acc']
        topo_acc = decoder_res['stop_acc']
        assm_loss, assm_acc = self.assm(x_batch, x_jtmpn_holder, z_mol_vecs, x_tree_mess)
        return {'loss': word_loss + topo_loss + assm_loss + beta * kl_div, 
                'kl_div': float(kl_div.numpy()), 
                'word_acc': word_acc, 
                'topo_acc': topo_acc, 
                'assm_acc': assm_acc}

    def assm(self, mol_batch, jtmpn_holder, x_mol_vecs, x_tree_mess):
        """
        Assemble subgraph
        Args:
            mol_batch(list): molecule trees.
            jtmpn_holder(tuple): (graph feature dict, batch idx).
            x_mol_vecs(tensor): graph latent represenation.
            x_tree_mess(tensor): tree latent represenation.
        Returns:
            graph assemble loss and accuracy.
        """
        jtmpn_holder, batch_idx = jtmpn_holder
        fatoms = jtmpn_holder['fatoms']
        fbonds = jtmpn_holder['fbonds']
        agraph = jtmpn_holder['agraph']
        bgraph = jtmpn_holder['bgraph']
        scope = jtmpn_holder['scope']
        batch_idx = paddle.to_tensor(batch_idx)

        cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, x_tree_mess)

        x_mol_vecs = paddle.index_select(axis=0, index=batch_idx, x=x_mol_vecs)
        x_mol_vecs = self.A_assm(x_mol_vecs) 
        scores = paddle.bmm(
            x_mol_vecs.unsqueeze(1),
            cand_vecs.unsqueeze(-1)
        ).squeeze()

        cnt, tot, acc = 0, 0, 0
        all_loss = []
        for i, mol_tree in enumerate(mol_batch):
            comp_nodes = [node for node in mol_tree.nodes if len(node.cands) > 1 and not node.is_leaf]
            cnt += len(comp_nodes)
            for node in comp_nodes:
                label = node.cands.index(node.label)
                ncand = len(node.cands)
                cur_score = paddle.slice(scores, [0], [tot], [tot + ncand])
                tot += ncand
                if float(cur_score[label].numpy()) >= float(cur_score.max().numpy()):
                    acc += 1

                label = paddle.to_tensor([label])
                all_loss.append(self.assm_loss(paddle.reshape(cur_score, shape=[1, -1]), label))

        all_loss = sum(all_loss) / len(mol_batch)
        return all_loss, acc * 1.0 / cnt

    def decode(self, x_tree_vecs, x_mol_vecs, prob_decode):
        """
        Decode smiles from latent space.
        Args:
            x_mol_vecs(tensor): graph latent represenation.
            x_tree_mess(tensor): tree latent represenation.
            prob_decode(bool): using bernoulli distribution in graph decode if prob_decode=true.
        Returns:
            smiles.
        """
        assert x_tree_vecs.shape[0] == 1 and x_mol_vecs.shape[0] == 1

        pred_root, pred_nodes = self.decoder.decode(x_tree_vecs, prob_decode)
        if len(pred_nodes) == 0:
            return None
        elif len(pred_nodes) == 1:
            return pred_root.smiles

        for i, node in enumerate(pred_nodes):
            node.nid = i + 1
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)

        scope = [(0, len(pred_nodes))]
        jtenc_holder, mess_dict = JTNNEncoder.tensorize_nodes(pred_nodes, scope)
        _, tree_mess = self.jtnn(*jtenc_holder)
        tree_mess = (tree_mess, mess_dict)  

        x_mol_vecs = self.A_assm(x_mol_vecs).squeeze()  

        cur_mol = copy_edit_mol(pred_root.mol)
        global_amap = [{}] + [{} for node in pred_nodes]
        global_amap[1] = {atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()}

        cur_mol, _ = self.dfs_assemble(tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [], pred_root, None,
                                       prob_decode, check_aroma=True)
        if cur_mol is None:
            cur_mol = copy_edit_mol(pred_root.mol)
            global_amap = [{}] + [{} for node in pred_nodes]
            global_amap[1] = {atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()}
            cur_mol, pre_mol = self.dfs_assemble(tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [], pred_root,
                                                 None, prob_decode, check_aroma=False)
            if cur_mol is None: cur_mol = pre_mol

        if cur_mol is None:
            return None

        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        return Chem.MolToSmiles(cur_mol) if cur_mol is not None else None

    def dfs_assemble(self, y_tree_mess, x_mol_vecs, all_nodes, cur_mol, global_amap, fa_amap, cur_node, fa_node,
                     prob_decode, check_aroma):
        """DFS in subgraph assembly"""
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid, a2, a1) for nid, a1, a2 in fa_amap if nid == cur_node.nid]
        cands, aroma_score = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0 or (sum(aroma_score) < 0 and check_aroma):
            return None, cur_mol

        cand_smiles, cand_amap = zip(*cands)

        aroma_score = paddle.to_tensor(aroma_score)
        cands = [(smiles, all_nodes, cur_node) for smiles in cand_smiles]

        if len(cands) > 1:
            jtmpn_holder = JTMPN.tensorize(cands, y_tree_mess[1])
            fatoms = jtmpn_holder['fatoms']
            fbonds = jtmpn_holder['fbonds']
            agraph = jtmpn_holder['agraph']
            bgraph = jtmpn_holder['bgraph']
            scope = jtmpn_holder['scope']
            cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, y_tree_mess[0])
            scores = paddle.mv(cand_vecs, x_mol_vecs) + aroma_score
        else:
            scores = paddle.to_tensor([1.0])

        if prob_decode:
            probs = paddle.squeeze(F.softmax(paddle.reshape(scores, shape=[1, -1]), axis=1)) + 1e-7  
            cand_idx = paddle.multinomial(probs, probs.numel())
        else:
            cand_idx = paddle.argsort(scores, descending=True)

        backup_mol = Chem.RWMol(cur_mol)
        pre_mol = cur_mol
        for i in range(cand_idx.numel()):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[int(cand_idx[i].numpy())]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id, ctr_atom, nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]

            cur_mol = attach_mols(cur_mol, children, [], new_global_amap) 
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None:
                continue

            has_error = False
            for nei_node in children:
                if nei_node.is_leaf:
                    continue
                tmp_mol, tmp_mol2 = self.dfs_assemble(y_tree_mess, x_mol_vecs, all_nodes, cur_mol, new_global_amap,
                                                      pred_amap, nei_node, cur_node, prob_decode, check_aroma)
                if tmp_mol is None:
                    has_error = True
                    if i == 0: pre_mol = tmp_mol2
                    break
                cur_mol = tmp_mol
            if not has_error: return cur_mol, cur_mol
        return None, pre_mol

