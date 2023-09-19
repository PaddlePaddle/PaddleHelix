import os
from paddle_code.data_loader import *
import time, argparse
import numpy as np
from collections import defaultdict as ddict
from ordered_set import OrderedSet
import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader

from paddle_code.model import coatt

class Main(object):
    def __init__(self, params):
        self.p = params
        self.save_path = os.path.join('paddle_saved', self.p.save)
        if self.p.gpu != '-1':
            self.device = paddle.set_device('gpu:' + self.p.gpu)
        else:
            self.device = paddle.set_device('cpu')

        self.load_data()
        if 'drkg' in self.p.dataset:
            self.ent_mm_emb, self.smiles_emb,self.structure_ent_emb,self.structure_rel_emb = drkg_multimodal_emb(self.ent2id,self.rel2id, self.device)
        else:
            raise
        self.model = self.add_model()
        self.optimizer = self.add_optimizer()
        self.scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=self.p.lr, milestones=[20], gamma=0.1, verbose=False)

        self.data_iter = {
            'train': DataLoader(
                TrainDataset(self.triples['train'], self.p),
                batch_size=self.p.batch_size,
                shuffle=True,
                num_workers=max(0, self.p.num_workers),
                drop_last=True, ),
            'valid_head': DataLoader(TestDataset(self.triples['valid_head'], self.p), batch_size=self.p.test_batch,
                                     num_workers=max(0, self.p.num_workers), ),
            'valid_tail': DataLoader(TestDataset(self.triples['valid_tail'], self.p), batch_size=self.p.test_batch,
                                     num_workers=max(0, self.p.num_workers)),
            'test_head': DataLoader(TestDataset(self.triples['test_head'], self.p), batch_size=self.p.test_batch,
                                    num_workers=max(0, self.p.num_workers)),
            'test_tail': DataLoader(TestDataset(self.triples['test_tail'], self.p), batch_size=self.p.test_batch,
                                    num_workers=max(0, self.p.num_workers)),
        }

    def add_optimizer(self):
        parameters = self.model.parameters()
        # params = []
        # for name, value in parameters.named_parameters():
        #     if name == 'rel_embed.weight' or name == 'ent_embed.weight':
        #         params += [{'params':value}]
        #     else:
        #         params += [{'params': value, 'weight_decay': self.p.l2}]
        if self.p.opt == 'adam':
            return paddle.optimizer.Adam(parameters=parameters, learning_rate=self.p.lr)
        else:
            return paddle.optimizer.SGD(parameters=parameters, learning_rate=self.p.lr, weight_decay=self.p.l2)

    def save_model(self, save_path):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.p)
        }
        paddle.save(state, self.save_path)

    def load_model(self, load_path):
        state = paddle.load(load_path)
        state_dict = state['state_dict']
        # self.best_val_mrr = state['best_val']['mrr']
        # self.best_val = state['best_val']

        self.model.set_state_dict(state_dict)
        self.optimizer.set_state_dict(state['optimizer'])

    def add_model(self):
        if self.p.model == 'coatt':
            model = coatt(self.p, self.ent2id, self.device, self.ent_mm_emb, self.smiles_emb)
        else:
            raise
        model.to(self.device)

        model.ent_embed.weight.set_value(self.structure_ent_emb)
        model.rel_embed.weight.set_value(self.structure_rel_emb)

        del self.structure_ent_emb
        del self.structure_rel_emb
        return model



    def load_data(self):
        ent_set, rel_set = OrderedSet(), OrderedSet()

        for split in ['train', 'test', 'valid']:
            for line in open('{}/{}.txt'.format(self.p.dataset, split), encoding='utf-8'):
                sub, rel, obj = line.strip().split('\t')
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)

        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}


        self.rel2id.update({rel + '_reverse': idx + len(self.rel2id) for idx, rel in enumerate(rel_set)})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.p.num_ent = len(self.ent2id)
        self.p.num_rel = len(self.rel2id) // 2

        self.data = ddict(list)
        sr2o = ddict(set)  
        for split in ['train', 'test', 'valid']:
            for line in open('{}/{}.txt'.format(self.p.dataset, split), encoding='utf-8'):
                sub_str, rel_str, obj_str = line.strip().split('\t')
                sub, rel, obj = int(self.ent2id[sub_str]), int(self.rel2id[rel_str]), int(self.ent2id[obj_str])
                self.data[split].append((sub, rel, obj))
                if split == 'test' or split == 'valid':
                    self.data[split + '_eliminate'].append((sub, rel, obj))

                if split == 'train':
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel + self.p.num_rel)].add(sub)
        self.data = dict(self.data)

        print(len(self.data['train']), len(self.data['test']), len(self.data['valid']), self.p.num_ent, self.p.num_rel)

        self.sr2o = {k: list(v) for k, v in sr2o.items()}  #  tail entity to list
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}

        self.triples = ddict(list)
        if self.p.strategy == 'one_to_n':
            for (sub, rel), obj in self.sr2o.items():
                self.triples['train'].append({'triple': (sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})
        else:
            for sub, rel, obj in self.data['train']:
                rel_inv = rel + self.p.num_rel
                sub_samp = len(self.sr2o[(sub, rel)]) + len(self.sr2o[(obj, rel_inv)])
                sub_samp = np.sqrt(1 / sub_samp) 

                self.triples['train'].append(
                    {'triple': (sub, rel, obj), 'label': self.sr2o[(sub, rel)], 'sub_samp': sub_samp})
                self.triples['train'].append(
                    {'triple': (obj, rel_inv, sub), 'label': self.sr2o[(obj, rel_inv)], 'sub_samp': sub_samp})

        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append(
                    {'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}_{}'.format(split, 'head')].append(
                    {'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

        self.triples = dict(self.triples)




    def run_epoch(self, epoch):
        self.model.train()
        losses = []
        train_iter = self.data_iter['train']

        for sub, rel, neg, label in train_iter:
            self.optimizer.clear_grad()
            if self.p.strategy == 'one_to_x':
                pred = self.model.forward(sub, rel, neg.long(),
                                          self.p.strategy)
                loss = self.model.loss(pred, label)
            else:
                pred = self.model.forward(sub, rel, None, self.p.strategy)
                loss = self.model.loss(pred, label)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        return losses

    def fit(self):
        self.best_val_mrr, self.best_val, self.best_epoch = 0., {}, 0.
        val_mrr = 0
        print('begin training')

        hit10_best = 0
        loss_for_paint = []

        for epoch in range(args.epoch):
            start = time.time()
            train_loss = self.run_epoch(epoch)
            end = time.time()
            loss_for_paint += train_loss
            print('[{},{:.8f},{:.0f}]'.format(epoch, np.mean(train_loss), end - start))

            if ((epoch + 1) % 20 == 0) and epoch>=40:
                start = time.time()
                results = self.evaluate('test')
                end = time.time()
                if hit10_best < results['hits@10']:
                    hit10_best = results['hits@10']
                    hit10_best_epoch = epoch
                    self.save_model(self.save_path)
                print(results['mrr'] * 100,
                    results['mr'],
                    results['hits@1'] * 100,
                    results['hits@3']* 100,
                    results['hits@10'] * 100)

    def get_combined_results(self, left_results, right_results):
        results = {}
        count = float(left_results['count'])

        results['left_mr'] = round(left_results['mr'] / count, 5)
        results['left_mrr'] = round(left_results['mrr'] / count, 5)
        results['right_mr'] = round(right_results['mr'] / count, 5)
        results['right_mrr'] = round(right_results['mrr'] / count, 5)
        results['mr'] = round((left_results['mr'] + right_results['mr']) / (2 * count), 5)
        results['mrr'] = round((left_results['mrr'] + right_results['mrr']) / (2 * count), 5)

        for k in range(10):
            results['left_hits@{}'.format(k + 1)] = left_results['hits@{}'.format(k + 1)] / count
            results['right_hits@{}'.format(k + 1)] = right_results['hits@{}'.format(k + 1)] / count
            results['hits@{}'.format(k + 1)] = (left_results['hits@{}'.format(k + 1)] + right_results['hits@{}'.format(k + 1)]) / (count * 2)

        return results

    def evaluate(self, split):
        self.model.eval()
        left_results = self.predict(split=split, mode='tail_batch')
        right_results = self.predict(split=split, mode='head_batch')
        results = self.get_combined_results(left_results, right_results)
        return results

    # def predict(self, split='valid', mode='tail_batch'):
    #     with paddle.no_grad():
    #         results = {}
    #         test_dataloader = self.data_iter['{}_{}'.format(split, mode.split('_')[0])]
    #         i = 0
    #         for sub, rel, obj, label in test_dataloader:
    #             print(i)
    #             i+= 1
    #             pred = self.model.forward(sub, rel, None, 'one_to_n')
    #             b_range = paddle.arange(pred.shape[0])
    #             target_pred = pred[b_range, obj]
    #             pred = paddle.where(paddle.equal(label, 1).astype('bool'), paddle.zeros_like(pred), pred)  # filter, change the correct entities to 0
    #             pred[b_range, obj] = target_pred
    #             ranks = 1 + paddle.argsort(paddle.argsort(pred, axis=1, descending=True), axis=1, descending=False)[
    #                 b_range, obj]
    #             ranks = ranks

    #             results['count'] = paddle.numel(ranks) + results.get('count', 0.0)
    #             results['mr'] = paddle.sum(ranks).item() + results.get('mr', 0.0)
    #             results['mrr'] = paddle.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
    #             for k in range(10):
    #                 results['hits@{}'.format(k + 1)] = paddle.numel(ranks[ranks <= (k + 1)]) + results.get(
    #                     'hits@{}'.format(k + 1), 0.0)
    #     return results

    def predict(self, split='valid', mode='tail_batch'):
        with paddle.no_grad():
            results = {}
            test_dataloader = self.data_iter['{}_{}'.format(split, mode.split('_')[0])]
            for sub, rel, obj, label in test_dataloader:
                pred = self.model.forward(sub, rel, None, 'one_to_n')
                b_range = np.arange(pred.shape[0])
                target_pred = pred[b_range, obj]
                label = label.numpy()
                pred = np.where(label == 1.0, np.zeros_like(pred), pred)
                pred[b_range, obj] = target_pred
                obj = obj.numpy()
                ranks = 1 + np.argsort(np.argsort(pred, axis=1)[:, ::-1], axis=1)[b_range, obj]                
                results['count'] = np.size(ranks) + results.get('count', 0.0)
                results['mr'] = np.sum(ranks) + results.get('mr', 0.0)
                results['mrr'] = np.sum(1.0 / ranks) + results.get('mrr', 0.0)
                for k in range(10):
                    results['hits@{}'.format(k + 1)] = np.size(ranks[ranks <= (k + 1)]) + results.get(
                        'hits@{}'.format(k + 1), 0.0)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser For Arguments",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data', dest="dataset", default='data/drkg/drkg_all') 
    parser.add_argument("--model", type=str, default='coatt') 
    parser.add_argument("--strategy", type=str, default='one_to_n') # one_to_n  one_to_x
    parser.add_argument("--save", default='coatt')
    parser.add_argument("--gpu", type=str, default='1')
    parser.add_argument("--num_workers", type=int, default=10,)
    parser.add_argument("--opt", type=str, default='adam',)
    parser.add_argument("--l2", type=float, default=0.0, )
    parser.add_argument("--epoch", default=500, type=int,)
    parser.add_argument('--seed', default=43, type=int)
    parser.add_argument('--label_smoothing', dest="label_smoothing", default=0., type=float)

    parser.add_argument("--embed_dim", type=int, default=500,) 
    parser.add_argument("--rel_dim", type=int, default=500,) 
    parser.add_argument("--fusion_dim", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.005,) #0.001
    parser.add_argument('--neg_num', default=1000, type=int,)
    parser.add_argument('--num_filt', dest="num_filt", default=128, type=int,)
    parser.add_argument('--ker_sz', dest="ker_sz", default=9, type=int,)
    parser.add_argument('--threshold', type=float, default=-0.5, )
    parser.add_argument('--num_head', type=int,default=2,)
    parser.add_argument('--interval', type=float, default=5.)
    parser.add_argument('--batch', dest="batch_size", default=300, type=int,)
    parser.add_argument('--test_batch', default=300, type=int,)

    args = parser.parse_args()
    

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    np.random.seed(args.seed)
    paddle.seed(args.seed)

    model = Main(args)
    model.fit()
