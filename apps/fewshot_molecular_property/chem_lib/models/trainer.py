import random
import os
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import joblib
import pgl.graph as G
from paddle.io import DataLoader

from .maml import MAML
from ..datasets import sample_meta_datasets, sample_test_datasets, MoleculeDataset
from ..utils import Logger

class Meta_Trainer(nn.Layer):
    def __init__(self, args, model):
        super(Meta_Trainer, self).__init__()

        self.args = args

        self.model = MAML(model, lr=args.inner_lr, first_order=not args.second_order, anil=False, allow_unused=True)

        self.optimizer = paddle.optimizer.AdamW(parameters=self.model.parameters(), learning_rate=args.meta_lr, weight_decay=args.weight_decay, grad_clip=nn.ClipGradByNorm(1))

        self.criterion = nn.CrossEntropyLoss()

        self.dataset = args.dataset
        self.test_dataset = args.test_dataset if args.test_dataset is not None else args.dataset
        self.data_dir = args.data_dir
        self.train_tasks = args.train_tasks
        self.test_tasks = args.test_tasks
        self.n_shot_train = args.n_shot_train
        self.n_shot_test = args.n_shot_test
        self.n_query = args.n_query

        self.in_lr = args.inner_lr

        self.emb_dim = args.emb_dim

        self.batch_task = args.batch_task

        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.inner_update_step = args.inner_update_step

        self.trial_path = args.trial_path
        trial_name = self.dataset + '_' + self.test_dataset + '@' + args.enc_gnn
        print(trial_name)
        logger = Logger(self.trial_path + '/results.txt', title=trial_name)
        log_names = ['Epoch']
        log_names += ['AUC-' + str(t) for t in args.test_tasks]
        log_names += ['AUC-Avg', 'AUC-Mid','AUC-Best']
        logger.set_names(log_names)
        self.logger = logger

        ###moleculedataset unknown###
        preload_train_data = {}
        if args.preload_train_data:
            print('preload train data')
            for task in self.train_tasks:
                dataset = MoleculeDataset(self.data_dir + self.dataset + "/new/" + str(task + 1),
                                          dataset=self.dataset)
                preload_train_data[task] = dataset
        preload_test_data = {}
        if args.preload_test_data:
            print('preload_test_data')
            for task in self.test_tasks:
                dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/new/" + str(task + 1),
                                          dataset=self.test_dataset)
                preload_test_data[task] = dataset
        self.preload_train_data = preload_train_data
        self.preload_test_data = preload_test_data
        if 'train' in self.dataset and args.support_valid:
            val_data_name = self.dataset.replace('train','valid')
            print('preload_valid_data')
            preload_val_data = {}
            for task in self.train_tasks:
                dataset = MoleculeDataset(self.data_dir + val_data_name + "/new/" + str(task + 1),
                                          dataset=val_data_name)
                preload_val_data[task] = dataset
            self.preload_valid_data = preload_val_data

        self.train_epoch = 0
        self.best_auc=0
        
        self.res_logs=[]

    def loader_to_samples(self, data):
        loader = DataLoader(data, batch_size=len(data), shuffle=False, num_workers=0)
        for samples in loader:
            samples=samples.to(self.device)
            return samples
    
    def get_data_sample(self,task_id, train=True):
        if train:
            task = self.train_tasks[task_id]
            if task in self.preload_train_data:
                dataset = self.preload_train_data[task]
            else:
                dataset = MoleculeDataset(self.data_dir + self.dataset + "/new/" + str(task + 1), dataset=self.dataset)

            s_data, q_data = sample_meta_datasets(dataset, self.dataset, task, self.n_shot_train, self.n_query)

            s_data_y = np.stack([i.y[0] for i in s_data.data_list])
            q_data_y = np.stack([i.y[0] for i in q_data.data_list])
            adapt_data = {'s_data': G.Graph.batch(s_data.data_list), 's_label': paddle.to_tensor(s_data_y), 'q_data': G.Graph.batch(q_data.data_list), 'q_label': paddle.to_tensor(q_data_y)}
            eval_data = { }
        else:
            task = self.test_tasks[task_id]
            if 'train' in self.dataset:
                dataset = self.preload_test_data[task]
                if self.args.support_valid:
                    val_dataset = self.preload_valid_data[task]
                    data_name = self.dataset.replace('train','valid')
                else:
                    val_dataset = self.preload_train_data[task]
                    data_name =self.dataset
                s_data, _, q_data_adapt = sample_test_datasets(val_dataset, data_name, task, self.n_shot_test, self.n_query, self.update_step_test)
                s_data = self.loader_to_samples(s_data)
                q_loader = DataLoader(dataset, batch_size=self.n_query, shuffle=True, num_workers=0)
                q_loader_adapt = DataLoader(q_data_adapt, batch_size=self.n_query, shuffle=True, num_workers=0)
                adapt_data = {'s_data': s_data, 's_label': s_data.y, 'data_loader': q_loader_adapt}
                eval_data = {'s_data': s_data, 's_label': s_data.y, 'data_loader': q_loader}
                return adapt_data, eval_data
            if task in self.preload_test_data:
                dataset = self.preload_test_data[task]
            else:
                dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/new/" + str(task + 1),
                                          dataset=self.test_dataset)
            s_data, q_data, q_data_adapt = sample_test_datasets(dataset, self.test_dataset, task, self.n_shot_test, self.n_query, self.update_step_test)
            
            s_data_y = np.stack([i.y[0] for i in s_data.data_list])

            q_loader = q_data.get_data_loader(batch_size=self.n_query, shuffle=True, num_workers=1)
            q_loader_adapt = q_data_adapt.get_data_loader(batch_size=self.n_query, shuffle=True, num_workers=1)

            adapt_data = {'s_data': G.Graph.batch(s_data.data_list), 's_label': paddle.to_tensor(s_data_y), 'data_loader': q_loader_adapt}
            eval_data = {'s_data': G.Graph.batch(s_data.data_list), 's_label': paddle.to_tensor(s_data_y), 'data_loader': q_loader}

        return adapt_data, eval_data

    def get_prediction(self, model, data, train=True):
        if train:
            s_logits, q_logits, adj, node_emb = model(data['s_data'], data['q_data'], data['s_label'])
            pred_dict = {'s_logits': s_logits, 'q_logits': q_logits, 'adj': adj, 'node_emb': node_emb}
        else:
            s_logits, logits,labels,adj_list,sup_labels = model.layers.forward_query_loader(data['s_data'], data['data_loader'], data['s_label'])
            pred_dict = {'s_logits':s_logits, 'logits': logits, 'labels': labels,'adj':adj_list,'sup_labels':sup_labels}

        return pred_dict

    def get_adaptable_weights(self, model, adapt_weight=None):
        if adapt_weight is None:
            adapt_weight = self.args.adapt_weight
        fenc = lambda x: x[0]== 'mol_encoder'
        frel = lambda x: x[0]== 'adapt_relation'
        fedge = lambda x: x[0]== 'adapt_relation' and 'layer_edge'  in x[1]
        fnode = lambda x: x[0]== 'adapt_relation' and 'layer_node'  in x[1]
        fclf = lambda x: x[0]== 'adapt_relation' and 'fc'  in x[1]
        if adapt_weight==0:
            flag=lambda x: not fenc(x)
        elif adapt_weight==1:
            flag=lambda x: not frel(x)
        elif adapt_weight==2:
            flag=lambda x: not (fenc(x) or frel(x))
        elif adapt_weight==3:
            flag=lambda x: not (fenc(x) or fedge(x))
        elif adapt_weight==4:
            flag=lambda x: not (fenc(x) or fnode(x))
        elif adapt_weight==5:
            flag=lambda x: not (fenc(x) or fnode(x) or fedge(x))
        elif adapt_weight==6:
            flag=lambda x: not (fenc(x) or fclf(x))
        else:
            flag= lambda x: True
        if self.train_epoch < self.args.meta_warm_step or self.train_epoch>self.args.meta_warm_step2:
            adaptable_weights = None
        else:
            adaptable_weights = []
            adaptable_names=[]
            for name, p in model.layers.named_parameters():
                names=name.split('.')
                if flag(names):
                    adaptable_weights.append(id(p))
                    adaptable_names.append(name)
        return adaptable_weights

    def get_loss(self, model, batch_data, pred_dict, train=True, flag = 0):
        n_support_train = self.args.n_shot_train
        n_support_test = self.args.n_shot_test
        n_query = self.args.n_query
        if not train:
            losses_adapt = self.criterion(pred_dict['s_logits'].reshape((2*n_support_test*n_query,2)),
                                          paddle.expand(batch_data['s_label'],[n_query,n_support_test*2]).reshape((1,2*n_support_test*n_query)).squeeze(0))
        else:
            if flag:
                losses_adapt = self.criterion(pred_dict['s_logits'].reshape((2*n_support_train*n_query,2)),
                                              paddle.expand(batch_data['s_label'],[n_query,n_support_train*2]).reshape((1,2*n_support_train*n_query)).squeeze(0))
            else:
                losses_adapt = self.criterion(pred_dict['q_logits'], batch_data['q_label'])

        if paddle.isnan(losses_adapt).any() or paddle.isinf(losses_adapt).any():
            print('!!!!!!!!!!!!!!!!!!! Nan value for supervised CE loss', losses_adapt)
            print(pred_dict['s_logits'])
            losses_adapt = paddle.zeros_like(losses_adapt)

        if self.args.reg_adj > 0:
            n_support = batch_data['s_label'].shape[0]
            adj = pred_dict['adj'][-1]
            if train:
                if flag:
                    s_label = paddle.expand(batch_data['s_label'], [n_query,batch_data['s_label'].shape[0]])
                    n_d = n_query * n_support
                    label_edge = model.layers.label2edge(s_label).reshape((n_d, -1))
                    pred_edge = adj[:,:,:-1,:-1].reshape((n_d, -1))
                else:
                    s_label = paddle.expand(batch_data['s_label'], [n_query,batch_data['s_label'].shape[0]])
                    q_label = batch_data['q_label'].unsqueeze(1)
                    total_label = paddle.concat([s_label, q_label], 1)
                    label_edge = model.layers.label2edge(total_label)[:,:,-1,:-1]
                    pred_edge = adj[:,:,-1,:-1]
            else:
                s_label = batch_data['s_label'].unsqueeze(0)
                n_d = n_support * self.args.rel_edge
                label_edge = model.layers.label2edge(s_label).reshape((n_d, -1))
                pred_edge = adj[:, :, :n_support, :n_support].mean(0).reshape((n_d, -1))
            adj_loss_val = F.mse_loss(pred_edge, label_edge)
            if paddle.isnan(adj_loss_val).any() or paddle.isinf(adj_loss_val).any():
                print('!!!!!!!!!!!!!!!!!!!  Nan value for adjacency loss', adj_loss_val)
                adj_loss_val = paddle.zeros_like(adj_loss_val)

            losses_adapt += self.args.reg_adj * adj_loss_val

        return losses_adapt

    def adapt_gradient_descent(self, model, lr, loss, approximate=True, memo=None):
        # copy the function from paddlefsl.utils.gradient_descent
        # Maps original data_ptr to the cloned tensor.
        # Useful when a model uses parameters from another model.
        memo = set() if memo is None else set(memo)
        # Do gradient descent on parameters
        gradients = []
        if len(model.layers.parameters()) != 0:
            gradients = paddle.grad(loss,
                                    model.layers.parameters(),
                                    retain_graph=not approximate,
                                    create_graph=not approximate,
                                    allow_unused=True)
        update_values = [- lr * grad if grad is not None else None for grad in gradients]
        for param, update in zip(model.layers.parameters(), update_values):
            if update is not None:
                param_ptr = id(param)
                if param_ptr in memo:
                    param.set_value(param.add(update))

    def train_step(self):

        self.train_epoch += 1

        task_id_list = list(range(len(self.train_tasks)))
        if self.batch_task > 0:
            batch_task = min(self.batch_task, len(task_id_list))
            task_id_list = random.sample(task_id_list, batch_task)
        data_batches={}
        for task_id in task_id_list:
            db = self.get_data_sample(task_id, train=True)
            data_batches[task_id]=db

        for k in range(self.update_step):
            losses_eval = []
            for task_id in task_id_list:
                train_data, _ = data_batches[task_id]
                model = self.model.clone()
                model.train()
                adaptable_weights = self.get_adaptable_weights(model)

                for inner_step in range(self.inner_update_step):
                    pred_adapt = self.get_prediction(model, train_data, train=True)
                    loss_adapt = self.get_loss(model, train_data, pred_adapt, train=True, flag = 1)

                    self.adapt_gradient_descent(model, self.in_lr, loss_adapt, memo = adaptable_weights)

                pred_eval = self.get_prediction(model, train_data, train=True)
                loss_eval = self.get_loss(model, train_data, pred_eval, train=True, flag = 0)

                losses_eval.append(loss_eval)

            losses_eval = paddle.stack(losses_eval)
            losses_eval = paddle.sum(losses_eval)

            losses_eval = losses_eval / len(task_id_list)

            self.optimizer.clear_grad()
            losses_eval.backward()
            self.optimizer.step()

            print('Train Epoch:',self.train_epoch,', train update step:', k, ', loss_eval:', float(losses_eval))

        return self.model.layers
        

    def test_step(self):
        step_results={'query_preds':[], 'query_labels':[], 'query_adj':[],'task_index':[]}
        auc_scores = []
        for task_id in range(len(self.test_tasks)):
            adapt_data, eval_data = self.get_data_sample(task_id, train=False)
            model = self.model.clone()
            if self.update_step_test>0:
                model.train()
                
                for i, batch in enumerate(adapt_data['data_loader']):
                    cur_adapt_data = {'s_data': adapt_data['s_data'], 's_label': adapt_data['s_label'],
                                    'q_data': G.Graph.batch(batch), 'q_label': None}

                    adaptable_weights = self.get_adaptable_weights(model)
                    pred_adapt = self.get_prediction(model, cur_adapt_data, train=True)
                    loss_adapt = self.get_loss(model, cur_adapt_data, pred_adapt, train=False)

                    self.adapt_gradient_descent(model, self.in_lr, loss_adapt, memo = adaptable_weights)

                    if i>= self.update_step_test-1:
                        break

            model.eval()
            with paddle.no_grad():
                pred_eval = self.get_prediction(model, eval_data, train=False)
                y_score = F.softmax(pred_eval['logits'],axis=-1).detach()[:,1]
                y_true = pred_eval['labels']
                if self.args.eval_support:
                    y_s_score = F.softmax(pred_eval['s_logits'],axis=-1).detach()[:,1]
                    y_s_true = eval_data['s_label']
                    y_score=paddle.concat([y_score, y_s_score])
                    y_true=paddle.concat([y_true, y_s_true])

                mm = paddle.metric.Auc()
                mm.update(preds = np.concatenate((1 - y_score.unsqueeze(-1).numpy(),y_score.unsqueeze(-1).numpy()),axis = 1), labels = y_true.unsqueeze(-1).numpy())
                auc = mm.accumulate()
                
            auc_scores.append(auc)

            print('Test Epoch:',self.train_epoch,', test for task:', task_id, ', AUC:', round(auc, 4))
            if self.args.save_logs:
                step_results['query_preds'].append(y_score.cpu().numpy())
                step_results['query_labels'].append(y_true.cpu().numpy())
                step_results['query_adj'].append(pred_eval['adj'].cpu().numpy())
                step_results['task_index'].append(self.test_tasks[task_id])

        mid_auc = np.median(auc_scores)
        avg_auc = np.mean(auc_scores)
        self.best_auc = max(self.best_auc,avg_auc)
        self.logger.append([self.train_epoch] + auc_scores  +[avg_auc, mid_auc,self.best_auc], verbose=False)

        print('Test Epoch:', self.train_epoch, ', AUC_Mid:', round(mid_auc, 4), ', AUC_Avg: ', round(avg_auc, 4),
              ', Best_Avg_AUC: ', round(self.best_auc, 4))
        
        if self.args.save_logs:
            self.res_logs.append(step_results)

        return self.best_auc

    def save_model(self):
        save_path = os.path.join(self.trial_path, f"step_{self.train_epoch}.pth")
        paddle.save(self.model.layers.state_dict(), save_path)
        print(f"Checkpoint saved in {save_path}")

    def save_result_log(self):
        joblib.dump(self.res_logs,self.args.trial_path+'/logs.pkl',compress=6)


    def conclude(self):
        df = self.logger.conclude()
        self.logger.close()
        print(df)