import torch
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from copy import deepcopy

class Manager(torch.nn.Module):
    def __init__(self,
                 arch,
                 taskcla,
                 args):
        super(Manager, self).__init__()
        self.arch = arch
        self.current_task = 0
        self.args = args
        self.class_incremental = self.args.class_incremental
        self.lr_patience = self.args.lr_patience
        self.lr_factor = self.args.lr_factor
        self.lr_min = self.args.lr_min
        self.fisher = {}
        self.params = {}
        # self.lamb = 0.75
        self.lamb = 500
        self.ce = torch.nn.CrossEntropyLoss()

        if self.class_incremental:
            self.predict = torch.nn.ModuleList()
            for task, n_class in taskcla:
                self.predict.append(torch.nn.Linear(1000,n_class))
        else:
            for task, n_class in taskcla:
                self.predict = torch.nn.Linear(1000,n_class)
                break
    
    def forward(self, features, task):
        h = self.arch(features)
        if self.class_incremental:
            logits = self.predict[task](h)
        else:
            logits = self.predict(h)
        return logits

    def calculate_fisher(self, train_dataloader, task):
        self.train()

        fisher = {}
        params = {}
        for n,p in self.named_parameters():
            fisher[n] = 0*p.data
            params[n] = 0*p.data

        for features, labels in train_dataloader:
            features, labels = features.to(self.args.device), labels.to(self.args.device)
            self.zero_grad()
            logits = self.forward(features, task)
            loss = self.ce(logits,labels)
            loss.backward()

            for n,p in self.named_parameters():
                if p.grad is not None:
                    pg = p.grad.data.clone().pow(2)
                    fisher[n] += pg

        for n,p in self.named_parameters():
            if p.grad is not None:
                pd = p.data.clone()
                params[n] = pd

        self.zero_grad()
        return fisher, params

    def train_with_eval(self, train_dataloader, val_dataloader, task):
        lr = self.args.lr
        patience = self.lr_patience
        self.opt = torch.optim.SGD(self.parameters(),lr=lr, momentum=0.9, weight_decay=5e-4)
        # self.opt = torch.optim.SGD(self.parameters(),lr=lr)
        best_loss = np.inf
        best_model = deepcopy(self.state_dict())

        for epoch in trange(self.args.epochs, leave=False):
            self.train()
            for features, labels in train_dataloader:
                features, labels = features.to(self.args.device), labels.to(self.args.device)
                self.zero_grad()
                logits = self.forward(features, task)
                loss = self.ce(logits,labels)
                if task != 0:
                    loss_ewc = 0
                    for t in range(task):
                        for n, p in self.named_parameters():
                            l = self.fisher[t][n]
                            l = l * (p - self.params[t][n]).pow(2)
                            loss_ewc += l.sum() 
                    loss = loss + self.lamb*loss_ewc
                loss.backward()
                self.opt.step()

            val_loss, acc, mif1, maf1 = self.evaluation(val_dataloader, task, valid = True)
            print()
            print('Val, Epoch:{}, Loss:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(epoch, val_loss, acc, mif1, maf1))
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(self.state_dict())
                patience = self.lr_patience
            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    if lr < self.lr_min:
                        break
                    patience = self.lr_patience
                    self.opt = torch.optim.SGD(self.parameters(),lr=lr, momentum=0.9, weight_decay=5e-4)
                    # self.opt = torch.optim.SGD(self.parameters(),lr=lr)
        self.load_state_dict(deepcopy(best_model))
        fisher, params = self.calculate_fisher(train_dataloader, task)
        self.current_task = task
        self.fisher[self.current_task] = fisher
        self.params[self.current_task] = params

    @torch.no_grad()
    def evaluation(self, test_dataloader, task, valid = False):
        self.eval()
        total_prediction = np.array([])
        total_labels = np.array([])
        total_loss = 0
        for features, labels in test_dataloader:
            features, labels = features.to(self.args.device), labels.to(self.args.device)
            logits = self.forward(features, task)
            loss = self.ce(logits,labels)
            prob, prediction = torch.max(logits, dim=1)
            total_loss = total_loss + loss.cpu().item() 
            total_labels = np.concatenate((total_labels, labels.cpu().numpy()), axis=0)
            total_prediction = np.concatenate((total_prediction, prediction.cpu().numpy()), axis=0)
        acc = accuracy_score(total_labels, total_prediction)
        mif1 = f1_score(total_labels, total_prediction, average='micro')
        maf1 = f1_score(total_labels, total_prediction, average='macro')
        if valid:
            return total_loss, round(acc*100,2), round(mif1*100,2), round(maf1*100,2)
        return round(acc*100,2), round(mif1*100,2), round(maf1*100,2)