import torch
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from copy import deepcopy


class Manager(torch.nn.Module):
    def __init__(self,
                 arch,
                 args):
        super(Manager, self).__init__()
        self.arch = arch
        self.current_task = 0
        self.args = args
        self.class_incremental = self.args.class_incremental
        self.lr_patience = self.args.lr_patience
        self.lr_factor = self.args.lr_factor
        self.lr_min = self.args.lr_min
        self.ce = torch.nn.CrossEntropyLoss()
    
    def forward(self, features, task):
        logits = self.arch(features, task)
        return logits

    def train_with_eval(self, train_dataloader, val_dataloader, task):
        self.train()
        lr = self.args.lr
        self.opt = torch.optim.SGD(self.arch.parameters(),lr=lr)
        best_loss = np.inf
        best_model = deepcopy(self.arch.state_dict())

        for epoch in trange(self.args.epochs, leave=False):
            for features, labels in train_dataloader:
                features, labels = features.to(self.args.device), labels.to(self.args.device)
                self.zero_grad()
                logits = self.forward(features, task)
                loss = self.ce(logits,labels)
                loss.backward()
                self.opt.step()

            val_loss, acc, mif1, maf1 = self.evaluation(val_dataloader, task, valid = True)
            print()
            print('Val, Epoch:{}, Loss:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(epoch, val_loss, acc, mif1, maf1))
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(self.arch.state_dict())
                patience = self.lr_patience
            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    if lr < self.lr_min:
                        break
                    patience = self.lr_patience
                    self.opt = torch.optim.SGD(self.arch.parameters(),lr=lr)
        self.arch.load_state_dict(deepcopy(best_model))

    @torch.no_grad()
    def evaluation(self, test_dataloader, task, valid = False):
        self.arch.eval()
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
        acc = accuracy_score(labels, prediction)
        mif1 = f1_score(labels, prediction, average='micro')
        maf1 = f1_score(labels, prediction, average='macro')
        if valid:
            return total_loss, round(acc*100,2), round(mif1*100,2), round(maf1*100,2)
        return round(acc*100,2), round(mif1*100,2), round(maf1*100,2)