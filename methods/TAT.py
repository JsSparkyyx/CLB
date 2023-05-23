import torch
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from copy import deepcopy
from utils import WarmUpLR
from torch.utils.tensorboard import SummaryWriter
import time

writer = SummaryWriter(f'./results/runs/{time.ctime()}/loss')

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
        self.ce = torch.nn.CrossEntropyLoss()
        self.lamb = 20
        self.beta = 2
        # embed_size = self.embed_size
        # heads = self.heads
        shared_size = 10
        embed_size = 64
        heads = 16

        self.shared_size = shared_size
        self.Q = torch.nn.ParameterList([torch.nn.Parameter(torch.randn((heads,1000,embed_size))) for i in range(len(taskcla)+shared_size)])
        self.K = torch.nn.ParameterList([torch.nn.Parameter(torch.randn((heads,1000,embed_size))) for i in range(len(taskcla)+shared_size)])
        self.V = torch.nn.ParameterList([torch.nn.Parameter(torch.randn((heads,1000,embed_size))) for i in range(len(taskcla)+shared_size)])
        self.norm_factor = 1/embed_size
        self.dropout = torch.nn.Dropout(p=0.5)
        for i in range(len(taskcla)):
            torch.nn.init.xavier_normal_(self.Q[i])
            torch.nn.init.xavier_normal_(self.K[i])
            torch.nn.init.xavier_normal_(self.V[i])

        if self.class_incremental:
            self.predict = torch.nn.ModuleList()
            for task, n_class in taskcla:
                self.predict.append(torch.nn.Linear(embed_size,n_class))
        else:
            for task, n_class in taskcla:
                self.predict = torch.nn.Linear(embed_size,n_class)
                break
    
    def forward(self, features, task):
        h = self.arch(features)
        # batch_size input_size
        # task_size, head_size, input_size, emb_size = self.task_key.shape
        h = torch.nn.functional.normalize(h, p=2, dim=1)
        embeddings = []
        for i in range(task+1+self.shared_size):
            K = self.dropout(self.K[i])
            Q = self.dropout(self.Q[i])
            V = self.dropout(self.V[i])
            x_key = torch.einsum('bi,hie->bhe',h,K)
            x_query = torch.einsum('bi,hie->bhe',h,Q)
            x_vector = torch.einsum('bi,hie->bhe',h,V)
            attention = torch.nn.functional.softmax(torch.einsum('bhe,bhe->bh',x_key,x_query)*self.norm_factor,dim=-1)
            final_rep = (x_vector * attention.unsqueeze(-1).expand(-1,-1,x_vector.shape[-1])).sum((1))
            embeddings.append(final_rep)
        embeddings = torch.stack(embeddings,dim=1).sum((1))
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        if self.class_incremental:
            logits = self.predict[task](embeddings)
        else:
            logits = self.predict(embeddings)
        return logits

    def compute_l2_loss(self):
        loss = 0
        for (n1, p1), (n2, p2) in zip(self.named_parameters(), self.previous_model.named_parameters()):
            # if n1.startswith('arch'):
            #     l = (p1 - p2).pow(2)
            #     loss += l.sum()
            if n1.startswith('K.') or n1.startswith('Q.') or n1.startswith('V.'):
                if int(n1[2]) < self.shared_size:
                    l = (p1 - p2).pow(2)
                    loss += l.sum()
        return loss

    def train_with_eval(self, train_dataloader, val_dataloader, task):
        lr = self.args.lr
        # patience = self.lr_patience
        patience = 10
        # self.opt = torch.optim.SGD(self.parameters(),lr=lr)
        # self.opt = torch.optim.SGD(self.parameters(),lr=lr, momentum=0.9, weight_decay=5e-4)
        self.opt = torch.optim.Adam(self.parameters(),lr=0.0001, weight_decay=5e-4)
        # self.train_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, milestones=[60,120,160], gamma=0.2)
        # self.warmup_scheduler = WarmUpLR(self.opt, len(train_dataloader) * self.args.warm)
        best_loss = np.inf
        best_model = deepcopy(self.state_dict())

        for epoch in trange(self.args.epochs, leave=False):
            self.train()
            for step, (features, labels) in enumerate(train_dataloader):
                features, labels = features.to(self.args.device), labels.to(self.args.device)
                self.zero_grad()
                logits = self.forward(features, task)
                loss = self.ce(logits,labels)
                writer.add_scalar(f'{self.args.method}/{task}/ce_loss',loss,step+len(train_dataloader)*epoch)
                if task != 0:
                    loss += self.lamb*self.compute_l2_loss()
                    self.previous_model.eval()
                    rep = self.arch(features)
                    with torch.no_grad():
                        pre_rep = self.previous_model.arch(features)
                    loss += self.beta*torch.norm(rep-pre_rep)
                    writer.add_scalar(f'{self.args.method}/{task}/l2_loss',self.lamb*self.compute_l2_loss(),step+len(train_dataloader)*epoch)
                    writer.add_scalar(f'{self.args.method}/{task}/distill_loss',self.beta*torch.norm(rep-pre_rep),step+len(train_dataloader)*epoch)
                loss.backward()
                self.opt.step()
            val_loss, acc, mif1, maf1 = self.evaluation(val_dataloader, task, valid = True)
            print()
            print('Val, Epoch:{}, Loss:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(epoch, val_loss, acc, mif1, maf1))
            if val_loss < best_loss:    
                best_loss = val_loss
                best_model = deepcopy(self.state_dict())
                # patience = self.lr_patience
                patience = 10
            else:
                patience -= 1
                if patience <= 0:
                    break
                    # lr /= self.lr_factor
                    # if lr < self.lr_min:
                    #     break
                    # patience = self.lr_patience
                    # self.opt = torch.optim.SGD(self.parameters(),lr=lr, momentum=0.9, weight_decay=5e-4)
        self.K[task+self.shared_size].requires_grad = False
        self.Q[task+self.shared_size].requires_grad = False
        self.V[task+self.shared_size].requires_grad = False
        self.load_state_dict(deepcopy(best_model))
        self.previous_model = deepcopy(self)
        self.previous_model.requires_grad = False

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