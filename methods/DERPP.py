import torch
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from copy import deepcopy
import torch.nn.functional as F

def reservoir(num_seen_examples, buffer_size):
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size: #total batch size
        return rand
    else:
        return -1
    
class buffer:
    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.attributes = ['features','labels','logits']

    def init_tensors(self,features,labels,logits):
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str == 'labels' else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))
    
    def is_empty(self):
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def store_data(self,features, labels, logits):
        if not hasattr(self, 'features'):
            self.init_tensors(features, labels, logits)
        
        for i in range(features.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.features[index] = features[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)

    def get_data(self, size):
        if size > min(self.num_seen_examples, self.features.shape[0]):
            size = min(self.num_seen_examples, self.features.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.features.shape[0]),
                                  size=size, replace=False)
        ret_tuple = (torch.stack([ee.cpu() for ee in self.features[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple
    
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
        self.alpha = 0.2
        self.beta = 0.5
        self.ce = torch.nn.CrossEntropyLoss()
        self.buffer_size = 5120
        self.buffer = buffer(self.buffer_size, args.device)
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



    def train_with_eval(self, train_dataloader, val_dataloader, task):
        self.train()
        lr = self.args.lr
        self.opt = torch.optim.SGD(self.parameters(),lr=lr, momentum=0.9, weight_decay=5e-4)
        # self.opt = torch.optim.Adam(self.parameters(),lr=0.001, weight_decay=5e-4)
        best_loss = np.inf
        best_model = deepcopy(self.state_dict())
        for epoch in trange(self.args.epochs, leave=False):
            for features, labels in train_dataloader:
                features, labels = features.to(self.args.device), labels.to(self.args.device)
                self.zero_grad()
                logits = self.forward(features, task)
                loss = self.ce(logits,labels)
                loss_reg = 0
                if not self.buffer.is_empty():
                    buf_features, buf_labels, buf_logits= self.buffer.get_data(self.buffer_size)
                    buf_output = self.forward(buf_features, task)
                    loss_reg = loss + self.alpha * F.mse_loss(buf_output, buf_logits)
                    loss_reg = loss + self.beta * F.cross_entropy(buf_output, buf_labels)
                loss = loss + loss_reg
                loss.backward(retain_graph=True)
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
                    self.opt = torch.optim.SGD(self.parameters(),lr=lr)
        self.load_state_dict(deepcopy(best_model))
        self.buffer.store_data(
            features=features,
            labels = labels,
            logits = logits,
        )


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