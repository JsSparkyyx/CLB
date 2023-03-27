import torch
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

class Manager(torch.nn.Module):
    def __init__(self,
                 in_feat,
                 taskcla,
                 arch,
                 args):
        super(Manager,self).__init__()
        self.arch = arch
        self.current_task = 0
        self.args = args
        self.class_incremental = self.args.class_incremental
        self.lr_patience = self.args.lr_patience
        self.lr_factor = self.args.lr_factor
        self.lr_min = self.args.lr_min
        self.lamb = 0.75
        self.smax = 400
        self.clipgrad = 10000
        self.mask_pre = None
        self.mask_back = None
        self.thres_cosh = 50
        self.thres_emb = 6
        n_hidden = 1000
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(in_feat,n_hidden)
        self.fc2=torch.nn.Linear(n_hidden,n_hidden)
        self.efc1=torch.nn.Embedding(len(taskcla),n_hidden)
        self.efc2=torch.nn.Embedding(len(taskcla),n_hidden)
        self.gate=torch.nn.Sigmoid()
        self.predict = torch.nn.ModuleList()
        for task, n_class, _ in taskcla:
            self.predict.append(torch.nn.Linear(n_hidden,n_class))
        self.ce=torch.nn.CrossEntropyLoss()
    
    def forward(self, g, features, task, s, mini_batch = False):
        h = self.arch(g, features, mini_batch)
        masks = self.mask(task,s=s)
        gfc1, gfc2 = masks
        h = self.get_feature(h,gfc1,gfc2)
        logits = self.predict[task](h)

        return logits, masks
    
    def get_feature(self,x,gfc1,gfc2):
        h=self.drop1(x.view(x.size(0),-1))
        h=self.drop2(self.relu(self.fc1(h)))
        h=h*gfc1.expand_as(h)
        h=self.drop2(self.relu(self.fc2(h)))
        h=h*gfc2.expand_as(h)
        return h

    def mask(self,task,s=1):
        gfc1=self.gate(s*self.efc1(torch.LongTensor([task]).cuda()))
        gfc2=self.gate(s*self.efc2(torch.LongTensor([task]).cuda()))
        return [gfc1,gfc2]

    def get_view_for(self,n,masks):

        gfc1,gfc2=masks

        if n=='fc1.weight':
            return gfc1.data.view(-1,1).expand_as(self.fc1.weight)
        elif n=='fc1.bias':
            return gfc1.data.view(-1)
        elif n=='fc2.weight':
            post=gfc2.data.view(-1,1).expand_as(self.fc2.weight)
            pre=gfc1.data.view(1,-1).expand_as(self.fc2.weight)
            return torch.min(post,pre)
        elif n=='fc2.bias':
            return gfc2.data.view(-1)

        return None
    
    def train_with_eval(self, dataloader, val_dataloader,g, features, task, labels, train_mask, val_mask, args):
        self.train()
        lr = self.args.lr
        self.opt = torch.optim.SGD(self.arch.parameters(),lr=lr)
        best_loss = np.inf
        best_model = deepcopy(self.arch.state_dict())
        for epoch in trange(args.epochs, leave=False):
            for step, (seed_nodes, output_nodes, blocks) in enumerate(dataloader):
                s=(self.smax-1/self.smax)*step/(g.num_nodes()//args.batch_size)+1/self.smax
                self.zero_grad()
                logits, masks = self.forward(blocks, features[seed_nodes], task, s, mini_batch = True)
                loss = self.ce(logits,labels[output_nodes])
                reg=0
                count=0
                if self.mask_pre is not None:
                    for m,mp in zip(masks,self.mask_pre):
                        aux=1-mp
                        reg+=(m*aux).sum()
                        count+=aux.sum()
                else:
                    for m in masks:
                        reg+=m.sum()
                        count+=np.prod(m.size()).item()
                reg/=count
                loss = loss + self.lamb*reg
                loss.backward()

                # Restrict layer gradients in backprop
                if task>0:
                    for n,p in self.named_parameters():
                        if n in self.mask_back:
                            p.grad.data*=self.mask_back[n]

                # Compensate embedding gradients
                for n,p in self.named_parameters():
                    if n.startswith('e'):
                        num=torch.cosh(torch.clamp(s*p.data,-self.thres_cosh,self.thres_cosh))+1
                        den=torch.cosh(p.data)+1
                        p.grad.data*=self.smax/s*num/den

                # Apply step
                torch.nn.utils.clip_grad_norm_(self.parameters(),self.clipgrad)
                self.opt.step()

                # Constrain embeddings
                for n,p in self.named_parameters():
                    if n.startswith('e'):
                        p.data=torch.clamp(p.data,-self.thres_emb,self.thres_emb)

            val_loss, acc, mif1, maf1 = self.evaluation(g,val_dataloader, task, valid = True)
            print()
            print('Val, Epoch:{}, Loss:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(epoch, loss.item(), acc, mif1, maf1))
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
        # Activations mask
        mask=self.mask(task,s=self.smax)
        for i in range(len(mask)):
            mask[i]=torch.autograd.Variable(mask[i].data.clone(),requires_grad=False)
        if task==0:
            self.mask_pre=mask
        else:
            for i in range(len(self.mask_pre)):
                self.mask_pre[i]=torch.max(self.mask_pre[i],mask[i])

        # Weights mask
        self.mask_back={}
        for n,_ in self.named_parameters():
            vals=self.get_view_for(n,self.mask_pre)
            if vals is not None:
                self.mask_back[n]=1-vals

    @torch.no_grad()
    def evaluation(self, g, test_dataloader, task, valid = False):
        self.arch.eval()
        total_prediction = np.array([])
        total_labels = np.array([])
        total_loss = 0
        for features, labels in test_dataloader:
            logits, masks = self.forward(g, features, task, self.smax)
            prob, prediction = torch.max(logits, dim=1)
            prediction = prediction[val_mask].cpu().numpy()
            labels = labels[val_mask].cpu().numpy()
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
