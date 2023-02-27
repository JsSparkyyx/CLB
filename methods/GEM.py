import torch
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import copy
import quadprog

class Manager(torch.nn.Module):
    def __init__(self,
                 in_feat,
                 taskcla,
                 arch,
                 args,
                 lr = 0.005,
                 weight_decay = 0.001):
        super(Manager, self).__init__()
        self.arch = arch
        self.current_task = 0
        self.margin = args.gem_margin

        self.predict = torch.nn.ModuleList()
        for task, n_class, _ in taskcla:
            self.predict.append(torch.nn.Linear(in_feat,n_class))

        self.ce = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        self.mem_mask = []
        self.mem_ft = []
        self.mem_label = []
        self.mem_g = []
        self.observed_tasks = []
        self.grad_dims = []
        for param in self.arch.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), len(taskcla)).cuda()
    
    def forward(self, g, features, task, mini_batch = False):
        h = self.arch(g, features, mini_batch)
        logits = self.predict[task](h)

        return logits

    def train_with_eval(self, g, features, task, labels, train_mask, val_mask, args):
        self.train()
        
        self.observed_tasks.append(task)
        for epoch in trange(args.epochs, leave=False):
            for old_task_i in self.observed_tasks[:-1]:
                self.zero_grad()
                # fwd/bwd on the examples in the memory
                output = self.forward(self.mem_g[old_task_i], self.mem_ft[old_task_i], old_task_i)
                old_task_loss = self.ce(
                                    output[self.mem_mask[old_task_i]],
                                    self.mem_label[old_task_i][self.mem_mask[old_task_i]])
                old_task_loss.backward()
                store_grad(self.arch.parameters, self.grads, self.grad_dims,
                                old_task_i)

            self.zero_grad()
            logits = self.forward(g, features, task)
            loss = self.ce(logits[train_mask],labels[train_mask])
            loss.backward()

            if len(self.observed_tasks) > 1:
                # copy gradient
                store_grad(self.arch.parameters, self.grads, self.grad_dims, task)
                indx = torch.cuda.LongTensor(self.observed_tasks[:-1])
                dotp = torch.mm(self.grads[:, task].unsqueeze(0),
                                self.grads.index_select(1, indx))
                if (dotp < 0).sum() != 0:
                    project2cone2(self.grads[:, task].unsqueeze(1),
                                self.grads.index_select(1, indx), self.margin)
                    # copy gradients back
                    overwrite_grad(self.arch.parameters, self.grads[:, task],
                                self.grad_dims)
            self.opt.step()

            if epoch % 50 == 0:
                acc, mif1, maf1 = self.evaluation(g, features, task, labels, val_mask)
                print()
                print('Val, Epoch:{}, Loss:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(epoch, loss.item(), acc, mif1, maf1))

        self.current_task = task
        self.mem_mask.append(train_mask.data.clone())
        self.mem_ft.append(features.data.clone())
        self.mem_label.append(labels.data.clone())
        self.mem_g.append(copy.deepcopy(g).to('cuda:0'))

    def batch_train_with_eval(self, dataloader, g, features, task, labels, train_mask, val_mask, args):
        self.train()
        
        self.observed_tasks.append(task)
        for epoch in trange(args.epochs, leave=False):
            for seed_nodes, output_nodes, blocks in dataloader:
                for old_task_i in self.observed_tasks[:-1]:
                    self.zero_grad()
                    # fwd/bwd on the examples in the memory
                    output = self.forward(self.mem_g[old_task_i], self.mem_ft[old_task_i], old_task_i)
                    old_task_loss = self.ce(
                                        output[self.mem_mask[old_task_i]],
                                        self.mem_label[old_task_i][self.mem_mask[old_task_i]])
                    old_task_loss.backward()
                    store_grad(self.arch.parameters, self.grads, self.grad_dims,
                                    old_task_i)

                self.zero_grad()
                logits = self.forward(blocks, features[seed_nodes], task, mini_batch = True)
                loss = self.ce(logits,labels[output_nodes])
                loss.backward()

                if len(self.observed_tasks) > 1:
                    # copy gradient
                    store_grad(self.arch.parameters, self.grads, self.grad_dims, task)
                    indx = torch.cuda.LongTensor(self.observed_tasks[:-1])
                    dotp = torch.mm(self.grads[:, task].unsqueeze(0),
                                    self.grads.index_select(1, indx))
                    if (dotp < 0).sum() != 0:
                        project2cone2(self.grads[:, task].unsqueeze(1),
                                    self.grads.index_select(1, indx), self.margin)
                        # copy gradients back
                        overwrite_grad(self.arch.parameters, self.grads[:, task],
                                self.grad_dims)
                self.opt.step()

            if epoch % 50 == 0:
                acc, mif1, maf1 = self.evaluation(g, features, task, labels, val_mask)
                print()
                print('Val, Epoch:{}, Loss:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(epoch, loss.item(), acc, mif1, maf1))

        self.current_task = task
        self.mem_mask.append(train_mask.data.clone())
        self.mem_ft.append(features.data.clone())
        self.mem_label.append(labels.data.clone())
        self.mem_g.append(copy.deepcopy(g).to('cuda:0'))

    @torch.no_grad()
    def evaluation(self, g, features, task, labels, val_mask):
        self.eval()
        logits = self.forward(g, features, task)
        prob, prediction = torch.max(logits, dim=1)
        prediction = prediction[val_mask].cpu().numpy()
        labels = labels[val_mask].cpu().numpy()
        acc = accuracy_score(labels, prediction)
        mif1 = f1_score(labels, prediction, average='micro')
        maf1 = f1_score(labels, prediction, average='macro')
        return round(acc*100,2), round(mif1*100,2), round(maf1*100,2)

def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.detach().cpu().t().double().numpy()
    gradient_np = gradient.detach().cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))