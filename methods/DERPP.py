import torch
import torch.nn.functional as F
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
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
        self.num_seen_samples = 0
        self.attributes = ['graph','features','labels','logits','masks']

    def init_tensors(self,
                        graph: torch.Tensor,
                        features: torch.Tensor,
                        labels: torch.Tensor,
                        logits: torch.Tensor,
                        masks: torch.Tensor):
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str == 'labels' else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))

    def store_data(self, g, features, labels, logits, masks):
        if not hasattr(self, 'examples'):
            self.init_tensors(g, features, labels, logits, masks)
        
        for i in range(features.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.features[index] = features[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if g is not None:
                    self.g[index] = g[i].to(self.device)
                if logits is not None:
                    self.masks[index] = masks[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)

    def get_data(self, size):
        if size > min(self.num_seen_examples, self.features.shape[0]):
            size = min(self.num_seen_examples, self.features.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple

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
        self.alpha = args.derpp_alpha
        self.beta = args.derpp_beta
        self.buffer_size = args.derpp_buffer_size
        self.buffer = buffer(self.buffer_size, args.device)
        self.class_incremental = args.class_incremental

        if self.class_incremental:
            self.predict = torch.nn.ModuleList()
            for task, n_class, _ in taskcla:
                self.predict.append(torch.nn.Linear(in_feat,n_class))
        else:
            self.predict = torch.nn.Linear(in_feat,taskcla)

        self.ce = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
    
    def forward(self, g, features, task, mini_batch = False):
        h = self.arch(g, features, mini_batch)
        if self.class_incremental:
            logits = self.predict[task](h)
        else:
            logits = self.predict(h)

        return logits

    def train_with_eval(self, g, features, task, labels, train_mask, val_mask, args):
        self.train()

        for epoch in trange(args.epochs, leave=False):
            self.zero_grad()
            logits = self.forward(g, features, task)
            loss = self.ce(logits[train_mask],labels[train_mask])
            loss_reg = 0
            if task != 0:
                buf_g, buf_features, buf_labels, buf_logits, buf_masks = self.buffer.get_data(self.buffer_size)
                buf_output = self.forward(buf_g, buf_features, task)

                loss_reg += self.alpha * F.mse_loss(buf_output, buf_logits)
                loss_reg += self.beta * F.cross_entropy(buf_output, buf_labels[buf_masks])

            loss = loss + loss_reg
            loss.backward()
            self.opt.step()

            if epoch % 50 == 0:
                acc, mif1, maf1 = self.evaluation(g, features, task, labels, val_mask)
                print()
                print('Val, Epoch:{}, Loss:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(epoch, loss.item(), acc, mif1, maf1))

        self.current_task = task
        self.buffer.store_data(
            graph=g,
            labels=labels,
            logits = logits.data ,
            masks=train_mask,
        )

    def batch_train_with_eval(self, dataloader, g, features, task, labels, train_mask, val_mask, args):
        self.train()

        for epoch in trange(args.epochs, leave=False):
            for seed_nodes, output_nodes, blocks in dataloader:
                self.zero_grad()
                logits = self.forward(blocks, features[seed_nodes], task, mini_batch = True)
                loss = self.ce(logits,labels[output_nodes])
                loss_reg = 0
                if task != 0:
                    buf_g, buf_features, buf_labels, buf_logits, buf_masks = self.buffer.get_data(self.buffer_size)
                    buf_output = self.forward(buf_g, buf_features, task)

                    loss_reg += self.alpha * F.mse_loss(buf_output, buf_logits)
                    loss_reg += self.beta * F.cross_entropy(buf_output, buf_labels[buf_masks])

                loss = loss + loss_reg
                loss.backward()
                self.opt.step()

            if epoch % 50 == 0:
                acc, mif1, maf1 = self.evaluation(g, features, task, labels, val_mask)
                print()
                print('Val, Epoch:{}, Loss:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(epoch, loss.item(), acc, mif1, maf1))

        self.current_task = task
        self.buffer.store_data(
            graph=g,
            labels=labels,
            logits = logits.data ,
            masks=train_mask,
        )

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