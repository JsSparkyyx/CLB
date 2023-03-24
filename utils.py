import torch
import random
import numpy as np
import pandas as pd
from torchvision import datasets,transforms
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def save_results(results,args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(os.path.join(args.save_path,args.dataset)):
        os.makedirs(os.path.join(args.save_path,args.dataset))
    if not os.path.exists(os.path.join(args.save_path,args.dataset,'detail')):
        os.makedirs(os.path.join(args.save_path,args.dataset,'detail'))
    path = os.path.join(args.save_path,args.dataset,'detail') + '/' + args.arch+'_'+args.method+'_'+str(args.num_tasks)+'_'+str(args.seed) + '.csv'
    results.to_csv(path,index=False)
    if args.method != 'Joint':
        LA = 0
        FM = 0
        for task in range(args.num_tasks):
            LA += float(results[(results['stage'] == task) & (results['task'] == task)]['accuracy'])
            if task != args.num_tasks - 1:
                FM += results[results['stage'] == args.num_tasks - 1]['accuracy'].max() - float(results[(results['stage'] == args.num_tasks - 1) & (results['task'] == task)]['accuracy'])
        ACC = results[results['stage'] == args.num_tasks - 1]['accuracy'].mean()
        LA = LA/args.num_tasks
        FM = FM/(args.num_tasks - 1)
        path = os.path.join(args.save_path,args.dataset) + '/' + args.arch+'_'+args.method+'_'+str(args.num_tasks)+'_overall' + '.csv'
        index = ""
        for i in args.index:
            index += str(i) + "->"
        index = index[:-2]
        print(index)
        with open(path, 'a') as f:
            f.write("{:.2f},{:.2f},{:.2f},{},{},{:.6f}\n".format(round(ACC,2),round(FM,2),round(LA,2),args.seed,index,args.time))
        print("{:.2f},{:.2f},{:.2f},{}\n".format(round(ACC,2),round(FM,2),round(LA,2),args.seed))

def data2dataloaders(data, args):
    train_dataloaders = []
    test_dataloaders = []
    val_dataloaders = []
    for task in range(args.num_tasks):
        train_samples = TensorDataset(
                        data[task]['train']['x'],
                        data[task]['train']['y'],
                        )
        test_samples = TensorDataset(
                    data[task]['test']['x'],
                    data[task]['test']['y'],
                    )
        val_samples = TensorDataset(
                    data[task]['valid']['x'],
                    data[task]['valid']['y'],
                    )
        train_dataloader = DataLoader(train_samples, sampler=RandomSampler(train_samples), batch_size=args.train_batch_size,pin_memory=True)
        val_dataloader = DataLoader(val_samples, sampler=SequentialSampler(val_samples), batch_size=args.test_batch_size,pin_memory=True)
        test_dataloader = DataLoader(test_samples, sampler=SequentialSampler(test_samples), batch_size=args.test_batch_size)
        train_dataloaders.append(train_dataloader)
        test_dataloaders.append(test_dataloader)
        val_dataloaders.append(val_dataloader)
    return train_dataloaders, test_dataloaders, val_dataloaders

def load_joint_data(args):
    from data.load_data import load_dataset
    if args.dataset == 'CIFAR100':
        taskcla = [(0,100)]
        size = [3,32,32]
        offset = 10
    elif args.dataset == 'PMNIST':
        taskcla = [(0,10)]
        size = [1, 28, 28]
        offset = 0
    elif args.dataset == 'SplitMNIST':
        taskcla = [(0,10)]
        size = [1, 28, 28]
        offset = 2
    else:
        raise(args.dataset + ' not supported.')
    unshuffled_data, _, size = load_dataset(args)
    data = {}
    data[0] = unshuffled_data[0]
    for task in range(1, args.num_tasks):
        data[0]['train']['x'] = torch.cat((data[0]['train']['x'],unshuffled_data[task]['train']['x']),0)
        data[0]['train']['y'] = torch.cat((data[0]['train']['y'],unshuffled_data[task]['train']['y'].add(task*offset)),0)
        data[0]['test']['x'] = torch.cat((data[0]['test']['x'],unshuffled_data[task]['test']['x']),0)
        data[0]['test']['y'] = torch.cat((data[0]['test']['y'],unshuffled_data[task]['test']['y'].add(task*offset)),0)
        data[0]['valid']['x'] = torch.cat((data[0]['valid']['x'],unshuffled_data[task]['valid']['x']),0)
        data[0]['valid']['y'] = torch.cat((data[0]['valid']['y'],unshuffled_data[task]['valid']['y'].add(task*offset)),0)
    train_dataloaders = []
    test_dataloaders = []
    val_dataloaders = []
    train_samples = TensorDataset(
                        data[0]['train']['x'],
                        data[0]['train']['y'],
                        )
    test_samples = TensorDataset(
                data[0]['test']['x'],
                data[0]['test']['y'],
                )
    val_samples = TensorDataset(
                data[0]['valid']['x'],
                data[0]['valid']['y'],
                )
    train_dataloader = DataLoader(train_samples, sampler=RandomSampler(train_samples), batch_size=args.train_batch_size,pin_memory=True)
    val_dataloader = DataLoader(val_samples, sampler=SequentialSampler(val_samples), batch_size=args.test_batch_size,pin_memory=True)
    test_dataloader = DataLoader(test_samples, sampler=SequentialSampler(test_samples), batch_size=args.test_batch_size)
    train_dataloaders.append(train_dataloader)
    test_dataloaders.append(test_dataloader)
    val_dataloaders.append(val_dataloader)
    return train_dataloaders, test_dataloaders, val_dataloaders, taskcla, size

# def load_joint_data(args):
#     data = {}
#     train_dataloaders = []
#     test_dataloaders = []
#     val_dataloaders = []
#     if args.dataset == 'CIFAR100':
#         mean=[x/255 for x in [125.3,123.0,113.9]]
#         std=[x/255 for x in [63.0,62.1,66.7]]
#         taskcla = [(0,100)]
#         size = [3,32,32]
#         offset = 10
#         train_samples = datasets.CIFAR100('./datasets/cf100/',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
#         test_samples = datasets.CIFAR100('./datasets/cf100/',train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
#         data={'train':{'x': [],'y': []}, 'valid':{'x': [],'y': []}}
#         loader = DataLoader(train_samples,batch_size=1,shuffle=False)
#         for image,target in loader:
#             n=target.numpy()[0]
#             data['train']['x'].append(image) # 255 
#             data['train']['y'].append(n)
#         data['train']['x']=torch.stack(data['train']['x']).view(-1,size[0],size[1],size[2])
#         data['train']['y']=torch.LongTensor(np.array(data['train']['y'],dtype=int)).view(-1)
#         r=np.arange(data['train']['x'].size(0))
#         r=np.array(shuffle(r,random_state=args.seed),dtype=int)
#         nvalid=int(0.1*len(r))
#         ivalid=torch.LongTensor(r[:nvalid])
#         itrain=torch.LongTensor(r[nvalid:])
#         data['valid']={}
#         data['valid']['x']=data['train']['x'][ivalid].clone()
#         data['valid']['y']=data['train']['y'][ivalid].clone()
#         data['train']['x']=data['train']['x'][itrain].clone()
#         data['train']['y']=data['train']['y'][itrain].clone()
#         train_samples = TensorDataset(
#                             data['train']['x'],
#                             data['train']['y'],
#                             )
#         val_samples = TensorDataset(
#                     data['valid']['x'],
#                     data['valid']['y'],
#                     )
#     elif args.dataset == 'PMNIST':
#         from data.load_data import load_dataset
#         taskcla = [(0,10)]
#         size = [1, 28, 28]
#         offset = 0
#         unshuffled_data, _, size = load_dataset(args)
#         data[0] = unshuffled_data[0]
#         for task in range(1, args.num_tasks):
#             data[0]['train']['x'] = torch.cat((data[0]['train']['x'],unshuffled_data[task]['train']['x']),0)
#             data[0]['train']['y'] = torch.cat((data[0]['train']['y'],unshuffled_data[task]['train']['y'].add(task*offset)),0)
#             data[0]['test']['x'] = torch.cat((data[0]['test']['x'],unshuffled_data[task]['test']['x']),0)
#             data[0]['test']['y'] = torch.cat((data[0]['test']['y'],unshuffled_data[task]['test']['y'].add(task*offset)),0)
#             data[0]['valid']['x'] = torch.cat((data[0]['valid']['x'],unshuffled_data[task]['valid']['x']),0)
#             data[0]['valid']['y'] = torch.cat((data[0]['valid']['y'],unshuffled_data[task]['valid']['y'].add(task*offset)),0)
#         train_samples = TensorDataset(
#                             data[0]['train']['x'],
#                             data[0]['train']['y'],
#                             )
#         test_samples = TensorDataset(
#                     data[0]['test']['x'],
#                     data[0]['test']['y'],
#                     )
#         val_samples = TensorDataset(
#                     data[0]['valid']['x'],
#                     data[0]['valid']['y'],
#                     )
#     elif args.dataset == 'SplitMNIST':
#         taskcla = [(0,10)]
#         size = [1, 28, 28]
#         offset = 2
#         train_samples = datasets.MNIST('./datasets/mnist/',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
#         test_samples = datasets.MNIST('./datasets/mnist/',train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
#         data={'train':{'x': [],'y': []}, 'valid':{'x': [],'y': []}}
#         loader = DataLoader(train_samples,batch_size=1,shuffle=False)
#         for image,target in loader:
#             n=target.numpy()[0]
#             data['train']['x'].append(image) # 255 
#             data['train']['y'].append(n)
#         data['train']['x']=torch.stack(data['train']['x']).view(-1,size[0],size[1],size[2])
#         data['train']['y']=torch.LongTensor(np.array(data['train']['y'],dtype=int)).view(-1)
#         r=np.arange(data['train']['x'].size(0))
#         r=np.array(shuffle(r,random_state=args.seed),dtype=int)
#         nvalid=int(0.1*len(r))
#         ivalid=torch.LongTensor(r[:nvalid])
#         itrain=torch.LongTensor(r[nvalid:])
#         data['valid']={}
#         data['valid']['x']=data['train']['x'][ivalid].clone()
#         data['valid']['y']=data['train']['y'][ivalid].clone()
#         data['train']['x']=data['train']['x'][itrain].clone()
#         data['train']['y']=data['train']['y'][itrain].clone()
#         train_samples = TensorDataset(
#                             data['train']['x'],
#                             data['train']['y'],
#                             )
#         val_samples = TensorDataset(
#                     data['valid']['x'],
#                     data['valid']['y'],
#                     )
#     else:
#         raise(args.dataset + ' not supported.')
#     train_dataloader = DataLoader(train_samples, sampler=RandomSampler(train_samples), batch_size=args.train_batch_size,pin_memory=True)
#     val_dataloader = DataLoader(val_samples, sampler=SequentialSampler(val_samples), batch_size=args.test_batch_size,pin_memory=True)
#     test_dataloader = DataLoader(test_samples, sampler=SequentialSampler(test_samples), batch_size=args.test_batch_size)  
#     train_dataloaders.append(train_dataloader)
#     test_dataloaders.append(test_dataloader)
#     val_dataloaders.append(val_dataloader)
#     return train_dataloaders, test_dataloaders, val_dataloaders, taskcla, size