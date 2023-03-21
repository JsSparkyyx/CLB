import torch
from torchvision import datasets,transforms
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random

def load_dataset(args):
    if args.dataset == 'CIFAR100':
        from data import CIFAR100 as dataset
    elif args.dataset == 'PMNIST':
        from data import PMNIST as dataset
    elif args.dataset == 'SplitMNIST':
        from data import SplitMNIST as dataset
    else:
        raise(args.dataset + ' not supported.')
    data,taskcla,size = dataset.get(args)
    return data,taskcla,size

def load_joint_data(args):
    if args.dataset == 'CIFAR100':
        taskcla = [(0,100)]
        size=[3,32,32]
    elif args.dataset == 'PMNIST':
        taskcla = [(0,10)]
        size = [1, 28, 28]
    elif args.dataset == 'SplitMNIST':
        taskcla = [(0,10)]
        size = [1, 28, 28]
    else:
        raise(args.dataset + ' not supported.')
    unshuffled_data, taskcla, size = load_dataset(args)
    data = {}
    data[0] = unshuffled_data[0]
    for task in range(1, args.num_tasks):
        data[0]['train']['x'] = torch.cat((data[0]['train']['x'],unshuffled_data[task]['train']['x']),0)
        data[0]['train']['y'] = torch.cat((data[0]['train']['y'],unshuffled_data[task]['train']['y']),0)
        data[0]['test']['x'] = torch.cat((data[0]['test']['x'],unshuffled_data[task]['test']['x']),0)
        data[0]['test']['y'] = torch.cat((data[0]['test']['y'],unshuffled_data[task]['test']['y']),0)
        data[0]['valid']['x'] = torch.cat((data[0]['valid']['x'],unshuffled_data[task]['valid']['x']),0)
        data[0]['valid']['y'] = torch.cat((data[0]['valid']['y'],unshuffled_data[task]['valid']['y']),0)
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


