import torch
import random
import numpy as np
import pandas as pd
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
    if args.method is not 'Joint':
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