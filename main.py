from utils import *
from init_parameters import init_parameters
from data.load_data import *
import importlib
from time import time
# from torch.utils.tensorboard import SummaryWriter

def main(args):
    # writer = SummaryWriter(f'./results/runs/lamb_distill_{args.lamb_distill}_{args.seed}/metrics')
    unshuffled_data, taskcla, size = load_dataset(args)
    arch = importlib.import_module(f'models.{args.arch}')
    arch = arch.NET(size, taskcla, args)
    manager = importlib.import_module(f'methods.{args.method}')
    manager = manager.Manager(arch, args).to(args.device)

    results = pd.DataFrame([],columns=['stage','task','accuracy','micro-f1','macro-f1','seed'])
    index = np.arange(args.num_tasks)
    np.random.shuffle(index)
    args.index = index
    print(args.index)
    data = {}
    for idx, i in enumerate(args.index):
        data[idx] = unshuffled_data[i]
    
    train_dataloaders, test_dataloaders, val_dataloaders = data2dataloaders(data, args)
    t0 = time()
    for task in range(args.num_tasks):
        print('Train task:{}'.format(task))
        manager.train_with_eval(train_dataloaders[task], val_dataloaders[task], task)
        for previous in range(task+1):
            acc, mif1, maf1 = manager.evaluation(test_dataloaders[previous], previous)
            print('Stage:{} Task:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(task, previous, acc, mif1, maf1))
            # writer.add_scalar(f'{args.method}/{previous}/acc',acc,task)
            # writer.add_scalar(f'{args.method}/{previous}/mif1',mif1,task)
            # writer.add_scalar(f'{args.method}/{previous}/maf1',maf1,previous)
            results.loc[len(results.index)] = [task,previous,acc,mif1,maf1,args.seed]
        # writer.add_scalar(f'{args.method}/mean_acc',results[results['stage'] == task]['accuracy'].mean(),task)
    t1 = time()
    args.time = t1 - t0
    save_results(results,args)

def joint_training(args):
    args.class_incremental = False
    train_dataloader, test_dataloader, val_dataloader, taskcla, size = load_joint_data(args)
    arch = importlib.import_module(f'models.{args.arch}')
    arch = arch.NET(size, taskcla, args)
    manager = importlib.import_module(f'methods.Finetune')
    manager = manager.Manager(arch, args).to(args.device)

    results = pd.DataFrame([],columns=['stage','task','accuracy','micro-f1','macro-f1','seed'])
    index = np.arange(args.num_tasks)
    np.random.shuffle(index)
    args.index = index
    print(args.index)

    t0 = time()
    manager.train_with_eval(train_dataloader, val_dataloader, 0)
    acc, mif1, maf1 = manager.evaluation(test_dataloader, 0)
    print('Stage:{} Task:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(0, 0, acc, mif1, maf1))
    results.loc[len(results.index)] = [0,0,acc,mif1,maf1,args.seed]
    t1 = time()
    args.time = t1 - t0
    save_results(results,args)

if __name__ == '__main__':
    args = init_parameters()

    args.device = 'cuda:{}'.format(str(args.gpu_id)) if torch.cuda.is_available() else 'cpu'
    args.class_incremental = True
    if args.dataset == 'PMNIST':
        args.class_incremental = False
        args.num_tasks = 10
    elif args.dataset == 'CIFAR100':
        args.num_tasks = 10
    elif args.dataset == 'SplitMNIST':
        args.num_tasks = 5
    set_seed(args.seed)

    if args.methods == 'Joint':
        joint_training(args)
        return
    main(args)
