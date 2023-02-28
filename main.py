from utils import *
from init_parameters import init_parameters
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from data.load_data import *
import importlib

# python main.py --method CR --arch GAT --dataset reddit --manner full_batch --seed 0 --epoch 3000 --lr 0.001 --weight_decay 5e-4
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
        print('Train task:{}'.format(task))
        manager.train_with_eval(train_dataloader, val_dataloader, task)
        for previous in range(task+1):
            acc, mif1, maf1 = manager.evaluation(test_dataloader, task)
            print('Stage:{} Task:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(task, previous, acc, mif1, maf1))
            # writer.add_scalar(f'{args.method}/{previous}/acc',acc,task)
            # writer.add_scalar(f'{args.method}/{previous}/mif1',mif1,task)
            # writer.add_scalar(f'{args.method}/{previous}/maf1',maf1,previous)
            results.loc[len(results.index)] = [task,previous,acc,mif1,maf1,args.seed]
        # writer.add_scalar(f'{args.method}/mean_acc',results[results['stage'] == task]['accuracy'].mean(),task)
    save_results(results,args)

if __name__ == '__main__':
    args = init_parameters()

    # torch.cuda.set_device(args.gpu_id)
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

    main(args)