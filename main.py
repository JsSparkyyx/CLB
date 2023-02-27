from utils import *
from init_parameters import init_parameters
from torch.utils.tensorboard import SummaryWriter
from data.load_data import *
import importlib

# python main.py --method CR --arch GAT --dataset reddit --manner full_batch --seed 0 --epoch 3000 --lr 0.001 --weight_decay 5e-4
def main(args):
    writer = SummaryWriter(f'./results/runs/lamb_distill_{args.lamb_distill}_{args.seed}/metrics')
    data, taskcla, size = load_dataset(args)
    arc = importlib.import_module(f'models.{args.arch}')
    arc = arc.NET(taskcla, args)
    manager = importlib.import_module(f'methods.{args.method}')
    manager = manager.Manager(args.gat_hidden*args.gat_head, taskcla, arc, args).to(args.device)

    results = pd.DataFrame([],columns=['stage','task','accuracy','micro-f1','macro-f1','seed'])

    for task in range(args.num_tasks):
        print('Train task:{}'.format(task))
        g, features, _, labels, train_mask, val_mask, test_mask = data[task].retrieve_data()
        manager.train_with_eval(g, features, task, labels, train_mask, val_mask, args)
        for previous in range(task+1):
            g, features, _, labels, train_mask, val_mask, test_mask = data[previous].retrieve_data()
            acc, mif1, maf1 = manager.evaluation(g, features, previous, labels, test_mask)
            print('Stage:{} Task:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(task, previous, acc, mif1, maf1))
            writer.add_scalar(f'{args.method}/{previous}/acc',acc,task)
            # writer.add_scalar(f'{args.method}/{previous}/mif1',mif1,task)
            # writer.add_scalar(f'{args.method}/{previous}/maf1',maf1,previous)
            results.loc[len(results.index)] = [task,previous,acc,mif1,maf1,args.seed]
        writer.add_scalar(f'{args.method}/mean_acc',results[results['stage'] == task]['accuracy'].mean(),task)
    save_results(results,args)

if __name__ == '__main__':
    args = init_parameters()

    # torch.cuda.set_device(args.gpu_id)
    args.device = 'cuda:{}'.format(str(args.gpu_id)) if torch.cuda.is_available() else 'cpu'
    args.fanouts = [int(i) for i in args.fanouts.split(',')]
    if args.dataset == 'cfd':
        args.class_incremental = False
    else:
        args.class_incremental = True
    set_seed(args.seed)

    main(args)