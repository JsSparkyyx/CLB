import torch
import random
import numpy as np
import pandas as pd
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
    path = os.path.join(args.save_path,args.dataset,'detail') + '/' + args.arch+'_'+args.method+'_'+str(args.num_tasks)+'_'+ args.manner +'_'+str(args.seed) + '.csv'
    results.to_csv(path,index=False)
    LA = 0
    FM = 0
    for task in range(args.num_tasks):
        LA += float(results[(results['stage'] == task) & (results['task'] == task)]['accuracy'])
        if task != args.num_tasks - 1:
            FM += results[results['stage'] == args.num_tasks - 1]['accuracy'].max() - float(results[(results['stage'] == args.num_tasks - 1) & (results['task'] == task)]['accuracy'])
    ACC = results[results['stage'] == args.num_tasks - 1]['accuracy'].mean()
    LA = LA/args.num_tasks
    FM = FM/(args.num_tasks - 1)
    path = os.path.join(args.save_path,args.dataset) + '/' + args.arch+'_'+args.method+'_'+str(args.num_tasks)+'_'+ args.manner +'_overall' + '.csv'
    if args.dataset == "cfd":
        index = ""
        for i in args.indexes:
            index += i + "->"
        index = index[:-2]
        print(index)
        with open(path, 'a') as f:
            f.write("{:.2f},{:.2f},{:.2f},{},{},{}\n".format(round(ACC,2),round(FM,2),round(LA,2),args.seed,index,args.lamb_distill))
        
    else:
        with open(path, 'a') as f:
            f.write("{:.2f},{:.2f},{:.2f},{}\n".format(round(ACC,2),round(FM,2),round(LA,2),args.seed))
    print("{:.2f},{:.2f},{:.2f},{}\n".format(round(ACC,2),round(FM,2),round(LA,2),args.seed))