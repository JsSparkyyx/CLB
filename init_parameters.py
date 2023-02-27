import argparse

def init_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['cora','amazon','reddit','cfd'], default='cfd')
    parser.add_argument('--arch', '--architecture', type=str, choices=['HTG','GAT','GCN','SAGE'], default='HTG')
    parser.add_argument('--method', type=str, choices=['CFD','Finetune','CR','EWC','HAT','GEM','MAS'], default='CFD')
    parser.add_argument('--num_tasks', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')


    args = parser.parse_args()
    return args