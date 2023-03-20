import argparse

def init_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['CIFAR100','SplitMNIST','PMNIST'], default='SplitMNIST')
    parser.add_argument('--arch', '--architecture', type=str, choices=['CNN'], default='CNN')
    parser.add_argument('--method', type=str, choices=['Finetune','Joint','CR','EWC','HAT','GEM','MAS'], default='Finetune')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_min', type=float, default=1e-5)
    parser.add_argument('--lr_patience', type=int, default=6)
    parser.add_argument('--lr_factor', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')

    args = parser.parse_args()
    return args