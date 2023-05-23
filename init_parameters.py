import argparse

def init_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['CIFAR100','SplitMNIST','PMNIST','CIFAR10'], default='SplitMNIST')
    parser.add_argument('--arch', '--architecture', type=str, choices=['CNN','ResNet'], default='CNN')
    parser.add_argument('--method', type=str, choices=['Finetune','Joint','CR','EWC','HAT','GEM','MAS','DERPP','TAT'], default='Finetune')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=512) 
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--lr_min', type=float, default=1e-5)
    parser.add_argument('--lr_patience', type=int, default=6)
    parser.add_argument('--lr_factor', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--warm', type=int, default=1)

    args = parser.parse_args()
    return args

# python main.py --method HAT --dataset SplitMNIST --lr 0.05 --lr_factor 3 --lr_min 1e-4 --lr_patience 5 0