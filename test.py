from torchvision.models import resnet18
from torchvision import datasets,transforms
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from utils import *
from init_parameters import init_parameters
from data.load_data import *
import importlib
from time import time

cf100_dir = './datasets/cf100/'
file_dir = './datasets/cf100/binary_cifar100'

args = init_parameters()

mean=[x/255 for x in [125.3,123.0,113.9]]
std=[x/255 for x in [63.0,62.1,66.7]]
size=[3,32,32]

train = datasets.CIFAR100(cf100_dir,train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
test = datasets.CIFAR100(cf100_dir,train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

train_dataloader = DataLoader(train, sampler=RandomSampler(train), batch_size=args.train_batch_size,pin_memory=True)
test_dataloader = DataLoader(test, sampler=SequentialSampler(test), batch_size=args.test_batch_size)
taskcla = [(0,100)]
args.class_incremental = False
args.device = 'cuda:{}'.format(str(args.gpu_id)) if torch.cuda.is_available() else 'cpu'
arch = importlib.import_module(f'models.ResNet')
arch = arch.NET(size, args)
manager = importlib.import_module(f'methods.Finetune')
manager = manager.Manager(arch, taskcla, args).to(args.device)

manager.train_with_eval(train_dataloader, test_dataloader, 0)

acc, mif1, maf1 = manager.evaluation(test_dataloader, 0)
print('Stage:{} Task:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(0, previous, acc, mif1, maf1))