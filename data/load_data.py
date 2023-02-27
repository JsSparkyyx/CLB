import torch
from torchvision import datasets,transforms
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random

def load_dataset(args):
    if args.dataset == 'CIFAR100':
        from data import CIFAR100
        data,taskcla,size = CIFAR100.get(args)
    else:
        raise(args.dataset + ' not supported.')
    return data,taskcla,size


