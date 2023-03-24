import sys
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet18

class NET(torch.nn.Module):

    def __init__(self, shape, args):
        super(NET,self).__init__()
        ncha, size = shape[0], shape[1]
        self.resnet = resnet18(pretrained=True)
        print('ResNet')
        return

    def forward(self, x):
        h = self.resnet(x)
        return h