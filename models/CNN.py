import sys
import torch
import torch.nn.functional as F
import numpy as np
# Alext net from
# https://github.com/joansj/hat/blob/master/src/networks/alexnet.py

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class Net(torch.nn.Module):

    def __init__(self,args,taskcla=None):
        super(Net,self).__init__()

        ncha = args.image_channel
        size = args.image_size

        self.taskcla=taskcla

        self.conv1=torch.nn.Conv2d(ncha,64,kernel_size=size//8)
        s=compute_conv_output_size(size,size//8)
        s=s//2
        self.conv2=torch.nn.Conv2d(64,128,kernel_size=size//10)
        s=compute_conv_output_size(s,size//10)
        s=s//2
        self.conv3=torch.nn.Conv2d(128,256,kernel_size=2)
        s=compute_conv_output_size(s,2)
        s=s//2
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()
        self.args = args


        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(256*s*s,2048)
        self.fc2=torch.nn.Linear(2048,2048)
        self.old_weight_norm = []

        if self.taskcla is not None:
            if 'dil' in args.scenario:
                self.last=torch.nn.Linear(2048,args.nclasses)
                self.merge_last=torch.nn.Linear(args.nclasses*2,args.nclasses)

            elif 'til' in args.scenario:
                self.last=torch.nn.ModuleList()
                self.merge_last=torch.nn.ModuleList()

                for t,n in self.taskcla:
                    self.last.append(torch.nn.Linear(2048,n))
                    self.merge_last.append(torch.nn.Linear(n*2,n))

        print('CNN')


        return

    def forward(self,x):
        h = self.features(x)

        if self.taskcla is None:
            if 'dil' in self.args.scenario:
                y = self.last(h)
            elif 'til' in self.args.scenario:
                y=[]
                for t,i in self.taskcla:
                    y.append(self.last[t](h))
            return y
        return h

    def features(self,x):
        h=self.maxpool(self.drop1(self.relu(self.conv1(x))))
        h=self.maxpool(self.drop1(self.relu(self.conv2(h))))
        h=self.maxpool(self.drop2(self.relu(self.conv3(h))))
        h=h.view(x.size(0),-1)
        h=self.drop2(self.relu(self.fc1(h)))
        h=self.drop2(self.relu(self.fc2(h)))
        return h