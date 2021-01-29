import torch
from torch import nn, optim
import torch.nn.functional as F
import pdb
import json
import argparse

with open("clpConfig.json") as f:
    data = f.read()
config = json.loads(data)

drp=config['dnn']['dropout']


class clpdnn(nn.Module):
    def __init__(self,M):
        super().__init__()
        self.fc1=nn.Linear(M,M//2)
        self.fc2=nn.Linear(M//2,M//2)
        self.fc3=nn.Linear(M//2,M//2)
        self.fc4=nn.Linear(M//2,2)
        self.drop=nn.Dropout(p=drp)
                
    def forward(self, x):
        x=F.leaky_relu(self.fc1(x))
        x=self.drop(x)
        x=F.leaky_relu(self.fc2(x))
#         x=self.drop(x)
#         x=F.leaky_relu(self.fc3(x))
        x=F.softmax(self.fc4(x))
        return x