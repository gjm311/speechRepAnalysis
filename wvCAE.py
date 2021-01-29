import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import pdb



class CAEenc(nn.Module):
    def __init__(self, dim=256, nc=1):
        super().__init__()
        self.conv1=nn.Conv2d(nc, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool=nn.MaxPool2d((2,2))
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2=nn.Conv2d(8,16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)        
        self.conv3=nn.Conv2d(16,32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4=nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        self.linear = nn.Linear(64*4*16, dim)

    def forward(self, x):

        x =F.leaky_relu((self.bn1(self.pool(self.conv1(x)))))
        x =F.leaky_relu((self.bn2(self.pool(self.conv2(x)))))
        x =F.leaky_relu((self.bn3(self.pool(self.conv3(x)))))
        x =F.leaky_relu((self.bn4(self.pool(self.conv4(x)))))
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

class CAEdec(nn.Module):
    def __init__(self, dim=256, nc=1):
        super().__init__()
        self.conv1=nn.ConvTranspose2d(64,32, kernel_size=3, stride=1, padding=(1,0), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2=nn.ConvTranspose2d(32,16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3=nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(8)
        self.conv4=nn.ConvTranspose2d(8, nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.linear = nn.Linear(dim,64*4*16)

    def forward(self, x):
            
        x = self.linear(x)
        x = x.view(x.size(0), 64,4,16)
        x = F.interpolate(x, scale_factor=2)
        x =F.leaky_relu((self.bn1(self.conv1(x))))
        x = F.interpolate(x, scale_factor=2)
        x =F.leaky_relu((self.bn2(self.conv2(x))))
        x = F.interpolate(x, scale_factor=2)
        x =F.leaky_relu((self.bn3(self.conv3(x))))
        x = F.interpolate(x, scale_factor=2)
        x =F.sigmoid((self.conv4(x)))

        return x[:,:,:,:256]


class wvCAEn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = CAEenc(dim=dim)
        self.decoder = CAEdec(dim=dim)

    def forward(self, x):
        bottleneck = self.encoder(x)
        x = self.decoder(bottleneck)
        return x, bottleneck


