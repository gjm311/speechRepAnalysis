import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np



class CAEenc(nn.Module):
    def __init__(self, dim=256, nc=1):
        super().__init__()
        self.conv1=nn.Conv2d(nc, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool=nn.MaxPool2d((2, 2))
        self.conv2=nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3=nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4=nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        self.linear = nn.Linear(128*8*7, dim)

    def forward(self, bb_x, nb_x):
        bb_x =F.leaky_relu((self.bn1(self.pool(self.conv1(bb_x)))))
        bb_x =F.leaky_relu((self.bn2(self.pool(self.conv2(bb_x)))))
        bb_x =F.leaky_relu((self.bn3(self.pool(self.conv3(bb_x)))))
        nb_x =F.leaky_relu((self.bn1(self.pool(self.conv1(nb_x)))))
        nb_x =F.leaky_relu((self.bn2(self.pool(self.conv2(nb_x)))))
        nb_x =F.leaky_relu((self.bn3(self.pool(self.conv3(nb_x)))))
        
        x=torch.cat((bb_x,nb_x),dim=0)
        x =F.leaky_relu((self.bn4(self.pool(self.conv4(x)))))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class CAEdec(nn.Module):
    def __init__(self, dim=256, nc=1):
        super().__init__()
        self.conv1=nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=(1,0), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2=nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3=nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4=nn.ConvTranspose2d(16, nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.linear = nn.Linear(dim,128*8*7)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), 128, 8, 7)
        x = F.interpolate(x, scale_factor=2)
        
        bb_x=x[0:x.size(0)//2,:,:,:]
        nb_x=x[x.size(0)//2:,:,:,:]
        
        bb_x =F.leaky_relu((self.bn1(self.conv1(bb_x))))
        bb_x = F.interpolate(bb_x, scale_factor=2)
        bb_x =F.leaky_relu((self.bn2(self.conv2(bb_x))))
        bb_x = F.interpolate(bb_x, scale_factor=2)
        bb_x =F.leaky_relu((self.bn3(self.conv3(bb_x))))
        bb_x = F.interpolate(bb_x, scale_factor=2)
        bb_x =F.sigmoid((self.conv4(bb_x)))
        nb_x =F.leaky_relu((self.bn1(self.conv1(nb_x))))
        nb_x = F.interpolate(nb_x, scale_factor=2)
        nb_x =F.leaky_relu((self.bn2(self.conv2(nb_x))))
        nb_x = F.interpolate(nb_x, scale_factor=2)
        nb_x =F.leaky_relu((self.bn3(self.conv3(nb_x))))
        nb_x = F.interpolate(nb_x, scale_factor=2)
        nb_x =F.sigmoid((self.conv4(nb_x)))
        return bb_x[:,:,:,0:-2],nb_x[:,:,:,0:-2]


class mcCAEn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = CAEenc(dim=dim)
        self.decoder = CAEdec(dim=dim)
        
    def forward(self, bb_x, nb_x, volta=0):
        bottleneck = self.encoder(bb_x,nb_x)
        bb_x,nb_x = self.decoder(bottleneck)
        if volta==0:
            return bb_x,nb_x
        else:
            return bb_x,nb_x,bottleneck

