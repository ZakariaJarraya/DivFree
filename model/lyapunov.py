import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons, make_circles
import numpy 








class lyapunov(nn.Module):

    def __init__(self, c1=0., c2=3., d=50, scale=1.):
        super(lyapunov, self).__init__()
        #self.c1=c1
        #self.c2=c2
        self.scale=nn.Parameter(torch.tensor(scale))
        self.c1 = nn.Parameter(0*torch.ones(1, d))
        self.c2 = nn.Parameter(3*torch.ones(1, d))
        self.d=d
        #self.c3 = nn.Parameter(6*torch.ones(1, 2))
        #self.c4 = nn.Parameter(9*torch.ones(1, 2))
    def forward(self, x):
        
        #x=torch.sum(torch.pow(x, 2),dim=1)

        
        return -torch.exp(-torch.sqrt(torch.sum(torch.pow(x-self.c1, 2), dim=-1))/(self.scale*math.sqrt(self.d)))-torch.exp(-torch.sqrt(torch.sum(torch.pow(x-self.c2, 2), dim=-1))/(self.scale*math.sqrt(self.d)))#-torch.exp(-torch.sum(torch.pow(x-self.c3, 2)/self.scale,dim=-1))-torch.exp(-torch.sum(torch.pow(x-self.c4, 2)/self.scale,dim=-1))


        



class dynamics_init(nn.Module):

    def __init__(self, c1=0., c2=3., d=50, scale=1.):
        super(dynamics_init, self).__init__()
        #self.c1=c1
        #self.c2=c2
        self.scale1=nn.Parameter(torch.tensor(scale))
        self.scale2=nn.Parameter(torch.tensor(scale))
        self.c1 = nn.Parameter(0*torch.ones(1, d))
        self.c2 = nn.Parameter(3*torch.ones(1, d))
        self.d=d
    def forward(self, x):
        
        return torch.transpose(torch.transpose(2*(x-self.c1)/self.scale1*math.sqrt(self.d), 0, 1)*(-torch.exp(-torch.sum(torch.pow((x-self.c1)/self.scale1*math.sqrt(self.d), 2),dim=-1))), 0,1)+torch.transpose(torch.transpose(2*((x-self.c2)/self.scale2*math.sqrt(self.d)), 0, 1)*(-torch.exp(-torch.sum(torch.pow((x-self.c2)/self.scale2*math.sqrt(self.d), 2),dim=-1))), 0,1)#+torch.transpose(torch.transpose(2*((x-self.c3)/1.5), 0, 1)*(-torch.exp(-torch.sum(torch.pow((x-self.c3)/1.5, 2),dim=-1))), 0,1)
        #return torch.transpose(torch.transpose(2*(x-self.c1)/2., 0, 1)*(-torch.exp(-torch.sum(torch.pow((x-self.c1)/2., 2),dim=-1))), 0,1)+torch.transpose(torch.transpose(2*((x-self.c2)/2.), 0, 1)*(-torch.exp(-torch.sum(torch.pow((x-self.c2)/2., 2),dim=-1))), 0,1)
  

class grad(nn.Module):
    def __init__(self, f, d):
        super(grad, self).__init__()
        self.f=f
        self.d=d

    def forward(self,x_batch):
        x_batch=torch.reshape(x_batch,(-1,self.d))
        l=[Variable(x_batch[i], requires_grad= True) for i in range(len(x_batch))]
            
        #f=[f(torch.stack([i])) for i in l]
        f=[self.f(i) for i in l]
        
        return -torch.reshape(torch.cat([torch.autograd.grad(f[i], l[i], create_graph=True)[0] for i in range(len(l))]),(len(x_batch),self.d))





class V(nn.Module):

    def __init__(self, dim_in, dim_hid, dim_out):
        super(V, self).__init__()
        self.layer1 = nn.Linear(dim_in, dim_hid)
        #self.res=nn.Linear(dim_hid, dim_hid)
        self.activation=nn.ReLU()
        self.layer2 = nn.Linear(dim_hid, dim_hid)
        self.layer3 = nn.Linear(dim_hid, dim_hid)
        self.layer4 = nn.Linear(dim_hid, dim_out)
    def forward(self, x):
        x=self.layer1(x)
        x=self.activation(x)
        x=self.layer2(x)
        x=self.activation(x)
        x=self.layer3(x)
        x=self.activation(x)
        
        return self.layer4(x)   





