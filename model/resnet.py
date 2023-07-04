import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons, make_circles
import numpy 









def rk4_solver(f, y0, t0, tf, h):
    # Runge-Kutta 4th Order Method [By Bottom Science]
    t = t0
    y = y0
    while t <= tf:
        k1 = h * f(y, y)
        k2 = h * f(y + k1/2, y + k1/2)
        k3 = h * f(y + k2/2, y + k2/2)
        k4 = h * f(y + k3, y + k3)
        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        t = t + h
    return y    




class resnet(nn.Module):
    def __init__(self, g, T, step):
        super(resnet, self).__init__()
        self.scheme=scheme
        self.g=g
        self.T=T
        self.p=p
        self.step=step
        #self.integ=integration()
        #self.reverse=integration_reverse()
        self.data=data
        self.finalLayer=nn.Linear(2,2)
        self.Identity=nn.Identity()
    def forward(self, x, t=10):
        #data_temp=self.data.clone() 
        #discretization = np.linspace(0, self.T, round((self.T)/(self.step)))    
        if self.scheme=='euler':  
            x=rk4_solver(self.g, x, 0, 0.1*t, self.step)
            x=self.Identity(x)
            logits=self.finalLayer(x)
            probas = torch.nn.functional.softmax(logits, dim=1)  
        return logits, probas
