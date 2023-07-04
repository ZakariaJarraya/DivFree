import torch.nn as nn
import math
from platform import java_ver
import torch.nn.functional as F
import matplotlib.animation as animation
from math import *
import torch
import matplotlib.pyplot as plt

import seaborn as sns
import torch.nn as nn
import numpy
from model import resnet as rs

import numpy as np
from trainer import resnet_trainer

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import resnet

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons, make_circles
import numpy 
from torchdiffeq import odeint



def div(module,x):
    return torch.autograd.functional.jacobian(module,x,create_graph=True).sum(dim=2).view(len(x),2,2).diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=1)









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



def rk4_solver(f, y0, t0, tf, h):
    # Runge-Kutta 4th Order Method [By Bottom Science]
    t = t0
    y = y0
    while t <= tf:
        k1 = h * f(y)
        k2 = h * f(y + k1/2)
        k3 = h * f(y + k2/2)
        k4 = h * f(y + k3)
        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        t = t + h
    return y    



class ResNet(nn.Module):
    def __init__(self, g, T, step):
        super(ResNet, self).__init__()
        if type(g)==tuple:
            self.g=g[0]
            self.par0=torch.nn.Parameter(g[1][0],requires_grad=True)
            self.par1=torch.nn.Parameter(g[1][1],requires_grad=True)
            self.par2=torch.nn.Parameter(g[1][2],requires_grad=True)
            self.par3=torch.nn.Parameter(g[1][3],requires_grad=True)
            self.par4=torch.nn.Parameter(g[1][4],requires_grad=True)
            self.par5=torch.nn.Parameter(g[1][5],requires_grad=True)
        else:
            self.g=g    
        self.T=T
        self.step=step
        self.finalLayer=nn.Linear(2,2)
        self.Identity=nn.Identity()
    def forward(self, x, t=10):
        x=rk4_solver(self.g, x, 0, 0.1*t, self.step)
        x=self.Identity(x)
        logits=self.finalLayer(x)
        probas = torch.nn.functional.softmax(logits, dim=1)  
        return logits, probas



def train_epoch(model, Y, X, criterion, opt, batch_size=100):
    model.train()
    #Y=dynamic_i(X)
    losses = []
    for beg_i in range(0, X.size(0), batch_size):
        x_batch = Variable(X[beg_i:beg_i + batch_size, :])
        y_batch = Y[beg_i:beg_i + batch_size, :]
        #x_batch = Variable(x_batch)
        #y_batch = Variable(y_batch)
        opt.zero_grad()
        # (1) Forward
        x_batch.requires_grad=True
        y_hat = model(x_batch)
        logits, probas = model(x_batch)
        loss=criterion(logits, y_batch.view(logits.size()[0]).long())
        # (2) Compute diff
        # (3) Compute gradients
        loss.backward(retain_graph=True)
        # (4) update weights
        opt.step()        
        #losses.append(loss.data.numpy())
    return losses

"""



def train_epoch(model, Y, X, criterion, opt, alpha, batch_size=100):
    model.train()
    #Y=dynamic_i(X)
    losses = []
    for beg_i in range(0, X.size(0), batch_size):
        x_batch = Variable(X[beg_i:beg_i + batch_size, :])
        y_batch = Y[beg_i:beg_i + batch_size, :]
        #x_batch = Variable(x_batch)
        #y_batch = Variable(y_batch)
        opt.zero_grad()
        # (1) Forward
        x_batch.requires_grad=True
        y_hat = model(x_batch)
        logits, probas = model(x_batch)
        penal = nn.MSELoss()
        loss=criterion(logits, y_batch.view(logits.size()[0]).long()) + alpha * penal(div(model.g,x_batch), 0*torch.ones(len(x_batch)).to('cuda'))
        # (2) Compute diff
        # (3) Compute gradients
        loss.backward(retain_graph=True)
        # (4) update weights
        opt.step()        
        #losses.append(loss.data.numpy())
    return losses






def train(model,X,Y,alpha,num_epochs=100):
    opt = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    e_losses = []
    for e in range(num_epochs):
        e_losses += train_epoch(model, Y, X, criterion, opt, alpha)
    #plt.plot(e_losses)          

"""

def train(model,X,Y,num_epochs=100):
    opt = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    e_losses = []
    for e in range(num_epochs):
        e_losses += train_epoch(model, Y, X, criterion, opt)
    #plt.plot(e_losses)       
    








def determinant(image,model,x):
    y=[]
    for i in x:
        y.append(torch.det(torch.autograd.functional.jacobian(image,i)))
    return torch.stack(y).view(-1,1)    


class aggreg(nn.Module):
    def __init__(self, liste):
        super(aggreg, self).__init__()
        self.liste=liste
    def forward(self, x):
        z=0
        t=0
        for i in self.liste:
            z+=i(x)[0]
            t+=i(x)[1]
        z=z/len(self.liste)
        t=t/len(self.liste)
        return z,t
