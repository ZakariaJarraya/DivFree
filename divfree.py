from platform import java_ver
import torch.nn.functional as F
import matplotlib.animation as animation
from math import *
import torch
import matplotlib.pyplot as plt
import umap.umap_ as umap
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
from functorch import *


upper=torch.triu(torch.ones(2, 2))-torch.diag(torch.ones(2))
lower=-torch.transpose(upper,0,1)


matrix=upper+lower
matrix=matrix.to('cuda')




class scalar(nn.Module):
    def __init__(self,ini,mid,out):
        super(scalar, self).__init__()
        self.layer1=torch.nn.Linear(ini,mid)
        self.layer2=torch.nn.Linear(mid,mid)
        self.layer3=torch.nn.Linear(mid,out)
        self.act=torch.nn.Tanh()
    def forward(self, x):      
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        x = self.layer3(x)
        return x


class dynamic(nn.Module):
    def __init__(self,fonction):
        super(dynamic, self).__init__()
        self.fonction=fonction
    def forward(self, t, x):      
        x=self.fonction(x)
        return x        




class transformation(nn.Module):
    def __init__(self,fonction):
        super(transformation, self).__init__()
        self.fonction=fonction
    def forward(self, t, x): 
        #x=torch.sum(torch.stack([torch.autograd.grad(self.fonction(x[i]),x, create_graph=True)[0] for i in range(len(x))]),dim=0)
        x=torch.autograd.grad(self.fonction(x),x,grad_outputs=torch.ones_like(self.fonction(x)), create_graph=True)[0]
        #x=x.view(-1,2)
        x=x.view(-1,2,1)
        new_matrix=torch.cat([matrix]*x.size()[0])
        new_matrix=new_matrix.view(x.size()[0],2,2)     
        return torch.bmm(new_matrix,x).view(-1,2)
        

import numpy as np
import sklearn
import torch
import sklearn.datasets
from sklearn.model_selection import train_test_split
import os

X, Y = sklearn.datasets.make_moons(n_samples=200, noise=0.1)



X = X.astype("float32")
X = X * 2 + np.array([-1, -0.2])
X = X.astype("float32")
X=torch.tensor(X).view(-1,2)
X=X.to('cuda')
X.requires_grad=True
Y=torch.tensor(Y).view(-1,1)
Y=Y.to('cuda')





def train_epoch(model, Y, X, criterion, opt, batch_size=1000):
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




def train(v,Y,X,num_epochs=30):
    opt = optim.Adam(v.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    e_losses = []
    for e in range(num_epochs):
        e_losses += train_epoch(v, Y, X, criterion, opt)
    #plt.plot(e_losses)  


def acc(network, liste, Y):
    acc=0
    indice=[]
    for i in range(len(Y)):
        if torch.argmax(network(liste[i].view(-1,2))[1])==Y[i]:
                acc+=1
                     
        else:
            pass
            indice.append(i)        
    return  (acc/len(Y))    
    
    
    

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook




class ResNet1(nn.Module):
    def __init__(self, scheme, g, p, data, T, step):
        super(ResNet1, self).__init__()
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
    
def variance_inter(X,Y):
    cond1=Y==1
    cond0=Y==0
    X0=X[cond0.nonzero(as_tuple=True)[0]]
    X1=X[cond1.nonzero(as_tuple=True)[0]]
    X1=X1.view(-1,2)
    X0=X0.view(-1,2)
    X=X.view(-1,2)
    X_mean=torch.mean(X,dim=0)
    X0_mean=torch.mean(X0,dim=0)
    X1_mean=torch.mean(X1,dim=0)
    return torch.cdist(X0_mean.view(-1,2), X_mean.view(-1,2), p=2.0)*(len(X0)/len(X))+torch.cdist(X1_mean.view(-1,2), X_mean.view(-1,2), p=2.0)*(len(X1)/len(X))

def variance_intra(X,Y):
    cond1=Y==1
    cond0=Y==0
    X0=X[cond0.nonzero(as_tuple=True)[0]]
    X1=X[cond1.nonzero(as_tuple=True)[0]]
    X1=X1.view(-1,2)
    X0=X0.view(-1,2)
    X=X.view(-1,2)
    return torch.var(X0)*(len(X0)/len(X))+torch.var(X1)*(len(X1)/len(X))



import math

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


fonction=scalar(2,128,1)

fonction=fonction.to(torch.device("cuda:0"))



class addition(nn.Module):
    def __init__(self,f1,f2):
        super(addition, self).__init__()
        self.f1=f1
        self.f2=f2
    def forward(self, t, x):     
        return self.f1(t,x)+self.f2(t,x)




#fonction.half()
fonction=scalar(2,128,1)

fonction=fonction.to(torch.device("cuda:0"))

F1=transformation(fonction)

F1=F1.to(torch.device("cuda:0"))

##

fonction=scalar(2,128,1)

fonction=fonction.to(torch.device("cuda:0"))

F2=transformation(fonction)

F2=F2.to(torch.device("cuda:0"))


##





X.requires_grad=True
res1=ResNet1('euler',F1,1, X, 1, 0.1)    
res1=res1.to('cuda')
train(res1,Y,X,100)    



fonction=scalar(2,128,2)
fonction=dynamic(fonction)

fonction=fonction.to(torch.device("cuda:0"))

X.requires_grad=True
res2=ResNet1('euler',fonction,1, X, 1., 0.1)    
res2=res2.to('cuda')
train(res2,Y,X,100)  
"""
fig, ax = plt.subplots(figsize=(4,4))
color= ['red' if l == 0. else 'blue' for l in Y]

from matplotlib.animation import FuncAnimation


x,y = np.meshgrid(np.linspace(-6,6,15),np.linspace(-6,6,15))
u=[]
h=[]
for j in torch.flatten(torch.from_numpy(np.transpose(y)[0])):
    for i in torch.flatten(torch.from_numpy(x[0])):
                    a=torch.tensor([[i,j]])
                    a=a.to('cuda')
                    a.requires_grad=True
                    u.append(float(res1.g(1,a.float())[0][0]))
                    h.append(float(res1.g(1,a.float())[0][1]))

u=np.array(u)
h=np.array(h)
u.resize(15,15)
h.resize(15,15)




def animate_reseau(i):
        ax.clear()
        #ax.set_xlim(1*np.min(xmin),1*np.max(xmax))
        #ax.set_ylim(1*np.min(ymin),1*np.max(ymax))
        x_min, x_max = X[:, 0].min()-10.5 , X[:, 0].max()+10.5  
        y_min, y_max = X[:, 1].min()-10.5 , X[:, 1].max()+10.5 
        h = 0.1
        # Generate a grid of points with distance h between them
        xx,yy=np.meshgrid(np.arange(x_min.cpu().detach(), x_max.cpu().detach(), h), np.arange(y_min.cpu().detach(), y_max.cpu().detach(), h))
        grille=torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float().to('cuda')
        grille.requires_grad=True
        Z = torch.argmax(res1(grille)[1],dim=1) 
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        ax.contourf(xx, yy, Z.cpu().detach().numpy(), 8, alpha=.75, cmap='RdBu')
        #pyplot.colorbar(c)
        res1.Identity.register_forward_hook(get_activation('Identity'))
        output = res1(X,t=i)
        ax.scatter(activation['Identity'][:,0].cpu().numpy(),activation['Identity'][:,1].cpu().numpy(),color=color)
        #ax.quiver(x,y,u,h)
        #pyplot.show()



p = FuncAnimation(fig, animate_reseau, interval=100, frames=range(10))
p.save('divfree.gif', writer='pillow')





def animate_reseau(i):
        ax.clear()
        #ax.set_xlim(1*np.min(xmin),1*np.max(xmax))
        #ax.set_ylim(1*np.min(ymin),1*np.max(ymax))
        x_min, x_max = X[:, 0].min()-2. , X[:, 0].max()+2. 
        y_min, y_max = X[:, 1].min()-2. , X[:, 1].max()+2. 
        h = 0.1
        # Generate a grid of points with distance h between them
        xx,yy=np.meshgrid(np.arange(x_min.cpu().detach(), x_max.cpu().detach(), h), np.arange(y_min.cpu().detach(), y_max.cpu().detach(), h))
        grille=torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float().to('cuda')
        grille.requires_grad=True
        V = res1.g(1,grille)
        Vx=V[:,0].reshape(xx.shape).detach().cpu().numpy()
        Vy=V[:,1].reshape(xx.shape).detach().cpu().numpy()
        V_norm = np.sqrt(Vx**2 + Vy**2)
        # Plot the contour and training examples
        c=pyplot.contourf(xx, yy, V_norm, 8, alpha=.75, cmap="YlGn")
        pyplot.colorbar(c)
        pyplot.quiver(xx, yy, Vx, Vy)
        res1.Identity.register_forward_hook(get_activation('Identity'))
        output = res1(X,t=i)
        ax.scatter(activation['Identity'][:,0].cpu().numpy(),activation['Identity'][:,1].cpu().numpy(),color=color)
        #ax.quiver(x,y,u,h)
        pyplot.show()











class density(nn.Module):
    def __init__(self,mean, variance):
        super(density, self).__init__()
        self.mean=mean
        self.variance=variance
    def forward(self, x):     
        inverse=torch.inverse(self.variance)   
        x=x.view(-1,2,1)
        mean=torch.cat([self.mean]*x.size()[0])
        mean=mean.view(x.size()[0],2,1)  
        new_inverse=torch.cat([inverse]*x.size()[0])
        new_inverse=new_inverse.view(x.size()[0],2,2)   
        return (torch.exp(-0.5*torch.bmm(x.view(-1,1,2),torch.bmm(new_inverse,x-mean).view(-1,2,1))).view(-1,1))/(2*pi*torch.det(self.variance))    
    



mean=torch.tensor([0.,0.])
covariance=torch.tensor([1.,0.,0.,1.]).view(2,2)
mean=mean.to(torch.device("cuda:0"))
covariance=covariance.to(torch.device("cuda:0"))


densité1=density(mean,covariance)
densité1=densité1.to(torch.device("cuda:0"))    
    







class KDE(nn.Module):
    def __init__(self,kernel,window,data):
        super(KDE, self).__init__()
        self.kernel=kernel
        self.window=window
        self.data=data
    def forward(self, x):      
        a=0
        for i in self.data:
             a+=kernel((x-i)/self.window)
        a=a/(self.window*len(self.data))     
        return a
    


        
def output1(x):
    return rk4_solver(res1.g, x, 0, 1, 0.1)

        
def output2(x):
    return rk4_solver(res2.g, x, 0, 1, 0.1)

def jac(x,fonction):
    return torch.autograd.functional.jacobian(fonction,x)


def det(x):
    return torch.det(x)    

deter1=[]    
for i in range(len(grille)):
    deter1.append(torch.det(jac(grille[i],output1)))    


c=pyplot.contourf(xx, yy, deter1.detach().cpu().numpy(), 8, alpha=.75, cmap="YlGn")    

"""


torch.mean(torch.norm(torch.sum(torch.stack([torch.autograd.grad(res1(X)[1][:,1][i],X, create_graph=True)[0] for i in range(10)]),dim=0),dim=1))


