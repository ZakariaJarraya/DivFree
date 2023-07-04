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
import geotorch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons, make_circles
import numpy 


%matplotlib inline


def bound_loss(y_pred,ubound,lbound):
    relu=nn.ReLU()
    lower_loss = torch.mean(relu((lbound-y_pred))) 
    upper_loss = torch.mean(relu((y_pred-ubound)))
    custom_loss= lower_loss + upper_loss
    return custom_loss

# define the part of the loss function implementing the PDE
#@tf.function
def grad_loss_eq(gradx, x_batch_train, vf_batch_train):
    relu=nn.ReLU()
    g_loss = torch.mean(relu(((( torch.sum(gradx*vf_batch_train,axis=1) ))))+1.)
    return g_loss

# define the upper bound for the boundary condition
def upperbound ( data ):
    return 10.*data[:,0]**2 + 10.*data[:,1]**2

# define the lower bound for the boundary condition
def lowerbound ( data ):
    return 0.1*data[:,0]**2 + 0.1*data[:,1]**2

# define 0 minimality loss
def zloss ( Lyap, data ):
    relu=nn.ReLU()
    return torch.mean(relu(Lyap(torch.tensor([0.,0.]))-Lyap(data)))+ torch.mean(relu(Lyap(torch.tensor([9.,9.]))-Lyap(data)))





def finalstate ( reseau, data ):
    #relu=nn.ReLU()
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook



    reseau.Identity.register_forward_hook(get_activation('Identity'))

    output = reseau(data,t=200)
    distance1=torch.sum(torch.pow(activation['Identity']-a.c1, 2),dim=-1)
    distance2=torch.sum(torch.pow(activation['Identity']-a.c2, 2),dim=-1)
    distance=torch.min(distance1,distance2)
    
    return torch.mean(distance)



    
class poids(nn.Module):

    def __init__(self, func=a):
        super(poids, self).__init__()
        self.func=func
    def forward(self, x):
        
        #x=torch.sum(torch.pow(x, 2),dim=1)
        distance1=torch.sum(torch.pow(x-self.func.c1, 2),dim=-1)
        distance2=torch.sum(torch.pow(x-self.func.c2, 2),dim=-1)
        distance=torch.min(distance1,distance2)
        
        poids=torch.mean(1/(1+distance))
        
        
        return poids