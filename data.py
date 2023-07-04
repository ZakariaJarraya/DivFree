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





def load_circles(class_size):
  
    return  make_circles(n_samples=class_size, noise=0.1, factor=0.1, random_state=1)

def load_moons(class_size):
    
    return  make_moons(n_samples=class_size, noise=0.3, random_state=0)






def samplers(mean1=0*torch.tensor([1.,1.]), mean2=5*torch.tensor([1.,1.]), covariance1=torch.eye(2), covariance2=torch.eye(2)):
    

    gaussian1= torch.distributions.multivariate_normal.MultivariateNormal(mean1, covariance1)

    gaussian2= torch.distributions.multivariate_normal.MultivariateNormal(mean2, covariance2)
    
    return gaussian1, gaussian2




def load_gaussians(mean1, mean2, covariance1, covariance2, class_size):


    sampler1, sampler2 = samplers(mean1, mean2, covariance1, covariance2)
    X1=[]
    X2=[]
    for i in range(class_size):
        X1.append(sampler1.sample())
        X2.append(sampler2.sample())
        
    X1=torch.stack(X1)   
    X2=torch.stack(X2)
    X = torch.cat((X1,X2), dim=0)
    Y1 = torch.zeros(class_size, 1)
    Y2 = torch.ones(class_size, 1)    
    Y = torch.cat([Y1, Y2], dim=0)
    
    return X, Y

  
