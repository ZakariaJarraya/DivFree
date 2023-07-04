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



class DATA:





    def __init__(self,distribution,class_size,mean1=torch.tensor([1.,1.]),mean2=torch.tensor([1.,1.]),covariance1=torch.eye(2),covariance2=torch.eye(2)):
  
        self.distribution = distribution      
        self.class_size = class_size
        self.mean1=mean1
        self.mean2=mean2
        self.covariance1=covariance1
        self.covariance2=covariance2



    def load(self):

        if self.distribution == 1:
            
            self.content=make_circles(n_samples=self.class_size, noise=0.1, factor=0.1, random_state=1)

        elif self.distribution == 2:
            
            self.content=make_moons(n_samples=self.class_size, noise=0.3, random_state=0)

        else:

            self.content=load_gaussians(self.mean1, self.mean2, self.covariance1, self.covariance2, self.class_size)




    def prepare(self):
        data,data_choice=self.content, self.distribution
        X=data[0]
        Y=data[1]
        if data_choice==3:
            a=X.numpy()
            b=Y.numpy()
        else:
            a=X
            b=Y
        X0=[]
        X1=[]

        if data_choice==3:
            for i in range(len(a)):
                if b[i].tolist()==[0.0]:
                    X0.append(a[i].tolist())
                else:    
                    X1.append(a[i].tolist())
        else:
            for i in range(len(a)):
                if int(b[i].tolist())==0:
                    X0.append(a[i].tolist())
                else:    
                    X1.append(a[i].tolist())    
            X=torch.from_numpy(X).float()
            Y=torch.from_numpy(Y).float().resize(len(Y),1)        
        X0=torch.tensor(X0)
        X1=torch.tensor(X1)

        return X,Y,X0,X1























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



def prepare(data,data_choice):

    X=data[0]
    Y=data[1]
    if data_choice==3:
        a=X.numpy()
        b=Y.numpy()
    else:
        a=X
        b=Y
    X0=[]
    X1=[]

    if data_choice==3:
        for i in range(len(a)):
            if b[i].tolist()==[0.0]:
                X0.append(a[i].tolist())
            else:    
                X1.append(a[i].tolist())
    else:
        for i in range(len(a)):
            if int(b[i].tolist())==0:
                X0.append(a[i].tolist())
            else:    
                X1.append(a[i].tolist())    
        X=torch.from_numpy(X).float()
        Y=torch.from_numpy(Y).float().resize(len(Y),1)        
    X0=torch.tensor(X0)
    X1=torch.tensor(X1)

    return X,Y,X0,X1
  
