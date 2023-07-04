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





if torch.cuda.is_available(): 
 dev = "cuda:0" 
else: 
 dev = "cpu" 
device = torch.device(dev)



upper=torch.triu(torch.ones(2, 2))-torch.diag(torch.ones(2))
lower=-torch.transpose(upper,0,1)


matrix=upper+lower
matrix=matrix.to(torch.device("cuda:0"))
#matrix=matrix.half()



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


class addition(nn.Module):
    def __init__(self,f1,f2,f3):
        super(addition, self).__init__()
        self.f1=f1
        self.f2=f2
        self.f3=f3
    def forward(self, x):     
        return self.f1(x)+self.f2(x)+self.f3(x)


        
def rnew_solver(f, p, y0, t0, tf, h):
    # Runge-Kutta 4th Order Method [By Bottom Science]
    t = t0
    y = y0
    while t <= tf:
        k1 = h * f(t, y)
        k2 = h * f(t + h/2, y + k1/2)
        k3 = h * f(t + h/2, y + k2/2)
        k4 = h * f(t + h, y + k3)
        #y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        z=y.clone()
        t_=t0
        while t_ <= t:
            #r1 = h * -f(t, z)
            #r2 = h * -f(t + h/2, z + r1/2)
            #r3 = h * -f(t + h/2, z + r2/2)
            #r4 = h * -f(t + h, z + r3)
            #z = z + (r1 + 2*r2 + 2*r3 + r4) / 6
            z=z-h*f(t_,z)
            t_ = t_ + h
        y = y + (((k1 + 2*k2 + 2*k3 + k4) / 6) +torch.sum(torch.stack([torch.autograd.grad(p(z[i]),y, create_graph=True)[0] for i in range(len(y))]),dim=0))
        t = t + h
    return y



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




class grad(nn.Module):
    def __init__(self,fonction):
        super(grad, self).__init__()
        self.fonction=fonction

    def forward(self,x):
        return torch.autograd.grad(self.fonction(x),x, grad_outputs=torch.ones_like(self.fonction(x)), create_graph=True)[0] 





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
        




fonction=scalar(2,128,1)

fonction=fonction.to(torch.device("cuda:0"))

#fonction.half()


F=transformation(fonction)

F=F.to(torch.device("cuda:0"))

#F.half()




#X=X.half()



###### Points in the plane ######


mean1=torch.tensor([[3.,0.]])
mean2=torch.tensor([[3.,0.]])
mean3=torch.tensor([[0.,0.]])
covariance1=torch.tensor([[[1.,0.],[0.,1.]]])
covariance2=torch.tensor([[[1.,0.],[0.,1.]]])
covariance3=torch.tensor([[[1.,0.],[0.,1.]]])





dist1=torch.distributions.multivariate_normal.MultivariateNormal(mean1, covariance1)

dist2=torch.distributions.multivariate_normal.MultivariateNormal(mean2, covariance2)

dist3=torch.distributions.multivariate_normal.MultivariateNormal(mean3, covariance3)


gauss1=[dist1.sample() for i in range(50)]

gauss2=[dist2.sample() for i in range(50)]

gauss1=gauss1+gauss2

gauss1=torch.stack(gauss1)
gauss1=gauss1.view(-1,2)


gauss3=[dist3.sample() for i in range(100)]
gauss3=torch.stack(gauss3)
gauss3=gauss3.view(-1,2)


X=torch.cat((gauss1,gauss3), dim=0)

Y=[torch.tensor([1.]) for i in range(100)]+[torch.tensor([0.]) for i in range(100)]
Y=torch.stack(Y)
Y=Y.view(-1,1)





r=torch.randperm(len(X))
X=X[r]
Y=Y[r]


X=X.to(torch.device("cuda:0"))
Y=Y.to(torch.device("cuda:0"))

#X.half()
#Y.half()


color= ['black' if l == 0. else 'green' for l in Y]


#plt.scatter(X.cpu()[:,0],X.cpu()[:,1],color=color)
#plt.show()


fonction=scalar(2,5,1)
fonction.to(torch.device("cuda:0"))
#fonction.half()

mean=torch.tensor([0.,0.])
covariance=torch.tensor([0.5,0.,0.,0.5]).view(2,2)
mean=mean.to(torch.device("cuda:0"))
covariance=covariance.to(torch.device("cuda:0"))


densité1=density(mean,covariance)
densité1=densité1.to(torch.device("cuda:0"))



mean=torch.tensor([6.,0.])
covariance=torch.tensor([1.,0.,0.,1.]).view(2,2)
mean=mean.to(torch.device("cuda:0"))
covariance=covariance.to(torch.device("cuda:0"))

densité2=density(mean,covariance)
densité2=densité2.to(torch.device("cuda:0"))



mean=torch.tensor([-6.,0.])
covariance=torch.tensor([1.,0.,0.,1.]).view(2,2)
mean=mean.to(torch.device("cuda:0"))
covariance=covariance.to(torch.device("cuda:0"))

densité3=density(mean,covariance)
densité3=densité3.to(torch.device("cuda:0"))




add=addition(densité1,densité2,densité3)
add=add.to(torch.device("cuda:0"))








class integration(nn.Module):
    def __init__(self,dynamic):
        super(integration, self).__init__()
        self.dynamic=dynamic
    def forward(self, x):  
            for i in range(1000):
                x=x+0.1*self.dynamic(x)    
            return x

class integration_reverse(nn.Module):
    def __init__(self):
        super(integration_reverse, self).__init__()
    def forward(self, x, scheme, dynamic, T, step): 
        discretization = np.linspace(0, T, round((T)/(step)))    
        if scheme=='euler':  
            for i in discretization:
                x=x+((step)*(-dynamic(x)))    
        return x


from numpy import exp, sqrt, array


def gaussian(d, bw): return exp(-0.5*((d/bw))**2) / (bw * math.sqrt(2*math.pi))



def distance(x, X): return sqrt(((x-X)**2).sum(1))




def grad(X,data):
    # Loop through every point
    #with torch.no_grad():
        Y=X.clone()
        for i in range(Y.size(0)):
            x=Y[i]
            # Find distance from point x to every other point in X
            dist = distance(x, data)
            # Use gaussian to turn into array of weights
            weight = gaussian(dist, 2.5)
            # Weighted sum (see next section for details)
            Y[i] = ((weight[:,None]*Y).sum(0) / weight.sum())-Y[i]
        return Y


def normalize(x,y,res1,add):
	 grad=torch.sum(torch.stack([torch.autograd.grad(add(y[i]),x, create_graph=True)[0] for i in range(len(x))]),dim=0)
	 quotient=torch.norm(res1.g(x),dim=1)/(torch.norm(grad,dim=1)+0.1)
	 quotient=quotient.view(-1,1)
	 return torch.cat((quotient,quotient),dim=1)*grad



class ResNet(nn.Module):
    def __init__(self, scheme, g, p, data, T, step):
        super(ResNet, self).__init__()
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
    def forward(self, x, t=5):
        #data_temp=self.data.clone() 
        #discretization = np.linspace(0, self.T, round((self.T)/(self.step)))    
        if self.scheme=='euler': 
            x=rnew_solver(self.g, self.p, x, 0, self.T, self.step)
            x=self.Identity(x)
            logits=self.finalLayer(x)
            probas = torch.nn.functional.softmax(logits, dim=1)  
        return logits, probas



 







def meanshift_iter(X):
    # Loop through every point
    for i, x in enumerate(X):
        # Find distance from point x to every other point in X
        dist = distance(x, X)
        # Use gaussian to turn into array of weights
        weight = gaussian(dist, 2.5)
        # Weighted sum (see next section for details)
        X[i] = (weight[:,None]*X).sum(0) / weight.sum()
    return X




    

def grad(X,data):
    # Loop through every point
    with torch.no_grad():
        Y=X.clone()
        for i in range(Y.size(0)):
            x=Y[i]
            # Find distance from point x to every other point in X
            dist = distance(x, data)
            # Use gaussian to turn into array of weights
            weight = gaussian(dist, 2.5)
            # Weighted sum (see next section for details)
            Y[i] = ((weight[:,None]*Y).sum(0) / weight.sum())-Y[i]
        return Y




from torch import exp, sqrt


def meanshift(data):
    X = np.copy(data)
    # Loop through a few epochs
    # A full implementation would automatically stop when clusters are stable
    for it in range(30): X = meanshift_iter(X)
    return X
    
    
  


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


X.requires_grad=True
res=ResNet('euler',F,densité1, X, 0.5, 0.05)    
res=res.to('cuda')
train(res,Y,X,1) 



activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


#x,y = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


fig, ax = plt.subplots(figsize=(4,4))
color= ['green' if l == 0. else 'black' for l in Y]

from matplotlib.animation import FuncAnimation

def animate_reseau(i):
        ax.clear()
        #ax.set_xlim(1*np.min(xmin),1*np.max(xmax))
        #ax.set_ylim(1*np.min(ymin),1*np.max(ymax))
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        h = 0.1
        # Generate a grid of points with distance h between them
        xx,yy=np.meshgrid(np.arange(x_min.cpu().detach(), x_max.cpu().detach(), h), np.arange(y_min.cpu().detach(), y_max.cpu().detach(), h))
        grille=torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float().to('cuda')
        grille.requires_grad=True
        Z = torch.argmax(res1(grille)[1],dim=1) 
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        ax.contourf(xx, yy, Z.cpu().detach().numpy(), 8, alpha=.75, cmap=plt.cm.Spectral)
        res1.Identity.register_forward_hook(get_activation('Identity'))
        output = res1(X,t=i)
        ax.scatter(activation['Identity'][:,0].cpu().numpy(),activation['Identity'][:,1].cpu().numpy(),color=color)
        plt.show()

#p = FuncAnimation(fig, animate_reseau, interval=200, frames=range(10))
#p.save('spiral_div.gif', writer='pillow')


res1.Identity.register_forward_hook(get_activation('Identity'))
output = res1(X,t=10)
ax.scatter(activation['Identity'][:,0].cpu().numpy(),activation['Identity'][:,1].cpu().numpy(),color=color)
plt.savefig('ee.png')




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


def limit_points(X,reseau=torch.nn.Identity(2)):
    i=0
    xmin=[]
    xmax=[]
    ymax=[]
    ymin=[]
    while i<=100:
        try:
            reseau.Identity.register_forward_hook(get_activation('Identity'))
            output = reseau(X,t=i)
            xmin.append(np.min(activation['Identity'][:,0].cpu().detach().numpy()))
            xmax.append(np.max(activation['Identity'][:,0].cpu().detach().numpy()))
            ymin.append(np.min(activation['Identity'][:,1].cpu().detach().numpy()))
            ymax.append(np.max(activation['Identity'][:,1].cpu().detach().numpy()))
            i+=1
        except AttributeError:
            output = reseau(X)    
            xmin.append(reseau(X)[:,1].cpu().detach().numpy())
            xmax.append(reseau(X)[:,0].cpu().detach().numpy())
            ymin.append(reseau(X)[:,1].cpu().detach().numpy())
            ymax.append(reseau(X)[:,1].cpu().detach().numpy())
            i+=1
    return np.min(xmin),np.max(xmax),np.min(ymin),np.max(ymax)


def plot_lyapunov(X,lyapunov,numpoints,reseau=torch.nn.Identity(2)):
    # define plotting range and mesh
    xmin,xmax,ymin,ymax = limit_points(X,reseau)
    _x = np.linspace(1.5*xmin, 1.5*xmax, numpoints)
    _y = np.linspace(1.5*ymin, 1.5*ymax, numpoints)
    _X, _Y = np.meshgrid(_x, _y)
    s= _X.shape
    Ze = np.zeros(s)
    Zp = np.zeros(s)
    DT = np.zeros((numpoints**2,2))
    # convert mesh into point vector for which the model can be evaluated
    c = 0
    for i in range(s[0]):
        for j in range(s[1]):
            DT[c,0] = _X[i,j]
            DT[c,1] = _Y[i,j]
            c = c+1
    Ep=torch.stack([lyapunov(i.to('cuda')) for i in torch.tensor(DT).float()])        
    c = 0
    for i in range(s[0]):
        for j in range(s[1]):
            #Ze[i,j] = Ee[c]
            Zp[i,j] = Ep[c]
            c = c+1
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # plot values V
    #ax.plot_surface(_X, _Y, Zp, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    ax.plot_wireframe(_X, _Y, Zp, rstride=7, cstride=7)  




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

#fonction.half()


F=transformation(fonction)

F=F.to(torch.device("cuda:0"))

X.requires_grad=True
res1=ResNet1('euler',F,add, X, 1, 0.1)    
res1=res1.to('cuda')
train(res1,Y,X,100)    



fonction=scalar(2,128,2)
fonction=dynamic(fonction)

fonction=fonction.to(torch.device("cuda:0"))

X.requires_grad=True
res2=ResNet1('euler',fonction,add, X, 1., 0.1)    
res2=res2.to('cuda')
train(res2,Y,X,100)  


def sample(densité):
    statement=False
    while statement==False:
            u=torch.rand(1)
            u.requires_grad=True
            u=u.to('cuda')
            y=(torch.rand(1,2)-0.5)*12
            y=y.view(-1,2)
            y=y.to('cuda')
            y.requires_grad=True
            current_value=0.25*densité2(y)+0.25*densité3(y)+0.5*densité1(y)
            #current_value=densité2(y)
            current_value=current_value.to('cuda')
            statement=(u<(current_value/2))
    return y.view(-1,2)        







def ratio(x,y,res):
    data=torch.cat((x,y),0)
    res.Identity.register_forward_hook(get_activation('Identity'))
    output = res(data,t=5)
    output=activation['Identity']
    numerator=torch.cdist(output[0].view(-1,2),output[1].view(-1,2))
    denumerator=torch.cdist(data[0].view(-1,2),data[1].view(-1,2))
    return numerator/denumerator
    



def lq_distortion(res,res_,densité):
    mean=0
    mean_=0
    for i in range(400):
        a=sample(densité)
        b=sample(densité)
        mean+=torch.pow(ratio(a,b,res),4)
        mean_+=torch.pow(ratio(a,b,res_),4)
    mean=mean/400
    mean_=mean_/400
    mean=torch.pow(mean,0.25)
    mean_=torch.pow(mean_,0.25)
    return mean, mean_



import numpy as np
from numpy import pi
# import matplotlib.pyplot as plt

N = 400
theta = np.sqrt(np.random.rand(N))*2*pi # np.linspace(0,2*pi,100)

r_a = 2*theta + pi
data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
x_a = data_a + np.random.randn(N,2)

r_b = -2*theta - pi
data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
x_b = data_b + np.random.randn(N,2)

res_a = np.append(x_a, np.zeros((N,1)), axis=1)
res_b = np.append(x_b, np.ones((N,1)), axis=1)

res = np.append(res_a, res_b, axis=0)
np.random.shuffle(res)


X=torch.cat((torch.tensor(x_a[:,0]).view(-1,1),torch.tensor(x_a[:,1]).view(-1,1)),dim=1)



X=torch.cat((X,torch.cat((torch.tensor(x_b[:,0]).view(-1,1),torch.tensor(x_b[:,1]).view(-1,1)),dim=1)),dim=0)



from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

n_samples = 300

samples, labels = make_circles(n_samples=n_samples, factor=.3, noise=.05)

bluecircle = samples[labels==0]
redcircle  = samples[labels==1]



X=torch.cat((torch.tensor(bluecircle[:,0]).view(-1,1),torch.tensor(bluecircle[:,1]).view(-1,1)),dim=1)
X=torch.cat((X,torch.cat((torch.tensor(redcircle[:,0]).view(-1,1),torch.tensor(redcircle[:,1]).view(-1,1)),dim=1)),dim=0)
X=X.to('cuda')
X.requires_grad=True
Y=torch.tensor(labels).to('cuda').view(-1,1)

plt.scatter(bluecircle[:, 0], bluecircle[:, 1], c='b', marker='o', s=10)
plt.scatter(redcircle[:, 0], redcircle[:, 1], c='r', marker='+', s=30)
plt.show()


def grad_penalty(res1,X):

        return torch.mean(torch.norm(torch.sum(torch.stack([torch.autograd.grad(res1(X)[1][:,1][i],X, create_graph=True)[0] for i in range(len(res2(X)[1][:,1]))]),dim=0),dim=1))




import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.optimize as opt

def spiral_data(nsamp):
    # Translated from
    # https://github.com/tensorflow/playground/blob/master/src/dataset.ts
    n = int(nsamp / 2)
    spiral_dat = np.zeros((nsamp, 4))
    cntr = 0
    # Generate the positive examples
    deltaT = 0
    for i in range(n):
        r = i / n * 5
        t = 1.75 * i / n * 2 * np.pi + deltaT
        x = r * np.sin(t)
        y = r * np.cos(t)
        spiral_dat[cntr,0] = x
        spiral_dat[cntr,1] = y
        spiral_dat[cntr,2] = 1
        cntr += 1
    # Generate the negative examples
    deltaT = np.pi
    for i in range(n):
        r = i / n * 5
        t = 1.75 * i / n * 2 * np.pi + deltaT
        x = r * np.sin(t)
        y = r * np.cos(t)
        spiral_dat[cntr,0] = x
        spiral_dat[cntr,1] = y
        spiral_dat[cntr,3] = 1
        cntr += 1
    return spiral_dat




X=torch.tensor(spiral_data(3000)[:,:2]).to('cuda')
X.requires_grad=True
X=X.view(-1,2)
X=X.float()
Y=torch.tensor(spiral_data(3000)[:,2]).to('cuda')
Y=Y.view(-1,1)


from divergence_free import *


bsz = 10
ndim = 2

module = nn.Sequential(
        nn.Linear(ndim, 128),
        nn.Tanh(),
        nn.Linear(128, 128),
        nn.Tanh(),
        nn.Linear(128, ndim),
    )

u_fn, params, A_fn = build_divfree_vector_field(module)

f = lambda t,x: u_fn(params, x)


